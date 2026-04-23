import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
try:
    import wandb
except ImportError:  # optional dependency for routing-model training
    wandb = None
import random
import pathlib
from st_saca.paths import DATA_DIR, ROUTING_CKPT_DIR
BASE = pathlib.Path(__file__).resolve().parent

# --- 超参数 (Hyperparameters) ---
EMBEDDING_DIM = 64  # 嵌入维度 best 256
N_HEADS = 8          # 多头注意力的头数
N_LAYERS = 3         # Encoder的层数 best 6
LEARNING_RATE = 1e-4 # 学习率
BATCH_SIZE = 512       # 假设的批量大小
START = [104.06, 30.67] # 起点坐标
NUM_EPOCHS = 5000      # 监督学习的训练轮数
RESTART_EVERY = 2000  # 余弦退火重启周期

class SACA_EncoderLayer(nn.Module):
    """
    SACA的Encoder层，按照论文描述（改进BN为LN）: MHA -> LN -> FF -> LN
    
    """
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        # 论文提到使用BN [cite: 213]，注意BN的维度处理
        # self.bn1 = nn.BatchNorm1d(embed_dim)
        # self.bn2 = nn.BatchNorm1d(embed_dim)

        # 修改为使用 LN
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        
        # 1. Multi-Head Attention + Residual
        attn_out, _ = self.mha(x, x, x)
        x = x + self.dropout1(attn_out)
        
        # 2. Batch Norm 1 [cite: 213]
        # BN1d需要 (B, C, L) 或 (B, C)，我们有 (B, L, C)
        # x_permuted = x.permute(0, 2, 1) # (B, C, L)
        # x_bn = self.bn1(x_permuted)
        # x = x_bn.permute(0, 2, 1) # (B, L, C)

        x = self.ln1(x)

        # 3. Feed Forward + Residual
        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        
        # 4. Batch Norm 2 [cite: 213]
        # x_permuted = x.permute(0, 2, 1)
        # x_bn = self.bn2(x_permuted)
        # x = x_bn.permute(0, 2, 1)
        
        x = self.ln2(x)

        return x

class AttentionRouteModel(nn.Module):
    """
    基于SACA论文的Attention路由模型 
    使用Encoder-Decoder架构
    """
    def __init__(self, embed_dim, n_heads, n_layers):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. 节点嵌入层
        # 将 (lon, lat) 2维坐标和站点下车人数 嵌入到 embed_dim 维
        self.node_embed = nn.Linear(3, embed_dim)
        
        # 2. Encoder
        self.encoder_layers = nn.ModuleList(
            [SACA_EncoderLayer(embed_dim, n_heads) for _ in range(n_layers)]
        )
        
        # 3. Decoder (Pointer Network)
        # 对应论文公式(11)的 W1, W2, v [cite: 218, 221]
        self.W1_embed = nn.Linear(embed_dim, embed_dim, bias=False) # h_k
        self.W2_decode = nn.Linear(embed_dim, embed_dim, bias=False) # h_i^d
        self.v = nn.Linear(embed_dim, 1, bias=False)                 # v^T

    def supervised_loss(
        self,
        label_route_local,          # (B, N_DEST) 监督标签，局部站点顺序 0..N_DEST-1
        station_weights,            # (B, N_DEST) 站点对应的乘客权重
        station_coords,             # (B, N_DEST, 2)
        start_coord,                # (B, 2)
        *,
        return_details=False
    ):
        """
        label_route_local: 来自数据集中“第一行 label”，表示在 station_coords 内的访问顺序
        station_coords:   对应的未排序坐标
        start_coord:      起点坐标
        """
        loc = station_coords
        start = start_coord
        B, N, _ = loc.shape
        weight = station_weights

        # 1. 组合目的地特征 (B, N, 3)
        node_features = torch.cat([loc, weight.unsqueeze(-1)], dim=2)

        # 2. 组合起点特征 (B, 1, 3) (起点权重为0)
        start_weight = torch.zeros(B, 1, 1, device=loc.device, dtype=loc.dtype)
        start_features = torch.cat([start.unsqueeze(1), start_weight], dim=2)
        
        # 3. 拼接所有节点
        all_nodes_features = torch.cat([start_features, node_features], dim=1)   # (B, N+1, 3)

        # 4. 嵌入 & Encoder
        enc = self.node_embed(all_nodes_features)

        for layer in self.encoder_layers:
            enc = layer(enc)

        W1_h = self.W1_embed(enc)                                # (B, N+1, D)
        mask = torch.zeros(B, N+1, dtype=torch.bool, device=loc.device)
        mask[:, 0] = True
        dec_h = enc[:, 0, :]
        batch_idx = torch.arange(B, device=loc.device)

        gt_full = label_route_local.long() + 1                   # 转为 all_nodes 索引，跳过起点
        step_logps = []
        for t in range(N):
            scores = self.v(torch.tanh(W1_h + self.W2_decode(dec_h).unsqueeze(1))).squeeze(-1)
            scores = scores.masked_fill(mask.clone(), -float('inf'))
            log_p = F.log_softmax(scores, dim=1)

            idx_t = gt_full[:, t]
            step_logps.append(log_p[batch_idx, idx_t])

            mask[batch_idx, idx_t] = True
            dec_h = enc[batch_idx, idx_t, :]

        logp_mat = torch.stack(step_logps, dim=1)                # (B, N)
        loss_sup = (-logp_mat.mean()).mean()                 # NLL

        if return_details:
            return loss_sup, {"per_step_logp": logp_mat, "sum_logp": logp_mat.sum(dim=1)}
        return loss_sup

    def forward(self, loc, start, weight):
        """
        前向传播 
        loc: (batch_size, n_dest, 2) - 目的地坐标
        start: (batch_size, 2) - 起点坐标
        weight: (batch_size, n_dest) - 乘客权重 
        """
        B, N_DEST, _ = loc.shape
        
        # --- *** 修改点：构造 3D 特征 (同 supervised_loss) *** ---
        # 1. 组合目的地特征 (B, N_DEST, 3)
        node_features = torch.cat([loc, weight.unsqueeze(-1)], dim=2)

        # 2. 组合起点特征 (B, 1, 3) (起点权重为0)
        start_weight = torch.zeros(B, 1, 1, device=loc.device, dtype=loc.dtype)
        start_features = torch.cat([start.unsqueeze(1), start_weight], dim=2)
        
        # 3. 拼接所有节点 (B, N_DEST + 1, 3)
        all_nodes_features = torch.cat([start_features, node_features], dim=1)
        
        # 2. 嵌入 (B, N+1, D)
        all_nodes_embed = self.node_embed(all_nodes_features) # <-- 使用 3D 特征
        
        # 3. Encoder (B, N+1, D)
        encoded_nodes = all_nodes_embed
        for layer in self.encoder_layers:
            encoded_nodes = layer(encoded_nodes)
            
        # 4. Decoder (Sequential Pointing)
        # (B, N+1, D)
        W1_h = self.W1_embed(encoded_nodes) 
        
        # 初始化
        mask = torch.zeros(B, N_DEST + 1, device=loc.device, dtype=torch.bool)
        route_indices = []
        log_probs = []
        
        # 初始解码器隐藏状态 = 起点(索引0)的嵌入
        decoder_hidden_state = encoded_nodes[:, 0, :]
        mask[:, 0] = True # 遮罩起点
        
        # 循环 N_DEST 次, 选出N个目的地
        for _ in range(N_DEST):
            W2_h_d = self.W2_decode(decoder_hidden_state) # (B, D)
            
            # 广播 W2_h_d 并与 W1_h 相加
            # (B, N+1, D) + (B, 1, D) -> (B, N+1, D)
            scores_raw = torch.tanh(W1_h + W2_h_d.unsqueeze(1))
            
            # 对应公式 v^T * ... 
            scores = self.v(scores_raw).squeeze(-1) # (B, N+1)
            
            # 应用掩码
            scores = scores.masked_fill(mask.clone(), -float('inf'))
            
            # Softmax转为概率
            probs = F.softmax(scores, dim=1)
            log_p = F.log_softmax(scores, dim=1)
            
            # --- 采样 (训练) 或 贪心 (评估) ---
            if self.training:
                # 采样
                next_node_idx = torch.multinomial(probs, 1).squeeze(1) # (B,)
            else:
                # 贪心 (评估模式下)
                next_node_idx = torch.argmax(probs, dim=1) # (B,)
            
            # 收集log_prob
            batch_indices = torch.arange(B, device=loc.device)
            log_probs.append(log_p[batch_indices, next_node_idx])
            
            # 更新
            route_indices.append(next_node_idx)
            mask[batch_indices, next_node_idx] = True
            decoder_hidden_state = encoded_nodes[batch_indices, next_node_idx, :]

        # (B, N_DEST)
        route_indices = torch.stack(route_indices, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        
        # 返回的 route_indices 是在 all_nodes_coords 中的索引
        # (例如 [3, 4, 1, 2], 0是起点)
        return route_indices, log_probs
    
def haversine_torch(lon1, lat1, lon2, lat2, device):
    """
    在Torch中批量计算Haversine距离 (单位: 公里)
    lon1, lat1, lon2, lat2: (batch_size,)
    """
    R = 6371.0 # 地球半径 (km)
    
    lon1, lat1, lon2, lat2 = map(torch.deg2rad, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    
    distance = R * c
    return distance

def calculate_avg_passenger_distance(route_indices, loc, start, weight, device):
    """
    计算平均乘客乘坐距离 
    route_indices: (B, N_DEST) - 模型输出的路由 (在all_nodes中的索引)
    loc: (B, N_DEST, 2)
    start: (B, 2)
    weight: (B, N_DEST)
    """
    B, N_DEST, _ = loc.shape
    
    # (B, N_DEST + 1, 2)
    all_nodes_coords = torch.cat([start.unsqueeze(1), loc], dim=1)
    
    # route_indices_full 包含起点 (B, N_DEST + 1)
    start_node = torch.zeros(B, 1, dtype=torch.long, device=device)
    route_indices_full = torch.cat([start_node, route_indices], dim=1)
    
    total_passenger_distance = torch.zeros(B, device=device)
    
    # ----------------------------------------------------
    # 高效的矢量化计算 (替代循环)
    # 1. 获取排序后的坐标 (B, N+1, 2)
    # (B, N+1, 1) -> (B, N+1, 2)
    idx_expanded = route_indices_full.unsqueeze(-1).expand(-1, -1, 2)
    sorted_coords = torch.gather(all_nodes_coords, 1, idx_expanded)
    
    # 2. 计算每段路程 (B, N)
    # A = sorted_coords[:, :-1] (B, N, 2)
    # B = sorted_coords[:, 1:] (B, N, 2)
    segment_dists = haversine_torch(
        sorted_coords[:, :-1, 0].reshape(-1), # lon1
        sorted_coords[:, :-1, 1].reshape(-1), # lat1
        sorted_coords[:, 1:, 0].reshape(-1),  # lon2
        sorted_coords[:, 1:, 1].reshape(-1),  # lat2
        device
    ).reshape(B, N_DEST)
    
    # 3. 计算累积路程 (B, N)
    # [dist1, dist1+dist2, dist1+dist2+dist3, ...]
    cumulative_bus_dist = torch.cumsum(segment_dists, dim=1)
    
    # 4. 获取排序后的乘客权重
    # route_indices 是 all_nodes 中的索引, weight 对应 loc
    # 所以索引要 -1 (因为 0 是起点)
    dest_indices = route_indices[:, :] - 1 # (B, N_DEST)
    # (B, N_DEST, 1)
    idx_expanded = dest_indices.unsqueeze(-1).expand(-1, -1, 1)
    # weight是 (B, N), gather需要 (B, N, 1)
    sorted_weights = torch.gather(weight.unsqueeze(-1), 1, idx_expanded).squeeze(-1)
    
    # 5. 计算总乘客距离
    # (B, N) * (B, N) -> (B, N)
    total_passenger_distance = (sorted_weights * cumulative_bus_dist).sum(dim=1)
    
    # 6. 计算平均
    total_passengers = weight.sum(dim=1)
    
    # 避免除以0
    avg_passenger_dist = total_passenger_distance / (total_passengers + 1e-6)
    
    return avg_passenger_dist

# 数据加载以及批次生成函数
def load_supervised_data(file_paths):
    """
    读取supervised_dataset_i_stations.json的数据其中i为站点数量，每个文件包含1024个样本
    数据加载类实现数据读取与批次生成
    """
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            samples = json.load(f)
            samples = [{
                'loc': np.array(sample['station_coords'], dtype=np.float32),
                'weight': np.array(sample['passengers'], dtype=np.float32),
                'label': np.array(sample['local_optimal_order'], dtype=np.float32),
                'start': np.array(START, dtype=np.float32),
                'label_cost': sample['avg_cost_per_passenger']
            } for sample in samples]
            data.append(samples)
    return data

def compute_normalization_stats(train_data_groups):
    """
    仅遍历训练集，计算 (lon, lat, weight) 三个特征的
    全局均值和标准差。
    """
    all_features = []
    
    for group in train_data_groups:
        for sample in group:
            # (lon, lat) 特征
            loc_features = sample['loc']  # (N, 2)
            # (weight) 特征
            weight_features = sample['weight'][:, np.newaxis]  # (N, 1)
            
            # 组合成 (lon, lat, weight)
            node_feats = np.concatenate([loc_features, weight_features], axis=1)  # (N, 3)
            all_features.append(node_feats)
            
            # 也要包含起点 (lon, lat, 0.0)
            start_feat = np.array(
                [[sample['start'][0], sample['start'][1], 0.0]], 
                dtype=np.float32
            )  # (1, 3)
            all_features.append(start_feat)
            
    # 将所有数据堆叠成一个大矩阵
    full_dataset_features = np.concatenate(all_features, axis=0)  # (Total_Nodes, 3)
    
    # 计算均值和标准差
    mean = np.mean(full_dataset_features, axis=0)
    std = np.std(full_dataset_features, axis=0)
    
    # 防止 std 为 0 (例如，如果所有 'weight' 都一样)
    std = np.where(std == 0, 1e-6, std)
    
    stats = {
        'mean': torch.tensor(mean, dtype=torch.float32),
        'std': torch.tensor(std, dtype=torch.float32)
    }
    return stats

def getbatch_supervised(data_groups, batch_size, device, stats):
    """
    从按文件分组的数据中随机取一组，再随机采样一个批次。
    """
    file_idx = random.randrange(len(data_groups))        # 选文件索引
    dataset = data_groups[file_idx]                      # 该文件全部样本
    replace = len(dataset) < batch_size
    samples = np.random.choice(dataset, batch_size, replace=replace)

    # 准备空的列表来收集增强后的数据
    loc_list = []
    weight_list = []
    label_list = []
    start_list = []

    for s in samples:
        loc = s['loc']       # (N, 2)
        weight = s['weight'] # (N,)
        label = s['label']   # (N,)
        N = loc.shape[0]

        # 1. 创建随机置换
        perm = np.random.permutation(N)
        
        # 2. 打乱输入
        loc_shuffled = loc[perm]
        weight_shuffled = weight[perm]

        # 3. 修正标签
        # 我们需要一个逆置换: inv_perm[新索引] = 旧索引
        # 不，我们需要: inv_perm[旧索引] = 新索引
        inv_perm = np.empty(N, dtype=np.int64)
        inv_perm[perm] = np.arange(N)
        
        # label是旧索引的顺序，将其转换为新索引的顺序
        # 例如: label[0] = 3 (意味着第一个去旧索引3)
        # 新索引是 inv_perm[3]
        label_shuffled = inv_perm[label.astype(np.int64)]

        # 4. 收集
        loc_list.append(loc_shuffled)
        weight_list.append(weight_shuffled)
        label_list.append(label_shuffled)
        start_list.append(s['start'])

    # 批量堆叠
    loc_batch   = torch.as_tensor(np.stack(loc_list, axis=0), device=device)
    weight_batch = torch.as_tensor(np.stack(weight_list, axis=0), device=device)
    start_batch = torch.as_tensor(np.stack(start_list, axis=0), device=device)
    label_batch = torch.as_tensor(np.stack(label_list, axis=0), device=device, dtype=torch.long)



    mean = stats['mean'].to(device)  # (3,)
    std = stats['std'].to(device)    # (3,)

    # (B, N, 2) - (1, 1, 2) / (1, 1, 2)
    loc_batch = (loc_batch - mean[None, None, :2]) / std[None, None, :2]
    # (B, N) - (1,) / (1,)
    weight_batch = (weight_batch - mean[2]) / std[2]
    # (B, 2) - (1, 2) / (1, 2)
    start_batch = (start_batch - mean[None, :2]) / std[None, :2]
    
    return loc_batch, start_batch, weight_batch, label_batch # 返回 weight_batch

@torch.no_grad()
def evaluate_supervised(model, data_groups, batch_size, device, stats):
    """按组遍历验证集，汇总序列级损失与token级NLL"""
    model.eval()
    total_seq_loss = 0.0
    total_seqs = 0
    token_nll_sum = 0.0
    token_cnt = 0

    mean = stats['mean'].to(device)  # (3,)
    std = stats['std'].to(device)    # (3,)

    for dataset in data_groups:
        for i in range(0, len(dataset), batch_size):
            samples = dataset[i:i+batch_size]
            loc_np = np.stack([s['loc'] for s in samples], axis=0)

            weight_np = np.stack([s['weight'] for s in samples], axis=0) 
            start_np = np.stack([s['start'] for s in samples], axis=0)
            label_np = np.stack([s['label'] for s in samples], axis=0).astype(np.int64)

            loc   = torch.as_tensor(loc_np, device=device)
        
            weight = torch.as_tensor(weight_np, device=device)
            start = torch.as_tensor(start_np, device=device)
            label = torch.as_tensor(label_np, device=device)

            # 归一化
            loc = (loc - mean[None, None, :2]) / std[None, None, :2]
            weight = (weight - mean[2]) / std[2]
            start = (start - mean[None, :2]) / std[None, :2]

            loss, details = model.supervised_loss(
                label_route_local=label, 
                station_coords=loc, 
                station_weights=weight, 
                start_coord=start, 
                return_details=True
            )
            total_seq_loss += loss.item() * len(samples)
            total_seqs += len(samples)

            # token 级 NLL（跨 batch、跨步累加）
            token_nll_sum += (-details["per_step_logp"]).sum().item()
            token_cnt += loc.shape[0] * loc.shape[1]

    avg_seq_loss = total_seq_loss / max(1, total_seqs)
    avg_token_nll = token_nll_sum / max(1, token_cnt)
    return avg_seq_loss, avg_token_nll

def split_grouped_data(data_groups, val_ratio=0.1, seed=42):
    """对每个文件内做切分，保持同N"""
    rng = np.random.default_rng(seed)
    train_groups, val_groups = [], []
    for dataset in data_groups:
        idx = np.arange(len(dataset))
        rng.shuffle(idx)
        val_size = max(1, int(len(dataset) * val_ratio))
        val_idx = idx[:val_size]
        train_idx = idx[val_size:]
        train_groups.append([dataset[i] for i in train_idx])
        val_groups.append([dataset[i] for i in val_idx])
    return train_groups, val_groups

@torch.no_grad()
def evaluate_model_cost(model, data_groups, batch_size, device, stats):
    """
    遍历所有数据，计算模型预测成本与标签成本的差异。
    """
    model.eval()
    
    # 将 stats 移动到 device
    mean = stats['mean'].to(device)
    std = stats['std'].to(device)

    all_pred_costs = []
    all_label_costs = []

    print("开始评估成本...")
    
    for dataset in data_groups:
        # 在每个 N 不同的组内分批
        for i in range(0, len(dataset), batch_size):
            samples = dataset[i:i+batch_size]
            
            # --- 1. 准备批次数据 (原始 & 标签) ---
            loc_list = [s['loc'] for s in samples]
            weight_list = [s['weight'] for s in samples]
            start_list = [s['start'] for s in samples]
            
            # 标签成本 (我们要对比的目标)
            label_cost_np = np.array([s['label_cost'] for s in samples])
            
            # 转换为张量
            loc_orig = torch.tensor(np.stack(loc_list), device=device, dtype=torch.float32)
            weight_orig = torch.tensor(np.stack(weight_list), device=device, dtype=torch.float32)
            start_orig = torch.tensor(np.stack(start_list), device=device, dtype=torch.float32)
            label_cost_tensor = torch.tensor(label_cost_np, device=device, dtype=torch.float32)

            # --- 2. 准备归一化数据 (用于模型输入) ---
            loc_norm = (loc_orig - mean[None, None, :2]) / std[None, None, :2]
            weight_norm = (weight_orig - mean[2]) / std[2]
            start_norm = (start_orig - mean[None, :2]) / std[None, :2]

            # --- 3. 模型预测 ---
            route_indices, _ = model(loc_norm, start_norm, weight_norm)
            
            # --- 4. 计算预测成本 (必须使用 *原始* 坐标) ---
            pred_cost_tensor = calculate_avg_passenger_distance(
                route_indices, loc_orig, start_orig, weight_orig, device
            )
            
            # --- 5. 收集结果 ---
            all_pred_costs.append(pred_cost_tensor.cpu())
            all_label_costs.append(label_cost_tensor.cpu())
            
    print("评估完成，正在计算指标...")

    # 将所有批次的结果合并
    final_pred_costs = torch.cat(all_pred_costs)
    final_label_costs = torch.cat(all_label_costs)
    
    # --- 6. 计算最终指标 ---
    
    # a. 数值误差 (Predicted - Label)
    absolute_error = final_pred_costs - final_label_costs
    
    # b. 近似比 (Optimality Gap) (Predicted / Label)
    # 我们加上 1e-6 以避免除以 0 (如果某个标签成本为0)
    approximation_ratio = final_pred_costs / (final_label_costs + 1e-6)

    # --- 7. 打印报告 ---
    print("\n--- 模型成本评估报告 ---")
    print(f"总样本数: {len(final_pred_costs)}")
    
    print("\n[ 平均数值误差 (Predicted - Label) ]")
    print(f"  平均误差: {absolute_error.mean().item():.4f} km")
    print(f"  (意味着模型预测的成本平均比标签高/低 {absolute_error.mean().item():.4f} km)")
    
    print("\n[ 平均近似比 (Predicted / Label) ]")
    print(f"  平均比率: {approximation_ratio.mean().item():.4f}")
    print(f"  (意味着模型预测的成本平均是标签成本的 {approximation_ratio.mean().item() * 100:.2f} %)")
    print(f"  (即 Optimality Gap 约为 {(approximation_ratio.mean().item() - 1) * 100:.2f} %)")
    print("----------------------------\n")

    return absolute_error.mean().item(), approximation_ratio.mean().item()

# --- 训练脚本 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_val_loss = float('inf')  # 初始化一个无限大的“最佳损失”
    ROUTING_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_PATH = ROUTING_CKPT_DIR / "best_model.pth"
    STATS_PATH = ROUTING_CKPT_DIR / "normalization_stats.pt"
    
    # 1. *** 修改点：增加正则化 ***
    model = AttentionRouteModel(EMBEDDING_DIM, N_HEADS, N_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # 增加L2正则化
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_mult=1,
        T_0=RESTART_EVERY,
        eta_min=1e-9  # 确保它不会降到0
    )

    # 2. 准备数据 
    file_paths = [DATA_DIR / "dataset_traincenter" / f"supervised_dataset_{n}_stations.json"
                  for n in range(5, 11)]
    data = load_supervised_data(file_paths)
    train_data, val_data = split_grouped_data(data, val_ratio=0.1, seed=42)

    # 3. 计算步数 
    STEPS_PER_EPOCH = sum(len(g) // BATCH_SIZE for g in train_data)

    # 4. 初始化 wandb
    if wandb is not None:
        wandb.init(
            project="AM",
            config=dict(
                embedding_dim=EMBEDDING_DIM, n_heads=N_HEADS, n_layers=N_LAYERS,
                lr=LEARNING_RATE, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
                steps_per_epoch=STEPS_PER_EPOCH
            )
        )
        wandb.watch(model, log="gradients", log_freq=100)

    # 归一化
    stats = compute_normalization_stats(train_data)
    
    # 5. 训练循环
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_seq_loss = 0.0
        token_nll_sum = 0.0
        token_cnt = 0

        for _ in range(STEPS_PER_EPOCH):
            loc, start, weight, label = getbatch_supervised(train_data, BATCH_SIZE, device, stats)
            
            loss, details = model.supervised_loss(
                label_route_local=label, 
                station_coords=loc, 
                station_weights=weight, # 传入
                start_coord=start, 
                return_details=True
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_seq_loss += loss.item()
            token_nll_sum += (-details["per_step_logp"]).sum().item()
            token_cnt += loc.shape[0] * loc.shape[1]

        scheduler.step()

        train_seq_loss = running_seq_loss / max(1, STEPS_PER_EPOCH)
        train_token_nll = token_nll_sum / max(1, token_cnt)

        # 验证
        val_seq_loss, val_token_nll = evaluate_supervised(model, val_data, BATCH_SIZE, device, stats)

        # 保存最佳模型
        if val_token_nll < best_val_loss:
            best_val_loss = val_token_nll  # 更新最佳损失
            torch.save(model.state_dict(), SAVE_PATH)
            torch.save(stats, STATS_PATH)
            best_epoch = epoch # 记录最佳epoch
            # print(f"Saved best model at epoch {epoch+1} with ValTokNLL: {val_token_nll:.4f}")

        # 记录到 wandb
        if wandb is not None:
            wandb.log({
                "train/loss_seq": train_seq_loss,
                "train/nll_token": train_token_nll,
                "val/loss_seq": val_seq_loss,
                "val/nll_token": val_token_nll,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch + 1
            })
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                f"TrainSeqLoss: {train_seq_loss:.4f} | TrainTokNLL: {train_token_nll:.4f} | "
                f"ValSeqLoss: {val_seq_loss:.4f} | ValTokNLL: {val_token_nll:.4f}")
    print(f"Training complete. Best validation token NLL: {best_val_loss:.4f} at epoch {best_epoch+1}")

    if wandb is not None:
        wandb.finish()
