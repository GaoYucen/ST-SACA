import sys
import torch
import torch.optim as optim
import pathlib
from tqdm import tqdm
from st_saca.paths import ROUTING_CKPT_DIR
from st_saca.routing import am

class TrainConfig:
    def __init__(self):
        # 此时可以放心改回 cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 必须与 AM.py 中的定义或需要的参数一致
        self.embedding_dim = 64
        self.n_heads = 8
        self.n_layers = 3
        
        # 训练参数
        self.batch_size = 64
        self.graph_size = 30  # 对应 num_destinations
        self.epochs = 100
        self.steps_per_epoch = 200
        self.lr = 1e-4
        
        # 路径
        self.model_dir = ROUTING_CKPT_DIR
        self.model_save_path = self.model_dir / "pomo_best_model.pth"
        self.stats_save_path = self.model_dir / "pomo_normalization_stats.pt"

# --- 8倍 Augmentation (不变) ---
def augment_xy_data_by_8_fold(xy_data):
    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    
    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((y, x), dim=2)
    dat3 = torch.cat((x, -y), dim=2)
    dat4 = torch.cat((-x, y), dim=2)
    dat5 = torch.cat((-x, -y), dim=2)
    dat6 = torch.cat((-y, -x), dim=2)
    dat7 = torch.cat((-y, x), dim=2)
    dat8 = torch.cat((y, -x), dim=2)

    return torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)

# --- 数据生成 (不变) ---
class DataGenerator:
    def __init__(self, config):
        self.config = config
    
    def get_batch(self):
        # 随机生成坐标 (0, 1)
        depot_xy = torch.rand(self.config.batch_size, 2, device=self.config.device)
        node_xy = torch.rand(self.config.batch_size, self.config.graph_size, 2, device=self.config.device)
        # 随机生成需求 (1~9 / 30)
        demand = torch.randint(1, 10, (self.config.batch_size, self.config.graph_size), device=self.config.device).float()
        demand = demand / 30.0 
        return depot_xy, node_xy, demand

# --- 修正后的距离计算函数 ---
def calculate_tour_length(depot, node_xy, route_indices):
    """
    计算路径长度
    depot: [Total_Batch, 2]
    node_xy: [Total_Batch, N, 2]
    route_indices: [Total_Batch, N]  <-- AM.py 输出范围是 [1, N]
    """
    batch_size, graph_size, _ = node_xy.size()
    
    # 构造完整坐标列表: Index 0 -> Depot, Index 1..N -> Nodes
    # shape: (B, N+1, 2)
    all_coords = torch.cat([depot.unsqueeze(1), node_xy], dim=1) 
    
    # [关键修正点] 
    # AM.py 输出的 route_indices 已经是 1~N 的形式 (指向 all_coords 的正确位置)
    # 所以直接使用，不需要 +1
    gather_index = route_indices 

    # 构造 Tour 序列: Depot(0) -> Path -> Depot(0)
    zeros = torch.zeros(batch_size, 1, dtype=torch.long, device=depot.device)
    tour_indices = torch.cat([zeros, gather_index, zeros], dim=1)
    
    # Gather 坐标
    tour_coords = all_coords.gather(1, tour_indices.unsqueeze(-1).expand(-1, -1, 2))
    
    # 计算相邻点欧氏距离
    dists = (tour_coords[:, 1:] - tour_coords[:, :-1]).norm(p=2, dim=2)
    return dists.sum(dim=1)

def train():
    config = TrainConfig()
    config.model_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化你的 AM 模型
    model = am.AttentionRouteModel(config.embedding_dim, config.n_heads, config.n_layers).to(config.device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    data_gen = DataGenerator(config)
    
    best_avg_length = float('inf')
    
    print(f"Start POMO Training on {config.device}...")
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        epoch_dist = 0
        
        pbar = tqdm(range(config.steps_per_epoch), desc=f"Epoch {epoch+1}")
        
        for _ in pbar:
            # 1. 获取数据
            depot, loc, demand = data_gen.get_batch()
            
            # 2. 归一化统计量 (模拟真实场景)
            loc_mean = loc.mean(dim=(0, 1), keepdim=True)
            loc_std = loc.std(dim=(0, 1), keepdim=True) + 1e-6
            
            loc_norm = (loc - loc_mean) / loc_std
            depot_norm = (depot - loc_mean.squeeze(1)) / loc_std.squeeze(1)
            
            # 3. POMO Augmentation (8x)
            aug_loc_norm = augment_xy_data_by_8_fold(loc_norm)
            aug_depot_norm = depot_norm.repeat(8, 1)
            aug_demand = demand.repeat(8, 1)
            
            # 原始坐标也增强 (用于算距离 Reward)
            aug_loc_raw = augment_xy_data_by_8_fold(loc)
            aug_depot_raw = augment_xy_data_by_8_fold(depot.unsqueeze(1)).squeeze(1)

            # 4. 模型前向传播
            # AM.py forward 返回: route_indices, log_probs
            output = model(aug_loc_norm, aug_depot_norm, aug_demand)
            
            if isinstance(output, tuple):
                route_indices, log_probs = output
            else:
                # 兼容性处理
                route_indices = output
                log_probs = None 

            # 5. 计算 Reward (Tour Length)
            # 这里的 route_indices 直接使用，不用偏移
            cost = calculate_tour_length(aug_depot_raw, aug_loc_raw, route_indices)
            
            # 6. POMO Loss 计算 (Baseline = Mean of 8 augmentations)
            cost_reshaped = cost.view(8, config.batch_size)
            baseline = cost_reshaped.mean(dim=0) # [B]
            advantage = cost_reshaped - baseline.unsqueeze(0) # [8, B]
            
            if log_probs is not None:
                log_prob_sum = log_probs.sum(dim=1) # [8*B]
                log_prob_reshaped = log_prob_sum.view(8, config.batch_size)
                
                # REINFORCE Loss
                loss = (advantage * log_prob_reshaped).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_dist += cost.mean().item()
            pbar.set_postfix({"AvgDist": f"{cost.mean().item():.3f}"})
            
        avg_dist = epoch_dist / config.steps_per_epoch
        print(f"Epoch {epoch+1} Avg Distance: {avg_dist:.4f}")
        
        # 保存最佳模型
        if avg_dist < best_avg_length:
            best_avg_length = avg_dist
            print(f"Saving best model to {config.model_save_path}...")
            torch.save(model.state_dict(), config.model_save_path)
            
            # 保存统计信息 (Fallback)
            stats = {
                'mean': torch.tensor([0.5, 0.5, 0.5], device=config.device), 
                'std': torch.tensor([0.28, 0.28, 0.28], device=config.device)
            }
            torch.save(stats, config.stats_save_path)

if __name__ == "__main__":
    train()