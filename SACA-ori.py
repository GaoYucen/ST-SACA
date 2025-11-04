import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from scipy.spatial import distance_matrix
import random
from collections import deque
import matplotlib.pyplot as plt

# 设备配置，使用mps
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    def __init__(self):
        # 业务场景参数（论文Section V）
        self.departure_station = np.array([104.06, 30.67])  # 成都火车站经纬度（模拟）
        self.num_destinations = 30  # 论文中聚类得到的30个目的地
        self.bus_capacity = 30  # 公交容量（论文Section V.B）
        self.bus_speed = 30  # 公交速度（km/h）
        self.beta_d = 0.001  # 单位距离成本（论文NP-hard证明部分）
        self.price_range = [0.0, 1.0]  # 价格范围（论文Section V.B）
        self.time_slot_duration = 1.0  # 时间步时长（小时）

        # SAC算法参数（论文Section IV.B）
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.005  # 目标网络软更新系数
        self.lr = 3e-4  # 学习率
        self.alpha = 0.2  # 熵正则化系数
        self.hidden_dim = 256  # 网络隐藏层维度
        self.batch_size = 256  # 经验回放批次大小
        self.memory_size = 100000  # 经验回放池容量

        # 注意力机制参数（论文Section IV.C）
        self.attention_heads = 4  # 多头注意力头数
        self.attention_layers = 2  # 注意力层数
        self.embedding_dim = 128  # 目的地嵌入维度

        # 奖励函数参数（论文公式8）
        self.lambda_or = 4.0  # ORR权重（论文敏感性分析最优值）

        # 训练参数
        self.episodes = 100  # 训练轮次
        self.time_slots_per_episode = 24  # 每轮次时间步数（模拟1天）
        self.target_entropy = -2 * self.num_destinations
        # 收敛与早停参数
        self.min_episodes = 80           # 最少训练轮次
        self.max_episodes = 1000         # 最多训练轮次（安全上限）
        self.conv_window = 10            # 收敛检测滑动窗口大小K
        self.conv_delta_ratio = 0.01     # 最近均值与前一均值的相对差阈值（1%）
        self.conv_std_ratio = 0.05       # 最近窗口标准差占比阈值（5%）
        self.eval_interval = 10          # 可选：评估打印间隔

class BusBookingEnv:
    def __init__(self, config):
        self.config = config
        self.init_destinations()  # 初始化目的地
        # 关键修复：仅创建一次注意力模块，并移至 MPS
        self.dispatcher = AttentionDispatcherRouter(
            config, self.dest_coords, self.dist_k
        ).to(device)  # 确保模块在 MPS 上
        self.reset()

    def init_destinations(self):
        """生成30个目的地的经纬度（模拟DBSCAN-PAM聚类结果）"""
        np.random.seed(42)
        self.dest_coords = self.config.departure_station + np.random.normal(0, 5, (self.config.num_destinations, 2))
        # 计算每个目的地到出发站的距离（km，基于经纬度粗略转换）
        self.dist_k = distance_matrix([self.config.departure_station], self.dest_coords)[0] * 111  # 1度≈111km
        self.dist_max = np.max(self.dist_k)
        self.dist_min = np.min(self.dist_k)

    def reset(self):
        """重置环境状态（每轮次开始）"""
        self.time_slots = []
        self.current_p = np.zeros(self.config.num_destinations)
        # 初始公交状态：{公交ID: [剩余返回时间, 容量]}
        self.buses = {i: [0.0, self.config.bus_capacity] for i in range(10)}  # 初始10辆公交
        # 初始潜在需求（每目的地的潜在乘客数，模拟夜间波动）
        self.N_p = np.random.poisson(20, self.config.num_destinations)  # 泊松分布模拟需求
        return self.get_state()

    def get_state(self):
        """获取当前状态（论文Section IV.B：状态定义）
        状态向量：[所有目的地潜在需求N_p, 所有公交剩余返回时间]
        """
        bus_remaining_time = [bus[0] for bus in self.buses.values()]
        state = np.concatenate([self.N_p, bus_remaining_time])
        # return torch.FloatTensor(state).to(device)
        return torch.FloatTensor(state)  # 移除 .to(device)

    def demand_function(self, p_k):
        """需求函数F_pk（论文公式3）：F_pk = 1 - (dist_max + dist_min - dist_k)/dist_max * p²"""
        multiplier = (self.dist_max + self.dist_min - self.dist_k) / self.dist_max
        F_pk = 1 - multiplier * (p_k ** 2)
        F_pk = np.clip(F_pk, 0, 1)  # 需求概率不能为负
        return self.N_p * F_pk  # D_k^t = N_pk * F_pk

    def supply_function(self, p_k, a_k):
        """供给函数（论文公式4）：S_k^t = N_s * a_k * F_sk，F_sk = (dist_max + dist_min - dist_k)/dist_max * p²"""
        # 计算当前可用座位数N_s（所有公交剩余容量之和）
        self.N_s = sum([bus[1] for bus in self.buses.values()])
        if self.N_s == 0:
            return np.zeros_like(p_k)
        
        multiplier = (self.dist_max + self.dist_min - self.dist_k) / self.dist_max
        F_sk = multiplier * (p_k ** 2)
        F_sk = np.clip(F_sk, 0, 1)
        return self.N_s * a_k * F_sk  # S_k^t

    def calculate_reward(self, O_k, D_k, cost_total):
        """计算奖励（论文公式8）：Reward = R^t + λ*ORR^t"""
        # 收入R^t = 总收款 - 总成本（论文公式4）
        p_k = self.current_p  # 当前价格向量
        revenue = np.sum(p_k * O_k)
        R_t = revenue - cost_total

        # 订单响应率ORR = 已派单订单数 / 潜在需求数
        ORR_t = np.sum(O_k) / (np.sum(D_k) + 1e-6)  # 避免除零

        return R_t + self.config.lambda_or * ORR_t, R_t, ORR_t

    def step(self, action):
        """环境一步交互（对应论文图3流程）
        action: (P, A) -> P: 价格向量(30,), A: 座位分配向量(30,)（sum(A)=1）
        """
        self.current_p, a_k = action
        a_k = np.clip(a_k, 0, 1)
        a_k = a_k / (np.sum(a_k) + 1e-6)  # 确保sum(A)=1

        # 1. 计算需求D、供给S、已派单订单O（论文公式3-5）
        D_k = self.demand_function(self.current_p)
        S_k = self.supply_function(self.current_p, a_k)
        O_k = np.minimum(D_k, S_k).astype(int)  # 整数订单数

        # 2. 若无可派单订单，直接更新状态
        if np.sum(O_k) == 0:
            self.update_bus_state({})  # 无公交出发
            next_state = self.get_state()
            return next_state, 0.0, False, {"revenue": 0.0, "orr": 0.0, "cost": 0.0, "total_distance": 0.0}

        # 3. 生成订单-目的地映射（O_k个订单对应第k个目的地）
        orders = []
        for k in range(self.config.num_destinations):
            orders.extend([k] * O_k[k])

        # 4. 调用注意力模块进行派单与路径规划
        bus_routes, bus_orders = self.dispatcher.dispatch(orders, self.buses)

        # 5. 计算总成本（论文公式4：cost_i = β_d * dis_i）
        cost_total = 0.0
        for route in bus_routes.values():
            dis_i = self.calculate_route_distance(route)
            cost_total += self.config.beta_d * dis_i

        # 6. 更新公交状态（剩余返回时间、容量）
        self.update_bus_state(bus_routes)

        # 7. 生成下一时间步的潜在需求（模拟需求波动）
        self.N_p = np.random.poisson(20 + np.sin(len(self.time_slots) * 0.5), self.config.num_destinations)
        self.time_slots.append(len(self.time_slots) + 1)

        # 8. 计算奖励与终止状态
        reward, revenue, orr = self.calculate_reward(O_k, D_k, cost_total)
        next_state = self.get_state()
        done = len(self.time_slots) >= self.config.time_slots_per_episode

        # 记录信息
        info = {
            "revenue": revenue,
            "orr": orr,
            "cost": cost_total,
            "total_distance": cost_total / self.config.beta_d,
            "orders_accepted": np.sum(O_k)
        }
        return next_state, reward, done, info

    def calculate_route_distance(self, route):
        """计算路径总距离（出发站→所有目的地→出发站）"""
        if len(route) == 0:
            return 0.0
        # 路径坐标序列：出发站 → 目的地1 → 目的地2 → ... → 出发站
        coords = [self.config.departure_station] + [self.dest_coords[k] for k in route] + [self.config.departure_station]
        return np.sum(distance_matrix(coords, coords)[np.arange(len(coords)-1), np.arange(1, len(coords))]) * 111

    def update_bus_state(self, bus_routes):
        """更新公交剩余返回时间（基于路径长度与速度）"""
        # 首先减少所有公交的剩余返回时间（当前时间步已过去）
        for bus_id in self.buses:
            self.buses[bus_id][0] = max(0.0, self.buses[bus_id][0] - self.config.time_slot_duration)

        # 为新出发的公交分配容量并计算返回时间
        available_buses = [bid for bid in self.buses if self.buses[bid][0] == 0.0]  # 已返回的公交
        for i, (bus_id, route) in enumerate(bus_routes.items()):
            if i >= len(available_buses):
                break  # 无可用公交，停止分配
            bid = available_buses[i]
            # 计算路径耗时（小时）= 距离 / 速度
            dis = self.calculate_route_distance(route)
            time_needed = dis / self.config.bus_speed
            self.buses[bid][0] = time_needed  # 更新剩余返回时间
            self.buses[bid][1] = self.config.bus_capacity  # 重置容量（假设公交返回后可复用）

    
class Actor(nn.Module):
    """策略网络：输入状态→输出价格P与座位分配A（连续动作）"""
    def __init__(self, config, state_dim, action_dim):
        super(Actor, self).__init__()
        self.config = config
        self.action_dim = action_dim  # 动作维度=2*num_destinations（P+A）

        # 状态编码器
        self.fc1 = nn.Linear(state_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # 价格P输出头（约束在[0, 1]）
        self.fc_p = nn.Linear(config.hidden_dim, config.num_destinations)
        # 座位分配A输出头（约束sum(A)=1）
        self.fc_a = nn.Linear(config.hidden_dim, config.num_destinations)

        # 动作噪声参数（用于探索）
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        """输出动作的均值与标准差（用于采样）"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # 价格P：输出均值（通过sigmoid约束在[0,1]）
        p_mean = torch.sigmoid(self.fc_p(x))
        # 座位分配A：输出均值（通过softmax约束sum=1）
        a_mean = F.softmax(self.fc_a(x), dim=-1)
        action_mean = torch.cat([p_mean, a_mean], dim=-1)
        # 动作标准差（所有维度共享）
        log_std_scalar = torch.tanh(x[:, :1] * 0.1)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std_scalar + 1)
        std = log_std.exp().expand_as(action_mean)
        return action_mean, std

    def sample(self, state):
        """采样动作（带重参数化技巧）"""
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样
        p_trans = torch.sigmoid(x_t[:, :self.config.num_destinations])
        a_trans = F.softmax(x_t[:, self.config.num_destinations:], dim=-1)
        y_t = torch.cat([p_trans, a_trans], dim=-1)
        log_prob = normal.log_prob(x_t).sum(dim=-1)
        log_prob -= torch.log(p_trans * (1 - p_trans) + 1e-6).sum(dim=-1)
        log_prob -= torch.log(a_trans + 1e-6).sum(dim=-1)
        return y_t, log_prob, mean


class Critic(nn.Module):
    """评价网络：输入状态+动作→输出Q值"""
    def __init__(self, config, state_dim, action_dim):
        super(Critic, self).__init__()
        self.config = config

        # 双Q网络（避免过估计）
        self.q1_net = self._build_q_net(state_dim, action_dim)
        self.q2_net = self._build_q_net(state_dim, action_dim)

    def _build_q_net(self, state_dim, action_dim):
        """构建单个Q网络"""
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )

    def forward(self, state, action):
        """输出两个Q值"""
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2

    def q1_forward(self, state, action):
        """仅输出Q1（用于目标网络更新）"""
        sa = torch.cat([state, action], dim=-1)
        return self.q1_net(sa)


class SAC:
    def __init__(self, config, state_dim, action_dim, dispatcher):
        self.config = config
        self.actor = Actor(config, state_dim, action_dim).to(device)
        self.critic = Critic(config, state_dim, action_dim).to(device)
        self.target_critic = Critic(config, state_dim, action_dim).to(device)
        self.dispatcher = dispatcher  # 保存注意力模块引用

        # 关键修复：合并 Actor 和 dispatcher 的参数到同一优化器（同属策略网络）
        self.optimizer_actor = optim.Adam(
            list(self.actor.parameters()) + list(self.dispatcher.parameters()),
            lr=config.lr
        )
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=config.lr)

        # 目标网络初始化（不变）
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(param.data)

        self.memory = deque(maxlen=config.memory_size)
        self.target_entropy = getattr(config, "target_entropy", -action_dim)
        self.log_alpha = torch.tensor(
            math.log(config.alpha),
            device=device,
            requires_grad=True,
        )
        self.optimizer_alpha = optim.Adam([self.log_alpha], lr=config.lr)

    def select_action(self, state, deterministic=False, greedy_samples=0):
        """选择动作（训练探索/确定性/贪心采样）"""
        state = state.unsqueeze(0).to(device)
        with torch.no_grad():
            if greedy_samples > 0:
                best_action = None
                best_q = -float("inf")
                for _ in range(greedy_samples):
                    candidate, _, _ = self.actor.sample(state)
                    q_val = self.critic.q1_forward(state, candidate).item()
                    if q_val > best_q:
                        best_q = q_val
                        best_action = candidate
                action_tensor = best_action if best_action is not None else self.actor(state)[0]
            elif deterministic:
                action_tensor = self.actor(state)[0]
            else:
                action_tensor = self.actor.sample(state)[0]
        action = action_tensor.detach().cpu().numpy()[0]
        # 拆分动作：前半部分为P（价格），后半部分为A（座位分配）
        p = action[:self.config.num_destinations]
        a = action[self.config.num_destinations:]
        return p, a

    def store_transition(self, state, action, reward, next_state, done):
        """存储经验到回放池"""
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        """更新SAC网络（每步训练调用）"""
        if len(self.memory) < self.config.batch_size:
            return  # 回放池未满，不更新

        # 1. 采样批次经验
        batch = random.sample(self.memory, self.config.batch_size)
        state = torch.cat([s.unsqueeze(0) for s, _, _, _, _ in batch], dim=0).to(device)
        action = torch.tensor(
            np.stack([a for _, a, _, _, _ in batch]),
            dtype=torch.float32,
            device=device,
        )
        reward = torch.FloatTensor([r for _, _, r, _, _ in batch]).unsqueeze(1).to(device)
        next_state = torch.cat([ns.unsqueeze(0) for _, _, _, ns, _ in batch], dim=0).to(device)
        done = torch.FloatTensor([d for _, _, _, _, d in batch]).unsqueeze(1).to(device)

        # 2. 计算目标Q值
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.target_critic(next_state, next_action)
            alpha = self.alpha
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob.unsqueeze(1)
            target_q = reward + (1 - done) * self.config.gamma * target_q

        # 3. 更新评价网络
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # 4. 更新策略网络（冻结评价网络）
        self.actor.train()
        self.critic.eval()
        action_pi, log_prob_pi, _ = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, action_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        alpha = self.alpha
        actor_loss = (alpha * log_prob_pi.unsqueeze(1) - q_pi).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        self.critic.train()

        alpha_loss = -(self.log_alpha * (log_prob_pi + self.target_entropy).detach()).mean()
        self.optimizer_alpha.zero_grad()
        alpha_loss.backward()
        self.optimizer_alpha.step()

        # 5. 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_critic": self.optimizer_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "optimizer_alpha": self.optimizer_alpha.state_dict(),
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.optimizer_actor.load_state_dict(state["optimizer_actor"])
        self.optimizer_critic.load_state_dict(state["optimizer_critic"])
        self.log_alpha.data.copy_(state["log_alpha"].to(device))
        self.optimizer_alpha.load_state_dict(state["optimizer_alpha"])


class MultiHeadAttention(nn.Module):
    """多头注意力层（论文Section IV.C）"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embed dim must be divisible by num heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 投影为Q, K, V（batch_size, seq_len, embed_dim）
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 注意力输出
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        return self.norm(attn_output)


class AttentionLayer(nn.Module):
    """注意力层（含多头注意力+前馈网络，论文公式9-10）"""
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ff_linear1 = nn.Linear(embed_dim, hidden_dim)
        self.ff_linear2 = nn.Linear(hidden_dim, embed_dim)
        self.ff_norm = nn.LayerNorm(embed_dim)
        self.ff_activation = nn.ReLU()

    def forward(self, x):
        # 多头注意力 + 残差连接
        x = x + self.attn(x)
        # 前馈网络 + 残差连接
        residual = x
        ff_output = self.ff_linear2(self.ff_activation(self.ff_linear1(x)))
        ff_output = self.ff_norm(ff_output)
        return residual + ff_output


class AttentionDispatcherRouter(nn.Module):
    """注意力派单与路径规划模块（论文Section IV.C）"""
    def __init__(self, config, dest_coords, dist_k):
        super().__init__()
        self.config = config
        self.dest_coords = dest_coords
        self.dist_k = dist_k

        # Encoder：目的地嵌入 + 多层注意力
        self.embedding = nn.Linear(2, config.embedding_dim)  # 经纬度→嵌入
        self.encoder = nn.Sequential(
            *[AttentionLayer(
                config.embedding_dim,
                config.attention_heads,
                config.hidden_dim
            ) for _ in range(config.attention_layers)]
        )

        # Decoder：路径生成（基于注意力权重）
        self.decoder_proj = nn.Linear(config.embedding_dim * 2, 1)  # 输入：目的地嵌入+公交状态嵌入

    def embed_destinations(self, dest_ids):
        """嵌入目的地（输入：目的地ID列表，输出：(1, seq_len, embed_dim)）"""
        if len(dest_ids) == 0:
            return torch.zeros(1, 0, self.config.embedding_dim).to(device)
        coords = torch.FloatTensor(self.dest_coords[dest_ids]).to(device)
        embed = self.embedding(coords).unsqueeze(0)  # (1, seq_len, embed_dim)
        return self.encoder(embed)

    def dispatch(self, orders, buses):
        """派单与路径规划
        输入：orders-订单列表（每个元素是目的地ID），buses-公交状态字典
        输出：bus_routes-公交路径字典{bus_id: [dest_id1, dest_id2, ...]}
        """
        if len(orders) == 0 or len(buses) == 0:
            return {}, {}

        # 1. 嵌入所有订单对应的目的地
        unique_dests = list(set(orders))
        dest_embeds = self.embed_destinations(unique_dests)  # (1, num_unique_dest, embed_dim)
        dest_to_idx = {d: i for i, d in enumerate(unique_dests)}

        # 2. 为每个公交分配订单并规划路径
        bus_routes = {}
        bus_orders = {}
        remaining_orders = orders.copy()
        available_buses = [bid for bid in buses if buses[bid][0] == 0.0]  # 可用公交

        for bus_id in available_buses:
            if len(remaining_orders) == 0:
                break
            bus_capacity = buses[bus_id][1]

            # 3. 选择不超过容量的订单（基于距离优先级：优先近距订单减少绕路）
            order_dists = [self.dist_k[d] for d in remaining_orders]
            sorted_indices = np.argsort(order_dists)[:bus_capacity]
            assigned_orders = [remaining_orders[i] for i in sorted_indices]
            # 从剩余订单中移除已分配订单
            remaining_orders = [remaining_orders[i] for i in range(len(remaining_orders)) if i not in sorted_indices]

            # 4. 为已分配订单规划路径（基于注意力权重排序）
            if len(assigned_orders) == 0:
                continue
            # 嵌入已分配订单的目的地
            assigned_dests = list(set(assigned_orders))
            assigned_embeds = self.embed_destinations(assigned_dests)  # (1, num_dest, embed_dim)
            # 计算注意力权重（目的地嵌入之间的相关性）
            attn_weights = torch.matmul(assigned_embeds, assigned_embeds.transpose(1, 2))  # (1, num_dest, num_dest)
            attn_weights = attn_weights.mean(dim=0).detach().cpu().numpy()  # (num_dest, num_dest)

            # 5. 贪心路径规划（从出发站最近的目的地开始，选择相关性最高的下一个）
            start_idx = np.argmin([self.dist_k[d] for d in assigned_dests])
            route = [assigned_dests[start_idx]]
            remaining_dests = [d for i, d in enumerate(assigned_dests) if i != start_idx]

            while remaining_dests:
                current_d = route[-1]
                current_idx = assigned_dests.index(current_d)
                # 选择与当前目的地相关性最高的未访问目的地
                next_idx = np.argmax([attn_weights[current_idx][assigned_dests.index(d)] for d in remaining_dests])
                next_d = remaining_dests.pop(next_idx)
                route.append(next_d)

            # 6. 记录公交路径与订单
            bus_routes[bus_id] = route
            bus_orders[bus_id] = assigned_orders

        return bus_routes, bus_orders

def evaluate_policy(agent, config, episodes=5):
    eval_env = BusBookingEnv(config)
    eval_env.dispatcher.load_state_dict(agent.dispatcher.state_dict())
    eval_env.dispatcher.to(device)
    # 将 total_reward 改为 avg_reward
    metrics = {"avg_reward": [], "revenue": [], "orr": [], "total_distance": []}
    for _ in range(episodes):
        state = eval_env.reset()
        totals = {"reward_sum": 0.0, "revenue": [], "orr": [], "total_distance": []}
        step_count = 0
        for _ in range(config.time_slots_per_episode):
            p, a = agent.select_action(state, greedy_samples=10)
            next_state, reward, done, info = eval_env.step((p, a))
            totals["reward_sum"] += reward
            step_count += 1
            totals["revenue"].append(info["revenue"])
            totals["orr"].append(info["orr"])
            totals["total_distance"].append(info["total_distance"])
            state = next_state
            if done:
                break
        # 记录 Avg Reward
        metrics["avg_reward"].append(float(totals["reward_sum"] / max(1, step_count)))
        metrics["revenue"].append(float(np.mean(totals["revenue"])) if totals["revenue"] else 0.0)
        metrics["orr"].append(float(np.mean(totals["orr"])) if totals["orr"] else 0.0)
        metrics["total_distance"].append(float(np.mean(totals["total_distance"])) if totals["total_distance"] else 0.0)
    summary = {k: float(np.mean(v)) for k, v in metrics.items()}
    print(f"Evaluation summary over {episodes} episodes: {summary}")
    return summary

def train_saca(config):
    # 修复：补充 MPS 兼容的随机种子
    np.random.seed(42)    # numpy 随机种子
    random.seed(42)       # Python 随机种子（经验回放采样用）
    torch.manual_seed(42) # PyTorch 全局种子
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(42)     # MPS 单独种子
        # torch.mps.manual_seed_all(42) # 多 MPS 设备兼容（如有）
    
    # 打印设备信息，确认是否使用 MPS（便于调试）
    print(f"Training on device: {device}")

    # 1. 初始化环境与SAC
    # ...（后续添加种子代码）
    env = BusBookingEnv(config)
    state_dim = config.num_destinations + len(env.buses) # 状态维度=需求数+公交数
    action_dim = 2 * config.num_destinations # 动作维度=价格数+座位分配数
    # 传入 env 的 dispatcher
    sac = SAC(config, state_dim, action_dim, env.dispatcher)

    # 2. 训练记录
    log = {
        "total_reward": [],
        "revenue": [],
        "orr": [],
        "total_distance": []
    }

    # 3. 训练循环
    for episode in range(config.max_episodes):
        state = env.reset()
        total_reward = 0.0
        episode_log = {k: [] for k in ["revenue", "orr", "total_distance"]}
        step_count = 0  # 新增：统计本 episode 的步数

        for t in range(config.time_slots_per_episode):
            # 4. 选择动作（P+A）
            p, a = sac.select_action(state, deterministic=False)  # 训练阶段保持探索
            action = np.concatenate([p, a])

            # 5. 环境交互
            next_state, reward, done, info = env.step((p, a))
            total_reward += reward
            step_count += 1

            # 6. 存储经验
            sac.store_transition(state, action, reward, next_state, done)

            # 7. 更新SAC网络（每5步更新一次，修复解包错误）
            if t % 5 == 0:
                update_result = sac.update()
                if update_result is not None:
                    critic_loss, actor_loss = update_result

            # 8. 记录信息
            for k in episode_log.keys():
                episode_log[k].append(info[k])
            state = next_state

            if done:
                break

        # 9. 记录每轮次平均指标（将 total_reward 转换为 Avg Reward）
        avg_reward_ep = total_reward / max(1, step_count)
        log["total_reward"].append(avg_reward_ep)  # 复用键，语义改为“平均奖励”
        for k in episode_log.keys():
            log[k].append(np.mean(episode_log[k]))

        # 10. 打印训练进度（改为 Avg Reward）
        if (episode + 1) % config.eval_interval == 0:
            print(f"Episode {episode+1}")
            print(f"  Avg Reward: {avg_reward_ep:.2f}")
            print(f"  Avg Revenue: {np.mean(episode_log['revenue']):.2f}")
            print(f"  Avg ORR: {np.mean(episode_log['orr']):.3f}")
            print(f"  Avg Distance: {np.mean(episode_log['total_distance']):.2f}km\n")

        # 早停判定：仍基于 log["total_reward"]（现为平均奖励）
        K = config.conv_window
        n_ep = len(log["total_reward"])
        if n_ep >= max(config.min_episodes, 2 * K):
            last_K = np.array(log["total_reward"][-K:])
            prev_K = np.array(log["total_reward"][-2*K:-K])
            ma_last = last_K.mean()
            ma_prev = prev_K.mean()
            delta = abs(ma_last - ma_prev)
            delta_thr = config.conv_delta_ratio * (abs(ma_last) + 1e-8)
            std_last = last_K.std()
            std_thr = config.conv_std_ratio * (abs(ma_last) + 1e-8)
            if (delta <= delta_thr) and (std_last <= std_thr):
                print(f"[Early Stopping] Converged at episode {episode+1}: "
                      f"MA_prev={ma_prev:.3f}, MA_last={ma_last:.3f}, "
                      f"delta={delta:.3f}<=thr={delta_thr:.3f}, "
                      f"std={std_last:.3f}<=thr={std_thr:.3f}")
                break

    # 11. 评估与绘图
    eval_summary = evaluate_policy(sac, config, episodes=5)
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(log["total_reward"])
    plt.title("Average Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")

    plt.subplot(2, 2, 2)
    plt.plot(log["revenue"])
    plt.title("Average Revenue per Time Slot")
    plt.xlabel("Episode")
    plt.ylabel("Revenue")

    plt.subplot(2, 2, 3)
    plt.plot(log["orr"])
    plt.title("Average ORR per Time Slot")
    plt.xlabel("Episode")
    plt.ylabel("ORR")

    plt.subplot(2, 2, 4)
    plt.plot(log["total_distance"])
    plt.title("Average Total Distance per Time Slot")
    plt.xlabel("Episode")
    plt.ylabel("Distance (km)")

    plt.tight_layout()
    plt.show()

    torch.save(sac.state_dict(), "sac_bus_booking.pth")

    return sac, env


# 启动训练
if __name__ == "__main__":
    config = Config()
    sac_agent, env = train_saca(config)