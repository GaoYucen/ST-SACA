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
import pathlib

# 引用生成数据的模块
import routing_model.gener_data as gd

BASEFILE = pathlib.Path(__file__).parent.resolve()

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------
# 1. 配置模块 (合并 GRC 参数与 ELG 业务参数)
# ------------------------------------------------------------------------

class Config:
    def __init__(self):
        # 业务场景参数
        self.departure_station = np.array([104.06, 30.67])  # 成都火车东站经纬度
        self.num_destinations = 30
        self.bus_capacity = 30
        self.bus_speed = 30
        self.beta_d = 0.15
        self.price_range = [0.0, 1.0]
        self.time_slot_duration = 1.0
        # 波动振幅
        self.demand_fluctuation = 10.0  # 潜在需求波动振幅
        # 波动频率
        self.demand_frequency = 1.0  # 潜在需求波动频率

        # 基础 RL 参数
        self.gamma = 0.99
        self.tau = 0.005
        self.lr = 3e-4
        self.alpha = 0.2
        self.hidden_dim = 256
        self.batch_size = 256
        self.memory_size = 100000

        # 注意力机制参数
        self.attention_heads = 4
        self.attention_layers = 2
        self.embedding_dim = 128

        # 奖励函数参数
        self.lambda_or = 4.0

        # 训练参数
        self.episodes = 100
        self.time_slots_per_episode = 24
        self.target_entropy = -2 * self.num_destinations
        self.min_episodes = 80
        self.max_episodes = 1000
        self.conv_window = 20
        self.conv_delta_ratio = 0.01
        self.conv_std_ratio = 0.05
        self.eval_interval = 10

        # --- GRC 算法特有参数 ---
        self.grc_lambda_goal = 1.0   # Goal Reaching Loss权重
        self.grc_lambda_rl = 1.0     # SAC Loss权重
        self.grc_K = 10              # 采样预测状态的数量
        self.grc_top_k = 2           # 选为 Goal State 的数量
        self.env_model_lr = 1e-3     # 环境模型学习率
        self.scoring_model_lr = 1e-3 # 评分模型学习率


# ------------------------------------------------------------------------
# 2. ELG 路由算法模块 (从 ELG.py 保留)
# ------------------------------------------------------------------------

class ELG_GlobalPolicy(nn.Module):
    """
    Global Policy (Based on POMO/Transformer)
    Learns from global information of the complete graph.
    """
    def __init__(self, embedding_dim, hidden_dim, num_heads, num_layers):
        super(ELG_GlobalPolicy, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Encoder: Transformer Encoder to process global graph
        self.input_proj = nn.Linear(2, embedding_dim) # Coordinates (x, y) -> Embedding
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder Components (Simplified for Inference Baseline)
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.single_head_attention = nn.Linear(embedding_dim, 1)

    def forward_encoder(self, x):
        h = self.input_proj(x)
        h = self.encoder(h) # (Batch, Seq_Len, Embed_Dim)
        return h

    def get_logits(self, current_node_embed, graph_embed):
        q = self.q_proj(current_node_embed)
        k = self.k_proj(graph_embed)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.embedding_dim)
        return scores.squeeze(1)


class ELG_LocalPolicy(nn.Module):
    """
    Local Policy (Transferrable)
    Learns from local topological features (Polar Coordinates) of K-nearest neighbors.
    """
    def __init__(self, embedding_dim, hidden_dim, num_heads):
        super(ELG_LocalPolicy, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_proj = nn.Linear(2, embedding_dim) 
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        self.context_vector = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.output_proj = nn.Linear(embedding_dim, 1)

    def forward(self, local_neighbors_coords):
        local_embed = self.input_proj(local_neighbors_coords) # (Batch, K, Dim)
        batch_size = local_embed.size(0)
        q = self.context_vector.expand(batch_size, -1, -1)
        attn_out, _ = self.multihead_attn(query=q, key=local_embed, value=local_embed)
        scores = torch.bmm(attn_out, local_embed.transpose(1, 2)) / math.sqrt(self.embedding_dim)
        return scores.squeeze(1)


class ELG_TSP_Solver(nn.Module):
    """
    Ensemble Solver for TSP Component.
    """
    def __init__(self, config):
        super(ELG_TSP_Solver, self).__init__()
        self.config = config
        embed_dim = 128
        self.global_policy = ELG_GlobalPolicy(embedding_dim=embed_dim, hidden_dim=128, num_heads=4, num_layers=2)
        self.local_policy = ELG_LocalPolicy(embedding_dim=embed_dim, hidden_dim=128, num_heads=4)
        self.K_neighbors = 10 
        self.xi_penalty = 10.0 

    def cartesian_to_polar(self, current_node, neighbors):
        diff = neighbors - current_node.unsqueeze(1)
        x = diff[:, :, 0]
        y = diff[:, :, 1]
        rho = torch.sqrt(x**2 + y**2 + 1e-8)
        theta = torch.atan2(y, x)
        max_rho = torch.max(rho, dim=1, keepdim=True)[0]
        rho_norm = rho / (max_rho + 1e-6)
        return torch.stack([rho_norm, theta], dim=-1)

    def forward(self, nodes, start_idx=0):
        batch_size, num_nodes, _ = nodes.size()
        global_embed = self.global_policy.forward_encoder(nodes)
        visited_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=nodes.device)
        curr_idx = torch.tensor([start_idx], device=nodes.device)
        tour = [curr_idx]
        visited_mask[0, start_idx] = True
        
        for _ in range(num_nodes - 1):
            curr_node_embed = global_embed[torch.arange(batch_size), curr_idx].unsqueeze(1)
            u_global = self.global_policy.get_logits(curr_node_embed, global_embed)
            
            curr_coords = nodes[torch.arange(batch_size), curr_idx]
            dists = torch.cdist(curr_coords.unsqueeze(1), nodes).squeeze(1)
            dists_masked = dists.clone()
            dists_masked[visited_mask] = float('inf')
            
            K = min(self.K_neighbors, num_nodes - len(tour))
            _, neighbor_indices = torch.topk(dists_masked, K, largest=False)
            
            penalty_mask = torch.ones_like(u_global, dtype=torch.bool)
            penalty_mask.scatter_(1, neighbor_indices, False)
            u_global = u_global - (self.xi_penalty * penalty_mask.float())

            neighbor_coords = torch.gather(nodes, 1, neighbor_indices.unsqueeze(-1).expand(-1, -1, 2))
            polar_inputs = self.cartesian_to_polar(curr_coords, neighbor_coords)
            u_local_partial = self.local_policy(polar_inputs)
            
            u_local = torch.zeros_like(u_global)
            u_local.scatter_(1, neighbor_indices, u_local_partial)
            
            u_ens = u_global + u_local
            u_ens[visited_mask] = -float('inf')
            
            next_node = torch.argmax(u_ens, dim=1)
            visited_mask.scatter_(1, next_node.unsqueeze(1), True)
            tour.append(next_node)
            curr_idx = next_node
            
        return torch.stack(tour, dim=1)


class ELGDispatcherRouter(nn.Module):
    """
    Dispatcher using ELG TSP Solver
    """
    def __init__(self, config, dest_coords, dist_k):
        super().__init__()
        self.config = config
        self.dest_coords = dest_coords
        self.dist_k = dist_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tsp_solver = ELG_TSP_Solver(config).to(self.device)
        self.tsp_solver.eval() # Inference mode

    def dispatch(self, orders, buses):
        if len(orders) == 0 or len(buses) == 0:
            return {}, {}

        bus_routes = {}
        bus_orders = {}
        remaining_orders = orders.copy()
        available_buses = [bid for bid in buses if buses[bid][0] == 0.0]

        for bus_id in available_buses:
            if len(remaining_orders) == 0:
                break
            bus_capacity = int(buses[bus_id][1])

            order_dists = [self.dist_k[d] for d in remaining_orders]
            sorted_indices = np.argsort(order_dists)[:bus_capacity]
            
            assigned_orders = [remaining_orders[i] for i in sorted_indices]
            
            keep_mask = np.ones(len(remaining_orders), dtype=bool)
            keep_mask[sorted_indices] = False
            remaining_orders = [remaining_orders[i] for i in range(len(remaining_orders)) if keep_mask[i]]

            if len(assigned_orders) == 0:
                continue

            unique_dests = list(set(assigned_orders))
            depot_loc = self.config.departure_station
            dests_loc = self.dest_coords[unique_dests]
            
            all_nodes_np = np.vstack([depot_loc, dests_loc])
            all_nodes_tensor = torch.tensor(all_nodes_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                tour_indices = self.tsp_solver(all_nodes_tensor, start_idx=0)
            
            pred_indices = tour_indices.squeeze(0).cpu().numpy().tolist()
            
            route = []
            for idx in pred_indices:
                if idx == 0: continue
                original_dest_id = unique_dests[idx - 1]
                route.append(original_dest_id)

            bus_routes[bus_id] = route
            bus_orders[bus_id] = assigned_orders

        return bus_routes, bus_orders


# ------------------------------------------------------------------------
# 3. 辅助模块 (Environment)
# ------------------------------------------------------------------------

class BusBookingEnv:
    def __init__(self, config):
        self.config = config
        self.init_destinations()
        # 使用 ELG 调度器
        self.dispatcher = ELGDispatcherRouter(
            config, self.dest_coords, self.dist_k
        ).to(device)
        self.reset()

    def init_destinations(self):
        np.random.seed(42)
        self.dest_coords = np.array(gd.readbusstations(gd.config.FILE_PATH))
        
        dep_lon = torch.full((self.config.num_destinations,), float(self.config.departure_station[0]), device=device)
        dep_lat = torch.full((self.config.num_destinations,), float(self.config.departure_station[1]), device=device)
        dest_lons = torch.tensor(self.dest_coords[:, 0], device=device, dtype=torch.float32)
        dest_lats = torch.tensor(self.dest_coords[:, 1], device=device, dtype=torch.float32)
        
        def haversine_torch(lon1, lat1, lon2, lat2):
            R = 6371.0
            dlon = torch.deg2rad(lon2 - lon1)
            dlat = torch.deg2rad(lat2 - lat1)
            a = torch.sin(dlat / 2)**2 + torch.cos(torch.deg2rad(lat1)) * torch.cos(torch.deg2rad(lat2)) * torch.sin(dlon / 2)**2
            c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
            return R * c

        self.dist_k = haversine_torch(dep_lon, dep_lat, dest_lons, dest_lats).detach().cpu().numpy()
        self.dist_max = np.max(self.dist_k)
        self.dist_min = np.min(self.dist_k)

    def reset(self):
        self.time_slots = []
        self.current_p = np.zeros(self.config.num_destinations)
        self.buses = {i: [0.0, self.config.bus_capacity] for i in range(10)}
        self.N_p = np.random.poisson(20, self.config.num_destinations)
        return self.get_state()

    def get_state(self):
        bus_remaining_time = [bus[0] for bus in self.buses.values()]
        state = np.concatenate([self.N_p, bus_remaining_time])
        return torch.FloatTensor(state)

    def demand_function(self, p_k):
        multiplier = (self.dist_max + self.dist_min - self.dist_k) / self.dist_max
        F_pk = 1 - multiplier * (p_k ** 2)
        F_pk = np.clip(F_pk, 0, 1)
        return self.N_p * F_pk

    def supply_function(self, p_k, a_k):
        self.N_s = sum([bus[1] for bus in self.buses.values()])
        if self.N_s == 0:
            return np.zeros_like(p_k)
        multiplier = (self.dist_max + self.dist_min - self.dist_k) / self.dist_max
        F_sk = multiplier * (p_k ** 2)
        F_sk = np.clip(F_sk, 0, 1)
        return self.N_s * a_k * F_sk

    def calculate_reward(self, O_k, D_k, cost_total):
        p_k = self.current_p
        revenue = np.sum(p_k * O_k)
        R_t = revenue - cost_total
        ORR_t = np.sum(O_k) / (np.sum(D_k) + 1e-6)
        return R_t + self.config.lambda_or * ORR_t, R_t, ORR_t

    def step(self, action):
        self.current_p, a_k = action
        a_k = np.clip(a_k, 0, 1)
        a_k = a_k / (np.sum(a_k) + 1e-6)

        D_k = self.demand_function(self.current_p)
        S_k = self.supply_function(self.current_p, a_k)
        O_k = np.minimum(D_k, S_k).astype(int)

        if np.sum(O_k) == 0:
            self.update_bus_state({})
            next_state = self.get_state()
            return next_state, 0.0, False, {"revenue": 0.0, "orr": 0.0, "cost": 0.0, "total_distance": 0.0}

        orders = []
        for k in range(self.config.num_destinations):
            orders.extend([k] * O_k[k])

        # 使用 ELG Dispatcher
        bus_routes, bus_orders = self.dispatcher.dispatch(orders, self.buses)

        cost_total = 0.0
        for route in bus_routes.values():
            dis_i = self.calculate_route_distance(route)
            cost_total += self.config.beta_d * dis_i

        self.update_bus_state(bus_routes)
        self.N_p = np.random.poisson(20 + self.config.demand_fluctuation * np.sin(len(self.time_slots) * self.config.demand_frequency), self.config.num_destinations)
        self.time_slots.append(len(self.time_slots) + 1)

        reward, revenue, orr = self.calculate_reward(O_k, D_k, cost_total)
        next_state = self.get_state()
        done = len(self.time_slots) >= self.config.time_slots_per_episode

        info = {
            "revenue": revenue,
            "orr": orr,
            "cost": cost_total,
            "total_distance": cost_total / self.config.beta_d,
            "orders_accepted": np.sum(O_k)
        }
        return next_state, reward, done, info

    def calculate_route_distance(self, route):
        if len(route) == 0: return 0.0
        path_coords = np.array(
            [self.config.departure_station] + 
            [self.dest_coords[k] for k in route] + 
            [self.config.departure_station]
        )
        lon1, lat1 = path_coords[:-1, 0], path_coords[:-1, 1]
        lon2, lat2 = path_coords[1:, 0], path_coords[1:, 1]
        R = 6371.0
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return np.sum(R * c)

    def update_bus_state(self, bus_routes):
        for bus_id in self.buses:
            self.buses[bus_id][0] = max(0.0, self.buses[bus_id][0] - self.config.time_slot_duration)
        available_buses = [bid for bid in self.buses if self.buses[bid][0] == 0.0]
        for i, (bus_id, route) in enumerate(bus_routes.items()):
            if i >= len(available_buses): break
            bid = available_buses[i]
            dis = self.calculate_route_distance(route)
            time_needed = dis / self.config.bus_speed
            self.buses[bid][0] = time_needed
            self.buses[bid][1] = self.config.bus_capacity

# ------------------------------------------------------------------------
# 4. GRC 核心模块 (模型与 Agent)
# ------------------------------------------------------------------------

class EnvironmentModel(nn.Module):
    """
    环境模型 E: (s_t, a_t) -> s_{t+1}
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(EnvironmentModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        next_state_pred = self.fc3(x)
        return next_state_pred

class ScoringModel(nn.Module):
    """
    评分模型 G: s -> score [0, 1]
    """
    def __init__(self, state_dim, hidden_dim=256):
        super(ScoringModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        score = torch.sigmoid(self.fc_out(x))
        return score

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        return self.norm(attn_output)

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ff_linear1 = nn.Linear(embed_dim, hidden_dim)
        self.ff_linear2 = nn.Linear(hidden_dim, embed_dim)
        self.ff_norm = nn.LayerNorm(embed_dim)
        self.ff_activation = nn.ReLU()

    def forward(self, x):
        x = x + self.attn(x)
        residual = x
        ff_output = self.ff_linear2(self.ff_activation(self.ff_linear1(x)))
        ff_output = self.ff_norm(ff_output)
        return residual + ff_output

class SpatialActor(nn.Module):
    def __init__(self, config, state_dim, action_dim, dest_coords):
        super(SpatialActor, self).__init__()
        self.config = config
        self.register_buffer("dest_coords", torch.tensor(dest_coords, dtype=torch.float32))
        
        self.station_embed_dim = 64
        self.input_proj = nn.Linear(3, self.station_embed_dim)
        
        self.attn_layer = AttentionLayer(
            embed_dim=self.station_embed_dim,
            num_heads=4,
            hidden_dim=128
        )
        
        self.num_buses = state_dim - config.num_destinations
        self.fusion_dim = self.station_embed_dim + self.num_buses
        
        self.fc1 = nn.Linear(self.fusion_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.fc_p = nn.Linear(config.hidden_dim, config.num_destinations)
        self.fc_a = nn.Linear(config.hidden_dim, config.num_destinations)

        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        batch_size = state.size(0)
        demands = state[:, :self.config.num_destinations]
        bus_states = state[:, self.config.num_destinations:]
        
        coords_batch = self.dest_coords.unsqueeze(0).expand(batch_size, -1, -1)
        demands_expanded = demands.unsqueeze(-1)
        station_features = torch.cat([demands_expanded, coords_batch], dim=-1)
        
        x_embed = self.input_proj(station_features)
        x_attn = self.attn_layer(x_embed)
        global_demand_feat = x_attn.mean(dim=1)
        
        fusion_input = torch.cat([global_demand_feat, bus_states], dim=-1)
        x = F.relu(self.fc1(fusion_input))
        x = F.relu(self.fc2(x))
        
        p_mean = torch.sigmoid(self.fc_p(x))
        a_mean = torch.sigmoid(self.fc_a(x))
        action_mean = torch.cat([p_mean, a_mean], dim=-1)
        
        log_std_scalar = torch.tanh(x[:, :1] * 0.1)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std_scalar + 1)
        std = log_std.exp().expand_as(action_mean)
        return action_mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        x_t = normal.rsample()
        
        p_trans = torch.sigmoid(x_t[:, :self.config.num_destinations])
        a_trans = torch.sigmoid(x_t[:, self.config.num_destinations:])
        y_t = torch.cat([p_trans, a_trans], dim=-1)
        
        log_prob = normal.log_prob(x_t).sum(dim=-1)
        correction = torch.log(p_trans * (1 - p_trans) + 1e-6).sum(dim=-1) + \
                     torch.log(a_trans * (1 - a_trans) + 1e-6).sum(dim=-1)
        log_prob -= correction
        return y_t, log_prob, mean

class Critic(nn.Module):
    def __init__(self, config, state_dim, action_dim):
        super(Critic, self).__init__()
        self.config = config
        self.q1_net = self._build_q_net(state_dim, action_dim)
        self.q2_net = self._build_q_net(state_dim, action_dim)

    def _build_q_net(self, state_dim, action_dim):
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1_net(sa), self.q2_net(sa)

    def q1_forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1_net(sa)

class GRC_SAC:
    """
    GRC 算法 Agent (集成到 ELG 框架中)
    """
    def __init__(self, config, state_dim, action_dim, dispatcher, dest_coords):
        self.config = config
        
        # 1. 初始化标准 SAC 组件
        self.actor = SpatialActor(config, state_dim, action_dim, dest_coords).to(device)
        self.critic = Critic(config, state_dim, action_dim).to(device)
        self.target_critic = Critic(config, state_dim, action_dim).to(device)
        self.dispatcher = dispatcher

        # 2. 初始化 GRC 组件
        self.env_model = EnvironmentModel(state_dim, action_dim).to(device)
        self.scoring_model = ScoringModel(state_dim).to(device)

        # 3. 优化器
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=config.lr)
        self.optimizer_env_model = optim.Adam(self.env_model.parameters(), lr=config.env_model_lr)
        self.optimizer_scoring = optim.Adam(self.scoring_model.parameters(), lr=config.scoring_model_lr)

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(param.data)

        self.memory = deque(maxlen=config.memory_size)
        self.target_entropy = getattr(config, "target_entropy", -action_dim)
        self.log_alpha = torch.tensor(math.log(config.alpha), device=device, requires_grad=True)
        self.optimizer_alpha = optim.Adam([self.log_alpha], lr=config.lr)

    def select_action(self, state, deterministic=False, greedy_samples=0):
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
        p = action[:self.config.num_destinations]
        a = action[self.config.num_destinations:]
        return p, a

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_env_model(self, state, action, next_state):
        pred_next_state = self.env_model(state, action)
        loss = F.mse_loss(pred_next_state, next_state)
        self.optimizer_env_model.zero_grad()
        loss.backward()
        self.optimizer_env_model.step()
        return loss.item()

    def update_scoring_model(self, state_batch, reward_batch):
        g = (reward_batch - reward_batch.min()) / (reward_batch.max() - reward_batch.min() + 1e-6)
        half = len(state_batch) // 2
        s1 = state_batch[:half]
        s2 = state_batch[half:2*half]
        g1 = g[:half]
        g2 = g[half:2*half]
        
        score1 = self.scoring_model(s1)
        score2 = self.scoring_model(s2)
        loss = ((g2 - g1) * score1 + (g1 - g2) * score2).mean()
        
        self.optimizer_scoring.zero_grad()
        loss.backward()
        self.optimizer_scoring.step()
        return loss.item()

    def update(self):
        if len(self.memory) < self.config.batch_size:
            return None

        batch = random.sample(self.memory, self.config.batch_size)
        state = torch.cat([s.unsqueeze(0) for s, _, _, _, _ in batch], dim=0).to(device)
        action = torch.tensor(np.stack([a for _, a, _, _, _ in batch]), dtype=torch.float32, device=device)
        reward = torch.FloatTensor([r for _, _, r, _, _ in batch]).unsqueeze(1).to(device)
        next_state = torch.cat([ns.unsqueeze(0) for _, _, _, ns, _ in batch], dim=0).to(device)
        done = torch.FloatTensor([d for _, _, _, _, d in batch]).unsqueeze(1).to(device)

        env_loss = self.update_env_model(state, action, next_state)
        scoring_loss = self.update_scoring_model(state, reward)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.target_critic(next_state, next_action)
            alpha = self.log_alpha.exp()
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob.unsqueeze(1)
            target_q = reward + (1 - done) * self.config.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        self.actor.train()
        self.critic.eval()
        self.env_model.eval()
        self.scoring_model.eval()

        action_pi, log_prob_pi, _ = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, action_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        alpha = self.log_alpha.exp()
        sac_loss = (alpha * log_prob_pi.unsqueeze(1) - q_pi).mean()

        K = self.config.grc_K
        top_k = self.config.grc_top_k
        state_expanded = state.repeat_interleave(K, dim=0)
        actions_k, _, _ = self.actor.sample(state_expanded)
        predicted_states = self.env_model(state_expanded, actions_k)
        scores = self.scoring_model(predicted_states).view(-1, K)
        _, top_indices = torch.topk(scores, top_k, dim=1)
        
        pred_states_reshaped = predicted_states.view(-1, K, predicted_states.size(-1))
        goal_loss = 0.0
        for b in range(state.size(0)):
            goals = pred_states_reshaped[b][top_indices[b]].detach()
            dists = torch.cdist(pred_states_reshaped[b].unsqueeze(0), goals.unsqueeze(0))
            min_dists = dists.min(dim=2)[0]
            goal_loss += min_dists.mean()
        goal_loss = goal_loss / state.size(0)

        total_actor_loss = self.config.grc_lambda_rl * sac_loss + self.config.grc_lambda_goal * goal_loss

        self.optimizer_actor.zero_grad()
        total_actor_loss.backward()
        self.optimizer_actor.step()
        self.critic.train()

        alpha_loss = -(self.log_alpha * (log_prob_pi + self.target_entropy).detach()).mean()
        self.optimizer_alpha.zero_grad()
        alpha_loss.backward()
        self.optimizer_alpha.step()

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

        return critic_loss.item(), sac_loss.item()

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "env_model": self.env_model.state_dict(),
            "scoring_model": self.scoring_model.state_dict(),
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_critic": self.optimizer_critic.state_dict(),
            "optimizer_env": self.optimizer_env_model.state_dict(),
            "optimizer_score": self.optimizer_scoring.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "optimizer_alpha": self.optimizer_alpha.state_dict(),
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.env_model.load_state_dict(state["env_model"])
        self.scoring_model.load_state_dict(state["scoring_model"])
        self.optimizer_actor.load_state_dict(state["optimizer_actor"])
        self.optimizer_critic.load_state_dict(state["optimizer_critic"])
        self.optimizer_env_model.load_state_dict(state["optimizer_env"])
        self.optimizer_scoring.load_state_dict(state["optimizer_score"])
        self.log_alpha.data.copy_(state["log_alpha"].to(device))
        self.optimizer_alpha.load_state_dict(state["optimizer_alpha"])

# ------------------------------------------------------------------------
# 5. 训练与评估流程
# ------------------------------------------------------------------------

def evaluate_policy(agent, config, episodes=5):
    eval_env = BusBookingEnv(config)
    # 对于 ELG，dispatcher 是无参数或独立参数的，这里简单加载 agent 对应的权重（如果有）
    # 但由于 GRC_SAC 的 dispatcher 是作为引用传入的，这里 eval_env 已经自己初始化了新的 dispatcher
    # 且 ELG_TSP_Solver 在当前实现中是 eval 模式的 baseline，不随 RL 训练，所以无需 load_state_dict
    
    metrics = {"avg_reward": [], "revenue": [], "orr": [], "total_distance": [], "cost": []}
    for _ in range(episodes):
        state = eval_env.reset()
        totals = {"reward_sum": 0.0, "revenue": [], "orr": [], "total_distance": [], "cost": []}
        step_count = 0
        for _ in range(config.time_slots_per_episode):
            p, a = agent.select_action(state, greedy_samples=10)
            next_state, reward, done, info = eval_env.step((p, a))
            totals["reward_sum"] += reward
            step_count += 1
            totals["revenue"].append(info["revenue"])
            totals["orr"].append(info["orr"])
            totals["total_distance"].append(info["total_distance"])
            totals["cost"].append(info["cost"])
            state = next_state
            if done: break
        metrics["avg_reward"].append(float(totals["reward_sum"] / max(1, step_count)))
        metrics["revenue"].append(float(np.mean(totals["revenue"])) if totals["revenue"] else 0.0)
        metrics["orr"].append(float(np.mean(totals["orr"])) if totals["orr"] else 0.0)
        metrics["total_distance"].append(float(np.mean(totals["total_distance"])) if totals["total_distance"] else 0.0)
        metrics["cost"].append(float(np.mean(totals["cost"])) if totals["cost"] else 0.0)
    summary = {k: float(np.mean(v)) for k, v in metrics.items()}
    print(f"Evaluation summary over {episodes} episodes: {summary}")
    return summary

def train_grc_elg(config, run_name="GRC_ELG"):
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print(f"Training GRC-ELG on device: {device}")

    env = BusBookingEnv(config)
    state_dim = config.num_destinations + len(env.buses)
    action_dim = 2 * config.num_destinations
    
    # 使用 GRC Agent
    agent = GRC_SAC(config, state_dim, action_dim, env.dispatcher, env.dest_coords)

    log = {
        "total_reward": [],
        "revenue": [],
        "orr": [],
        "total_distance": [],
        "cost": []
    }

    for episode in range(config.max_episodes):
        state = env.reset()
        total_reward = 0.0
        episode_log = {k: [] for k in ["revenue", "orr", "total_distance", "cost"]}
        step_count = 0

        for t in range(config.time_slots_per_episode):
            p, a = agent.select_action(state, deterministic=False)
            action = np.concatenate([p, a])
            next_state, reward, done, info = env.step((p, a))
            total_reward += reward
            step_count += 1
            agent.store_transition(state, action, reward, next_state, done)

            if t % 5 == 0:
                agent.update()

            for k in episode_log.keys():
                episode_log[k].append(info[k])
            state = next_state
            if done: break

        avg_reward_ep = total_reward / max(1, step_count)
        log["total_reward"].append(avg_reward_ep)
        for k in episode_log.keys():
            log[k].append(np.mean(episode_log[k]))

        if (episode + 1) % config.eval_interval == 0:
            print(f"Episode {episode+1}")
            print(f"  Avg Reward: {avg_reward_ep:.2f}")
            print(f"  Avg Revenue: {np.mean(episode_log['revenue']):.2f}")
            print(f"  Avg ORR: {np.mean(episode_log['orr']):.3f}")
            print(f"  Avg Distance: {np.mean(episode_log['total_distance']):.2f}km\n")

        # 早停逻辑
        K = config.conv_window
        if len(log["total_reward"]) >= max(config.min_episodes, 2 * K):
            last_K = np.array(log["total_reward"][-K:])
            prev_K = np.array(log["total_reward"][-2*K:-K])
            if abs(last_K.mean() - prev_K.mean()) < config.conv_delta_ratio * abs(last_K.mean()) and \
               last_K.std() < 0.1 * abs(last_K.mean()):
                print(f"Converged at episode {episode+1}")
                break

    eval_summary = evaluate_policy(agent, config, episodes=5)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = f"{run_name}_{timestamp}"

    log_dir = BASEFILE / "log"
    log_dir.mkdir(exist_ok=True)
    
    import csv
    csv_file = log_dir / f'grc_elg_training_log_{timestamp}.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        keys = list(log.keys())
        writer.writerow(['Episode'] + keys)
        for i in range(len(log[keys[0]])):
            row = [i + 1] + [f"{log[k][i]:.4f}" for k in keys]
            writer.writerow(row)

    plt.figure(figsize=(18, 12))
    
    # 1. Reward
    plt.subplot(2, 3, 1)
    plt.plot(log["total_reward"], label='Reward', color='blue')
    plt.title("Average Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 2. Revenue
    plt.subplot(2, 3, 2)
    plt.plot(log["revenue"], label='Revenue', color='green')
    plt.title("Average Revenue per Time Slot")
    plt.xlabel("Episode")
    plt.ylabel("Revenue")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 3. Cost
    plt.subplot(2, 3, 3)
    plt.plot(log["cost"], label='Cost', color='red')
    plt.title("Average Cost per Time Slot")
    plt.xlabel("Episode")
    plt.ylabel("Cost")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 4. ORR
    plt.subplot(2, 3, 4)
    plt.plot(log["orr"], label='ORR', color='orange')
    plt.title("Average ORR per Time Slot")
    plt.xlabel("Episode")
    plt.ylabel("ORR")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 5. Distance
    plt.subplot(2, 3, 5)
    plt.plot(log["total_distance"], label='Distance', color='purple')
    plt.title("Average Distance per Time Slot")
    plt.xlabel("Episode")
    plt.ylabel("Distance (km)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 6. Revenue vs Cost (Profit Analysis)
    plt.subplot(2, 3, 6)
    plt.plot(log["revenue"], label='Revenue', color='green', alpha=0.7)
    plt.plot(log["cost"], label='Cost', color='red', alpha=0.7)
    plt.title("Revenue vs Cost Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # plot_file = log_dir / f'grc_elg_results_{timestamp}.png'
    # plt.savefig(plot_file, dpi=300)
    # print(f"Training results plot saved to '{plot_file}'")

    # plt.show()

    return agent, env

if __name__ == "__main__":
    config = Config()
    train_grc_elg(config)