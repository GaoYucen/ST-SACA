import torch
import numpy as np
from st_saca.agents import st_saca as SACA
import pathlib
import csv
import random
from st_saca.paths import ensure_output_dir

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GreedyDispatcher:
    """
    【消融模块严格版】
    1. Dispatch (派单): 完全复用 SACA 逻辑，选取离出发站最近的 k 个订单。
    2. Routing (路由): 对这 k 个订单使用贪心算法 (最近邻) 排序，而非注意力模型。
    """
    def __init__(self, dest_coords, departure_station, dist_k):
        # 注意：这里多传入了一个 dist_k，这是 SACA 环境中预计算好的【出发站到各点距离】
        self.dest_coords = dest_coords
        self.departure_station = departure_station
        self.dist_k = dist_k 

    def _haversine_distance(self, loc1, locs2):
        """计算球面距离: loc1(2,) -> locs2(N,2)"""
        R = 6371.0
        # 维度适配
        if loc1.ndim == 1:
            loc1 = loc1.reshape(1, 2)
        
        lon1, lat1 = loc1[:, 0], loc1[:, 1]
        lon2, lat2 = locs2[:, 0], locs2[:, 1]
        
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def dispatch(self, orders, buses):
        if not orders or not buses:
            return {}, {}

        bus_routes = {}
        bus_orders = {}
        remaining_orders = orders.copy()
        available_buses = [bid for bid in buses if buses[bid][0] == 0.0]

        for bus_id in available_buses:
            if not remaining_orders:
                break
            
            capacity = int(buses[bus_id][1])

            # ==========================================
            # Phase 1: 派单 (Dispatch) - 严格对齐 SACA
            # ==========================================
            # 策略：总是优先选择离【出发站】最近的订单上车
            
            # 获取剩余订单到出发站的距离
            # self.dist_k 是预计算好的数组 (num_destinations,)
            current_order_dists = [self.dist_k[d] for d in remaining_orders]
            
            # 按距离排序，取前 capacity 个
            # argsort 返回的是 current_order_dists 的索引，即 remaining_orders 的索引
            sorted_indices = np.argsort(current_order_dists)[:capacity]
            
            # 锁定这辆车的乘客 (Assigned Orders)
            assigned_orders = [remaining_orders[i] for i in sorted_indices]
            
            # 从总池子中移除这些订单
            # 技巧：从后往前删，防止索引偏移，或者用列表推导式重建
            # 这里为了安全，使用 mask 移除
            keep_mask = np.ones(len(remaining_orders), dtype=bool)
            keep_mask[sorted_indices] = False
            remaining_orders = [remaining_orders[i] for i in range(len(remaining_orders)) if keep_mask[i]]

            if not assigned_orders:
                continue

            # ==========================================
            # Phase 2: 路由 (Routing) - 贪心策略
            # ==========================================
            # 策略：对 assigned_orders 进行 TSP 路径规划 (最近邻法)
            
            route_sequence = []
            
            # 待规划的站点池 (可能有重复站点，因为多个订单可能去同一个地方)
            # 贪心算法通常处理的是唯一坐标点，然后处理下车人数
            # 但为了简单，我们把每个订单看作一个必须访问的任务
            
            # 复制一份待访问列表
            to_visit = assigned_orders.copy()
            current_loc = self.departure_station
            
            while to_visit:
                # 1. 计算当前位置到所有【待访问点】的距离
                # 注意：to_visit 里存的是 dest_id
                target_coords = self.dest_coords[to_visit] # (M, 2)
                dists = self._haversine_distance(current_loc, target_coords)
                
                # 2. 找最近的一个
                nearest_idx = np.argmin(dists)
                next_dest_id = to_visit[nearest_idx]
                
                # 3. 加入路径
                route_sequence.append(next_dest_id)
                
                # 4. 移动当前位置
                current_loc = self.dest_coords[next_dest_id]
                
                # 5. 移除已访问
                to_visit.pop(nearest_idx)

            bus_routes[bus_id] = route_sequence
            bus_orders[bus_id] = assigned_orders

        return bus_routes, bus_orders

class AblationEnv(SACA.BusBookingEnv):
    """
    继承原始环境，但覆盖初始化方法，
    强制使用 GreedyDispatcher 替代 AttentionDispatcherRouter
    """
    def __init__(self, config):
        # 1. 调用父类初始化
        super().__init__(config)
        
        # 2. 【核心修改】覆盖调度器
        print(">>> [Ablation Info] Loading GreedyDispatcher (Haversine) instead of AttentionRouter...")
        self.dispatcher = GreedyDispatcher(self.dest_coords, self.config.departure_station, self.dist_k)
        
        # 确保重置
        self.reset()

def train_ablation_wo_route():
    # 1. 配置
    config = SACA.Config()
    config.lambda_or = 0.1
    config.lr = 3e-3  
    run_name = "wo_AttnRoute"
    
    # 2. 初始化消融环境
    env = AblationEnv(config)
    
    # 3. 初始化 SAC Agent
    state_dim = config.num_destinations + len(env.buses)
    action_dim = 2 * config.num_destinations
    
    sac_agent = SACA.SAC(config, state_dim, action_dim, env.dispatcher, env.dest_coords)
    
    print(f"=== Starting Ablation Training: {run_name} (Spherical Distance) ===")

    # 4. 训练循环
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
            p, a = sac_agent.select_action(state, deterministic=False)
            action = np.concatenate([p, a])
            
            # 环境步进 (Greedy + Haversine)
            next_state, reward, done, info = env.step((p, a))
            total_reward += reward
            step_count += 1
            
            sac_agent.store_transition(state, action, reward, next_state, done)
            
            if t % 5 == 0:
                sac_agent.update()
            
            for k in episode_log.keys():
                episode_log[k].append(info[k])
            
            state = next_state
            if done:
                break
        
        # --- 聚合数据 ---
        avg_reward_ep = total_reward / max(1, step_count)
        log["total_reward"].append(avg_reward_ep)
        for k in episode_log.keys():
            log[k].append(np.mean(episode_log[k]))
        
        # --- 打印进度 ---
        if (episode + 1) % config.eval_interval == 0:
            print(f"Episode {episode+1}")
            print(f"  Avg Reward: {avg_reward_ep:.2f}")
            print(f"  Avg Revenue: {np.mean(episode_log['revenue']):.2f}")
            print(f"  Avg ORR: {np.mean(episode_log['orr']):.3f}")
            print(f"  Avg Distance: {np.mean(episode_log['total_distance']):.2f}km")
            print(f"  Avg Cost: {np.mean(episode_log['cost']):.2f}")
            print(f"  (Mode: {run_name})\n")

        # --- 早停机制 ---
        K = config.conv_window
        n_ep = len(log["total_reward"])
        if n_ep >= max(config.min_episodes, 2 * K):
            last_K = np.array(log["total_reward"][-K:])
            prev_K = np.array(log["total_reward"][-2*K:-K])
            ma_last = last_K.mean()
            ma_prev = prev_K.mean()
            delta = abs(ma_last - ma_prev)
            
            delta_thr = config.conv_delta_ratio * (abs(ma_last) + 1e-8)
            std_thr = 0.1 * (abs(ma_last) + 1e-8)
            std_last = last_K.std()
            
            if (delta <= delta_thr) and (std_last <= std_thr):
                print(f"[Early Stopping] Converged at episode {episode+1}")
                break

    # 5. 保存结果
    save_log(log, run_name)
    print("Training Finished.")

def save_log(log, run_name):
    path = ensure_output_dir("logs")
    path.mkdir(exist_ok=True)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = path / f"ablation_log_{run_name}_{timestamp}.csv"
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        keys = list(log.keys())
        writer.writerow(['Episode'] + keys)
        num_records = len(log[keys[0]])
        for i in range(num_records):
            row = [i+1] + [f"{log[k][i]:.4f}" for k in keys]
            writer.writerow(row)
    print(f"Log saved to {filename}")

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    train_ablation_wo_route()
