import random
import math
import os, pathlib
from st_saca.paths import STATIONS_FILE, require_file
BASE = pathlib.Path(__file__).resolve().parent

class config:
    BUS_SIZE = 50 # Bus的容量
    FILE_PATH = STATIONS_FILE
    STATIONNUM = [16, 17] # 一次发车，经停站点数量范围
    TIANFUPOS = [104.06, 30.67] # 天府机场经纬度
    DATANUM = 1024 # 数据集大小

def readbusstations(file_path):
    bus_stations = []
    with open(require_file(file_path, 'Station medoid file'), 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(',')
            # 跳过第一行表头
            if line.startswith("Station_ID,Longitude,Latitude"):
                continue
            if len(parts) >= 3:
                longitude = float(parts[1])
                latitude = float(parts[2])
                bus_stations.append([longitude, latitude])
    return bus_stations

def generate_passengers_distributed(num_stations, total_passengers):
    """
    使用归一化的随机权重来分配乘客数，确保总和正确。
    此方法速度非常快，不受总乘客数的影响。

    参数:
        num_stations (int): 站点数量。
        total_passengers (int): 总乘客人数。

    返回:
        list[int]: 长度为 num_stations 的列表，总和为 total_passengers。
    """
    if num_stations <= 0:
        return []
    if total_passengers == 0:
        return [0] * num_stations
    if num_stations == 1:
        return [total_passengers]

    # 1. 为每个站点生成一个随机权重
    weights = [random.random() for _ in range(num_stations)]
    total_weight = sum(weights)

    # 2. (处理极端情况) 如果所有随机权重都是0，则把所有乘客放在第一站
    if total_weight == 0:
        result = [0] * num_stations
        result[0] = total_passengers
        return result

    # 3. 按比例计算每个站点的理想 (浮点数) 乘客数
    ideal_shares = [(w / total_weight) * total_passengers for w in weights]
    
    # 4. 先将每个站点的乘客数向下取整
    stations = [int(share) for share in ideal_shares]
    
    # 5. 计算因为取整而“丢失”的乘客数 (余数)
    remainder = total_passengers - sum(stations)
    
    # 6. 将余数分配给“损失”了最多小数部分的站点
    #    (这可以确保分配是公平的，而不是随机地把余数加到某个站)
    fractional_parts = [(ideal_shares[i] - stations[i]) for i in range(num_stations)]
    
    # 获取按小数部分大小排序的站点索引
    indices_sorted_by_fraction = sorted(
        range(num_stations),
        key=lambda k: fractional_parts[k],
        reverse=True
    )
    
    # 将余下的乘客 (remainder 个人) 一人一个地分配给小数部分最大的那些站点
    for i in range(remainder):
        stations[indices_sorted_by_fraction[i]] += 1
        
    return stations

class TspPassengerTimeSolver:
    def __init__(self, stations, passengers, start_pos):
        # 1. 初始化输入数据
        self.start_pos = start_pos
        self.stations = stations
        
        # 2. 整合所有坐标点
        # self.all_points[0] 是起点
        # self.all_points[1] 是 stations[0]
        # self.all_points[2] 是 stations[1], ...
        self.all_points = [self.start_pos] + self.stations
        self.num_points = len(self.all_points) # 站点数 + 1 (起点)

        # 3. 整合乘客数据，与 all_points 对齐
        # self.passengers_at_station[0] = 0 (起点无人下车)
        # self.passengers_at_station[1] = passengers[0], ...
        self.passengers_at_station = [0] + passengers
        self.total_passengers = sum(passengers) # BUS_SIZE

        # 4. 预计算距离矩阵
        self.dist_matrix = self._compute_dist_matrix()
        
        # 5. 初始化解
        self.bestcost = float('inf')
        self.bestroute = []

    def _distance(self, p1, p2):
        """计算地球上两点间的球面距离"""
        R = 6371.0  # 地球半径，单位：公里
        lat1 = math.radians(p1[1])
        lon1 = math.radians(p1[0])
        lat2 = math.radians(p2[1])
        lon2 = math.radians(p2[0])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _compute_dist_matrix(self):
        """预计算所有点对之间的距离矩阵"""
        matrix = [[0.0] * self.num_points for _ in range(self.num_points)]
        for i in range(self.num_points):
            for j in range(i + 1, self.num_points):
                dist = self._distance(self.all_points[i], self.all_points[j])
                matrix[i][j] = dist
                matrix[j][i] = dist
        return matrix

    def solve(self):
        """开始求解"""
        # visited 数组用于跟踪已访问的节点
        visited = [False] * self.num_points
        # 标记起点 (index 0) 为已访问
        visited[0] = True 
        
        # 从起点 (node 0) 开始搜索
        self.dfs(
            current_node=0,           # 当前所在的节点索引
            current_path=[0],         # 当前已走过的路径 (索引)
            current_cost=0,           # 当前累积的成本 (乘客*距离)
            remaining_passengers=self.total_passengers, # 当前车上剩余乘客
            visited=visited
        )
        
        # 返回最终解
        return self.bestroute, self.bestcost

    # ------------------------------------------------------------------
    # 这是您要求的 dfs 函数
    # ------------------------------------------------------------------
    def dfs(self, current_node, current_path, current_cost, remaining_passengers, visited):
        """
        使用分支定界法进行深度优先搜索
        
        参数:
        current_node (int): 当前所在的节点索引 (在 self.all_points 中)
        current_path (list): 从起点开始的路径列表
        current_cost (float): 到达此节点的累积成本 (总乘客等待时间)
        remaining_passengers (int): 正好在 current_node 之前，车上剩余的乘客数
        visited (list[bool]): 标记节点是否已在当前路径中
        """
        
        # --- 1. 定界 (Bound) ---
        # 剪枝：如果当前成本已超过已知最优解，则此路不通
        if current_cost >= self.bestcost:
            return # 剪枝

        # --- 2. 基线条件 (Base Case) ---
        # 如果路径长度等于总点数，说明所有点都已访问
        if len(current_path) == self.num_points:
            # 我们找到了一个完整的解
            if current_cost < self.bestcost:
                self.bestcost = current_cost
                # 将路径 [0, 2, 1] 转换为 [1, 0] (站点的0-based索引)
                self.bestroute = [idx - 1 for idx in current_path[1:]]
            return

        # --- 3. 定界 (Lower Bound) - 核心 ---
        # 计算一个"下界"：当前成本 + 访问所有剩余节点所需的 *最小可能成本*
        
        unvisited_stations = []
        for i in range(1, self.num_points): # 遍历所有站点 (跳过起点0)
            if not visited[i]:
                unvisited_stations.append(i)

        if unvisited_stations:
            # 一个简单但有效的下界 (Admissible Heuristic):
            # 至少，我们必须从 current_node 移动到 *最近的* 那个未访问站点。
            # 并且，这趟路程 *至少* 会被 *所有* 剩余乘客乘坐。
            min_dist_to_next = min(self.dist_matrix[current_node][u] for u in unvisited_stations)
            
            # g_cost = current_cost (已花费成本)
            # h_cost = min_dist_to_next * remaining_passengers (预估未来最小成本)
            lower_bound = current_cost + (min_dist_to_next * remaining_passengers)

            if lower_bound >= self.bestcost:
                return # 剪枝 (Bound)

        # --- 4. 分支 (Branch) ---
        # 递归探索所有未访问的邻居
        
        # (优化：优先访问最近的邻居，这是一种贪心策略，有助于更快找到一个好的 'bestcost'，从而更早地剪枝)
        neighbors = sorted(unvisited_stations, key=lambda u: self.dist_matrix[current_node][u])

        for next_node in neighbors:
            # (理论上，neighbors 列表中的都未被访问，但双重检查无害)
            # if not visited[next_node]: # <== 已被 sorted(unvisited_stations, ...) 保证
            
            # --- 计算访问 next_node 的成本 ---
            
            # a. 这段路程的距离
            segment_distance = self.dist_matrix[current_node][next_node]
            
            # b. 这段路程的成本 = 距离 * 乘坐这段路的乘客数
            segment_cost = segment_distance * remaining_passengers
            
            # c. 更新递归调用的参数
            new_cost = current_cost + segment_cost
            new_remaining_passengers = remaining_passengers - self.passengers_at_station[next_node]
            
            # d. 递归 (Branch)
            visited[next_node] = True
            self.dfs(
                next_node,
                current_path + [next_node], # 添加新节点到路径
                new_cost,
                new_remaining_passengers,
                visited
            )
            
            # e. 回溯 (Backtrack)
            visited[next_node] = False

def generate_routelist(bus_stations, n_dest=None):
    '''
    随机生成DATANUM公交线路,并且返回他的最优经停序列
    用于构建监督学习数据集
    
    返回格式:
    data = [
        {
            'station_ids': [3, 7, 12, ...],           # 原始无序的站点ID（在bus_stations中的索引）
            'station_coords': [[lon1, lat1], ...],    # 对应的站点坐标
            'passengers': [5, 12, 8, ...],            # 每个站点的下车人数
            'optimal_order': [12, 3, 7, ...],         # TSP最优顺序（站点ID，不是局部下标）
            'optimal_cost': 1234.56                   # 最优成本
            'local_optimal_order': [2, 0, 1, ...]     # TSP最优顺序（局部下标，从0开始）
            'avg_cost_per_passenger': 56.78           # 平均每位乘客的成本
        },
        ...
    ]
    '''
    data = []
    
    print(f"开始生成 {config.DATANUM} 条监督数据...")
    print("=" * 60)

    for i in range(config.DATANUM):
        # 1. 随机选择经停站点数量
        if n_dest is not None:
            num_stations = n_dest
        else:
            num_stations = random.randint(config.STATIONNUM[0], config.STATIONNUM[1])

        # 2. 从所有公交站点中随机不重复选择站点
        # 记录：站点在 bus_stations 中的索引（这是真实的站点ID）
        selected_station_indices = random.sample(range(len(bus_stations)), num_stations)
        
        # 3. 获取这些站点的坐标
        selected_station_coords = [bus_stations[idx] for idx in selected_station_indices]
        
        # 4. 随机生成下车人数分布
        people_counts = generate_passengers_distributed(num_stations, config.BUS_SIZE)
        
        # 5. 使用TSP求解器计算最优路径
        solver = TspPassengerTimeSolver(
            stations=selected_station_coords,
            passengers=people_counts,
            start_pos=config.TIANFUPOS
        )
        
        # bestroute 是局部索引（0到num_stations-1）
        local_optimal_order, optimal_cost = solver.solve()
        
        # 6. 将局部索引转换为全局站点ID
        # local_optimal_order[i] 是在 selected_station_coords 中的位置
        # 我们需要转换为在 bus_stations 中的实际ID
        global_optimal_order = [selected_station_indices[local_idx] for local_idx in local_optimal_order]
        
        # 7. 构建数据项
        data_item = {
            'station_ids': selected_station_indices,       # 原始无序的站点ID
            'station_coords': selected_station_coords,     # 站点坐标
            'passengers': people_counts,                   # 下车人数
            'global_optimal_order': global_optimal_order,  # 最优顺序（全局站点ID）
            'optimal_cost': optimal_cost,                  # 最优成本
            'local_optimal_order': local_optimal_order,     # 最优顺序（局部站点ID）
            'avg_cost_per_passenger': optimal_cost / config.BUS_SIZE  # 平均每位乘客的成本
        }
        
        data.append(data_item)
        
        # 打印进度
        if (i + 1) % max(1, config.DATANUM // 10) == 0:
            print(f"进度: {i+1}/{config.DATANUM} | "
                  f"站点数: {num_stations} | "
                  f"成本: {optimal_cost:.2f} | "
                  f"平均: {optimal_cost/config.BUS_SIZE:.2f}")
    
    print("=" * 60)
    print(f"✓ 数据集生成完成！共 {len(data)} 条样本")
    
    return data

def save():
    for i in range(config.STATIONNUM[0], config.STATIONNUM[1] + 1):
        # 保存为json文件
        N_dest = i
        # datanum = config.DATANUM
        datanum = 2 ** 10
        config.DATANUM = datanum
        bus_stations = readbusstations(config.FILE_PATH)
        dataset = generate_routelist(bus_stations, n_dest=N_dest)
        file_name = BASE.parent / "dataset" / f"supervised_dataset_{N_dest}_stations.json"
        import json
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

def generate_single_data(numstations):
    bus_stations = readbusstations(config.FILE_PATH)
    # 1. 定义您的输入
    TIANFUPOS = config.TIANFUPOS           # 起点坐标
    # 随机选取几个站点
    stations = random.sample(bus_stations, numstations)    # 站点坐标列表
    passengers = generate_passengers_distributed(len(stations), config.BUS_SIZE)   # 对应站点的下车人数
    BUS_SIZE = sum(passengers)     # 总乘客数

    # 2. 创建求解器实例
    solver = TspPassengerTimeSolver(stations, passengers, TIANFUPOS)

    # 3. 求解
    # (您在问题中定义的全局变量现在被封装在 solver 对象中)
    bestroute, bestcost = solver.solve()

    # 4. 打印结果
    print(f"站点坐标: {stations}")
    print(f"下车人数: {passengers}")
    print(f"起点: {TIANFUPOS}")
    print("---")
    print(f"最佳路径 (站点索引): {bestroute}")
    print(f"最小总乘客时间 (Cost): {bestcost:.4f}")
    print(f"最小平均乘客时间: {(bestcost / BUS_SIZE):.4f}")
    return stations, passengers, TIANFUPOS, bestroute, bestcost

if __name__ == "__main__":
    # save()
    numstations = 18
    generate_single_data(numstations)

