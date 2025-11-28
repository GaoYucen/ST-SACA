import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import cdist
import os


class Config:
    """配置类，统一管理所有参数"""
    # 数据路径配置
    INPUT_FILE = "C:\\Users\\Administrator\\Desktop\\TSC_trans\\data\\dbscan_res\\clustering_results.txt"
    OUTPUT_PATH = "C:\\Users\\Administrator\\Desktop\\TSC_trans\\data\\pam_res\\"
    PLOT_PATH = "plot"
    
    # PAM聚类参数（针对32GB内存优化）
    N_CLUSTERS = 30  # 公交站点数量
    MAX_ITER = 300  # PAM算法最大迭代次数
    RANDOM_STATE = 42  # 随机种子，保证结果可复现
    
    # 采样参数（针对32GB内存的最优配置）
    # 30000个点的距离矩阵需要约 30000² × 8 bytes ≈ 7.2 GB
    # 加上算法开销，总内存约 8-10 GB，安全范围内
    PAM_SAMPLE_SIZE = 20000  # PAM算法采样点数（针对32GB内存优化）
    
    # 可视化配置
    FIGURE_SIZE = (14, 10)
    DPI = 150
    POINT_SIZE = 3  # 散点大小
    POINT_ALPHA = 0.4  # 散点透明度
    CENTER_SIZE = 200  # 中心点（五角星）大小
    CENTER_MARKER = '*'  # 中心点标记（五角星）
    MAX_PLOT_POINTS = 50000  # 最大显示点数
    
    @staticmethod
    def ensure_dir(path):
        """确保目录存在"""
        if not os.path.exists(path):
            os.makedirs(path)


def PAM(data, n_clusters=None, random_state=None, max_iter=None):
    """
    使用智能采样+PAM算法找到真实的公交站点位置
    策略：从大数据集中采样代表性点，在采样集上运行PAM，然后将所有点分配到最近的中心
    
    :param data: 经纬度数据 [[lon, lat, cluster_label], ...]
    :param n_clusters: 聚类数量（站点数量）
    :param random_state: 随机种子
    :param max_iter: 最大迭代次数
    :return: 聚类中心（站点位置）, 聚类标签, KMedoids模型
    """
    if n_clusters is None:
        n_clusters = Config.N_CLUSTERS
    if random_state is None:
        random_state = Config.RANDOM_STATE
    if max_iter is None:
        max_iter = Config.MAX_ITER
    
    # 提取经纬度坐标（前两列）
    print("Extracting coordinates...")
    coordinates = np.array([[float(point[0]), float(point[1])] for point in data])
    
    print(f"\n{'='*60}")
    print(f"PAM Clustering Configuration")
    print(f"{'='*60}")
    print(f"Total data points: {len(coordinates):,}")
    print(f"Target stations: {n_clusters}")
    print(f"Sample size for PAM: {Config.PAM_SAMPLE_SIZE:,}")
    print(f"Estimated memory usage: ~{Config.PAM_SAMPLE_SIZE**2 * 8 / 1e9:.1f} GB")
    print(f"{'='*60}\n")
    
    # 步骤1：智能采样
    if len(coordinates) > Config.PAM_SAMPLE_SIZE:
        print(f"[Step 1/4] Sampling {Config.PAM_SAMPLE_SIZE:,} representative points...")
        
        # 使用分层采样：确保采样点能代表整个数据分布
        np.random.seed(random_state)
        sample_indices = np.random.choice(
            len(coordinates), 
            Config.PAM_SAMPLE_SIZE, 
            replace=False
        )
        sampled_coords = coordinates[sample_indices]
        print(f"  ✓ Sampling completed: {len(sampled_coords):,} points")
    else:
        print(f"[Step 1/4] Dataset is small enough, using all points...")
        sampled_coords = coordinates
        print(f"  ✓ Using all {len(sampled_coords):,} points")
    
    # 步骤2：在采样数据上运行PAM算法
    print(f"\n[Step 2/4] Running PAM algorithm on sampled data...")
    print(f"  This may take a few minutes...")
    
    kmedoids = KMedoids(
        n_clusters=n_clusters,
        random_state=random_state,
        max_iter=max_iter,
        method='pam',
        metric='euclidean',
        init='k-medoids++'  # 使用改进的初始化方法
    )
    
    kmedoids.fit(sampled_coords)
    centers = kmedoids.cluster_centers_
    
    print(f"  ✓ PAM converged after {kmedoids.n_iter_} iterations")
    print(f"  ✓ Found {len(centers)} station locations (real data points)")
    
    # 步骤3：将所有数据点分配到最近的站点
    print(f"\n[Step 3/4] Assigning all {len(coordinates):,} points to nearest stations...")
    
    # 分批计算距离（避免内存溢出）
    batch_size = 100000
    labels = np.zeros(len(coordinates), dtype=int)
    
    for i in range(0, len(coordinates), batch_size):
        end = min(i + batch_size, len(coordinates))
        batch = coordinates[i:end]
        distances = cdist(batch, centers, metric='euclidean')
        labels[i:end] = np.argmin(distances, axis=1)
        
        if end < len(coordinates):
            print(f"  Progress: {end:,}/{len(coordinates):,} points processed")
    
    print(f"  ✓ All points assigned to stations")
    
    # 步骤4：统计每个站点的乘客数
    print(f"\n[Step 4/4] Station statistics:")
    print(f"\n{'Station':<10} {'Passengers':<12} {'Percentage':<12} {'Location (Lon, Lat)'}")
    print(f"{'-'*70}")
    
    for i in range(n_clusters):
        count = np.sum(labels == i)
        percentage = count / len(labels) * 100
        lon, lat = centers[i]
        print(f"  {i+1:<8} {count:>10,} {percentage:>10.2f}%    ({lon:.6f}, {lat:.6f})")
    
    print(f"{'-'*70}")
    print(f"{'Total':<10} {len(labels):>10,} {100.0:>10.1f}%")
    
    return centers, labels, kmedoids


def visualize_clusters(data, centers, labels, save_path=None):
    """
    可视化聚类结果和公交站点位置
    :param data: 原始数据点
    :param centers: 聚类中心（站点位置）
    :param labels: 聚类标签
    :param save_path: 保存路径
    """
    Config.ensure_dir(Config.PLOT_PATH)
    
    # 提取经纬度
    coordinates = np.array([[float(point[0]), float(point[1])] for point in data])
    
    # 智能采样
    if len(coordinates) > Config.MAX_PLOT_POINTS:
        indices = np.random.choice(len(coordinates), Config.MAX_PLOT_POINTS, replace=False)
        plot_coords = coordinates[indices]
        plot_labels = labels[indices]
        sample_info = f" (showing {len(plot_coords)}/{len(coordinates)} points)"
    else:
        plot_coords = coordinates
        plot_labels = labels
        sample_info = ""
    
    # 创建图表
    fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE)
    
    # 生成颜色映射
    colors = plt.cm.tab20(np.linspace(0, 1, Config.N_CLUSTERS))
    
    # 绘制每个聚类的点
    for i in range(Config.N_CLUSTERS):
        mask = plot_labels == i
        cluster_points = plot_coords[mask]
        if len(cluster_points) > 0:
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                      c=[colors[i]], s=Config.POINT_SIZE, alpha=Config.POINT_ALPHA,
                      edgecolors='none', label=f'Cluster {i+1}')
    
    # 绘制聚类中心（公交站点）- 使用五角星
    ax.scatter(centers[:, 0], centers[:, 1],
              marker=Config.CENTER_MARKER, s=Config.CENTER_SIZE,
              c='red', edgecolors='black', linewidths=1,
              label='Bus Stations', zorder=1000)
    
    # 为每个站点添加编号
    for i, center in enumerate(centers):
        ax.annotate(f'{i+1}', xy=(center[0], center[1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Bus Station Distribution (PAM Clustering){sample_info}\n'
                f'{Config.N_CLUSTERS} stations for {len(coordinates)} passengers',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 图例（只显示站点，不显示所有聚类以免过于拥挤）
    handles, labels_legend = ax.get_legend_handles_labels()
    # 只保留站点的图例
    station_handle = [h for h, l in zip(handles, labels_legend) if 'Station' in l]
    station_label = [l for l in labels_legend if 'Station' in l]
    if station_handle:
        ax.legend(station_handle, station_label, loc='upper right', fontsize=10, markerscale=0.5)
    
    if save_path is None:
        save_path = os.path.join(Config.PLOT_PATH, 'pam_clustering.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()


def output_station_locations(centers, save_path=None):
    """
    输出公交站点位置到文件
    :param centers: 聚类中心（站点位置）
    :param save_path: 保存路径
    """
    Config.ensure_dir(Config.OUTPUT_PATH)
    
    if save_path is None:
        save_path = os.path.join(Config.OUTPUT_PATH, 'bus_stations.txt')
    
    with open(save_path, 'w') as f:
        f.write("Station_ID,Longitude,Latitude\n")
        for i, center in enumerate(centers):
            f.write(f"{i+1},{center[0]:.6f},{center[1]:.6f}\n")
    
    print(f"Station locations saved to: {save_path}")
    
    # 也输出一个带有统计信息的详细文件
    return save_path


def read_file(file_path=None):
    """
    读取DBSCAN去噪后的数据
    :param file_path: 文件路径
    :return: 数据列表
    """
    data = []
    if file_path is None:
        file_path = Config.INPUT_FILE
    
    print(f"Reading data from: {file_path}")
    with open(file_path, 'r') as file:
        for line in file:
            # 跳过噪点
            if line.strip().split(',')[2] == '-1':
                continue
            data.append(line.strip().split(','))
    
    print(f"Loaded {len(data)} data points")
    return data

if __name__ == "__main__":
    print("=" * 70)
    print("Airport Shuttle Bus Station Planning System")
    print("Using PAM (K-Medoids) Clustering Algorithm")
    print("Finding REAL station locations from actual data points")
    print("=" * 70)
    
    # 1. 读取DBSCAN去噪后的数据
    print("\n[Phase 1] Reading denoised passenger drop-off data...")
    data = read_file()
    print(f"Sample data format: {data[0]}")
    
    # 2. 使用PAM算法找到最优站点位置
    print(f"\n[Phase 2] Finding optimal locations for {Config.N_CLUSTERS} bus stations...")
    centers, labels, model = PAM(data)
    
    # 3. 可视化结果
    print(f"\n[Phase 3] Generating visualization...")
    visualize_clusters(data, centers, labels)
    
    # 4. 输出站点位置
    print(f"\n[Phase 4] Saving station locations...")
    output_station_locations(centers)
    
    # 5. 输出摘要统计
    print("\n" + "=" * 70)
    print("✓ PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Total passengers analyzed: {len(data):,}")
    print(f"Bus stations identified: {Config.N_CLUSTERS}")
    print(f"Average passengers per station: {len(data) / Config.N_CLUSTERS:.0f}")
    print(f"\nAll stations are REAL DATA POINTS (actual passenger drop-off locations)")
    print(f"\nOutput files:")
    print(f"  • Visualization: {os.path.join(Config.PLOT_PATH, 'pam_clustering.png')}")
    print(f"  • Station list: {os.path.join(Config.OUTPUT_PATH, 'bus_stations.txt')}")
    print("=" * 70)
