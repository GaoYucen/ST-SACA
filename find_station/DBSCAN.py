import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import os


class Config:
    """配置类，统一管理所有参数"""
    # 数据路径配置
    DATA_PATH = "C:\\Users\\Administrator\\Desktop\\TSC_trans\\data\\total_ride_request\\"
    PLOT_PATH = "plot"
    OUTPUT_PATH = "C:\\Users\\Administrator\\Desktop\\TSC_trans\\data\\dbscan_res\\"
    
    # DBSCAN参数配置
    EPS = 0.0009  # DBSCAN半径参数（经纬度单位）
    MIN_SAMPLES = 50  # 最小样本数
    
    # 日期范围配置
    START_DAY = 1
    END_DAY = 7
    
    # K距离图参数
    K_NEIGHBORS = 50  # K距离图中的K值
    
    # 可视化配置
    FIGURE_SIZE = (12, 5)
    DPI = 150  # 提高分辨率
    POINT_SIZE = 2  # 增大点的大小
    POINT_ALPHA = 0.6  # 点的透明度
    MAX_PLOT_POINTS = 100000  # 可视化时最大显示点数（超过则采样）
    
    @staticmethod
    def ensure_plot_dir():
        """确保plot目录存在"""
        if not os.path.exists(Config.PLOT_PATH):
            os.makedirs(Config.PLOT_PATH)


def dbscan_denoise(data, eps=None, min_samples=None):
    """
    实现DBSCAN聚类算法，去除噪声点
    :param data: 经纬度数据列表 [[lon1, lat1], [lon2, lat2], ...]
    :param eps: DBSCAN半径参数
    :param min_samples: 最小样本数
    :return: 去噪后的数据, 原始数据, 聚类标签
    """
    if eps is None:
        eps = Config.EPS
    if min_samples is None:
        min_samples = Config.MIN_SAMPLES
    
    # 转换为numpy数组
    data_array = np.array(data)
    
    # 执行DBSCAN聚类
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(data_array)
    
    # 过滤噪声点（标签为-1的点）
    mask = labels != -1
    denoised_data = data_array[mask]
    
    print(f"Original data points: {len(data_array)}")
    print(f"Noise points: {np.sum(labels == -1)}")
    print(f"Denoised data points: {len(denoised_data)}")
    print(f"Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
    
    return denoised_data, data_array, labels


def plot_k_distance(data, k=None, save_path=None):
    """
    绘制K距离图，用于确定DBSCAN的eps参数
    :param data: 经纬度数据
    :param k: K值（默认使用Config中的配置）
    :param save_path: 保存路径
    """
    if k is None:
        k = Config.K_NEIGHBORS
    
    Config.ensure_plot_dir()
    
    data_array = np.array(data)
    
    # 计算每个点到第k个最近邻的距离
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(data_array)
    distances, indices = neighbors.kneighbors(data_array)
    
    # 取第k个邻居的距离并排序
    k_distances = distances[:, -1]
    k_distances = np.sort(k_distances)[::-1]  # 降序排列
    
    # 自动确定合适的Y轴显示范围（聚焦于有意义的区间）
    # 过滤掉极端值（前1%的异常大值），聚焦于大部分数据的分布
    percentile_99 = np.percentile(k_distances, 99)
    percentile_95 = np.percentile(k_distances, 95)
    percentile_90 = np.percentile(k_distances, 90)
    
    # 使用95分位数的1.5倍作为Y轴上限，确保能看到"肘部"
    y_max = min(percentile_95 * 1.5, percentile_99)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：完整视图
    ax1.plot(range(len(k_distances)), k_distances, 'b-', linewidth=1)
    ax1.set_xlabel('Data Points (sorted by K-distance, descending)', fontsize=12)
    ax1.set_ylabel(f'{k}-th Nearest Neighbor Distance', fontsize=12)
    ax1.set_title(f'K-distance Graph - Full View (K={k})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 添加参考线
    if Config.EPS is not None:
        ax1.axhline(y=Config.EPS, color='r', linestyle='--', linewidth=2,
                   label=f'Current EPS={Config.EPS}')
    ax1.axhline(y=percentile_95, color='orange', linestyle=':', linewidth=1.5,
               label=f'95th percentile={percentile_95:.4f}')
    ax1.legend(loc='upper right')
    
    # 右图：聚焦视图（去除极端值，便于观察"肘部"）
    ax2.plot(range(len(k_distances)), k_distances, 'b-', linewidth=1.5)
    ax2.set_xlabel('Data Points (sorted by K-distance, descending)', fontsize=12)
    ax2.set_ylabel(f'{k}-th Nearest Neighbor Distance', fontsize=12)
    ax2.set_title(f'K-distance Graph - Zoomed View (Y-axis limited)\nFind the "Elbow": Where curve drops sharply then flattens', 
                 fontsize=14)
    ax2.set_ylim(0, y_max)
    ax2.grid(True, alpha=0.3)
    
    # 添加参考线和标注
    if Config.EPS is not None:
        ax2.axhline(y=Config.EPS, color='r', linestyle='--', linewidth=2,
                   label=f'Current EPS={Config.EPS}')
    
    # 添加常见的建议eps范围标注
    suggested_eps_low = percentile_90
    suggested_eps_high = percentile_95
    ax2.axhspan(suggested_eps_low, suggested_eps_high, alpha=0.2, color='green',
               label=f'Suggested range: [{suggested_eps_low:.4f}, {suggested_eps_high:.4f}]')
    
    ax2.legend(loc='upper right', fontsize=10)
    
    # 添加统计信息文本
    stats_text = f'Statistics:\n'
    stats_text += f'90th percentile: {percentile_90:.4f}\n'
    stats_text += f'95th percentile: {percentile_95:.4f}\n'
    stats_text += f'99th percentile: {percentile_99:.4f}\n'
    stats_text += f'Max distance: {k_distances[0]:.4f}'
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path is None:
        save_path = os.path.join(Config.PLOT_PATH, 'k_distance_graph.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
    print(f"\nK-distance graph saved to: {save_path}")
    print(f"\nSuggested EPS range: [{suggested_eps_low:.4f}, {suggested_eps_high:.4f}]")
    print(f"Current EPS setting: {Config.EPS}")
    if Config.EPS < suggested_eps_low:
        print(f"⚠️  Current EPS may be too small - strict clustering, may produce more noise points")
    elif Config.EPS > suggested_eps_high:
        print(f"⚠️  Current EPS may be too large - will merge distant points, clustering may not be precise")
    else:
        print(f"✓ Current EPS is within the suggested range")
    plt.close()


def visualize_denoise(original_data, denoised_data, labels=None, save_path=None):
    """
    可视化去噪前后的数据对比
    :param original_data: 原始数据
    :param denoised_data: 去噪后数据
    :param labels: 聚类标签（用于显示噪声点）
    :param save_path: 保存路径
    """
    Config.ensure_plot_dir()
    
    # 智能采样：如果数据点太多，随机采样保持可视化清晰
    def sample_data(data, max_points):
        if len(data) > max_points:
            indices = np.random.choice(len(data), max_points, replace=False)
            return data[indices], indices
        return data, np.arange(len(data))
    
    original_sample, orig_indices = sample_data(original_data, Config.MAX_PLOT_POINTS)
    denoised_sample, denoise_indices = sample_data(denoised_data, Config.MAX_PLOT_POINTS)
    
    fig, axes = plt.subplots(1, 2, figsize=Config.FIGURE_SIZE)
    
    # 去噪前
    axes[0].scatter(original_sample[:, 0], original_sample[:, 1], 
                   c='blue', s=Config.POINT_SIZE, alpha=Config.POINT_ALPHA, edgecolors='none')
    axes[0].set_xlabel('Longitude', fontsize=10)
    axes[0].set_ylabel('Latitude', fontsize=10)
    sample_info = f" (showing {len(original_sample)}/{len(original_data)})" if len(original_sample) < len(original_data) else ""
    axes[0].set_title(f'Before Denoising\nTotal: {len(original_data)} points{sample_info}', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 去噪后
    axes[1].scatter(denoised_sample[:, 0], denoised_sample[:, 1], 
                   c='green', s=Config.POINT_SIZE, alpha=Config.POINT_ALPHA, edgecolors='none')
    axes[1].set_xlabel('Longitude', fontsize=10)
    axes[1].set_ylabel('Latitude', fontsize=10)
    noise_count = len(original_data) - len(denoised_data)
    sample_info = f" (showing {len(denoised_sample)}/{len(denoised_data)})" if len(denoised_sample) < len(denoised_data) else ""
    axes[1].set_title(f'After Denoising\nRemaining: {len(denoised_data)}, Removed: {noise_count}{sample_info}', 
                     fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    if save_path is None:
        save_path = os.path.join(Config.PLOT_PATH, 'denoise_comparison.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
    print(f"Denoising comparison saved to: {save_path}")
    plt.close()
    
    # 如果提供了标签，额外绘制一个显示聚类结果的图
    if labels is not None:
        plot_clusters(original_data, labels)

def plot_clusters(data, labels, save_path=None):
    """
    绘制聚类结果，不同聚类用不同颜色，噪声点用黑色
    :param data: 数据点
    :param labels: 聚类标签
    :param save_path: 保存路径
    """
    Config.ensure_plot_dir()
    
    # 智能采样
    def sample_by_label(data, labels, max_points):
        unique_labels = set(labels)
        sampled_data = []
        sampled_labels = []
        
        # 计算每个类别应该采样多少点
        points_per_label = max_points // len(unique_labels)
        
        for label in unique_labels:
            mask = labels == label
            label_data = data[mask]
            
            if len(label_data) > points_per_label:
                indices = np.random.choice(len(label_data), points_per_label, replace=False)
                sampled_data.append(label_data[indices])
                sampled_labels.extend([label] * points_per_label)
            else:
                sampled_data.append(label_data)
                sampled_labels.extend([label] * len(label_data))
        
        return np.vstack(sampled_data), np.array(sampled_labels)
    
    if len(data) > Config.MAX_PLOT_POINTS:
        plot_data, plot_labels = sample_by_label(data, labels, Config.MAX_PLOT_POINTS)
        sample_info = f" (showing {len(plot_data)}/{len(data)})"
    else:
        plot_data, plot_labels = data, labels
        sample_info = ""
    
    plt.figure(figsize=(10, 8))
    
    # 获取唯一的聚类标签
    unique_labels = set(plot_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # 噪声点用黑色
            col = 'black'
        
        class_member_mask = (plot_labels == k)
        xy = plot_data[class_member_mask]
        
        if k == -1:
            plt.scatter(xy[:, 0], xy[:, 1], c=[col], s=Config.POINT_SIZE, 
                       alpha=0.3, edgecolors='none', label='Noise')
        else:
            plt.scatter(xy[:, 0], xy[:, 1], c=[col], s=Config.POINT_SIZE, 
                       alpha=Config.POINT_ALPHA, edgecolors='none')
    
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1)
    plt.title(f'DBSCAN Clustering Results{sample_info}\n'
             f'Clusters: {n_clusters}, Noise points: {n_noise}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 只显示噪声点的图例
    if -1 in unique_labels:
        plt.legend(loc='best', markerscale=3)
    
    if save_path is None:
        save_path = os.path.join(Config.PLOT_PATH, 'cluster_results.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
    print(f"Clustering results saved to: {save_path}")
    plt.close()

def output_clustering_results(data, labels, save_path=None):
    """
    输出聚类结果到指定路径
    :param data: 数据点
    :param labels: 聚类标签
    :param save_path: 保存路径
    """
    if save_path is None:
        save_path = os.path.join(Config.OUTPUT_PATH, 'clustering_results.txt')

    with open(save_path, 'w') as f:
        for point, label in zip(data, labels):
            f.write(f"{point[0]},{point[1]},{label}\n")

    print(f"Clustering results output to: {save_path}")

def read_data(file_path=None, startday=None, endday=None):
    """
    读取指定日期范围内的下车点经纬度数据
    :param file_path: 数据文件路径前缀
    :param startday: 起始日期（包含）
    :param endday: 结束日期（包含）
    :return: 下车点经纬度数据列表
    """
    if file_path is None:
        file_path = Config.DATA_PATH
    if startday is None:
        startday = Config.START_DAY
    if endday is None:
        endday = Config.END_DAY
    
    data = []
    for day in range(startday, endday + 1):
        # 按行读取数据，数据分隔符为逗号，一行八个字段，第一个字段为id，后面七个字段为float数值
        filename = file_path + "order_2016110" + str(day)
        with open(filename, 'r') as file:
            for line in file:
                fields = line.strip().split(',')
                if len(fields) != 8:
                    continue
                point_id = fields[0]
                coordinates = list(map(float, fields[1:]))
                data.append((point_id, coordinates))
    # 返回下车点经纬度数据
    res = [coords[-3:-1] for _, coords in data if len(coords) >= 3]
    return res

def denoise_data():
    data = read_data()
    denoised_data, _, _ = dbscan_denoise(data)
    return denoised_data

if __name__ == "__main__":
    print("=" * 60)
    print("Start DBSCAN Denoising Process")
    print("=" * 60)
    
    # 读取数据
    print(f"\n1. Reading data (Date range: {Config.START_DAY} - {Config.END_DAY})")
    data = read_data()
    print(f"Data points loaded: {len(data)}")
    print("Data sample:", data[:5])
    
    # 绘制K距离图
    print(f"\n2. Plotting K-distance graph (K={Config.K_NEIGHBORS})")
    plot_k_distance(data)
    
    # 执行DBSCAN去噪
    print(f"\n3. Executing DBSCAN denoising (eps={Config.EPS}, min_samples={Config.MIN_SAMPLES})")
    denoised_data, original_data, labels = dbscan_denoise(data)
    
    # 可视化去噪前后对比
    print("\n4. Generating visualization charts")
    visualize_denoise(original_data, denoised_data, labels)
    
    print("\n" + "=" * 60)
    print("Processing completed! All charts saved to plot folder")
    print("=" * 60)

    # 输出聚类结果
    # output_clustering_results(original_data, labels)
