import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import time

# 尝试导入训练函数，如果不存在也不影响绘图
try:
    from train import main as train_main
except ImportError:
    print("Warning: train.py not found. Only plotting mode is available.")

# ==========================================
# 1. 全局配置与风格定义
# ==========================================
LOG_DIR = "log"
MAX_EPISODES = 80  # 绘图截断点

# 实验参数配置 (对应实验运行的先后顺序)
# Index 0 -> Subplot 1, Index 1 -> Subplot 2...
UNCERTAINTY_CONFIGS = [
    (1, 0.5),
    (1, 1),
    (10, 1),
    (5, 1)
]


# 算法定义 (标签: 文件名模式)
# 这里的 Key 将作为图例名称
ALGO_MAP = {
    "SACA (Spatial)": "saca_training_log_*.csv",
    "MLP (Baseline)": "mlp_training_log_*.csv",
    "GRC-ELG": "grc_elg_training_log_*.csv",
    "JDRL-POMO": "jdrl_training_log_*.csv"
}

# 算法对应的绘图风格 (严格对齐 plot.py)
# Key 必须与 ALGO_MAP 的 Key 一致
ALGO_STYLES = {
    "SACA (Spatial)": {"color": "red", "linestyle": "-"},
    "MLP (Baseline)": {"color": "gray", "linestyle": "--"},
    "GRC-ELG":        {"color": "blue", "linestyle": "-."},
    "JDRL-POMO":      {"color": "green", "linestyle": ":"}
}

# ==========================================
# 2. 实验执行模块
# ==========================================
def run_experiments():
    """
    依次执行4组不同参数的训练。
    """
    print(f"Starting experiments sequence with {len(UNCERTAINTY_CONFIGS)} configurations...")
    print("=" * 60)

    for i, (A, w) in enumerate(UNCERTAINTY_CONFIGS):
        print(f"\n[Run {i+1}/{len(UNCERTAINTY_CONFIGS)}] executing with A={A}, w={w}...")
        try:
            # train.py 会一次性生成所有算法的日志
            train_main(A=A, w=w)
            
            # 休眠确保时间戳差异，防止文件排序出错
            time.sleep(2) 
        except Exception as e:
            print(f"Error executing parameters A={A}, w={w}: {e}")

    print("\nAll experiments completed.")

# ==========================================
# 3. 数据加载与整理模块
# ==========================================
def get_sorted_log_files(pattern, count=4):
    """获取指定模式最新的 count 个文件 (按时间升序: 旧->新)"""
    search_path = os.path.join(LOG_DIR, pattern)
    files = glob.glob(search_path)
    if not files:
        return []
    # 按修改时间排序
    sorted_files = sorted(files, key=os.path.getmtime)
    # 取最后 count 个
    return sorted_files[-count:]

def load_and_organize_data():
    """
    核心逻辑：
    1. 读取每个算法最新的4个文件。
    2. 将它们“转置”，按配置(Config)分组。
    
    返回结构:
    organized_data[config_index] = {
        "SACA": df1,
        "GRC": df2,
        ...
    }
    """
    organized_data = [{} for _ in range(len(UNCERTAINTY_CONFIGS))]
    
    print("Loading data from logs...")
    
    for algo_name, file_pattern in ALGO_MAP.items():
        # 获取该算法对应的4个文件（假设按顺序对应4个配置）
        files = get_sorted_log_files(file_pattern, count=len(UNCERTAINTY_CONFIGS))
        
        if len(files) < len(UNCERTAINTY_CONFIGS):
            print(f"Warning: Not enough files for {algo_name}. Found {len(files)}, need {len(UNCERTAINTY_CONFIGS)}.")
        
        for i, filepath in enumerate(files):
            # 防止索引越界
            if i >= len(UNCERTAINTY_CONFIGS): break
            
            try:
                df = pd.read_csv(filepath)
                df = df.iloc[:MAX_EPISODES] # 统一截断
                
                # 将数据放入对应的配置槽位中
                organized_data[i][algo_name] = df
                # print(f"  - Mapped {os.path.basename(filepath)} to Config {i+1} ({algo_name})")
                
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                
    return organized_data

# ==========================================
# 4. 绘图与分析模块
# ==========================================
def plot_performance():
    # 1. 准备数据
    # data_by_config 是一个列表，长度为4，每个元素是一个字典 {算法名: DataFrame}
    data_by_config = load_and_organize_data()
    
    # 2. 设置画布
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # 3. 遍历 4 个配置 (对应 4 个子图)
    for cfg_idx, (A, w) in enumerate(UNCERTAINTY_CONFIGS):
        ax = axes[cfg_idx]
        current_config_data = data_by_config[cfg_idx]
        
        # 子图标题：显示当前的参数环境
        ax.set_title(f"Environment: A={A}, w={w}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        
        # 在当前子图中，画出所有算法的曲线
        # 遍历 ALGO_MAP 确保图例顺序一致
        for algo_name in ALGO_MAP.keys():
            if algo_name in current_config_data:
                df = current_config_data[algo_name]
                style = ALGO_STYLES[algo_name]
                
                if "total_reward" in df.columns:
                    ax.plot(df["Episode"], df["total_reward"],
                            label=algo_name,
                            color=style["color"],
                            linestyle=style["linestyle"],
                            linewidth=2,
                            alpha=0.9)
            else:
                # 如果缺少数据，可以选择打印警告
                pass

        ax.legend(loc='lower right') # 图例显示算法名
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(LOG_DIR, "uncertainty_comparison_by_config.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {output_path}")
    plt.close()

    # ==========================================
    # 5. 生成统计表格
    # ==========================================
    print("\n" + "="*60)
    print("Performance Summary (Avg Reward over last 10 episodes)")
    print("Rows: Parameters (Environment) | Cols: Algorithms")
    print("="*60)
    
    table_rows = []
    row_indices = []
    
    for cfg_idx, (A, w) in enumerate(UNCERTAINTY_CONFIGS):
        row_data = {}
        row_indices.append(f"A={A}, w={w}")
        
        current_data = data_by_config[cfg_idx]
        
        for algo_name in ALGO_MAP.keys():
            if algo_name in current_data:
                df = current_data[algo_name]
                if "total_reward" in df.columns and len(df) > 0:
                    val = df["total_reward"].tail(10).mean()
                    row_data[algo_name] = f"{val:.2f}"
                else:
                    row_data[algo_name] = "N/A"
            else:
                row_data[algo_name] = "-"
        
        table_rows.append(row_data)
        
    summary_df = pd.DataFrame(table_rows, index=row_indices)
    
    # 调整列顺序以匹配 ALGO_MAP 定义的顺序
    summary_df = summary_df[list(ALGO_MAP.keys())]

    # ==========================================
    # 按列最大值计算损失百分比
    # ==========================================
    formatted_df = summary_df.copy()

    for col in formatted_df.columns:
        # 将字符串转为 float（忽略 N/A 和 -）
        numeric_col = pd.to_numeric(formatted_df[col], errors="coerce")
        
        if numeric_col.notna().any():
            max_val = numeric_col.max()
            
            def format_with_loss(x):
                if pd.isna(x):
                    return "-"
                loss_pct = (max_val - x) / max_val * 100 if max_val != 0 else 0
                return f"{x:.2f} (-{loss_pct:.1f}%)"
            
            formatted_df[col] = numeric_col.apply(format_with_loss)

    
    print(formatted_df)
    print("="*60)
    formatted_df.to_csv(os.path.join(LOG_DIR, "uncertainty_stats.csv"))

    # ==========================================
    # SACA vs Best Baseline Gap Table
    # ==========================================
    gap_rows = []

    for idx in summary_df.index:
        saca_val = float(summary_df.loc[idx, "SACA (Spatial)"])
        
        # 选出当前行中，除 SACA 之外的最强 baseline
        baseline_vals = summary_df.loc[idx, [
            "MLP (Baseline)",
            "GRC-ELG",
            "JDRL-POMO"
        ]].astype(float)
        
        best_baseline = baseline_vals.max()
        
        gap_pct = (saca_val - best_baseline) / abs(best_baseline) * 100
        
        gap_rows.append({
            "SACA (Spatial)": f"{saca_val:.2f}",
            "Best Baseline": f"{best_baseline:.2f}",
            "Gap (%)": f"{gap_pct:+.1f}%"
        })

    gap_df = pd.DataFrame(gap_rows, index=summary_df.index)

    print("\n" + "="*60)
    print("SACA vs Best Baseline Gap Analysis")
    print("="*60)
    print(gap_df)
    print("="*60)

    # gap_df.to_csv(os.path.join(LOG_DIR, "saca_gap_vs_best.csv"))


# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    # 1. 运行实验 (已注释，直接使用现有数据)
    run_experiments()
    
    # 2. 绘图
    plot_performance()