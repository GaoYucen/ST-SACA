import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import SACA
import SACA_baseline

def run_comparison():
    # 1. 运行 Baseline 模型
    print("Running Baseline Model (MLP Actor)...")
    config_baseline = SACA_baseline.Config()
    config_baseline.max_episodes = 100  # 确保对比轮数一致
    # 禁用 plt.show() 以免阻塞
    plt.show = lambda: None 
    SACA_baseline.train_saca(config_baseline, run_name="baseline")

    # 2. 运行 Spatial 模型
    print("\nRunning Spatial Model (Spatial Actor)...")
    config_spatial = SACA.Config()
    config_spatial.max_episodes = 100
    SACA.train_saca(config_spatial, run_name="spatial")

    # 3. 查找生成的 CSV 文件
    # 假设文件名格式为 training_log_baseline_YYYYMMDD_HHMMSS.csv
    log_dir = "log"
    baseline_files = glob.glob(os.path.join(log_dir, "training_log_baseline_*.csv"))
    spatial_files = glob.glob(os.path.join(log_dir, "training_log_spatial_*.csv"))

    if not baseline_files or not spatial_files:
        print("Error: Could not find log files.")
        return

    # 取最新的文件
    baseline_csv = sorted(baseline_files)[-1]
    spatial_csv = sorted(spatial_files)[-1]

    print(f"\nComparing {baseline_csv} vs {spatial_csv}")

    # 4. 读取数据
    df_baseline = pd.read_csv(baseline_csv)
    df_spatial = pd.read_csv(spatial_csv)

    # 5. 绘图对比
    metrics = ["total_reward", "revenue", "cost", "orr"]
    titles = ["Average Reward", "Average Revenue", "Average Cost", "Average ORR"]
    
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        plt.plot(df_baseline["Episode"], df_baseline[metric], label="Baseline (MLP)", linestyle="--", color="gray")
        plt.plot(df_spatial["Episode"], df_spatial[metric], label="Spatial (Attention)", color="red", linewidth=2)
        plt.title(titles[i])
        plt.xlabel("Episode")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "comparison_result.png"), dpi=300)
    print(f"Comparison plot saved to '{os.path.join(log_dir, 'comparison_result.png')}'")

if __name__ == "__main__":
    run_comparison()
