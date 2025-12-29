import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

def plot_comparison():
    log_dir = "log"
    read_dir = "figure/compare"
    max_episodes = 80  # 统一截断量，只绘制前80条数据
    
    # ====== 字体大小参数（统一调整） ======
    FONT_SIZE = 20  # 修改这个值即可调整所有字体大小
    
    # 应用全局字体设置
    plt.rcParams.update({
        'font.size': FONT_SIZE,           # 基础字体大小
        'axes.titlesize': FONT_SIZE + 2,  # 标题稍大
        'axes.labelsize': FONT_SIZE,  # 坐标轴标签
        'xtick.labelsize': FONT_SIZE, # X轴刻度
        'ytick.labelsize': FONT_SIZE, # Y轴刻度
        'legend.fontsize': FONT_SIZE - 8, # 图例
    })
    
    # 定义需要对比的模型及其对应的文件前缀和绘图样式
    # 格式: (文件前缀模式, 图例标签, 颜色, 线型)
    models_config = [
        ("saca_training_log_*.csv", "ST-SACA", "red", "-"),
        ("mlp_training_log_*.csv", "SACA", "gray", "--"),
        ("grc_elg_training_log_*.csv", "GRC-ELG", "blue", "-."),
        ("jdrl_training_log_*.csv", "JDRL-POMO", "green", ":")
    ]
    
    data_to_plot = []
    
    print("Searching for log files in:", read_dir)
    
    for pattern, label, color, linestyle in models_config:
        search_path = os.path.join(read_dir, pattern)
        files = glob.glob(search_path)
        
        if not files:
            print(f"Warning: No files found for pattern '{pattern}'")
            continue
            
        # 获取最早的文件
        latest_file = sorted(files)[0]
        print(f"Found for {label}: {latest_file}")
        
        try:
            df = pd.read_csv(latest_file)
            
            # 应用截断
            df = df.iloc[:max_episodes]
            
            data_to_plot.append({
                "df": df,
                "label": label,
                "color": color,
                "linestyle": linestyle
            })
        except Exception as e:
            print(f"Error reading {latest_file}: {e}")

    if not data_to_plot:
        print("No valid data found to plot.")
        return

    # 绘图指标
    metrics = ["total_reward", "revenue", "cost", "orr"]
    titles = ["Average Reward", "Average Profit", "Average Cost", "Average ORR"]
    
    # 计算并打印对比表格
    print("\n" + "="*60)
    print(f"Performance Comparison (Average over last 10 episodes of the first {max_episodes})")
    print("Baseline for %: Worst performing model = 100%")
    print("="*60)

    summary_data = {}
    
    for metric in metrics:
        # 1. 计算每个模型的平均值
        model_means = {}
        for item in data_to_plot:
            df = item["df"]
            if metric in df.columns:
                # 取最后10条数据计算平均值
                model_means[item["label"]] = df[metric].tail(10).mean()
            else:
                model_means[item["label"]] = float('nan')
        
        # 2. 确定基准值 (最差的模型)
        valid_values = [v for v in model_means.values() if not pd.isna(v)]
        if not valid_values:
            continue
            
        if metric == "cost":
            # Cost 越小越好，所以最大值是最差的
            baseline_value = max(valid_values)
        else:
            # 其他指标越大越好，所以最小值是最差的
            baseline_value = min(valid_values)
            
        # 3. 格式化输出
        metric_column = []
        for item in data_to_plot:
            label = item["label"]
            val = model_means[label]
            if pd.isna(val):
                metric_column.append("N/A")
            else:
                if baseline_value == 0:
                    percentage = 0.0
                else:
                    percentage = (val / baseline_value) * 100
                metric_column.append(f"{val:.2f} ({percentage:.1f}%)")
        
        # 将 'revenue' 的 key 改为 'profit'
        key_name = "profit" if metric == "revenue" else metric
        summary_data[key_name] = metric_column

    # 创建汇总 DataFrame
    model_names = [item["label"] for item in data_to_plot]
    summary_df = pd.DataFrame(summary_data, index=model_names)
    print(summary_df)
    print("="*60 + "\n")

    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        for item in data_to_plot:
            df = item["df"]
            if metric in df.columns:
                plt.plot(df["Episode"], df[metric], 
                         label=item["label"], 
                         color=item["color"], 
                         linestyle=item["linestyle"], 
                         linewidth=2)
            else:
                print(f"Metric '{metric}' not found for {item['label']}")
        
        plt.title(titles[i])
        plt.xlabel("Episode")
        # 如果是 revenue 指标，Y轴标签显示为 Profit
        ylabel = "Profit" if metric == "revenue" else metric
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # 保存图片 (已注释，改为分别保存)
    # output_filename = "comparison_result_4_models.png"
    # # 添加时间戳
    # timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    # output_filename = f"comparison_result_4_models_{timestamp}.png"
    # output_path = os.path.join(log_dir, output_filename)
    # # plt.savefig(output_path, dpi=300)
    # print(f"\nComparison plot saved to '{output_path}'")
    
    # 显示图片 (如果在支持显示的终端/IDE中)
    plt.show()
    # plt.close()

    # 分别保存四个指标的图片
    print("\nSaving separate plots for each metric...")
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(10, 6))
        
        for item in data_to_plot:
            df = item["df"]
            if metric in df.columns:
                plt.plot(df["Episode"], df[metric], 
                         label=item["label"], 
                         color=item["color"], 
                         linestyle=item["linestyle"], 
                         linewidth=2)
        
        # plt.title(titles[i])
        plt.xlabel("Episode")
        # 如果是 revenue 指标，Y轴标签显示为 Profit
        ylabel = "Profit" if metric == "revenue" else metric
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        single_output_filename = f"comparison_{metric}.pdf"
        single_output_path = os.path.join(read_dir, single_output_filename)
        plt.savefig(single_output_path, dpi=600)
        print(f"Saved '{single_output_path}'")
        plt.close()

if __name__ == "__main__":
    plot_comparison()
