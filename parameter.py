import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import SACA
import tqdm

def train_saca_model(x=None):
    config = SACA.Config()
    if x is None:
        ORR = [0, 1, 2, 3, 4, 5]
    else:
        ORR = x
    for orr in tqdm.tqdm(ORR):
        config.lambda_or = orr
        SACA.train_saca(config)

def plot_saca_parameters(x=None):
    log_dir = "log"
    # 获取所有相关的日志文件
    pattern = os.path.join(log_dir, "saca_training_log_*.csv")
    files = glob.glob(pattern)
    
    # 确保至少有6个文件
    if len(files) < 6:
        print(f"Error: Expected at least 6 log files, found {len(files)}.")
        return

    # 按修改时间排序，取最新的6个
    # 文件名示例：saca_training_log_20251218_210118.csv
    # 按日期时间(YYYYMMDD_HHMMSS)排序，取最新6个
    def _extract_dt_key(path: str) -> int:
        name = os.path.splitext(os.path.basename(path))[0]
        # 取最后两段：YYYYMMDD_HHMMSS
        date_part, time_part = name.split("_")[-2], name.split("_")[-1]
        return int(f"{date_part}{time_part}")  # 20251218210118

    sorted_files = sorted(files, key=_extract_dt_key, reverse=True)[:6]

    # 如果希望后续 i 对应 lambda_vals 的顺序仍为 0..5（从小到大），则反转成旧->新
    sorted_files = list(reversed(sorted_files))
    
    if x is None:
        lambda_vals = [0, 1, 2, 3, 4, 5]
    else:
        lambda_vals = x
    orrs = []
    revenues = []

    print("Processing files:")
    for i, f in enumerate(sorted_files):
        print(f"  Lambda={lambda_vals[i]}: {f}")
        try:
            df = pd.read_csv(f)
            # 取最后10个episode的平均值
            if len(df) > 0:
                last_10 = df.tail(10)
                orrs.append(last_10["orr"].mean())
                revenues.append(last_10["revenue"].mean())
            else:
                orrs.append(0)
                revenues.append(0)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            orrs.append(0)
            revenues.append(0)

    # 开始绘图
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制 ORR (左轴)
    color = 'tab:orange'
    ax1.set_xlabel('Lambda OR (Weight of ORR)')
    ax1.set_ylabel('Average ORR', color=color, fontsize=12)
    line1 = ax1.plot(lambda_vals, orrs, color=color, marker='o', linewidth=2, label='ORR')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # 绘制 Revenue (右轴)
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Average Revenue', color=color, fontsize=12)
    line2 = ax2.plot(lambda_vals, revenues, color=color, marker='s', linestyle='--', linewidth=2, label='Revenue')
    ax2.tick_params(axis='y', labelcolor=color)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', fontsize=10)

    plt.title("Sensitivity Analysis: Lambda OR vs. ORR & Revenue", fontsize=14)
    plt.tight_layout()
    
    output_path = os.path.join(log_dir, "parameter_sensitivity_orr.pdf")
    plt.savefig(output_path, dpi=600)
    print(f"\nPlot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    x = [0, 100, 200, 300, 400, 500]
    # train_saca_model(x)
    plot_saca_parameters(x)

