# 实现对比STSCAC和SACA在平稳状态和非平稳状态下的表现
# 计算两个算法在平稳状态下，车辆数量为10和20的情况
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import time
import numpy as np
from st_saca.paths import OUTPUT_DIR

def run():
    from st_saca.agents import st_saca as SACA
    from st_saca.agents import saca_baseline as SACA_baseline
    # 训练非平稳ST-SACA和SACA模型
    SACA_config = SACA.Config()
    SACA_baseline_config = SACA_baseline.Config()
    SACA_config.demand_fluctuation = 10  # 非平稳参数
    SACA_config.demand_frequency = 1
    SACA_baseline_config.demand_fluctuation = 10
    SACA_baseline_config.demand_frequency = 1
    SACA.train_saca(SACA_config)
    SACA_baseline.train_saca(SACA_baseline_config)

    # 训练平稳ST-SACA和SACA模型
    SACA_config_stationary = SACA.Config()
    SACA_baseline_config_stationary = SACA_baseline.Config()
    SACA_config_stationary.demand_fluctuation = 0  # 平稳参数
    SACA_config_stationary.demand_frequency = 0
    SACA_baseline_config_stationary.demand_fluctuation = 0
    SACA_baseline_config_stationary.demand_frequency = 0
    SACA.train_saca(SACA_config_stationary)
    SACA_baseline.train_saca(SACA_baseline_config_stationary)

def run2():
    from st_saca.agents import st_saca as SACA
    from st_saca.agents import saca_baseline as SACA_baseline
    # 训练平稳ST-SACA和SACA模型，座位数为30
    SACA_config_stationary = SACA.Config()
    SACA_baseline_config_stationary = SACA_baseline.Config()
    SACA_config_stationary.demand_fluctuation = 0  # 平稳参数
    SACA_config_stationary.demand_frequency = 0
    SACA_config_stationary.bus_capacity = 30
    SACA_baseline_config_stationary.demand_fluctuation = 0
    SACA_baseline_config_stationary.demand_frequency = 0
    SACA_baseline_config_stationary.bus_capacity = 30
    SACA.train_saca(SACA_config_stationary)
    SACA_baseline.train_saca(SACA_baseline_config_stationary)

    # 训练平稳ST-SACA和SACA模型，座位数为60
    SACA_config_stationary.bus_capacity = 60
    SACA_baseline_config_stationary.bus_capacity = 60
    SACA.train_saca(SACA_config_stationary)
    SACA_baseline.train_saca(SACA_baseline_config_stationary)


def make_stationary_table_last10():
    """读取最新两份日志并生成平稳场景对比表。

    约定：
    - 最新文件：20 buses
    - 次新文件：10 buses
    - 指标：Profit(=revenue), Cost, ORR 的最后10步均值
    """

    log_dir = str(OUTPUT_DIR / "logs")
    last_n = 10
    max_episodes = 80  # 先截断到前80步，再取最后10步(71-80)统计

    # 这里沿用你项目中的命名：mlp_training_log_* 代表 SACA，saca_training_log_* 代表 ST-SACA
    algo_patterns = [
        ("SACA", os.path.join(log_dir, "mlp_training_log_*.csv")),
        ("ST-SACA", os.path.join(log_dir, "saca_training_log_*.csv")),
    ]

    def _extract_dt_key(path: str) -> int:
        name = os.path.splitext(os.path.basename(path))[0]
        date_part, time_part = name.split("_")[-2], name.split("_")[-1]
        return int(f"{date_part}{time_part}")

    def _get_latest_two(files: list[str], algo_name: str) -> tuple[str, str]:
        if not files:
            raise FileNotFoundError(f"No log files found for {algo_name} in '{log_dir}'.")
        files_sorted = sorted(files, key=_extract_dt_key, reverse=True)
        if len(files_sorted) < 2:
            raise FileNotFoundError(
                f"Need at least 2 log files for {algo_name} (latest=20 buses, second=10 buses), but found {len(files_sorted)}."
            )
        return files_sorted[0], files_sorted[1]

    def _metrics_last10(df: pd.DataFrame) -> dict:
        required = ["revenue", "cost", "orr"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns {missing}. Found columns: {list(df.columns)}")

        # 固定统计窗口：先截断到前 max_episodes（默认80），避免取到全局最后的日志
        df = df.iloc[:max_episodes]
        tail = df.tail(last_n) if len(df) >= last_n else df
        return {
            "Profit": float(tail["revenue"].mean()),
            "Cost": float(tail["cost"].mean()),
            "ORR": float(tail["orr"].mean()),
        }

    rows = []
    selected_files = {}

    for algo_name, pattern in algo_patterns:
        files = glob.glob(pattern)
        latest_20, second_10 = _get_latest_two(files, algo_name)
        selected_files[algo_name] = {"20": latest_20, "10": second_10}

        df_20 = pd.read_csv(latest_20)
        df_10 = pd.read_csv(second_10)
        m20 = _metrics_last10(df_20)
        m10 = _metrics_last10(df_10)

        rows.append(
            {
                "Method": algo_name,
                "10 Profit": m10["Profit"],
                "10 Cost": m10["Cost"],
                "10 ORR": m10["ORR"],
                "20 Profit": m20["Profit"],
                "20 Cost": m20["Cost"],
                "20 ORR": m20["ORR"],
            }
        )

    table_df = pd.DataFrame(rows)
    # 列顺序固定一下
    table_df = table_df[["Method", "10 Profit", "10 ORR", "10 Cost", "20 Profit", "20 ORR", "20 Cost"]]

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(log_dir, f"stationary_table_last10_{timestamp}.csv")
    # table_df.to_csv(out_csv, index=False)

    print("\nSelected log files (latest=20 buses, second=10 buses):")
    for algo_name, info in selected_files.items():
        print(f"  {algo_name} 20 buses: {info['20']}")
        print(f"  {algo_name} 10 buses: {info['10']}")

    print(f"\nTable saved to: {out_csv}\n")
    # 控制台打印一份，便于复制
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", lambda x: f"{x:.4f}",
    ):
        print(table_df)

    return table_df

def plot_comparison():
    log_dir = str(OUTPUT_DIR / "logs")
    max_episodes = 80  # 只绘制前80步
    pattern_saca = os.path.join(log_dir, "mlp_training_log_*.csv")
    pattern_stsaca = os.path.join(log_dir, "saca_training_log_*.csv")

    files_saca = glob.glob(pattern_saca)
    files_stsaca = glob.glob(pattern_stsaca)

    def _extract_dt_key(path: str) -> int:
        name = os.path.splitext(os.path.basename(path))[0]
        date_part, time_part = name.split("_")[-2], name.split("_")[-1]
        return int(f"{date_part}{time_part}")

    def _get_latest_two(files: list[str], algo_name: str) -> tuple[str, str]:
        """Return (stationary_latest, non_stationary_second_latest)."""
        if not files:
            raise FileNotFoundError(f"No log files found for {algo_name} in '{log_dir}'.")
        files_sorted = sorted(files, key=_extract_dt_key, reverse=True)
        if len(files_sorted) < 2:
            raise FileNotFoundError(
                f"Need at least 2 log files for {algo_name} (latest=stationary, second=non-stationary), "
                f"but found {len(files_sorted)}."
            )
        return files_sorted[0], files_sorted[1]

    # 最新=平稳，次新=非平稳
    latest_saca_stationary, latest_saca_nonstationary = _get_latest_two(files_saca, "SACA")
    latest_stsaca_stationary, latest_stsaca_nonstationary = _get_latest_two(files_stsaca, "ST-SACA")

    df_saca_sta = pd.read_csv(latest_saca_stationary)
    df_saca_non = pd.read_csv(latest_saca_nonstationary)
    df_st_sta = pd.read_csv(latest_stsaca_stationary)
    df_st_non = pd.read_csv(latest_stsaca_nonstationary)

    episode_col = "Episode"
    metric_col = "revenue"

    # 统一截断：只绘制前 max_episodes 条
    df_saca_sta = df_saca_sta.iloc[:max_episodes]
    df_saca_non = df_saca_non.iloc[:max_episodes]
    df_st_sta = df_st_sta.iloc[:max_episodes]
    df_st_non = df_st_non.iloc[:max_episodes]

    for algo, df in [
        ("SACA(non)", df_saca_non),
        ("SACA(sta)", df_saca_sta),
        ("ST-SACA(non)", df_st_non),
        ("ST-SACA(sta)", df_st_sta),
    ]:
        if episode_col not in df.columns or metric_col not in df.columns:
            raise KeyError(
                f"{algo} log missing required columns: '{episode_col}' and '{metric_col}'. "
                f"Found columns: {list(df.columns)}"
            )

    plt.figure(figsize=(10, 6))
    plt.plot(df_saca_sta[episode_col], df_saca_sta[metric_col], label="SACA (Stationary)", color="blue", linestyle="--")
    plt.plot(df_saca_non[episode_col], df_saca_non[metric_col], label="SACA (Non-stationary)", color="blue", linestyle="-")
    plt.plot(df_st_sta[episode_col], df_st_sta[metric_col], label="ST-SACA (Stationary)", color="red", linestyle="--")
    plt.plot(df_st_non[episode_col], df_st_non[metric_col], label="ST-SACA (Non-stationary)", color="red", linestyle="-")
    plt.xlabel("Episode")
    plt.ylabel("Profit")
    plt.title("Profit curves: ST-SACA vs SACA (Non-stationary vs Stationary)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(log_dir, f"STvsSACA_stationary_vs_nonstationary_{timestamp}.pdf")
    plt.savefig(output_path, dpi=600)
    print("\nSelected log files:")
    print(f"  SACA Stationary:     {latest_saca_stationary}")
    print(f"  ST-SACA Stationary:     {latest_stsaca_stationary}")
    print(f"  SACA Non-stationary: {latest_saca_nonstationary}")
    print(f"  ST-SACA Non-stationary: {latest_stsaca_nonstationary}")
    print(f"\nPlot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    # run()  # 运行实验
    # run2()  # 运行实验（需要 torch）
    make_stationary_table_last10()
    # plot_comparison()

