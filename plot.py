import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class PlotCurveStyle:
    """单条曲线样式配置。"""

    label: str
    color: str
    linestyle: str
    linewidth: float = 2.0
    marker: Optional[str] = None


@dataclass(frozen=True)
class PlotStyle:
    """本脚本绘图样式统一配置（论文出图/调参友好）。"""

    # ====== 字体大小参数（统一调整） ======
    font_size: int = 20
    title_size_delta: int = 2

    # 图注/图例
    legend_loc: str = "best"
    legend_fontsize: Optional[int] = None  # None -> 跟随 rcParams/基础字号
    show_legend: bool = True

    # 网格
    show_grid: bool = True
    grid_alpha: float = 0.3

    # 画布
    overview_figsize: Tuple[float, float] = (15, 10)
    single_figsize: Tuple[float, float] = (10, 6)

    # 保存
    save_dpi: int = 600
    save_bbox_inches: str = "tight"


@dataclass(frozen=True)
class PlotConfig:
    """日志读取与输出相关配置。"""

    log_dir: str = "log"
    read_dir: str = "figure/compare"
    max_episodes: int = 80  # 统一截断量，只绘制前 max_episodes 条数据
    # 指标顺序（决定了子图/单图保存时 i=0..3 的含义）
    # i=0 -> total_reward（Reward）
    # i=1 -> revenue（Profit）
    # i=2 -> cost（Cost）
    # i=3 -> orr（ORR）
    metrics: Tuple[str, ...] = ("total_reward", "revenue", "cost", "orr")
    # overview 2x2 组图每个子图的标题（与 metrics 一一对应）
    titles: Tuple[str, ...] = (
        "Average Reward",
        "Average Profit",
        "Average Cost",
        "Average ORR",
    )
    # overview 2x2 子图的 y 轴标签（不要绑定到 metrics 名字）
    ylabels: Tuple[str, ...] = ("Reward", "Profit", "Cost", "ORR")
    # overview 2x2 子图的图例位置（可逐子图单独设置；None 表示使用 PlotStyle.legend_loc）
    overview_legend_locs: Tuple[Optional[str], ...] = (None, None, None, "upper left")

    # 单独保存的 4 张 PDF：y 轴标签与图例位置（可与 overview 分开配置）
    # - single_ylabels：每张单图的 y 轴标签
    # - single_legend_locs：每张单图的 legend 位置（None 表示使用 PlotStyle.legend_loc）
    single_ylabels: Tuple[str, ...] = ("Reward", "Profit", "Cost", "ORR")
    single_legend_locs: Tuple[Optional[str], ...] = (None, None, None, "upper left")
    tail_n: int = 10

    # 保存单指标图
    single_output_pattern: str = "comparison_{metric}.pdf"


def _apply_mpl_style(style: PlotStyle) -> None:
    """把可调的字号/图例字号统一写入 matplotlib rcParams。"""

    legend_fs = style.legend_fontsize if style.legend_fontsize is not None else style.font_size
    plt.rcParams.update(
        {
            "font.size": style.font_size,
            "axes.titlesize": style.font_size + style.title_size_delta,
            "axes.labelsize": style.font_size,
            "xtick.labelsize": style.font_size,
            "ytick.labelsize": style.font_size,
            "legend.fontsize": legend_fs,
        }
    )

def plot_comparison(
    *,
    cfg: PlotConfig = PlotConfig(),
    style: PlotStyle = PlotStyle(),
    model_styles: Optional[Sequence[Tuple[str, PlotCurveStyle]]] = None,
) -> None:
    _apply_mpl_style(style)

    if len(cfg.metrics) != len(cfg.titles):
        raise ValueError("PlotConfig.metrics 与 PlotConfig.titles 长度必须一致")
    if len(cfg.metrics) != len(cfg.ylabels):
        raise ValueError("PlotConfig.metrics 与 PlotConfig.ylabels 长度必须一致")
    if len(cfg.metrics) != len(cfg.overview_legend_locs):
        raise ValueError("PlotConfig.metrics 与 PlotConfig.overview_legend_locs 长度必须一致")
    if len(cfg.metrics) != len(cfg.single_ylabels):
        raise ValueError("PlotConfig.metrics 与 PlotConfig.single_ylabels 长度必须一致")
    if len(cfg.metrics) != len(cfg.single_legend_locs):
        raise ValueError("PlotConfig.metrics 与 PlotConfig.single_legend_locs 长度必须一致")

    # 定义需要对比的模型及其对应的文件前缀和绘图样式
    # 格式: (文件前缀模式, 曲线样式)
    if model_styles is None:
        model_styles = (
            (
                "saca_training_log_*.csv",
                PlotCurveStyle(label="ST-SACA", color="red", linestyle="-"),
            ),
            (
                "mlp_training_log_*.csv",
                PlotCurveStyle(label="SACA", color="gray", linestyle="--"),
            ),
            (
                "grc_elg_training_log_*.csv",
                PlotCurveStyle(label="GRC-ELG", color="blue", linestyle="-."),
            ),
            (
                "jdrl_training_log_*.csv",
                PlotCurveStyle(label="JDRL-POMO", color="green", linestyle=":"),
            ),
        )
    
    data_to_plot = []
    
    print("Searching for log files in:", cfg.read_dir)
    
    for pattern, curve_style in model_styles:
        search_path = os.path.join(cfg.read_dir, pattern)
        files = glob.glob(search_path)
        
        if not files:
            print(f"Warning: No files found for pattern '{pattern}'")
            continue
            
        # 获取最早的文件
        latest_file = sorted(files)[0]
        print(f"Found for {curve_style.label}: {latest_file}")
        
        try:
            df = pd.read_csv(latest_file)
            
            # 应用截断
            df = df.iloc[: cfg.max_episodes]
            
            data_to_plot.append({
                "df": df,
                "style": curve_style,
            })
        except Exception as e:
            print(f"Error reading {latest_file}: {e}")

    if not data_to_plot:
        print("No valid data found to plot.")
        return

    # 绘图指标
    # 计算并打印对比表格
    print("\n" + "="*60)
    print(
        f"Performance Comparison (Average over last {cfg.tail_n} episodes of the first {cfg.max_episodes})"
    )
    print("Baseline for %: Worst performing model = 100%")
    print("="*60)

    summary_data = {}
    
    for metric in cfg.metrics:
        # 1. 计算每个模型的平均值
        model_means = {}
        for item in data_to_plot:
            df = item["df"]
            if metric in df.columns:
                # 取最后 tail_n 条数据计算平均值
                model_means[item["style"].label] = df[metric].tail(cfg.tail_n).mean()
            else:
                model_means[item["style"].label] = float("nan")
        
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
            label = item["style"].label
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
    model_names = [item["style"].label for item in data_to_plot]
    summary_df = pd.DataFrame(summary_data, index=model_names)
    print(summary_df)
    print("="*60 + "\n")

    plt.figure(figsize=style.overview_figsize)
    
    for i, metric in enumerate(cfg.metrics):
        plt.subplot(2, 2, i+1)
        
        for item in data_to_plot:
            df = item["df"]
            if metric in df.columns:
                s: PlotCurveStyle = item["style"]
                plt.plot(
                    df["Episode"],
                    df[metric],
                    label=s.label,
                    color=s.color,
                    linestyle=s.linestyle,
                    linewidth=s.linewidth,
                    marker=s.marker,
                )
            else:
                print(f"Metric '{metric}' not found for {item['style'].label}")
        
        plt.title(cfg.titles[i])
        plt.xlabel("Episode")
        # y 轴标签从配置读取（不与 metrics 绑定）
        plt.ylabel(cfg.ylabels[i])
        if style.show_legend:
            loc = cfg.overview_legend_locs[i] or style.legend_loc
            plt.legend(loc=loc)
        if style.show_grid:
            plt.grid(True, alpha=style.grid_alpha)

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
    for i, metric in enumerate(cfg.metrics):
        plt.figure(figsize=style.single_figsize)
        
        for item in data_to_plot:
            df = item["df"]
            if metric in df.columns:
                s: PlotCurveStyle = item["style"]
                plt.plot(
                    df["Episode"],
                    df[metric],
                    label=s.label,
                    color=s.color,
                    linestyle=s.linestyle,
                    linewidth=s.linewidth,
                    marker=s.marker,
                )
        
        # plt.title(titles[i])
        plt.xlabel("Episode")
        # 单图的 y 轴标签从配置读取（不与 metrics 绑定）
        plt.ylabel(cfg.single_ylabels[i])
        if style.show_legend:
            loc = cfg.single_legend_locs[i] or style.legend_loc
            plt.legend(loc=loc)
        if style.show_grid:
            plt.grid(True, alpha=style.grid_alpha)
        plt.tight_layout()
        
        os.makedirs(cfg.read_dir, exist_ok=True)
        single_output_filename = cfg.single_output_pattern.format(metric=metric)
        single_output_path = os.path.join(cfg.read_dir, single_output_filename)
        plt.savefig(
            single_output_path, dpi=style.save_dpi, bbox_inches=style.save_bbox_inches
        )
        print(f"Saved '{single_output_path}'")
        plt.close()

if __name__ == "__main__":
    plot_comparison()
