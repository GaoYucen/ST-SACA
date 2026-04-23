import glob
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from st_saca.agents import st_saca as SACA
from st_saca.paths import OUTPUT_DIR
import tqdm


@dataclass(frozen=True)
class PlotCurveStyle:
    """单条曲线样式配置。"""

    color: str
    marker: str
    linestyle: str = "-"
    linewidth: float = 2.0
    label: str = ""


@dataclass(frozen=True)
class PlotStyle:
    """本脚本的绘图统一样式配置（便于论文图调参）。"""

    # 画布
    figsize: Tuple[float, float] = (10, 6)

    # 字号
    label_fontsize: int = 25
    legend_fontsize: int = 20

    # 网格
    show_grid: bool = True
    grid_alpha: float = 0.3

    # 曲线样式
    orr_curve: PlotCurveStyle = PlotCurveStyle(
        color="tab:orange", marker="o", linestyle="-", linewidth=2.0, label="ORR"
    )
    profit_curve: PlotCurveStyle = PlotCurveStyle(
        color="tab:green", marker="s", linestyle="--", linewidth=2.0, label="Profit"
    )

    # 轴与图例
    x_label: str = "Weight of ORR"
    y1_label: str = "Average ORR"
    y2_label: str = "Average Profit"
    legend_loc: str = "lower right"

    # 保存
    save_dpi: int = 600
    save_tight_layout: bool = True
    save_bbox_inches: str = "tight"


@dataclass(frozen=True)
class PlotConfig:
    """与日志读取/输出路径相关的配置。"""

    log_dir: str = str(OUTPUT_DIR / "logs")
    log_glob: str = "saca_training_log_*.csv"
    output_name: str = "parameter_sensitivity_orr.pdf"
    expected_files: int = 6
    tail_n: int = 10


def _apply_mpl_style(style: PlotStyle) -> None:
    """可选：把一些常用字号写入 rcParams，保证后续扩展图也一致。"""

    plt.rcParams.update(
        {
            "axes.labelsize": style.label_fontsize,
            "xtick.labelsize": style.label_fontsize,
            "ytick.labelsize": style.label_fontsize,
            "legend.fontsize": style.legend_fontsize,
        }
    )

def train_saca_model(x: Optional[Sequence[float]] = None) -> None:
    config = SACA.Config()
    if x is None:
        ORR = [0, 1, 2, 3, 4, 5]
    else:
        ORR = x
    for orr in tqdm.tqdm(ORR):
        config.lambda_or = orr
        SACA.train_saca(config)

def plot_saca_parameters(
    x: Optional[Sequence[float]] = None,
    *,
    cfg: PlotConfig = PlotConfig(),
    style: PlotStyle = PlotStyle(),
) -> None:
    _apply_mpl_style(style)

    # 获取所有相关的日志文件
    pattern = os.path.join(cfg.log_dir, cfg.log_glob)
    files = glob.glob(pattern)
    
    # 确保至少有 expected_files 个文件
    if len(files) < cfg.expected_files:
        print(
            f"Error: Expected at least {cfg.expected_files} log files, found {len(files)}."
        )
        return

    # 按修改时间排序，取最新的6个
    # 文件名示例：saca_training_log_20251218_210118.csv
    # 按日期时间(YYYYMMDD_HHMMSS)排序，取最新6个
    def _extract_dt_key(path: str) -> int:
        name = os.path.splitext(os.path.basename(path))[0]
        # 取最后两段：YYYYMMDD_HHMMSS
        date_part, time_part = name.split("_")[-2], name.split("_")[-1]
        return int(f"{date_part}{time_part}")  # 20251218210118

    sorted_files = sorted(files, key=_extract_dt_key, reverse=True)[: cfg.expected_files]

    # 如果希望后续 i 对应 lambda_vals 的顺序仍为 0..5（从小到大），则反转成旧->新
    sorted_files = list(reversed(sorted_files))
    
    lambda_vals: Sequence[float] = [0, 1, 2, 3, 4, 5] if x is None else x
    orrs = []
    revenues = []

    print("Processing files:")
    for i, f in enumerate(sorted_files):
        print(f"  Lambda={lambda_vals[i]}: {f}")
        try:
            df = pd.read_csv(f)
            # 取最后 tail_n 个 episode 的平均值
            if len(df) > 0:
                tail_df = df.tail(cfg.tail_n)
                orrs.append(tail_df["orr"].mean())
                revenues.append(tail_df["revenue"].mean())
            else:
                orrs.append(0)
                revenues.append(0)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            orrs.append(0)
            revenues.append(0)

    # 开始绘图
    os.makedirs(cfg.log_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=style.figsize)

    # 绘制 ORR (左轴)
    ax1.set_xlabel(style.x_label)
    ax1.set_ylabel(style.y1_label, color=style.orr_curve.color, fontsize=style.label_fontsize)
    line1 = ax1.plot(
        lambda_vals,
        orrs,
        color=style.orr_curve.color,
        marker=style.orr_curve.marker,
        linestyle=style.orr_curve.linestyle,
        linewidth=style.orr_curve.linewidth,
        label=style.orr_curve.label,
    )
    ax1.tick_params(axis="y", labelcolor=style.orr_curve.color)
    if style.show_grid:
        ax1.grid(True, alpha=style.grid_alpha)

    # 绘制 Revenue (右轴)
    ax2 = ax1.twinx()
    ax2.set_ylabel(
        style.y2_label, color=style.profit_curve.color, fontsize=style.label_fontsize
    )
    line2 = ax2.plot(
        lambda_vals,
        revenues,
        color=style.profit_curve.color,
        marker=style.profit_curve.marker,
        linestyle=style.profit_curve.linestyle,
        linewidth=style.profit_curve.linewidth,
        label=style.profit_curve.label,
    )
    ax2.tick_params(axis="y", labelcolor=style.profit_curve.color)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc=style.legend_loc, fontsize=style.legend_fontsize)

    # plt.title("Sensitivity Analysis: Lambda OR vs. ORR & Profit", fontsize=14)
    if style.save_tight_layout:
        plt.tight_layout()
    
    output_path = os.path.join(cfg.log_dir, cfg.output_name)
    plt.savefig(output_path, dpi=style.save_dpi, bbox_inches=style.save_bbox_inches)
    print(f"\nPlot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    x = [0, 100, 200, 300, 400, 500]
    # train_saca_model(x)
    plot_saca_parameters(x)

