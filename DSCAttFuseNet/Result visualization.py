# -*- coding: utf-8 -*-
"""
Multi-method line plots per metric for image fusion papers.
- CSV schema: Method, <Metric1>, <Metric2>, ...
- One figure per metric (no subplots), legend on the right, markers distinguish methods.
- No seaborn, no manual colors. Uses matplotlib only.
- Saves PNG and PDF.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======== 1) Paths (EDIT THESE) ========
CSV_PATH = r"./Metrics/output/LLVIP/fusion_metrics_LLVIP_all_methods.csv"   # <- 修改为你的CSV路径
OUT_DIR  = r"./Metrics/output/figs_LLVIP"               # <- 修改为你的输出目录

# ======== 2) Load & sanitize ========
df_raw = pd.read_csv(CSV_PATH)
assert "Method" in df_raw.columns, "CSV must contain a 'Method' column."

# 只保留数值型指标列
num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
metrics = [c for c in num_cols if c.lower() not in ["index", "idx"]]
df = df_raw[["Method"] + metrics].copy()

# 方法顺序（可按需要调整）
# —— 统一方法命名（修正别名/错拼/大小写）——
_alias = {
    'densfuse': 'DenseFuse',
    'densefuse': 'DenseFuse',
    'densfuse-net': 'DenseFuse',

    'psfuison': 'PSFusion',   # 常见错拼
    'psfusion': 'PSFusion',

    'mgff': 'MGFF',
    'piafusion': 'PIAFusion',
    'rfn-nest': 'RFN-Nest',
    'sdnet': 'SDNet',
    'seafusion': 'SeAFusion',
    'swinfusion': 'SwinFusion',
    'ours': 'our',
    'our': 'our',
}
def _normalize(name: str) -> str:
    key = str(name).strip().lower()
    return _alias.get(key, name)

df["Method"] = df["Method"].apply(_normalize)

# —— 偏好顺序：用于排序；不在列表中的“新增方法”会自动追加在末尾 ——
preferred_order = ['DenseFuse','MGFF','PIAFusion','PSFusion',
                   'RFN-Nest','SDNet','SeAFusion','SwinFusion','our']

present = df["Method"].tolist()
present_unique = []
for m in present:               # 保留出现顺序的去重
    if m not in present_unique:
        present_unique.append(m)

order_index = {name: i for i, name in enumerate(preferred_order)}
methods = sorted(
    present_unique,
    key=lambda x: order_index.get(x, len(preferred_order) + present_unique.index(x))
)

print("[INFO] Methods in CSV (normalized):", methods)


# 输出目录
out_dir = Path(OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

# ======== 3) 绘图参数 ========
markers = ['o','s','D','^','v','<','>','P','X','*','h','H','d']  # 区分方法
LINEWIDTH = 1.2
MARKERSIZE = 3.0
FIGSIZE = (8.5, 4.6)  # 把图像拉宽，给右侧图例留空间
RIGHT_PAD = 0.78      # 后面用 subplots_adjust 预留右边空白
MAX_TICKS = 12        # 控制 x 轴最多显示多少个刻度（避免太挤）

FONTSIZE = 10

plt.rcParams.update({
    "font.size": FONTSIZE,
    "axes.titlesize": FONTSIZE+2,
    "axes.labelsize": FONTSIZE,
    "legend.fontsize": FONTSIZE-2,
})

# ======== 4) 画单个指标的“多方法折线图” ========
def plot_metric_panel(metric_name: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    legend_entries = []

    for i, method in enumerate(methods):
        vals = df.loc[df["Method"] == method, metric_name].reset_index(drop=True).values
        x = np.arange(1, len(vals) + 1)
        # 折线 + 标记（不指定颜色）
        ax.plot(x, vals,
                marker=markers[i % len(markers)],
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE)

        legend_entries.append(f"{method} (mean={np.nanmean(vals):.3f})")

    # 稀疏 x 轴刻度（最多 MAX_TICKS 个）
    ticks = np.linspace(1, len(vals), min(MAX_TICKS, len(vals)), dtype=int)
    ax.set_xticks(ticks)

    ax.set_xlabel("Image Index")
    ax.set_ylabel(f"{metric_name} (↑)")
    ax.set_title(metric_name)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # 右侧外置长图例 + 缩短图例线条长度
    ax.legend(
        legend_entries,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        handlelength=1.8,
        markerscale=0.9,
        borderaxespad=0.0
    )

    # 关键：右边留白，避免图例挤压绘图区
    fig.subplots_adjust(right=RIGHT_PAD)
    fig.tight_layout()

    png_path = out_dir / f"{metric_name}_lines.png"
    pdf_path = out_dir / f"{metric_name}_lines.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ======== 5) 批量出图 & 导出统计表 ========
# 均值±标准差表（便于生成表4-1）
mean_df = df.groupby("Method")[metrics].mean().loc[methods]
std_df  = df.groupby("Method")[metrics].std().loc[methods]
mean_std_str = mean_df.round(3).astype(str) + " ± " + std_df.round(3).astype(str)
mean_std_str.to_csv(out_dir / "table_mean_std.csv")

# 名次表（1=最好）
rank_df = mean_df.rank(ascending=False, method='min')
rank_df.to_csv(out_dir / "table_ranks.csv")

# 出图
for m in metrics:
    plot_metric_panel(m)

print("Done. Figures saved to:", out_dir)
