#!/usr/bin/env python3
"""Python版 vs 原版 CONOP 参数扫描对比图。"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conop_py.plotting import setup_chinese_font
setup_chinese_font()

# ── 读数据 ──────────────────────────────────────────────────
orig_rows = list(csv.DictReader(open(ROOT / "scripts" / "summary.csv")))
py_rows = list(csv.DictReader(open(ROOT / "results_py" / "sweep" / "sweep_ordinal.csv")))

tags = ["baseline", "ratio_099", "ratio_095", "temp_500", "temp_100", "steps_1200", "steps_0300"]
labels = ["baseline\n(0.98/250/600)", "RATIO=0.99", "RATIO=0.95",
          "TEMP=500", "TEMP=100", "STEPS=1200", "STEPS=300"]

def get_vals(rows, key):
    """提取每组3次运行的均值与范围。"""
    means, mins, maxs = [], [], []
    for tag in tags:
        vals = []
        for i in range(1, 4):
            for r in rows:
                if r["实验组"] == tag and r["run_id"] == f"run_{i}":
                    vals.append(float(r[key]))
        arr = np.array(vals)
        means.append(arr.mean())
        mins.append(arr.min())
        maxs.append(arr.max())
    return np.array(means), np.array(mins), np.array(maxs)

orig_mean, orig_min, orig_max = get_vals(orig_rows, "best_fit")
py_mean, py_min, py_max = get_vals(py_rows, "ordinal_score")

# ── 绘图 ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

x = np.arange(len(tags))
w = 0.35

# 左图：分组条形图（均值 ± 范围）
ax = axes[0]
bars1 = ax.bar(x - w/2, orig_mean, w, label="CONOP 原版 (Level)", color="#E74C3C", alpha=0.85)
bars2 = ax.bar(x + w/2, py_mean, w, label="Python 版 (Ordinal)", color="#3498DB", alpha=0.85)

# 误差线：min/max
for i, (bar, lo, hi) in enumerate(zip(bars1, orig_min, orig_max)):
    ax.plot([bar.get_x() + bar.get_width()/2]*2, [lo, hi], 'k-', linewidth=1.2)
for i, (bar, lo, hi) in enumerate(zip(bars2, py_min, py_max)):
    ax.plot([bar.get_x() + bar.get_width()/2]*2, [lo, hi], 'k-', linewidth=1.2)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Best Fit")
ax.set_title("所有 21 次运行结果：均值和波动范围")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.25)

# 标注关键
ax.annotate("RATIO=0.95\n剧烈波动(220~265)", xy=(2, 265), xytext=(1.2, 280),
            fontsize=8, color="#C0392B",
            arrowprops=dict(arrowstyle="->", color="#C0392B"))
ax.annotate("RATIO=0.99\n收敛不足", xy=(1, 240), xytext=(-0.3, 235),
            fontsize=8, color="#C0392B",
            arrowprops=dict(arrowstyle="->", color="#C0392B"))
ax.annotate("STEPS=300\n劣化明显", xy=(6, 280), xytext=(5.2, 290),
            fontsize=8, color="#C0392B",
            arrowprops=dict(arrowstyle="->", color="#C0392B"))

# 右图：参数敏感性相关性散点
ax2 = axes[1]
# 归一化：以各自 baseline 为基准
o_norm = orig_mean / orig_mean[0]
p_norm = py_mean / py_mean[0]
ax2.scatter(o_norm, p_norm, c=range(7), cmap="tab10", s=120, zorder=5, edgecolors="k", linewidth=0.5)
for i, lab in enumerate(tags):
    ax2.annotate(lab, (o_norm[i], p_norm[i]), fontsize=8,
                 textcoords="offset points", xytext=(5, 5))
ax2.plot([0.96, 1.20], [0.96, 1.20], 'k--', alpha=0.3, label="y=x（理想一致）")
# 线性拟合
from numpy.polynomial.polynomial import polyfit
b, m = polyfit(o_norm, p_norm, 1)
x_fit = np.linspace(0.97, 1.19, 50)
ax2.plot(x_fit, b + m*x_fit, 'r-', alpha=0.4, linewidth=1, label=f"趋势线 (斜率={m:.2f})")

ax2.set_xlabel("原版 CONOP（相对 baseline）")
ax2.set_ylabel("Python 版（相对 baseline）")
ax2.set_title("参数敏感性相关性（归一化到各自 baseline）")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.25)

# 相关性注释
corr = np.corrcoef(o_norm, p_norm)[0, 1]
ax2.text(0.05, 0.92, f"Pearson r = {corr:.3f}", transform=ax2.transAxes, fontsize=11,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
out = ROOT / "results_py" / "sweep" / "comparison.png"
plt.savefig(out, dpi=140)
print(f"图片已保存: {out}")
