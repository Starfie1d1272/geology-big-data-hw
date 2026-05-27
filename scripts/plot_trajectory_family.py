#!/usr/bin/env python3
"""轨迹族图 + 收敛步统计。

输入：results/<tag>/run_*/trajectory.txt（21 × 3 = 63 文件）
输出：results_py/
  - trajectory_family.png  21 条轨迹按参数组着色叠加
  - convergence_stats.csv  每次运行的收敛步统计

用法：
    uv run python scripts/plot_trajectory_family.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conop_py.plotting import setup_chinese_font, load_conop_trajectory


TAGS = ["baseline", "ratio_099", "ratio_095", "temp_500", "temp_100", "steps_1200", "steps_0300"]
LABELS = ["baseline\n(0.98/250/600)", "RATIO=0.99", "RATIO=0.95",
          "TEMP=500", "TEMP=100", "STEPS=1200", "STEPS=300"]
COLORS = ["#2d6a4f", "#e63946", "#f4a261", "#457b9d", "#a8dadc", "#6a4c93", "#e08b57"]


def find_convergence_step(temps, bests, threshold=0.001):
    """找到 best_fit 最后一次改善超过 threshold 的步数。

    注意：load_conop_trajectory 反转后 bests[0] 是最优值（最小），
    bests[-1] 是初始值（最大）。我们找首次达到最终最佳值的步数。
    """
    if len(bests) < 2:
        return 0
    target = min(bests)  # 最优值（最后一个改善到的最优）
    # 从后往前（从初始值往最优值方向）找首次达到 target 的步数
    # 反转后 bests 从最优到初始，所以从后往前实际上是从初始往最优
    for i in range(len(bests) - 1, -1, -1):
        if abs(bests[i] - target) < threshold:
            return i
    return 0


def main():
    setup_chinese_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── 左图：轨迹族图 ──
    ax = axes[0]
    conv_data = []  # [(tag, run, converge_step, best_fit)]

    for tag_idx, tag in enumerate(TAGS):
        for run in range(1, 4):
            traj_path = ROOT / "results" / tag / f"run_{run}" / "trajectory.txt"
            if not traj_path.exists():
                continue
            steps, temps, currents, bests = load_conop_trajectory(traj_path)
            if not steps:
                continue

            # 归一化到 0-1（各 runs steps 数不同）
            x = [s / max(steps) for s in steps]
            ax.plot(x, bests, color=COLORS[tag_idx], alpha=0.6, linewidth=0.8)

            conv_step = find_convergence_step(temps, bests)
            final_best = bests[0] if bests else None  # bests[0] = 最终值（反转后第一个）
            conv_data.append({
                "tag": tag, "run": run, "n_steps": len(steps),
                "converge_step": conv_step,
                "converge_ratio": conv_step / max(steps) if max(steps) > 0 else 0,
                "best_fit": final_best,
            })

    # 图例
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color=COLORS[i], label=LABELS[i])
                      for i in range(len(TAGS))]
    ax.legend(handles=legend_handles, fontsize=7)
    ax.set_xlabel("Cooling Step (归一化)")
    ax.set_ylabel("Best Fit (Level + Teaser)")
    ax.set_title("收敛轨迹族图（21 次 CONOP 运行）")
    ax.grid(alpha=0.3)

    # ── 右图：收敛步直方图 ──
    ax2 = axes[1]
    for tag_idx, tag in enumerate(TAGS):
        vals = [d["converge_ratio"] for d in conv_data if d["tag"] == tag]
        if not vals:
            continue
        ax2.scatter([tag_idx] * len(vals), vals, color=COLORS[tag_idx],
                    alpha=0.7, s=40, zorder=3)
        mean_val = sum(vals) / len(vals)
        ax2.plot([tag_idx - 0.2, tag_idx + 0.2], [mean_val, mean_val],
                 color=COLORS[tag_idx], linewidth=2)
    ax2.set_xticks(range(len(TAGS)))
    ax2.set_xticklabels(LABELS, fontsize=7)
    ax2.set_ylabel("收敛点（占总步数比例）")
    ax2.set_title("收敛步分布（点=单次, 线=均值）")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    png_path = ROOT / "results_py" / "trajectory_family.png"
    fig.savefig(png_path, dpi=200)
    print(f"→ {png_path}")
    plt.close(fig)

    # ── 收敛步统计表 ──
    csv_path = ROOT / "results_py" / "convergence_stats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "tag", "run", "best_fit", "n_steps", "converge_step", "converge_ratio"])
        w.writeheader()
        w.writerows(conv_data)
    print(f"→ {csv_path}")

    # 摘要
    print()
    print("── 收敛步统计摘要（均值）──")
    for tag in TAGS:
        vals = [d for d in conv_data if d["tag"] == tag]
        if not vals:
            continue
        mean_ratio = sum(d["converge_ratio"] for d in vals) / len(vals)
        mean_fit = sum(d["best_fit"] for d in vals) / len(vals)
        print(f"  {tag:<15s}  收敛比={mean_ratio:.2f}  best_fit均值={mean_fit:.2f}")


if __name__ == "__main__":
    main()
