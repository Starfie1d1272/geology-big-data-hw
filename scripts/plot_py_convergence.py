"""Python SA 收敛曲线 + 与原版 CONOP 对比图。

输入：
    results_py/traj_baseline_s42.csv          # Python 版（conop one --out-traj 生成）
    results/baseline/run_1/trajectory.txt     # 原版 CONOP（Windows 端跑的）

输出：
    results_py/convergence.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conop_py.plotting import (
    setup_chinese_font, load_python_trajectory, load_conop_trajectory,
)


def main() -> None:
    setup_chinese_font()
    py_traj = ROOT / "results_py" / "traj_baseline_s42.csv"
    orig_traj = ROOT / "results" / "baseline" / "run_1" / "trajectory.txt"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左：misfit 收敛曲线
    ax1 = axes[0]
    steps, temps_py, cur_py, best_py = load_python_trajectory(py_traj)
    ax1.plot(steps, cur_py, label="current", alpha=0.4, linewidth=0.7)
    ax1.plot(steps, best_py, label="best (Python, Ordinal)", linewidth=1.5, color="C0")

    if orig_traj.exists():
        try:
            _, _, _, best_orig = load_conop_trajectory(orig_traj)
            ax1.plot(range(len(best_orig)), best_orig,
                     label="best (CONOP 原版, LEVEL)",
                     linewidth=1.5, color="C3", alpha=0.7)
        except Exception as e:
            print(f"warn: 原版 trajectory 读取失败: {e}", file=sys.stderr)

    ax1.set_xlabel("降温步")
    ax1.set_ylabel("Misfit")
    ax1.set_title("收敛曲线对比")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 右：温度衰减（对数纵轴）
    ax2 = axes[1]
    ax2.semilogy(steps, temps_py, color="C2")
    ax2.set_xlabel("降温步")
    ax2.set_ylabel("温度 (log)")
    ax2.set_title("模拟退火温度衰减 (RATIO=0.98)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = ROOT / "results_py" / "convergence.png"
    plt.savefig(out, dpi=120)
    print(f"图片已保存: {out}")


if __name__ == "__main__":
    main()
