#!/usr/bin/env python3
"""绘制 Python 版 CONOP 的收敛曲线，与原版对比。"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = ["PingFang HK", "STHeiti", "Heiti TC", "sans-serif"]
mpl.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent


def load_traj(path):
    steps, temps, currents, bests = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            temps.append(float(row["temperature"]))
            currents.append(float(row["current"]))
            bests.append(float(row["best"]))
    return steps, temps, currents, bests


def load_orig_traj(path):
    """原版 trajectory.txt: 'Temperature  Current  Best' 形式。"""
    temps, currents, bests = [], [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                t = float(parts[0])
                c = float(parts[1])
                b = float(parts[2])
            except ValueError:
                continue
            temps.append(t)
            currents.append(c)
            bests.append(b)
    # 原版温度从高到低，这里反转为按"迭代步"递增
    return list(range(len(temps))), temps[::-1], currents[::-1], bests[::-1]


def main():
    py_traj = ROOT / "results_py" / "traj_baseline_s42.csv"
    orig_traj = ROOT / "results" / "baseline" / "run_1" / "trajectory.txt"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左：misfit 收敛曲线
    ax1 = axes[0]
    steps, temps_py, cur_py, best_py = load_traj(py_traj)
    ax1.plot(steps, cur_py, label="current", alpha=0.4, linewidth=0.7)
    ax1.plot(steps, best_py, label="best (Python, Ordinal)", linewidth=1.5, color="C0")

    if orig_traj.exists():
        try:
            _, _, _, best_orig = load_orig_traj(orig_traj)
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

    # 右：温度衰减
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
