"""共享绘图工具：中文字体配置 + trajectory 文件解析。

被 scripts/plot_*.py 复用，避免重复样板。
"""
from __future__ import annotations

import csv
from pathlib import Path


def setup_chinese_font() -> None:
    """matplotlib 用中文字体（macOS）。在任何画图脚本开头调一次。"""
    import matplotlib as mpl
    mpl.rcParams["font.family"] = ["PingFang HK", "STHeiti", "Heiti TC", "sans-serif"]
    mpl.rcParams["axes.unicode_minus"] = False


def load_python_trajectory(path: str | Path):
    """Python 版 SA trajectory.csv （由 scripts/conop.py one 或 run_multistart 生成）。

    返回 (steps, temps, currents, bests)。
    """
    steps, temps, currents, bests = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            temps.append(float(row["temperature"]))
            currents.append(float(row["current"]))
            bests.append(float(row["best"]))
    return steps, temps, currents, bests


def load_conop_trajectory(path: str | Path):
    """原版 CONOP9 trajectory.txt 格式：每行 'Temperature  Current  Best'。

    原版温度从高到低 → 这里反转为按"降温步"递增。
    返回 (steps, temps, currents, bests)。
    """
    temps, currents, bests = [], [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                t, c, b = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue
            temps.append(t)
            currents.append(c)
            bests.append(b)
    steps = list(range(len(temps)))
    return steps, temps[::-1], currents[::-1], bests[::-1]
