#!/usr/bin/env python3
"""接受率曲线：对比 STARTEMP=100/250/500 的接受率 vs 温度。

用法：
    uv run python scripts/plot_acceptance_rate.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conop_py.anneal import AnnealConfig, anneal
from conop_py.cost import level_misfit
from conop_py.io import parse_loadfile, parse_events, infer_taxa_from_observations


def main():
    data = ROOT / "CONOP-run"
    obs = parse_loadfile(data / "loadfile.dat")
    ents = parse_events(data / "events.txt", taxon_ids=infer_taxa_from_observations(obs))

    configs = [
        ("TEMP=100", 100),
        ("TEMP=250", 250),
        ("TEMP=500", 500),
    ]

    trajectories = {}
    for label, startemp in configs:
        cfg = AnnealConfig(startemp=startemp, ratio=0.98, steps=600, trials=300,
                           seed=42, coex_penalty=4)
        res = anneal(ents, obs, cfg, misfit_fn=level_misfit, verbose=False)
        # Parse trajectory for accepted/proposed/temperature
        steps = []
        for pt in res.trajectory:
            steps.append({
                "step": pt.cooling_step,
                "T": pt.temperature,
                "accepted": pt.accepted,
                "proposed": pt.proposed or 300,
                "accept_rate": pt.accepted / max(pt.proposed or 300, 1),
                "best_fit": pt.best_fit,
            })
        trajectories[label] = steps
        print(f"  {label}: 最后 best_fit={res.best_fit:.0f}  {len(steps)} 步")

    # 画图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from conop_py.plotting import setup_chinese_font
    setup_chinese_font()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"TEMP=100": "#a8dadc", "TEMP=250": "#457b9d", "TEMP=500": "#e63946"}

    # 左图：接受率 vs ln(T)
    ax = axes[0]
    for label, steps in trajectories.items():
        temps = np.array([s["T"] for s in steps])
        rates = np.array([s["accept_rate"] for s in steps])
        ln_t = np.log(np.maximum(temps, 1e-3))
        ax.plot(ln_t, rates, color=colors[label], label=label, linewidth=1.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50%")
    ax.set_xlabel("ln(T)")
    ax.set_ylabel("接受率 (accepted / proposed)")
    ax.set_title("接受率 vs 对数温度")
    ax.legend()
    ax.grid(alpha=0.3)

    # 右图：接受率 vs 降温步数
    ax2 = axes[1]
    for label, steps in trajectories.items():
        x = [s["step"] for s in steps]
        rates = [s["accept_rate"] for s in steps]
        ax2.plot(x, rates, color=colors[label], label=label, linewidth=1.5)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50%")
    ax2.set_xlabel("降温步数 (cooling step)")
    ax2.set_ylabel("接受率")
    ax2.set_title("接受率 vs 降温步数")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    png_path = ROOT / "results_py" / "acceptance_rate.png"
    fig.savefig(png_path, dpi=200)
    print(f"→ {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
