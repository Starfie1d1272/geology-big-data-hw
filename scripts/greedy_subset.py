#!/usr/bin/env python3
"""更新 greedy_subset 结果：用 steps=600 重跑第 8-12 步正向。

输出覆盖 results_py/greedy_subset/ 下的 CSV 和 PNG。
"""
from __future__ import annotations

import csv
import statistics
import sys
import time as time_mod
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conop_py.anneal import AnnealConfig, anneal
from conop_py.io import parse_loadfile, parse_events, infer_taxa_from_observations

DATA_DIR = ROOT / "CONOP-run"
OUT_DIR = ROOT / "results_py" / "greedy_subset"
CFG = AnnealConfig(startemp=250, ratio=0.98, steps=600, trials=300,
                    seed=42, coex_penalty=4)
SEEDS = [42, 17, 99]
ORDER = [8, 9, 7, 6, 5, 11, 12, 4, 3, 1, 10, 2]


def run_fits(ents, obs_list):
    fits = []
    for seed in SEEDS:
        cfg = replace(CFG, seed=seed)
        res = anneal(ents, obs_list, cfg, misfit_fn=lambda ctx: __import__('conop_py.cost', fromlist=['level_misfit']).level_misfit(ctx), verbose=False)
        fits.append(res.best_fit)
    return fits


def main():
    import conop_py.cost as cost_mod
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    obs_all = parse_loadfile(DATA_DIR / "loadfile.dat")
    ents = parse_events(DATA_DIR / "events.txt",
                        taxon_ids=infer_taxa_from_observations(obs_all))
    by_sec = defaultdict(list)
    for o in obs_all:
        by_sec[o.section_id].append(o)

    rows = []
    print(f"{'step':>4s} {'n_sec':>5s} {'加入':>10s} {'median':>6s} {'min':>5s} {'max':>5s} {'种子':>20s}")
    print("-" * 65)

    for i in range(1, len(ORDER) + 1):
        selected = ORDER[:i]
        obs_list = [o for s in selected for o in by_sec[s]]
        fits = run_fits(ents, obs_list)
        med = statistics.median(fits)
        added = str(selected[-1])
        print(f"{i:4d} {len(selected):5d} {added:>10s} {med:6.0f} {min(fits):5.0f} {max(fits):5.0f} {str([round(f,1) for f in fits]):>20s}")

        rows.append({
            "direction": "forward",
            "step": i,
            "n_sections": len(selected),
            "sections": str(selected),
            "median_fit": round(med, 1),
            "min_fit": round(min(fits), 1),
            "max_fit": round(max(fits), 1),
            "fits": str([round(f, 1) for f in fits]),
            "added_section": added,
        })

    # 写 CSV
    csv_path = OUT_DIR / "greedy_subset.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "direction", "step", "n_sections", "sections",
            "median_fit", "min_fit", "max_fit", "fits", "added_section",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"\n→ {csv_path}")

    # 画图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from conop_py.plotting import setup_chinese_font
        setup_chinese_font()

        fig, ax = plt.subplots(figsize=(10, 6))
        ns = [r["n_sections"] for r in rows]
        meds = [r["median_fit"] for r in rows]
        mins = [r["min_fit"] for r in rows]
        maxs = [r["max_fit"] for r in rows]

        ax.plot(ns, meds, "o-", color="#2d6a4f", linewidth=2, markersize=7, label="正向添加")
        ax.fill_between(ns, mins, maxs, color="#2d6a4f", alpha=0.12)

        # 标注每次加的剖面号
        for i, r in enumerate(rows):
            if i > 0:
                ax.annotate(f"+sec{r['added_section']}", (ns[i], meds[i]),
                            textcoords="offset points", xytext=(5, 10),
                            fontsize=7, alpha=0.7)

        ax.set_xlabel("剖面数")
        ax.set_ylabel("Level 中位数（3 seeds）")
        ax.set_title("正向贪心添加（steps=600）\n从最一致剖面逐次加入最矛盾剖面")
        ax.set_xticks(range(1, 13))
        ax.legend()
        ax.grid(alpha=0.3)

        png_path = OUT_DIR / "greedy_subset.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=200)
        print(f"→ {png_path}")
        plt.close(fig)
    except Exception as e:
        print(f"  ⚠ 画图失败: {e}")


if __name__ == "__main__":
    main()
