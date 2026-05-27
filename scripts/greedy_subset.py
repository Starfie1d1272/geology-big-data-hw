#!/usr/bin/env python3
"""贪心添加/删除实验：逐次加入或删除剖面，观察 Level 变化曲线。

正向：从最一致的剖面开始，逐次加入次一致的
反向：从全部 12 个开始，逐次删掉最矛盾的

目的：区分"SA 收敛困难"和"真实数据矛盾"。
  - 平滑上升 → SA 收敛问题
  - 突变跳跃 → 数据真实矛盾

用法：
    uv run python scripts/greedy_subset.py
"""
from __future__ import annotations

import csv
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conop_py.anneal import AnnealConfig, anneal
from conop_py.cost import ConopContext, build_section_observations, level_misfit, ordinal_misfit, coexistence_violations
from conop_py.io import parse_loadfile, parse_events, infer_taxa_from_observations

N_SEEDS = 3
SEEDS = [42, 17, 99]
DATA_DIR = ROOT / "CONOP-run"
OUT_DIR = ROOT / "results_py" / "greedy_subset"
CFG = AnnealConfig(startemp=250, ratio=0.98, steps=300, trials=200,
                    seed=42, coex_penalty=4)

# 按矛盾程度排序（来自 Jackknife Δ，最不矛盾 → 最矛盾）
JACKKNIFE_ORDER = [8, 9, 7, 6, 5, 11, 12, 4, 3, 1, 10, 2]


def run_fits(ents, obs_list, seeds):
    """给定观测列表和种子列表，返回每次运行的 best_fit 列表。"""
    fits = []
    for seed in seeds:
        cfg = replace(CFG, seed=seed)
        res = anneal(ents, obs_list, cfg, misfit_fn=level_misfit, verbose=False)
        fits.append(res.best_fit)
    return fits


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    obs_all = parse_loadfile(DATA_DIR / "loadfile.dat")
    ents = parse_events(DATA_DIR / "events.txt",
                        taxon_ids=infer_taxa_from_observations(obs_all))

    # 12 个剖面的观测分组
    by_sec: dict[int, list] = defaultdict(list)
    for o in obs_all:
        by_sec[o.section_id].append(o)

    all_sec_ids = sorted(by_sec.keys())

    # ── 正向添加 ──
    print("=== 正向添加 ===")
    forward_rows = []
    for i in range(1, len(JACKKNIFE_ORDER) + 1):
        selected = JACKKNIFE_ORDER[:i]
        jack_obs = [o for s in selected for o in by_sec[s]]
        fits = run_fits(ents, jack_obs, SEEDS)
        med = statistics.median(fits)
        print(f"  step {i:2d}: {len(selected):2d} 个剖面  Level median={med:.0f}  raw={fits}")
        forward_rows.append({
            "direction": "forward", "step": i, "n_sections": len(selected),
            "sections": str(selected), "median_fit": round(med, 1),
            "min_fit": round(min(fits), 1), "max_fit": round(max(fits), 1),
            "fits": str([round(f, 1) for f in fits]),
        })

    # ── 反向删除 ──
    print("\n=== 反向删除 ===")
    reverse_rows = []
    # 从全部开始，逐次删掉最矛盾的
    remaining = set(all_sec_ids)
    for i in range(0, len(JACKKNIFE_ORDER) + 1):
        if i == 0:
            selected = list(remaining)  # all 12
        else:
            removed = JACKKNIFE_ORDER[i - 1]
            remaining.discard(removed)
            selected = sorted(remaining)

        jack_obs = [o for s in selected for o in by_sec[s]]
        fits = run_fits(ents, jack_obs, SEEDS)
        med = statistics.median(fits)
        print(f"  step {i:2d}: {len(selected):2d} 个剖面  Level median={med:.0f}  raw={fits}")
        reverse_rows.append({
            "direction": "reverse", "step": i, "n_sections": len(selected),
            "sections": str(selected), "median_fit": round(med, 1),
            "min_fit": round(min(fits), 1), "max_fit": round(max(fits), 1),
            "fits": str([round(f, 1) for f in fits]),
        })

    # ── 写 CSV ──
    csv_path = OUT_DIR / "greedy_subset.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "direction", "step", "n_sections", "sections",
            "median_fit", "min_fit", "max_fit", "fits",
        ])
        w.writeheader()
        w.writerows(forward_rows + reverse_rows)
    print(f"\n→ {csv_path}")

    # ── 画图 ──
    try:
        from conop_py.plotting import init_plot, save_plot
        import matplotlib.pyplot as plt
        init_plot()

        fig, ax = plt.subplots(figsize=(10, 6))

        fwd = forward_rows
        rev = reverse_rows

        # 正向：横轴 = 剖面数
        ax.plot([r["n_sections"] for r in fwd], [r["median_fit"] for r in fwd],
                "o-", color="#2d6a4f", label="正向添加", linewidth=2, markersize=6)
        # 正向误差带
        ax.fill_between([r["n_sections"] for r in fwd],
                        [r["min_fit"] for r in fwd], [r["max_fit"] for r in fwd],
                        color="#2d6a4f", alpha=0.15)

        # 反向：横轴 = 剖面数 (从 12 到 1)
        rev_n = [r["n_sections"] for r in rev]
        ax.plot(rev_n, [r["median_fit"] for r in rev],
                "s--", color="#e63946", label="反向删除", linewidth=2, markersize=6)
        ax.fill_between(rev_n,
                        [r["min_fit"] for r in rev], [r["max_fit"] for r in rev],
                        color="#e63946", alpha=0.15)

        ax.set_xlabel("使用的剖面数")
        ax.set_ylabel("Level 中位数")
        ax.set_title("正反向贪心实验：Level vs 剖面数\n(3 seeds, 线条=median, 阴影=range)")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xticks(range(1, 13))

        png_path = OUT_DIR / "greedy_subset.png"
        save_plot(fig, png_path)
        print(f"→ {png_path}")
    except Exception as e:
        print(f"  ⚠ 画图失败: {e}")


if __name__ == "__main__":
    main()
