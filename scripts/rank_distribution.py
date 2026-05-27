#!/usr/bin/env python3
"""per-event rank 分布 + 共识序列表 + violin 图。

输入：50 个 bestsoln_s*.dat（multistart 输出）
输出：results_py/rank_distribution/
  - consensus.csv         共识序列表（事件名, rank中位数, 95%CI宽度, 标注）
  - rank_violin.png       小提琴图（事件按共识 rank 排序）

用法：
    uv run python scripts/rank_distribution.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conop_py.io import parse_solution, solution_to_sequence, parse_loadfile, parse_events, infer_taxa_from_observations

DATA_DIR = ROOT / "results_py" / "multistart" / "2026-05-26_190029Z_b-region-test"
OUT_DIR = ROOT / "results_py" / "rank_distribution"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载实体名（用于可读的事件标签） ──
    obs = parse_loadfile(ROOT / "CONOP-run" / "loadfile.dat")
    ents = parse_events(ROOT / "CONOP-run" / "events.txt",
                        taxon_ids=infer_taxa_from_observations(obs))
    names: dict[int, str] = {e.id: e.name.split()[0] if e.name else f"T{e.id}" for e in ents}

    # ── 2. 解析所有 50 个 bestsoln ──
    soln_files = sorted(DATA_DIR.glob("bestsoln_s*.dat"))
    print(f"找到 {len(soln_files)} 个 bestsoln 文件")

    # event_ranks[event_key] = [rank_in_sol_1, rank_in_sol_2, ...]
    event_ranks: dict[tuple[int, int], list[int]] = defaultdict(list)
    for path in soln_files:
        sol = parse_solution(path)
        seq = solution_to_sequence(sol)
        for pos, ev in enumerate(seq, 1):  # 1-indexed
            event_ranks[ev].append(pos)

    n_events = len(event_ranks)
    n_sols = len(soln_files)
    print(f"共 {n_events} 个事件 × {n_sols} 个解")

    # ── 3. 计算共识序列 ──
    # 按 rank 中位数排序
    rows = []
    for ev_key, ranks in event_ranks.items():
        arr = np.array(ranks)
        median = float(np.median(arr))
        ci_lo = float(np.percentile(arr, 2.5))
        ci_hi = float(np.percentile(arr, 97.5))
        ci_width = ci_hi - ci_lo
        eid, etype = ev_key
        tag = names.get(eid, f"E{eid}")
        type_label = {1: "FAD", 2: "LAD", 4: "EVT", 5: "ASH"}.get(etype, f"T{etype}")
        label = f"{tag}_{type_label}"
        rows.append({
            "event_label": label,
            "entity_id": eid,
            "event_type": etype,
            "rank_median": median,
            "rank_ci_lo": ci_lo,
            "rank_ci_hi": ci_hi,
            "ci_width": ci_width,
            "n_solutions": len(ranks),
        })

    # 按 rank 中位数排序
    rows.sort(key=lambda r: r["rank_median"])

    # ── 4. 写共识序列表 ──
    csv_path = OUT_DIR / "consensus.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "event_label", "entity_id", "event_type",
            "rank_median", "rank_ci_lo", "rank_ci_hi", "ci_width", "n_solutions",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"→ {csv_path}  ({len(rows)} 事件)")

    # ── 5. 打印高置信 / 高矛盾事件 ──
    sorted_by_width = sorted(rows, key=lambda r: r["ci_width"])
    print()
    print("── 高置信事件（区间最窄 Top 10） ──")
    for r in sorted_by_width[:10]:
        print(f"  {r['event_label']:<20s}  rank中位数={r['rank_median']:6.1f}  95%CI=[{r['rank_ci_lo']:.0f},{r['rank_ci_hi']:.0f}]  宽度={r['ci_width']:.0f}")
    print()
    print("── 矛盾事件（区间最宽 Top 10） ──")
    for r in sorted_by_width[-10:]:
        print(f"  {r['event_label']:<20s}  rank中位数={r['rank_median']:6.1f}  95%CI=[{r['rank_ci_lo']:.0f},{r['rank_ci_hi']:.0f}]  宽度={r['ci_width']:.0f}")

    # ── 6. violin 图 ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from conop_py.plotting import setup_chinese_font
        setup_chinese_font()

        # 取前 30 个事件（最左侧）和事件标签
        plot_rows = rows  # 全部
        event_labels = [r["event_label"] for r in plot_rows]
        all_ranks = [event_ranks[(r["entity_id"], r["event_type"])] for r in plot_rows]

        fig, ax = plt.subplots(figsize=(14, 8))
        parts = ax.violinplot(all_ranks, positions=range(len(plot_rows)),
                              showmeans=False, showmedians=True)

        # 着色：按区间宽度
        widths = np.array([r["ci_width"] for r in plot_rows])
        max_w = max(widths) if max(widths) > 0 else 1
        for i, pc in enumerate(parts["bodies"]):
            w = widths[i]
            # 绿(窄)→红(宽)
            r = min(1, w / max_w * 1.5)
            g = max(0, 1 - w / max_w * 1.5)
            pc.set_facecolor((r, g, 0.2, 0.7))
            pc.set_edgecolor("none")

        ax.set_xticks(range(len(plot_rows)))
        ax.set_xticklabels(event_labels, rotation=90, fontsize=5)
        ax.set_ylabel("Composite Rank")
        ax.set_title("per-event Rank 分布（50 次多重启）\n绿色=高置信, 红色=高矛盾")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        png_path = OUT_DIR / "rank_violin.png"
        fig.savefig(png_path, dpi=200)
        print(f"→ {png_path}")
        plt.close(fig)
    except Exception as e:
        print(f"  ⚠ 画图失败 (matplotlib?): {e}")


if __name__ == "__main__":
    main()
