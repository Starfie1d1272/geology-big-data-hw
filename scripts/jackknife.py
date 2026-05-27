#!/usr/bin/env python3
"""Jackknife 稳健性检验：依次去掉 12 个剖面之一，看 Level 和序列变化。

原理：系统性地去掉每个剖面的观测数据，重新跑 SA，
  如果某个剖面去掉后 best_fit 显著下降（更好）→ 该剖面是主要矛盾源
  如果某个剖面去掉后 best_fit 显著上升（更差）→ 该剖面提供了关键约束
  如果 consensus 序列大幅重组 → 结论依赖于该剖面

用法：
    uv run python scripts/jackknife.py

输出：results_py/jackknife/
  - summary.csv    每组 5 次重启的统计
  - report.txt     可读报告
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

from conop_py.anneal import AnnealConfig, anneal, build_anchor_order
from conop_py.cost import ConopContext, build_section_observations, level_misfit, ordinal_misfit, coexistence_violations
from conop_py.io import parse_loadfile, parse_events, infer_taxa_from_observations

N_SEEDS = 5
SEEDS = [42, 17, 99, 123, 7]
DATA_DIR = ROOT / "CONOP-run"
OUT_DIR = ROOT / "results_py" / "jackknife"
ALLCFG = AnnealConfig(startemp=250, ratio=0.98, steps=300, trials=200,
                       seed=42, coex_penalty=4)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载完整数据 ──
    obs_all = parse_loadfile(DATA_DIR / "loadfile.dat")
    ents = parse_events(DATA_DIR / "events.txt",
                        taxon_ids=infer_taxa_from_observations(obs_all))
    section_ids = sorted({o.section_id for o in obs_all})
    print(f"完整数据: {len(section_ids)} 个剖面, {len(obs_all)} 条观测")

    # ── 2. 加载剖面名映射 ──
    from conop_py.io import parse_sections
    secs = parse_sections(DATA_DIR / "sections.txt")
    sec_names = {s.id: s.name for s in secs}

    # ── 3. 预跑完整数据作为基线 ──
    print(f"\n跑基线 (完整数据集, {N_SEEDS} 次)...")
    baseline_results = []
    for seed in SEEDS:
        cfg = replace(ALLCFG, seed=seed)
        res = anneal(ents, obs_all, cfg, misfit_fn=level_misfit, verbose=False)
        baseline_results.append(res.best_fit)
    baseline_median = statistics.median(baseline_results)
    print(f"  基线 Level: median={baseline_median:.0f}  range=[{min(baseline_results):.0f}, {max(baseline_results):.0f}]")

    # 基线 consensus 序列
    section_obs_all = build_section_observations(obs_all)

    # ── 4. Jackknife 主循环 ──
    rows = []
    for sec_id in section_ids:
        sec_name = sec_names.get(sec_id, f"Sec{sec_id}")
        # 过滤掉该剖面的观测
        jack_obs = [o for o in obs_all if o.section_id != sec_id]
        n_removed = len(obs_all) - len(jack_obs)
        print(f"\n[{sec_id}/{len(section_ids)}] 去掉 {sec_name} (sec {sec_id}, -{n_removed} 条观测)")

        fits = []
        all_seqs = []
        for seed in SEEDS:
            cfg = replace(ALLCFG, seed=seed)
            t0 = time.perf_counter()
            res = anneal(ents, jack_obs, cfg, misfit_fn=level_misfit, verbose=False)
            elapsed = time.perf_counter() - t0
            fits.append(res.best_fit)
            all_seqs.append(res.best_sequence)

        med = statistics.median(fits)
        delta = med - baseline_median  # + = 变量差(没了约束), - = 变好(了矛盾)
        min_f, max_f = min(fits), max(fits)

        # consensus 变化：去掉该剖面后，序列变化多大？
        # 用 median sequence vs baseline median sequence 的 Spearman ρ
        jack_ctx = ConopContext.build(
            all_seqs[0],  # 用第一个 seed 的解做比较（简化）
            build_section_observations(jack_obs),
        )
        jack_level = level_misfit(jack_ctx)
        jack_ord = ordinal_misfit(jack_ctx)
        jack_coex = coexistence_violations(jack_ctx)

        arrow = "🔴 变差" if delta > 5 else "🟢 变好" if delta < -5 else "⚪ 无显著变化"
        print(f"  Level: {med:.0f} (Δ={delta:+.0f})  range=[{min_f:.0f},{max_f:.0f}]  {arrow}")
        print(f"  Ordinal={jack_ord:.0f}  Coex={jack_coex}")

        rows.append({
            "sec_id": sec_id,
            "sec_name": sec_name,
            "n_removed": n_removed,
            "median_fit": round(med, 1),
            "delta": round(delta, 1),
            "min_fit": round(min_f, 1),
            "max_fit": round(max_f, 1),
            "ordinal": round(jack_ord),
            "coex": jack_coex,
            "arrow": arrow,
        })

    # ── 5. 写 summary.csv ──
    csv_path = OUT_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sec_id", "sec_name", "n_removed",
            "median_fit", "delta", "min_fit", "max_fit",
            "ordinal", "coex", "arrow",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"\n→ {csv_path}")

    # ── 6. 写可读报告 ──
    report_path = OUT_DIR / "report.txt"
    with open(report_path, "w") as f:
        f.write("Jackknife 稳健性检验结果\n")
        f.write(f"基线 Level（完整数据集）: median={baseline_median:.0f}\n")
        f.write(f"方法：依次去掉 12 个剖面之一，每个跑 {N_SEEDS} 次 Level-only SA\n")
        f.write(f"Δ > +5 = 去掉后变差（该剖面提供关键约束）\n")
        f.write(f"Δ < -5 = 去掉后变好（该剖面是矛盾主要来源）\n\n")
        f.write(f"{'剖面':<8s} {'名':<8s} {'去掉观测':>8s} {'median':>8s} {'Δ':>6s} {'min':>6s} {'max':>6s} {'Ord':>6s} {'Coex':>5s} {'含义':<16s}\n")
        f.write("-" * 90 + "\n")
        for r in sorted(rows, key=lambda x: abs(x["delta"]), reverse=True):
            f.write(f"{r['sec_id']:<8d} {r['sec_name']:<8s} {r['n_removed']:>8d} "
                    f"{r['median_fit']:>8.1f} {r['delta']:>+6.1f} {r['min_fit']:>6.1f} {r['max_fit']:>6.1f} "
                    f"{r['ordinal']:>6d} {r['coex']:>5d} {r['arrow']:<16s}\n")
        f.write("-" * 90 + "\n\n")

        # 按 Δ 排序
        f.write("按 Δ 降序（去掉后恶化最严重 → 关键剖面的依据）：\n")
        for r in sorted(rows, key=lambda x: x["delta"], reverse=True):
            f.write(f"  sec {r['sec_id']} ({r['sec_name']}): Δ={r['delta']:+.1f} {r['arrow']}\n")
        f.write("\n")
        f.write("按 Δ 升序（去掉后改善最显著 → 矛盾源）：\n")
        for r in sorted(rows, key=lambda x: x["delta"]):
            f.write(f"  sec {r['sec_id']} ({r['sec_name']}): Δ={r['delta']:+.1f} {r['arrow']}\n")

    print(f"→ {report_path}")

    # ── 7. 终端摘要 ──
    print(f"\n{'='*80}")
    print("Jackknife 摘要（按 |Δ| 降序）")
    print(f"{'剖面':>4s} {'名':<8s} {'Δ Level':>8s} {'含义':<20s}")
    print("-" * 50)
    for r in sorted(rows, key=lambda x: abs(x["delta"]), reverse=True):
        print(f"{r['sec_id']:>4d} {r['sec_name']:<8s} {r['delta']:>+8.1f} {r['arrow']:<20s}")


if __name__ == "__main__":
    main()
