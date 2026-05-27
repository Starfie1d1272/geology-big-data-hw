#!/usr/bin/env python3
"""AGE 锚点裁判：关掉锚点约束跑 SA，看锚点是否自然保持。

原理：如果菊石数据本身足够强，即便关闭锚点约束，
SA 找到的最优解也应自然保持 K-Pg 界线、ASH 层的正确顺序。
如果关掉后顺序错乱，说明锚点是必要约束。

用法：
    uv run python scripts/anchor_judge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conop_py.anneal import AnnealConfig, anneal, build_anchor_order
from conop_py.cost import ConopContext, build_section_observations, level_misfit
from conop_py.io import parse_loadfile, parse_events, infer_taxa_from_observations


def main():
    data = ROOT / "CONOP-run"
    obs = parse_loadfile(data / "loadfile.dat")
    ents = parse_events(data / "events.txt", taxon_ids=infer_taxa_from_observations(obs))
    section_obs = build_section_observations(obs)

    anchor_order = build_anchor_order(obs)
    anchor_keys = set(anchor_order)
    print(f"锚点事件: {len(anchor_order)} 个")
    for i, a in enumerate(anchor_order, 1):
        print(f"  {i}. {a}")

    print()
    print(f"{'seed':>6s}  {'Level':>6s}  {'锚点顺序正确':>12s}  {'错误位置数':>10s}")
    print("-" * 40)

    for seed in [42, 17, 99, 123, 7]:
        cfg = AnnealConfig(startemp=250, ratio=0.98, steps=300, trials=200,
                           seed=seed, coex_penalty=4)
        res = anneal(ents, obs, cfg, misfit_fn=level_misfit, verbose=False,
                     use_anchors=False)

        # 检查锚点顺序
        actual = [ev for ev in res.best_sequence if ev in anchor_keys]
        order_ok = actual == anchor_order

        # 统计位置偏差
        if not order_ok:
            wrong_pos = sum(1 for a, b in zip(actual, anchor_order) if a != b)
        else:
            wrong_pos = 0

        print(f"{seed:>6d}  {res.best_fit:>6.0f}  {'✓' if order_ok else '✗':>12}  {wrong_pos:>10d}")

    print()
    print("结论：如果全部 ✓ → 菊石数据自洽，锚点只是辅助验证")
    print("      如果有 ✗ → 锚点是必要约束，没有它解不可靠")


if __name__ == "__main__":
    main()
