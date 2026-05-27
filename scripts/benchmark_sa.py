"""SA 基准 benchmark：固定 seed + 小预算，测一次完整 anneal 耗时。

用于对比优化前后的性能。运行：
    uv run --with-requirements requirements.txt python scripts/benchmark_sa.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conop_py.anneal import AnnealConfig, anneal
from conop_py.cost import ordinal_misfit
from conop_py.io import (
    infer_taxa_from_observations, parse_events, parse_loadfile,
)


def main(steps: int = 100, trials: int = 100, seed: int = 42):
    data = ROOT / "CONOP-run"
    obs = parse_loadfile(data / "loadfile.dat")
    entities = parse_events(data / "events.txt",
                            taxon_ids=infer_taxa_from_observations(obs))

    cfg = AnnealConfig(startemp=250.0, ratio=0.98,
                       steps=steps, trials=trials, seed=seed)

    print(f"配置: steps={steps} trials={trials} seed={seed}")
    print(f"内层 iter 总数: {steps * trials:,}")
    t0 = time.perf_counter()
    res = anneal(entities, obs, cfg, misfit_fn=ordinal_misfit, verbose=False)
    elapsed = time.perf_counter() - t0

    print(f"耗时: {elapsed:.2f} s")
    print(f"best_fit: {res.best_fit:.2f}")
    print(f"iters/sec: {steps*trials/elapsed:,.0f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.steps, args.trials, args.seed)
