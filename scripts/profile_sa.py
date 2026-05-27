"""快速 profile SA 热路径，找出真正的瓶颈。"""
from __future__ import annotations

import cProfile
import pstats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conop_py.anneal import AnnealConfig, anneal
from conop_py.cost import ordinal_misfit
from conop_py.io import infer_taxa_from_observations, parse_events, parse_loadfile


def main():
    data = ROOT / "CONOP-run"
    obs = parse_loadfile(data / "loadfile.dat")
    ents = parse_events(data / "events.txt", taxon_ids=infer_taxa_from_observations(obs))
    cfg = AnnealConfig(steps=50, trials=200, seed=42)

    pr = cProfile.Profile()
    pr.enable()
    anneal(ents, obs, cfg, misfit_fn=ordinal_misfit, verbose=False)
    pr.disable()

    ps = pstats.Stats(pr).sort_stats("cumulative")
    ps.print_stats(20)


if __name__ == "__main__":
    main()
