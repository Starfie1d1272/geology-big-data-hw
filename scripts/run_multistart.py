"""B10: 多进程并行多重启 SA。

用途：跑 N 次独立 SA（不同 seed），输出每次的 bestsoln + best_fit + trajectory，
供后续 consensus / jackknife / rank-distribution 等不确定性分析使用。

运行：
    uv run --with-requirements requirements.txt python scripts/run_multistart.py \
        --n 50 --steps 600 --trials 300 --tag multistart-v1

输出目录：results_py/multistart/<timestamp>_<tag>/
    ├── manifest.json              # 实验参数 + git commit
    ├── summary.csv                # seed,best_fit,steps_run,elapsed_s
    ├── bestsoln_s<seed>.dat       # N 个最优解（CONOP 格式）
    └── traj_s<seed>.csv           # N 条轨迹（可选 --save-trajectories）
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conop_py.anneal import AnnealConfig, anneal
from conop_py.cost import ordinal_misfit
from conop_py.io import infer_taxa_from_observations, parse_events, parse_loadfile


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()[:12]
    except Exception:
        return "unknown"


def _run_one(args: tuple) -> dict:
    """单次 SA 运行（在子进程中执行）。返回汇总 dict。"""
    seed, cfg_dict, save_traj, out_dir = args
    cfg = AnnealConfig(**cfg_dict, seed=seed)

    data = ROOT / "CONOP-run"
    obs = parse_loadfile(data / "loadfile.dat")
    ents = parse_events(data / "events.txt",
                        taxon_ids=infer_taxa_from_observations(obs))

    t0 = time.perf_counter()
    res = anneal(ents, obs, cfg, misfit_fn=ordinal_misfit, verbose=False)
    elapsed = time.perf_counter() - t0

    # 保存 bestsoln
    out = Path(out_dir)
    soln_path = out / f"bestsoln_s{seed}.dat"
    with open(soln_path, "w") as f:
        for pos, (eid, etype) in enumerate(res.best_sequence, 1):
            f.write(f"{eid:>6d}{etype:>6d}{pos:>6d}\n")

    # 可选：保存轨迹
    traj_path = None
    if save_traj:
        traj_path = out / f"traj_s{seed}.csv"
        with open(traj_path, "w") as f:
            f.write("step,temperature,current,best,accepted,proposed\n")
            for p in res.trajectory:
                f.write(f"{p.cooling_step},{p.temperature:.6f},"
                        f"{p.current_fit:.4f},{p.best_fit:.4f},"
                        f"{p.accepted},{p.proposed}\n")

    return {
        "seed": seed,
        "best_fit": float(res.best_fit),
        "steps_run": len(res.trajectory),
        "elapsed_s": round(elapsed, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="重启次数")
    ap.add_argument("--startemp", type=float, default=250.0)
    ap.add_argument("--ratio", type=float, default=0.98)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--early-stop-patience", type=int, default=80,
                    help="0 关闭，>0 启用早停")
    ap.add_argument("--workers", type=int, default=0,
                    help="并行进程数，0=自动 (cpu_count-1)")
    ap.add_argument("--seed-base", type=int, default=10000)
    ap.add_argument("--tag", type=str, default="multistart")
    ap.add_argument("--save-trajectories", action="store_true")
    args = ap.parse_args()

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    cfg_dict = dict(
        startemp=args.startemp, ratio=args.ratio,
        steps=args.steps, trials=args.trials,
        early_stop_patience=args.early_stop_patience,
        use_fast_ordinal=True,
    )

    # 输出目录
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ")
    out = ROOT / "results_py" / "multistart" / f"{ts}_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)

    # 写 manifest
    manifest = {
        "timestamp_utc": ts,
        "git_commit": _git_commit(),
        "n_restarts": args.n,
        "workers": workers,
        "cfg": cfg_dict,
        "seed_base": args.seed_base,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    seeds = [args.seed_base + i for i in range(args.n)]
    tasks = [(s, cfg_dict, args.save_trajectories, str(out)) for s in seeds]

    print(f"启动 {args.n} 次重启 × {workers} 进程")
    print(f"配置: STARTEMP={args.startemp} RATIO={args.ratio} "
          f"STEPS={args.steps} TRIALS={args.trials} "
          f"early_stop={args.early_stop_patience}")
    print(f"输出: {out.relative_to(ROOT)}")

    t0 = time.perf_counter()
    rows = []
    if workers == 1:
        for t in tasks:
            rows.append(_run_one(t))
            print(f"  seed={rows[-1]['seed']:>5d}  best={rows[-1]['best_fit']:7.2f}  "
                  f"steps={rows[-1]['steps_run']:>3d}  t={rows[-1]['elapsed_s']:.1f}s")
    else:
        with mp.Pool(workers) as pool:
            for r in pool.imap_unordered(_run_one, tasks):
                rows.append(r)
                print(f"  [{len(rows):>3d}/{args.n}] seed={r['seed']:>5d}  "
                      f"best={r['best_fit']:7.2f}  steps={r['steps_run']:>3d}  "
                      f"t={r['elapsed_s']:.1f}s")

    elapsed = time.perf_counter() - t0
    rows.sort(key=lambda r: r["seed"])

    # 写 summary.csv
    import csv
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "best_fit", "steps_run", "elapsed_s"])
        w.writeheader()
        w.writerows(rows)

    fits = [r["best_fit"] for r in rows]
    print()
    print(f"完成。墙钟 {elapsed:.1f} s，CPU 加速 ≈ {sum(r['elapsed_s'] for r in rows)/elapsed:.1f}×")
    print(f"best_fit: min={min(fits):.2f}  median={sorted(fits)[len(fits)//2]:.2f}  "
          f"max={max(fits):.2f}  std={(sum((x-sum(fits)/len(fits))**2 for x in fits)/len(fits))**0.5:.2f}")


if __name__ == "__main__":
    main()
