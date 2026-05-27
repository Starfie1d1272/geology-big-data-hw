"""统一 CLI 入口：把零散的 SA / 分析脚本聚合成子命令。

用法：
    uv run --with-requirements requirements.txt python scripts/conop.py <command> [args]

子命令：
    validate           对 CONOP-run/ 的数据做完整性校验
    eval  <soln>       给定 bestsoln.dat 路径，计算 ordinal/level/eventual penalty
    one                跑一次 SA（默认 baseline 配置，保存 trajectory + bestsoln）
    bench              单次 SA 计时基准
    multistart         多进程并行 N 次重启
    sweep              参数组扫描（旧 run_py_sweep.py，未改动）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conop_py.io import (
    parse_loadfile, parse_events, parse_sections, parse_solution,
    solution_to_sequence, infer_taxa_from_observations, validate_dataset,
)


def _load_data(data_dir: Path):
    obs = parse_loadfile(data_dir / "loadfile.dat")
    taxon_ids = infer_taxa_from_observations(obs)
    ents = parse_events(data_dir / "events.txt", taxon_ids=taxon_ids)
    secs = parse_sections(data_dir / "sections.txt")
    return obs, ents, secs


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_validate(args):
    obs, ents, secs = _load_data(Path(args.data_dir))
    errs = validate_dataset(obs, ents, secs, strict=False)
    if errs:
        print(f"⚠ 发现 {len(errs)} 个问题:")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    print(f"✓ 数据集校验通过  "
          f"({len(secs)} sections, {len(ents)} entities, {len(obs)} observations)")


def cmd_eval(args):
    """对一个 bestsoln.dat 计算所有 penalty。"""
    from conop_py.cost import (
        ConopContext, build_section_observations,
        ordinal_misfit, level_misfit, eventual_misfit, coexistence_violations,
    )
    obs, _, _ = _load_data(Path(args.data_dir))
    sol = parse_solution(args.soln)
    seq = solution_to_sequence(sol)
    section_obs = build_section_observations(obs)
    ctx = ConopContext.build(seq, section_obs)

    ordi = ordinal_misfit(ctx)
    lev = level_misfit(ctx)
    evt = eventual_misfit(ctx)
    coex = coexistence_violations(ctx)
    print(f"解: {args.soln}")
    print(f"  长度: {len(seq)} 个事件")
    print(f"  Ordinal:  {ordi:>8.1f}  (CONOP9 bestsoln=367)")
    print(f"  Level:    {lev:>8.1f}  (CONOP9 真值=237)")
    print(f"  Eventual: {evt:>8.1f}  (CONOP9 真值=353)")
    print(f"  共存违反: {coex}")


def cmd_one(args):
    """跑一次 SA，保存 trajectory + bestsoln。"""
    from conop_py.anneal import AnnealConfig, anneal
    from conop_py.cost import ordinal_misfit, level_misfit

    obs, ents, _ = _load_data(Path(args.data_dir))

    if args.mode == "level":
        misfit_fn = level_misfit
        mode_label = "Level-only"
    else:
        misfit_fn = ordinal_misfit
        mode_label = "Ordinal"

    cfg = AnnealConfig(
        startemp=args.startemp, ratio=args.ratio,
        steps=args.steps, trials=args.trials, seed=args.seed,
        early_stop_patience=args.early_stop_patience,
        coex_penalty=args.coex_penalty,
    )
    print(f"配置: {mode_label}  STARTEMP={cfg.startemp} RATIO={cfg.ratio} "
          f"STEPS={cfg.steps} TRIALS={cfg.trials} seed={cfg.seed}"
          + (f"  coex_penalty={cfg.coex_penalty}" if cfg.coex_penalty > 0 else ""))
    t0 = time.perf_counter()
    res = anneal(ents, obs, cfg, misfit_fn=misfit_fn, verbose=args.verbose)
    el = time.perf_counter() - t0
    print(f"best_fit={res.best_fit:.2f}  steps_run={len(res.trajectory)}  耗时 {el:.2f}s")

    if args.out_soln:
        out = Path(args.out_soln)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for pos, (eid, etype) in enumerate(res.best_sequence, 1):
                f.write(f"{eid:>6d}{etype:>6d}{pos:>6d}\n")
        print(f"→ {out}")
    if args.out_traj:
        out = Path(args.out_traj)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write("step,temperature,current,best,accepted,proposed\n")
            for p in res.trajectory:
                f.write(f"{p.cooling_step},{p.temperature:.6f},{p.current_fit:.4f},"
                        f"{p.best_fit:.4f},{p.accepted},{p.proposed}\n")
        print(f"→ {out}")


def cmd_bench(args):
    """委托到 benchmark_sa.py。"""
    import scripts.benchmark_sa as m
    m.main(args.steps, args.trials, args.seed)


def cmd_multistart(args):
    """委托到 run_multistart.py。"""
    sys.argv = ["run_multistart.py",
                "--n", str(args.n), "--tag", args.tag,
                "--steps", str(args.steps), "--trials", str(args.trials),
                "--startemp", str(args.startemp), "--ratio", str(args.ratio),
                "--workers", str(args.workers),
                "--early-stop-patience", str(args.early_stop_patience),
                "--seed-base", str(args.seed_base)]
    if args.save_trajectories:
        sys.argv.append("--save-trajectories")
    import scripts.run_multistart as m
    m.main()


def cmd_sweep(args):
    """委托到 run_py_sweep.py（7 组参数 × 3 seed = 21 次扫描）。"""
    sys.argv = ["run_py_sweep.py", "--mode", args.mode]
    if args.tag:
        sys.argv += ["--tag", args.tag]
    if args.no_anchors:
        sys.argv.append("--no-anchors")
    import scripts.run_py_sweep as m
    m.main()


def cmd_plot_conv(args):
    """委托到 plot_py_convergence.py（单次轨迹收敛图）。"""
    import scripts.plot_py_convergence as m
    m.main()


def cmd_plot_sweep(args):
    """委托到 plot_sweep_comparison.py（21 次扫描对比图）。"""
    # 该脚本目前是顶层执行（无 main 函数），用 runpy 触发
    import runpy
    runpy.run_module("scripts.plot_sweep_comparison", run_name="__main__")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_common(p):
    p.add_argument("--data-dir", default="CONOP-run")


def build_parser():
    ap = argparse.ArgumentParser(prog="conop", description="CONOP Python 工具集")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate", help="数据集完整性校验")
    _add_common(p)
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("eval", help="评估 bestsoln.dat 的 penalty")
    _add_common(p)
    p.add_argument("soln", help="bestsoln.dat 路径")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("one", help="单次 SA 运行")
    _add_common(p)
    p.add_argument("--mode", choices=["ordinal", "level"], default="ordinal")
    p.add_argument("--coex-penalty", type=float, default=0)
    p.add_argument("--startemp", type=float, default=250.0)
    p.add_argument("--ratio", type=float, default=0.98)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--early-stop-patience", type=int, default=0)
    p.add_argument("--out-soln")
    p.add_argument("--out-traj")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=cmd_one)

    p = sub.add_parser("bench", help="单次 SA 计时基准")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("multistart", help="多进程多重启")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--tag", default="multistart")
    p.add_argument("--startemp", type=float, default=250.0)
    p.add_argument("--ratio", type=float, default=0.98)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--early-stop-patience", type=int, default=80)
    p.add_argument("--seed-base", type=int, default=10000)
    p.add_argument("--save-trajectories", action="store_true")
    p.set_defaults(fn=cmd_multistart)

    p = sub.add_parser("sweep", help="7 组参数 × 3 seed = 21 次扫描")
    p.add_argument("--mode", choices=["ordinal", "weighted", "level", "both"],
                   default="ordinal")
    p.add_argument("--tag", default="", help="文件夹后缀标签")
    p.add_argument("--no-anchors", action="store_true",
                   help="关闭 AGE/ASH 锚点约束")
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("plot-conv", help="Python 轨迹收敛图 → results_py/trajectory/convergence.png")
    p.set_defaults(fn=cmd_plot_conv)

    p = sub.add_parser("plot-sweep",
                       help="21 次扫描对比图 → results_py/sweep/comparison.png")
    p.set_defaults(fn=cmd_plot_sweep)

    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
