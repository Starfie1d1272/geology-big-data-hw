#!/usr/bin/env python3
"""Python 版 CONOP 参数扫描脚本。

镜像原版 batch_run.bat 的 7 组参数 × 3 次重复，全自动运行（无需手动操作）。
结果保存到 results_py/sweep/，并与原版 summary.csv 横向对比。

用法：
    uv run python scripts/run_py_sweep.py
    uv run python scripts/run_py_sweep.py --mode ordinal       # 默认
    uv run python scripts/run_py_sweep.py --mode weighted      # 加权 Ordinal
    uv run python scripts/run_py_sweep.py --mode level         # Level penalty 优化（改进版）
    uv run python scripts/run_py_sweep.py --mode both          # 两种都跑，输出对比
结果保存到 results_py/sweep/<timestamp>/，每次运行自动创建新文件夹，
不会覆盖历史数据。文件夹内含：
  - ordinal.csv / level.csv / weighted.csv  结果数据
  - manifest.json                           运行元信息
"""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conop_py.io import parse_loadfile, parse_events, infer_taxa_from_observations
from conop_py.cost import (
    ConopContext, build_section_observations,
    ordinal_misfit, level_misfit, eventual_misfit, combined_misfit,
    build_pairwise_support, weighted_ordinal_misfit,
)
from conop_py.anneal import anneal, AnnealConfig

# ── 与 batch_run.bat 完全对应的 7 组参数 ──────────────────────────────────────
PARAM_GROUPS = [
    # (tag,          ratio,  startemp, steps)
    ("baseline",     0.980,  250,      600),
    ("ratio_099",    0.990,  250,      600),
    ("ratio_095",    0.950,  250,      600),
    ("temp_500",     0.980,  500,      600),
    ("temp_100",     0.980,  100,      600),
    ("steps_1200",   0.980,  250,      1200),
    ("steps_0300",   0.980,  250,      300),
]

SEEDS = [42, 17, 99]   # 对应原版的 run_1/2/3（原版无固定 seed，这里用固定 seed 保证可重现）
TRIALS = 300           # 与原版 TRIALS=300 一致


def load_orig_summary(path: Path) -> list[dict]:
    """读取原版 summary.csv。"""
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def run_sweep(mode: str, use_anchors: bool = True) -> list[dict]:
    """运行全部 21 次扫描，返回结果列表。

    mode:
        ordinal  — 优化 ordinal_misfit
        weighted — 优化加权 ordinal_misfit
        level    — 多目标：level + eventual（比例 2:1）
    """
    obs = parse_loadfile(ROOT / "CONOP-run" / "loadfile.dat")
    entities = parse_events(
        ROOT / "CONOP-run" / "events.txt",
        taxon_ids=infer_taxa_from_observations(obs),
    )
    section_obs = build_section_observations(obs)

    # 选择 misfit 函数
    misfit_weights = None
    if mode == "weighted":
        misfit_fn = None
        use_weighted = True
    elif mode == "level":
        misfit_fn = None
        misfit_weights = {'ordinal': 3.0, 'level': 1.0, 'eventual': 1.0}
        use_weighted = False
    else:  # ordinal (default)
        misfit_fn = ordinal_misfit
        use_weighted = False

    results = []
    total = len(PARAM_GROUPS) * len(SEEDS)
    done = 0

    for tag, ratio, startemp, steps in PARAM_GROUPS:
        for run_idx, seed in enumerate(SEEDS, 1):
            done += 1
            cfg = AnnealConfig(
                startemp=startemp, ratio=ratio, steps=steps,
                trials=TRIALS, seed=seed,
            )
            kw = dict(
                misfit_fn=misfit_fn,
                misfit_weights=misfit_weights,
                use_weighted=use_weighted,
                use_anchors=use_anchors,
                verbose=False,
            )
            print(f"[{done:2d}/{total}] {tag} run_{run_idx}  "
                  f"RATIO={ratio} STARTEMP={startemp} STEPS={steps} seed={seed} "
                  f"mode={mode}",
                  end=" ... ", flush=True)

            res = anneal(entities, obs, cfg, **kw)

            # 用 ConopContext 统一评估
            eval_ctx = ConopContext.build(res.best_sequence, section_obs)
            ord_score = ordinal_misfit(eval_ctx)
            lev_score = level_misfit(eval_ctx)
            evt_score = eventual_misfit(eval_ctx)

            print(f"best_fit={res.best_fit:.2f}  "
                  f"ordinal={ord_score:.1f}  level={lev_score:.1f}  "
                  f"eventual={evt_score:.1f}")

            results.append({
                "实验组": tag, "run_id": f"run_{run_idx}",
                "seed": seed, "RATIO": ratio, "STARTEMP": startemp, "STEPS": steps,
                "mode": mode,
                "best_fit_reported": res.best_fit,
                "ordinal_score": ord_score,
                "level_score": lev_score,
                "eventual_score": evt_score,
            })

    return results


def print_comparison(py_results: list[dict], orig_rows: list[dict], mode: str) -> None:
    """横向对比 Python 版与原版结果。"""
    # 原版数据按 (tag, run_id) 索引
    orig = {(r["实验组"], r["run_id"]): float(r["best_fit"]) for r in orig_rows}

    print()
    print("=" * 80)
    print(f"{'实验组':<14} {'run':>5} {'原版(Level)':>12} {'Python(Ordinal)':>16} {'差值%':>8}")
    print("-" * 80)

    for row in py_results:
        tag = row["实验组"]
        run_id = row["run_id"]
        py_ord = row["ordinal_score"]
        orig_val = orig.get((tag, run_id), None)
        diff_str = ""
        if orig_val:
            diff = (py_ord - orig_val) / orig_val * 100
            diff_str = f"{diff:+.1f}%"
        orig_str = f"{orig_val:.2f}" if orig_val else "—"
        print(f"{tag:<14} {run_id:>5} {orig_str:>12} {py_ord:>16.2f} {diff_str:>8}")

    print("=" * 80)
    print()
    print("注：原版优化 Level Penalty，Python 版优化 Ordinal Penalty，两者量纲不同，")
    print("    差值仅供参考（说明两种方法找到的序列质量的相对差异）。")


def save_csv(results: list[dict], out_path: Path) -> None:
    if not results:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  -> {out_path}")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Python CONOP 参数扫描")
    ap.add_argument("--mode", choices=["ordinal", "weighted", "level", "both"], default="ordinal")
    ap.add_argument("--no-anchors", action="store_true", help="关闭 AGE/ASH 锚点约束")
    ap.add_argument("--tag", default="", help="文件夹后缀标签，如 baseline、paper-v1")
    args = ap.parse_args()

    use_anchors = not args.no_anchors
    orig_rows = load_orig_summary(ROOT / "scripts" / "summary.csv")

    # 创建时间戳文件夹
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ")
    folder_name = f"{ts}_{args.tag}" if args.tag else ts
    out_dir = ROOT / "results_py" / "sweep" / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir}")

    if args.mode in ("ordinal", "both"):
        print("\n── Ordinal 模式扫描 ──")
        results_ord = run_sweep("ordinal", use_anchors=use_anchors)
        save_csv(results_ord, out_dir / "ordinal.csv")
        if orig_rows:
            print_comparison(results_ord, orig_rows, "ordinal")

    if args.mode in ("level", "both"):
        print("\n── 多目标模式扫描（ordinal ×3 + level + eventual）──")
        results_lev = run_sweep("level", use_anchors=use_anchors)
        save_csv(results_lev, out_dir / "level.csv")
        if orig_rows:
            print_comparison(results_lev, orig_rows, "level")

    if args.mode in ("weighted", "both"):
        print("\n── 加权 Ordinal 模式扫描 ──")
        results_wt = run_sweep("weighted", use_anchors=use_anchors)
        save_csv(results_wt, out_dir / "weighted.csv")
        if orig_rows:
            print_comparison(results_wt, orig_rows, "weighted")

    # 写 manifest
    manifest = {
        "timestamp": ts,
        "tag": args.tag or None,
        "mode": args.mode,
        "anchors": use_anchors,
        "git_commit": _git_commit(),
        "param_groups": [
            {"tag": tag, "ratio": ratio, "startemp": se, "steps": st}
            for tag, ratio, se, st in PARAM_GROUPS
        ],
        "seeds": SEEDS,
        "trials": TRIALS,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n  -> {out_dir / 'manifest.json'}")
    print(f"\n全部结果: {out_dir}")


if __name__ == "__main__":
    main()
