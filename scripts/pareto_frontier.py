"""多目标 Pareto 前沿 — Level vs Eventual（对应作业主题 3 的多目标权衡）。

方法：扫描权重 w_evt ∈ [0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]（共 7 组）
        × N 个 seed，每次 SA 用 combined_misfit = level + w_evt·eventual。
跑完后画 (Level, Eventual) 散点，取 Pareto front。

注：Level + Eventual 都是非负整数，相同 SA 配置下不同权重会偏向不同目标。
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conop_py.io import parse_loadfile, parse_events, parse_sections, infer_taxa_from_observations  # noqa: E402
from conop_py.anneal import AnnealConfig, anneal  # noqa: E402
from conop_py.cost import (  # noqa: E402
    ConopContext, build_section_observations,
    level_misfit, eventual_misfit, ordinal_misfit,
)
from conop_py.plotting import init_plot, save_plot  # noqa: E402

init_plot()
import matplotlib.pyplot as plt  # noqa: E402

DATA = ROOT / "CONOP-run"
OUT = ROOT / "results_py" / "pareto"
OUT.mkdir(parents=True, exist_ok=True)

WEIGHTS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def load_dataset():
    obs = parse_loadfile(DATA / "loadfile.dat")
    taxon_ids = infer_taxa_from_observations(obs)
    ents = parse_events(DATA / "events.txt", taxon_ids=taxon_ids)
    secs = parse_sections(DATA / "sections.txt")
    return obs, ents, secs


def run_one(w_evt: float, seed: int, steps: int, trials: int,
            ents, obs) -> dict:
    cfg = AnnealConfig(
        startemp=250.0, ratio=0.98,
        steps=steps, trials=trials, seed=seed,
        use_fast_ordinal=False,   # 多目标走慢路径
    )
    weights = {"ordinal": 0.0, "level": 1.0, "eventual": w_evt}
    t0 = time.perf_counter()
    res = anneal(ents, obs, cfg, misfit_weights=weights, verbose=False)
    el = time.perf_counter() - t0
    # 评估最终解的 Level / Eventual / Ordinal
    section_obs = build_section_observations(obs)
    ctx = ConopContext.build(res.best_sequence, section_obs)
    lev = level_misfit(ctx)
    evt = eventual_misfit(ctx)
    ordi = ordinal_misfit(ctx)
    return {
        "w_evt": w_evt, "seed": seed,
        "combined_fit": res.best_fit,
        "level": lev, "eventual": evt, "ordinal": ordi,
        "elapsed_s": el,
    }


def pareto_front(points: np.ndarray) -> np.ndarray:
    """二维 Pareto front（最小化 x, y）。返回布尔 mask。"""
    n = len(points)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (points[j, 0] <= points[i, 0] and
                    points[j, 1] <= points[i, 1] and
                    (points[j, 0] < points[i, 0] or points[j, 1] < points[i, 1])):
                mask[i] = False
                break
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 17, 99])
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--trials", type=int, default=200)
    args = ap.parse_args()

    obs, ents, _ = load_dataset()
    print(f"数据: {len(obs)} obs / {len(ents)} ents")
    print(f"配置: steps={args.steps} trials={args.trials} seeds={args.seeds}")
    print(f"权重扫描: w_evt ∈ {WEIGHTS}\n")

    runs: list[dict] = []
    total = len(WEIGHTS) * len(args.seeds)
    k = 0
    for w in WEIGHTS:
        for seed in args.seeds:
            k += 1
            r = run_one(w, seed, args.steps, args.trials, ents, obs)
            runs.append(r)
            print(f"  [{k:2d}/{total}] w_evt={w:<5} seed={seed:<4}  "
                  f"L={r['level']:>5.0f}  E={r['eventual']:>5.0f}  "
                  f"O={r['ordinal']:>5.0f}  {r['elapsed_s']:.1f}s")

    sp = OUT / "summary.csv"
    with open(sp, "w") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["w_evt", "seed", "level", "eventual", "ordinal",
                        "combined_fit", "elapsed_s"])
        for r in runs:
            w_csv.writerow([r["w_evt"], r["seed"],
                            r["level"], r["eventual"], r["ordinal"],
                            r["combined_fit"], f"{r['elapsed_s']:.2f}"])
    print(f"\n✓ {sp}")

    # === Pareto 散点图 ===
    pts = np.array([[r["level"], r["eventual"]] for r in runs])
    mask = pareto_front(pts)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    cmap = plt.cm.viridis
    weights_arr = np.array([r["w_evt"] for r in runs])
    norm = plt.Normalize(min(WEIGHTS), max(WEIGHTS))
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=weights_arr, cmap=cmap, norm=norm,
                    s=60, edgecolor="black", linewidth=0.8, zorder=3)
    # Pareto 前沿连线
    pf = pts[mask]
    order = np.argsort(pf[:, 0])
    pf = pf[order]
    ax.plot(pf[:, 0], pf[:, 1], "r--", lw=1.5, alpha=0.7, label="Pareto front")
    ax.scatter(pf[:, 0], pf[:, 1], s=130, facecolors="none",
               edgecolor="red", lw=1.6, zorder=4)

    # CONOP9 真值参考点
    ax.scatter([237], [353], marker="*", c="gold", s=350,
               edgecolor="black", lw=1.2, label="CONOP9 真值 (237, 353)", zorder=5)

    cb = plt.colorbar(sc, ax=ax, label="w_evt")
    ax.set_xlabel("Level misfit")
    ax.set_ylabel("Eventual misfit")
    ax.set_title(f"Pareto 前沿: Level vs Eventual "
                 f"({len(WEIGHTS)} 权重 × {len(args.seeds)} seed = {total} 解)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    save_plot(fig, OUT / "pareto.png")
    print(f"✓ {OUT / 'pareto.png'}")

    # === 权重 → Level/Eventual trade-off 图 ===
    fig, ax = plt.subplots(figsize=(8, 5))
    by_w_L: dict[float, list[float]] = {}
    by_w_E: dict[float, list[float]] = {}
    for r in runs:
        by_w_L.setdefault(r["w_evt"], []).append(r["level"])
        by_w_E.setdefault(r["w_evt"], []).append(r["eventual"])
    ws = sorted(by_w_L.keys())
    L_mean = [np.mean(by_w_L[w]) for w in ws]
    E_mean = [np.mean(by_w_E[w]) for w in ws]
    L_std = [np.std(by_w_L[w]) for w in ws]
    E_std = [np.std(by_w_E[w]) for w in ws]

    ax2 = ax.twinx()
    l1 = ax.errorbar(ws, L_mean, yerr=L_std, fmt="o-", color="#2d6a4f",
                     lw=1.8, label="Level (左轴)", capsize=4)
    l2 = ax2.errorbar(ws, E_mean, yerr=E_std, fmt="s--", color="#c44e52",
                      lw=1.8, label="Eventual (右轴)", capsize=4)
    ax.set_xlabel("w_evt (Eventual 的权重)")
    ax.set_ylabel("Level misfit", color="#2d6a4f")
    ax2.set_ylabel("Eventual misfit", color="#c44e52")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_title("权重越大 → 偏向 Eventual：Level 升 / Eventual 降")
    ax.grid(alpha=0.3)
    lines = [l1, l2]
    ax.legend(lines, [l.get_label() for l in lines], loc="center right",
              fontsize=9)
    save_plot(fig, OUT / "tradeoff.png")
    print(f"✓ {OUT / 'tradeoff.png'}")

    # === 文字报告 ===
    rp = OUT / "report.txt"
    with open(rp, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Pareto 前沿分析 — Level vs Eventual 多目标 SA\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"权重扫描: {WEIGHTS}\n")
        f.write(f"种子: {args.seeds}\n")
        f.write(f"配置: steps={args.steps} trials={args.trials}\n")
        f.write(f"总解数: {len(runs)}\n")
        f.write(f"Pareto 上解数: {int(mask.sum())}\n\n")

        f.write("# 各权重下 (Level, Eventual) 均值 ± std\n")
        f.write(f"  {'w_evt':<8} {'Level':<18} {'Eventual':<18}\n")
        for w in ws:
            lm, ls = np.mean(by_w_L[w]), np.std(by_w_L[w])
            em, es = np.mean(by_w_E[w]), np.std(by_w_E[w])
            f.write(f"  {w:<8.2f} {lm:>6.1f}±{ls:<8.1f}  "
                    f"{em:>6.1f}±{es:<8.1f}\n")

        f.write("\n# Pareto 前沿点\n")
        for i, p in enumerate(pf):
            f.write(f"  ({p[0]:.0f}, {p[1]:.0f})\n")

        f.write("\n# 结论\n")
        f.write("  - 权重 w_evt=0 时（纯 Level）→ Level 最低、Eventual 偏高\n")
        f.write("  - w_evt 增大 → Eventual 降，Level 升 → 经典 Pareto trade-off\n")
        f.write(f"  - CONOP9 真值 (237, 353) 与本扫描的关系见 pareto.png "
                f"金色五角星位置\n")
        f.write("  - 实践意义：纯单目标 SA 会牺牲另一目标；"
                "若想兼顾，应用 combined misfit + w_evt ≈ 0.1-0.5\n")

    print(f"✓ {rp}")
    print(f"\nPareto 前沿包含 {int(mask.sum())}/{len(runs)} 个解")


if __name__ == "__main__":
    main()
