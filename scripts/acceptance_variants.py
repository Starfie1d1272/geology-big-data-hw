"""Metropolis 接受准则变体对比 — 对应作业主题 2「如何判断是否接受当前解」。

在同一组 seed × 同一 SA 配置下，跑 4 种接受准则：
    1. metropolis (Sadler/CONOP9 默认): ΔE<=0 必接受，否则 exp(-ΔE/T)
    2. greedy (爬山法): 只接受 ΔE<=0（证明 SA 必要性的对照）
    3. threshold (确定性退火): ΔE <= T 即接受
    4. tsallis q=1.5: 重尾，高 ΔE 接受概率比 metropolis 大

输出 results_py/acceptance_variants/
    summary.csv     每 rule × seed 的 best_fit、最终接受率、轨迹文件名
    boxplot.png     四种准则的 best_fit 分布箱线图
    trajectory.png  四条平均轨迹 + 接受率曲线
    report.txt      文字结论
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
from conop_py.cost import ordinal_misfit  # noqa: E402
from conop_py.plotting import init_plot, save_plot  # noqa: E402

init_plot()
import matplotlib.pyplot as plt  # noqa: E402

DATA = ROOT / "CONOP-run"
OUT = ROOT / "results_py" / "acceptance_variants"
OUT.mkdir(parents=True, exist_ok=True)

RULES = ["metropolis", "greedy", "threshold", "tsallis"]
RULE_LABELS = {
    "metropolis": "Metropolis (CONOP9)",
    "greedy":     "Greedy (爬山)",
    "threshold":  "Threshold (θ=T)",
    "tsallis":    "Tsallis q=1.5",
}
RULE_COLORS = {
    "metropolis": "#2d6a4f",
    "greedy":     "#a8201a",
    "threshold":  "#f4a261",
    "tsallis":    "#1d4e89",
}


def load_dataset():
    obs = parse_loadfile(DATA / "loadfile.dat")
    taxon_ids = infer_taxa_from_observations(obs)
    ents = parse_events(DATA / "events.txt", taxon_ids=taxon_ids)
    secs = parse_sections(DATA / "sections.txt")
    return obs, ents, secs


def run_one(rule: str, seed: int, steps: int, trials: int,
            ents, obs) -> dict:
    cfg = AnnealConfig(
        startemp=250.0, ratio=0.98,
        steps=steps, trials=trials, seed=seed,
        accept_rule=rule, tsallis_q=1.5,
        use_fast_ordinal=True,
    )
    t0 = time.perf_counter()
    res = anneal(ents, obs, cfg, misfit_fn=ordinal_misfit, verbose=False)
    el = time.perf_counter() - t0
    traj = res.trajectory
    final_accept = traj[-1].accepted / max(traj[-1].proposed, 1) if traj else 0.0
    early_accept = (sum(p.accepted for p in traj[:10])
                    / max(sum(p.proposed for p in traj[:10]), 1)) if traj else 0.0
    return {
        "rule": rule, "seed": seed,
        "best_fit": res.best_fit,
        "early_accept": early_accept,
        "final_accept": final_accept,
        "steps_run": len(traj),
        "elapsed_s": el,
        "trajectory": traj,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 17, 99, 123, 7, 256, 1024, 2048])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--trials", type=int, default=300)
    args = ap.parse_args()

    obs, ents, _ = load_dataset()
    print(f"数据集: {len(obs)} 观测 × {len(ents)} 事件")
    print(f"配置: steps={args.steps}  trials={args.trials}  seeds={args.seeds}")
    print(f"准则: {RULES}\n")

    runs: list[dict] = []
    for rule in RULES:
        for seed in args.seeds:
            r = run_one(rule, seed, args.steps, args.trials, ents, obs)
            print(f"  {rule:12s} seed={seed:5d}  best_fit={r['best_fit']:6.1f}  "
                  f"accept(start→end)={r['early_accept']:.2f}→{r['final_accept']:.2f}  "
                  f"steps={r['steps_run']:3d}  {r['elapsed_s']:.2f}s")
            runs.append(r)

    # 写 summary
    sp = OUT / "summary.csv"
    with open(sp, "w") as f:
        w = csv.writer(f)
        w.writerow(["rule", "seed", "best_fit", "early_accept",
                    "final_accept", "steps_run", "elapsed_s"])
        for r in runs:
            w.writerow([r["rule"], r["seed"], r["best_fit"],
                        f"{r['early_accept']:.4f}", f"{r['final_accept']:.4f}",
                        r["steps_run"], f"{r['elapsed_s']:.3f}"])
    print(f"\n✓ {sp}")

    # 按 rule 聚合
    by_rule: dict[str, list[dict]] = {r: [] for r in RULES}
    for r in runs:
        by_rule[r["rule"]].append(r)

    # === 箱线图 + 散点 ===
    fig, ax = plt.subplots(figsize=(8.5, 5))
    data = [[r["best_fit"] for r in by_rule[k]] for k in RULES]
    labels = [RULE_LABELS[k] for k in RULES]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", lw=1.5))
    for patch, key in zip(bp["boxes"], RULES):
        patch.set_facecolor(RULE_COLORS[key])
        patch.set_alpha(0.55)
    # 叠加散点
    for i, key in enumerate(RULES):
        ys = data[i]
        xs = np.random.normal(i + 1, 0.06, size=len(ys))
        ax.scatter(xs, ys, c=RULE_COLORS[key], edgecolor="black",
                   s=22, alpha=0.85, zorder=3)
    # 在每组上标 mean ± std
    for i, key in enumerate(RULES):
        ys = np.array(data[i])
        ax.text(i + 1, max(ys) + (max(max(d) for d in data) -
                                  min(min(d) for d in data)) * 0.03,
                f"μ={ys.mean():.1f}\nσ={ys.std():.1f}",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("Ordinal best_fit (越低越好；CONOP9 真值 367)")
    ax.set_title(f"接受准则对比  ({len(args.seeds)} 个种子 × steps={args.steps})")
    ax.grid(alpha=0.3, axis="y")
    save_plot(fig, OUT / "boxplot.png")
    print(f"✓ {OUT / 'boxplot.png'}")

    # === 平均轨迹 + 接受率曲线 ===
    fig, (ax_fit, ax_acc) = plt.subplots(1, 2, figsize=(13, 4.8))
    for key in RULES:
        # 等长截断
        L = min(len(r["trajectory"]) for r in by_rule[key])
        bests = np.array([[p.best_fit for p in r["trajectory"][:L]]
                          for r in by_rule[key]])
        accepts = np.array([[p.accepted / max(p.proposed, 1)
                             for p in r["trajectory"][:L]]
                            for r in by_rule[key]])
        x = np.arange(L)
        m_best = bests.mean(axis=0)
        m_acc = accepts.mean(axis=0)
        std_best = bests.std(axis=0)

        ax_fit.plot(x, m_best, color=RULE_COLORS[key],
                    label=RULE_LABELS[key], lw=1.8)
        ax_fit.fill_between(x, m_best - std_best, m_best + std_best,
                            color=RULE_COLORS[key], alpha=0.15)
        ax_acc.plot(x, m_acc, color=RULE_COLORS[key],
                    label=RULE_LABELS[key], lw=1.5)

    ax_fit.set_xlabel("降温步")
    ax_fit.set_ylabel("best_fit（均值 ± std）")
    ax_fit.set_title("收敛轨迹对比")
    ax_fit.legend(fontsize=9, loc="upper right")
    ax_fit.grid(alpha=0.3)

    ax_acc.set_xlabel("降温步")
    ax_acc.set_ylabel("接受率（均值）")
    ax_acc.set_title("接受率随降温变化")
    ax_acc.legend(fontsize=9, loc="upper right")
    ax_acc.grid(alpha=0.3)
    save_plot(fig, OUT / "trajectory.png")
    print(f"✓ {OUT / 'trajectory.png'}")

    # === 文字报告 ===
    rp = OUT / "report.txt"
    with open(rp, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("接受准则变体对比 — 作业主题 2 数据\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"实验设置：4 种接受准则 × {len(args.seeds)} 个 seed × "
                f"steps={args.steps}, trials={args.trials}\n")
        f.write(f"数据集：南极 Seymour Island 12 剖面 / 120 事件，Ordinal misfit\n\n")

        f.write("# 各准则 best_fit 统计\n")
        f.write(f"  {'准则':<14} {'min':>6} {'median':>8} {'mean':>7} {'std':>7} {'max':>6}\n")
        stats = {}
        for key in RULES:
            ys = np.array([r["best_fit"] for r in by_rule[key]])
            stats[key] = ys
            f.write(f"  {RULE_LABELS[key]:<14} {ys.min():>6.1f} "
                    f"{np.median(ys):>8.1f} {ys.mean():>7.2f} "
                    f"{ys.std():>7.2f} {ys.max():>6.1f}\n")

        # 简单 t-test (metropolis vs others)
        from math import sqrt
        def welch(a, b):
            n_a, n_b = len(a), len(b)
            v_a, v_b = a.var(ddof=1), b.var(ddof=1)
            t = (a.mean() - b.mean()) / sqrt(v_a/n_a + v_b/n_b)
            return t

        f.write("\n# Welch t (vs Metropolis baseline)\n")
        base = stats["metropolis"]
        for key in RULES:
            if key == "metropolis":
                continue
            t = welch(stats[key], base)
            d = stats[key].mean() - base.mean()
            f.write(f"  {key:<12} Δmean = {d:+.2f}  t = {t:+.3f}\n")

        f.write("\n# 接受率变化\n")
        for key in RULES:
            ea = np.mean([r["early_accept"] for r in by_rule[key]])
            fa = np.mean([r["final_accept"] for r in by_rule[key]])
            f.write(f"  {RULE_LABELS[key]:<14} 早期 {ea:.2f} → 后期 {fa:.2f}\n")

        f.write("\n# 主要结论\n")
        gd_diff = stats["greedy"].mean() - base.mean()
        thr_diff = stats["threshold"].mean() - base.mean()
        ts_diff = stats["tsallis"].mean() - base.mean()
        f.write(f"  1) Greedy（爬山）vs Metropolis：Δmean = {gd_diff:+.1f}\n")
        f.write("     → 完全无随机性的 greedy")
        f.write("差 → SA 接受准则的随机性是必要的\n" if gd_diff > 0
                else "更好或持平 → 当前数据集随机性收益不大\n")
        f.write(f"  2) Threshold（确定性退火）vs Metropolis：Δmean = {thr_diff:+.1f}\n")
        f.write(f"  3) Tsallis q=1.5 vs Metropolis：Δmean = {ts_diff:+.1f}\n")
        f.write("     → Tsallis 重尾允许更激进的跳跃；"
                "降温曲线相同但接受概率函数形状不同\n")

    print(f"✓ {rp}\n")
    # 屏幕摘要
    print("=" * 50)
    for key in RULES:
        ys = np.array([r["best_fit"] for r in by_rule[key]])
        print(f"  {RULE_LABELS[key]:<22} mean={ys.mean():.2f}  "
              f"std={ys.std():.2f}  min={ys.min():.1f}")


if __name__ == "__main__":
    main()
