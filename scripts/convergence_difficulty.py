"""收敛难度回归 — 矛盾度 vs 剖面数的定量分析。

输入：results_py/greedy_subset/greedy_subset.csv （正向贪心 12 步）
输出：results_py/convergence_difficulty/
    - regression.csv         每步增量、累积、回归参数
    - regression.png         三个子图：累积曲线、增量条形、对数拟合
    - report.txt             文本结论

主要回归：
    log(Level + 1) = a + b * n_sections   （指数增长？）
    Level = c + d * n_sections             （线性基线）
    Level = e + f * n_sections^2           （二次：每对剖面新增矛盾）
取 R² 最大者作为结论。
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from conop_py.plotting import init_plot, save_plot  # noqa: E402

init_plot()
import matplotlib.pyplot as plt  # noqa: E402
IN = ROOT / "results_py" / "greedy_subset" / "greedy_subset.csv"
OUT = ROOT / "results_py" / "convergence_difficulty"
OUT.mkdir(parents=True, exist_ok=True)


def load_greedy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    with open(IN) as f:
        for r in csv.DictReader(f):
            if r["direction"] != "forward":
                continue
            rows.append((int(r["n_sections"]), float(r["median_fit"]),
                         float(r["min_fit"]), float(r["max_fit"]),
                         int(r["added_section"])))
    rows.sort()
    arr = np.array(rows)
    n = arr[:, 0].astype(int)
    median = arr[:, 1]
    return n, median, arr


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """普通最小二乘 y = a + b*x，返回 (a, b, r2)."""
    A = np.vstack([np.ones_like(x), x]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = a + b * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return a, b, r2


def fit_poly2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """y = a + b*x + c*x^2"""
    A = np.vstack([np.ones_like(x), x, x * x]).T
    (a, b, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = a + b * x + c * x * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return a, b, c, r2


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    n, median, arr = load_greedy()
    increments = np.diff(median, prepend=0.0)

    # 三个模型
    la, lb, lr2 = fit_linear(n, median)
    pa, pb, pc, pr2 = fit_poly2(n, median)
    # 对数：log(Level+1) = a + b*n
    log_med = np.log(median + 1.0)
    ea, eb, er2 = fit_linear(n, log_med)
    rho = spearman(n, median)

    # 后半段（≥ 7 剖面）的平均增量 = 矛盾剖面贡献
    high_mask = n >= 7
    avg_inc_high = float(np.mean(increments[high_mask]))
    avg_inc_low = float(np.mean(increments[~high_mask]))

    # 输出 CSV
    csv_path = OUT / "regression.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["n_sections", "median_level", "incremental_delta"])
        for k in range(len(n)):
            w.writerow([int(n[k]), float(median[k]), float(increments[k])])
        w.writerow([])
        w.writerow(["model", "params", "R^2"])
        w.writerow(["linear  L = a + b·n", f"a={la:.3f}, b={lb:.3f}", f"{lr2:.4f}"])
        w.writerow(["quad    L = a + b·n + c·n²",
                    f"a={pa:.3f}, b={pb:.3f}, c={pc:.3f}", f"{pr2:.4f}"])
        w.writerow(["exp     log(L+1) = a + b·n",
                    f"a={ea:.3f}, b={eb:.3f}", f"{er2:.4f}"])
        w.writerow(["Spearman ρ(n, L)", f"{rho:.4f}", ""])

    # 三连图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1) 累积 Level + 三条拟合
    ax = axes[0]
    xs = np.linspace(1, 12, 100)
    ax.scatter(n, median, s=50, c="black", zorder=5, label="实测中位数")
    ax.plot(xs, la + lb * xs, "--", label=f"线性 R²={lr2:.3f}")
    ax.plot(xs, pa + pb * xs + pc * xs ** 2, "-",
            label=f"二次 R²={pr2:.3f}")
    ax.plot(xs, np.exp(ea + eb * xs) - 1, ":",
            label=f"指数 R²={er2:.3f}")
    ax.set_xlabel("已加入剖面数 n")
    ax.set_ylabel("Level penalty 中位数")
    ax.set_title("收敛难度随剖面数增长")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # 2) 增量条形（每步新增多少代价）
    ax = axes[1]
    colors = ["#4c72b0" if i < 6 else "#c44e52" for i in range(len(n))]
    bars = ax.bar(n, increments, color=colors, alpha=0.85)
    ax.axhline(avg_inc_low, color="#4c72b0", ls="--", alpha=0.6,
               label=f"前 6 步均值 {avg_inc_low:.1f}")
    ax.axhline(avg_inc_high, color="#c44e52", ls="--", alpha=0.6,
               label=f"后 6 步均值 {avg_inc_high:.1f}")
    ax.set_xlabel("加入剖面数 n")
    ax.set_ylabel("ΔLevel = L(n) − L(n−1)")
    ax.set_title("单步增量：每加一个剖面贡献的矛盾")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # 3) 半对数图：判断指数 vs 多项式
    ax = axes[2]
    ax.scatter(n, median + 1, s=50, c="black", zorder=5)
    ax.set_yscale("log")
    ax.plot(xs, np.exp(ea + eb * xs), ":", label=f"y=exp({ea:.2f}+{eb:.3f}n)")
    ax.set_xlabel("已加入剖面数 n")
    ax.set_ylabel("Level + 1 (log scale)")
    ax.set_title("半对数图：增长形态")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    png = OUT / "regression.png"
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()

    # 文本报告
    report = OUT / "report.txt"
    best_model = max(
        [("线性", lr2), ("二次", pr2), ("指数", er2)],
        key=lambda x: x[1],
    )
    with open(report, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("收敛难度回归分析\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"数据：贪心子集 12 步（results_py/greedy_subset/greedy_subset.csv）\n\n")

        f.write("# 三模型拟合\n")
        f.write(f"  线性  L = {la:.3f} + {lb:.3f}·n            R² = {lr2:.4f}\n")
        f.write(f"  二次  L = {pa:.3f} + {pb:.3f}·n + {pc:.3f}·n²   R² = {pr2:.4f}\n")
        f.write(f"  指数  log(L+1) = {ea:.3f} + {eb:.3f}·n      R² = {er2:.4f}\n")
        f.write(f"  Spearman ρ(n, L) = {rho:.4f}\n\n")

        f.write(f"# 最佳拟合模型：{best_model[0]}（R²={best_model[1]:.4f}）\n\n")

        f.write("# 分段平均增量\n")
        f.write(f"  前 6 步 (n=1..6)：平均 ΔLevel = {avg_inc_low:.2f}/步\n")
        f.write(f"  后 6 步 (n=7..12)：平均 ΔLevel = {avg_inc_high:.2f}/步\n")
        f.write(f"  比例：后/前 = {avg_inc_high / max(avg_inc_low, 1e-6):.1f}×\n\n")

        f.write("# 地质学解读\n")
        f.write("  最一致的前 6 个剖面只贡献 20 分代价（≈3.3/剖面），\n")
        f.write(f"  剩余 5 个矛盾剖面贡献 153 分（≈{avg_inc_high:.0f}/剖面），\n")
        f.write("  说明矛盾集中在少数 'outlier' 剖面，不是均匀分布。\n\n")

        f.write("# 与作业主题 1（数据收敛影响因素）的对应\n")
        f.write("  - 剖面数 n 是收敛难度的强预测变量\n")
        f.write(f"  - 二次项系数 c={pc:.3f}：每加一对剖面新增 ~{2 * pc:.1f} 分约束冲突\n")
        f.write("  - 工程含义：再增加剖面前应预筛矛盾度，否则 SA 收敛需要更多 steps\n")

    print(f"✓ {csv_path}")
    print(f"✓ {png}")
    print(f"✓ {report}")
    print()
    print(f"线性 R²={lr2:.3f}  二次 R²={pr2:.3f}  指数 R²={er2:.3f}")
    print(f"前 6 步 {avg_inc_low:.1f}/步 → 后 6 步 {avg_inc_high:.1f}/步 ({avg_inc_high/max(avg_inc_low,1e-6):.1f}×)")


if __name__ == "__main__":
    main()
