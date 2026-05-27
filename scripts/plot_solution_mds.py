#!/usr/bin/env python3
"""解距离矩阵 + MDS 投影。

输入：50 个 bestsoln_s*.dat（multistart 目录）
      事件名映射（events.txt）
输出：results_py/rank_distribution/
  - solution_mds.png   MDS 2D 投影（点=解，颜色=best_fit）
  - solution_mds.csv   投影坐标

用法：
    uv run python scripts/plot_solution_mds.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "results_py" / "multistart" / "2026-05-26_190029Z_b-region-test"
OUT_DIR = ROOT / "results_py" / "rank_distribution"

from conop_py.io import parse_solution, solution_to_sequence


def spearman_rank_correlation(seq_a: list, seq_b: list) -> float:
    """两个序列之间的 Spearman 秩相关。值域 [-1, 1]，1=完全一致。"""
    n = len(seq_a)
    pos_a = {ev: i for i, ev in enumerate(seq_a)}
    pos_b = {ev: i for i, ev in enumerate(seq_b)}

    # 公共事件
    common = [ev for ev in seq_a if ev in pos_b]
    if len(common) < 3:
        return 0.0

    ranks_a = np.array([pos_a[ev] for ev in common], dtype=float)
    ranks_b = np.array([pos_b[ev] for ev in common], dtype=float)

    # Spearman ρ = Pearson on ranks
    n_c = len(common)
    d = ranks_a - ranks_b
    rho = 1 - 6 * np.sum(d ** 2) / (n_c * (n_c ** 2 - 1))
    return float(rho)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. 解析所有解 ──
    soln_files = sorted(DATA_DIR.glob("bestsoln_s*.dat"))
    print(f"读取 {len(soln_files)} 个解...")

    sequences = {}
    for path in soln_files:
        seed = int(path.stem.split("_s")[1])
        sol = parse_solution(path)
        seq = solution_to_sequence(sol)
        sequences[seed] = seq

    seeds = sorted(sequences.keys())
    n = len(seeds)

    # ── 2. 距离矩阵（1 - ρ）──
    print(f"计算 {n}×{n} 距离矩阵...")
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho = spearman_rank_correlation(sequences[seeds[i]], sequences[seeds[j]])
            dist = 1.0 - max(-1.0, min(1.0, rho))  # [0, 2]
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    print(f"  距离范围: [{dist_matrix.min():.3f}, {dist_matrix.max():.3f}]")

    # ── 3. MDS 投影到 2D ──
    try:
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=42, normalized_stress=False)
        coords = mds.fit_transform(dist_matrix)
        print(f"  MDS stress: {mds.stress_:.2f}")
    except ImportError:
        # Fallback: 手动 MDS via eigendecomposition
        print("  sklearn 不可用，用 eigendecomposition MDS...")
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ (dist_matrix ** 2) @ H
        eigvals, eigvecs = np.linalg.eigh(B)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        coords = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))

    # ── 4. 彩色：每个解的 best_fit ──
    summary_path = DATA_DIR / "summary.csv"
    best_fits = {}
    if summary_path.exists():
        with open(summary_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                best_fits[int(row["seed"])] = float(row["best_fit"])
    colors = [best_fits.get(s, 250) for s in seeds]

    # ── 5. 画图 ──
    from conop_py.plotting import init_plot
    import matplotlib.pyplot as plt
    init_plot()

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=colors, cmap="RdYlGn_r",
                    s=40, alpha=0.8, edgecolors="k", linewidth=0.3)
    cbar = plt.colorbar(sc, ax=ax, label="best_fit (Ordinal)")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title("50 次多重启解的距离投影 (Spearman ρ → MDS)")
    ax.grid(alpha=0.3)

    from conop_py.plotting import save_plot
    png_path = OUT_DIR / "solution_mds.png"
    save_plot(fig, png_path)
    print(f"→ {png_path}")

    # ── 6. 写坐标 CSV ──
    csv_path = OUT_DIR / "solution_mds.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "best_fit", "mds_x", "mds_y"])
        for i, s in enumerate(seeds):
            w.writerow([s, best_fits.get(s, ""), coords[i, 0], coords[i, 1]])
    print(f"→ {csv_path}")


if __name__ == "__main__":
    main()
