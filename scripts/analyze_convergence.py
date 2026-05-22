#!/usr/bin/env python3
"""CONOP 收敛轨迹分析：对比 Mac 原始数据和台式机第一次运行的结果"""

import re
import matplotlib.pyplot as plt
import numpy as np

def parse_trajectory(path):
    steps = []
    best_fits = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("Temperature"):
                continue
            parts = stripped.split()
            if len(parts) >= 3:
                step = float(parts[0])  # temperature 作为 step proxy
                misfit = float(parts[1])
                best = float(parts[2])
                steps.append(step)
                best_fits.append(best)
    # 翻转：温度从高到低，转为从低到高（迭代步数）
    steps = steps[::-1]
    best_fits = best_fits[::-1]
    return np.arange(len(steps)), np.array(steps), np.array(best_fits)

# 原始数据（Mac 上第一次 push 的，Git 历史里能找到）
# 但简单点：我们直接从第二次运行中提取第一次运行的数据
# 实际我们只有两次运行：一次 Mac，一次台式机
# 布局: 第一次 push 的数据在 Git HEAD~1
import subprocess
proj = "/Users/starfie1d/GitHub/geology-big-data-hw"

# 我们需要第一次的 trajectory
# 用 git show HEAD~1:CONOP-run/trajectory.txt
old_traj = subprocess.run(
    ["git", "show", "0f14b19:CONOP-run/trajectory.txt"],
    capture_output=True, text=True, cwd=proj
).stdout

with open("/tmp/traj_old.txt", "w") as f:
    f.write(old_traj)

iters1, temps1, best1 = parse_trajectory("/tmp/traj_old.txt")
iters2, temps2, best2 = parse_trajectory(f"{proj}/CONOP-run/trajectory.txt")

print(f"第一次运行（原始数据）:")
print(f"  初始代价: {best1[0]:.2f}, 最终: {best1[-1]:.2f}, 收敛比: {(1 - best1[-1]/best1[0])*100:.1f}%")
print(f"  迭代步数: {len(best1)}")

print(f"\n这次运行（台式机）:")
print(f"  初始代价: {best2[0]:.2f}, 最终: {best2[-1]:.2f}, 收敛比: {(1 - best2[-1]/best2[0])*100:.1f}%")
print(f"  迭代步数: {len(best2)}")

print(f"\n最终解差异: {best1[-1]:.2f} vs {best2[-1]:.2f}")
print(f"台式机结果更优: {'是' if best2[-1] < best1[-1] else '否'} (差 {abs(best2[-1]-best1[-1]):.2f})")

# 画图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：收敛曲线对比
ax1.plot(iters1, best1, label=f'原始 (Best={best1[-1]:.1f})', alpha=0.7, linewidth=0.5)
ax1.plot(iters2, best2, label=f'台式机 (Best={best2[-1]:.1f})', alpha=0.7, linewidth=0.5)
ax1.set_xlabel('迭代步数')
ax1.set_ylabel('最优代价 (Best Fit)')
ax1.set_title('CONOP 收敛曲线对比')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右图：温度衰减
ax2.plot(iters1, temps1, label=f'原始 RATIO=0.98', alpha=0.7)
ax2.set_xlabel('迭代步数')
ax2.set_ylabel('温度')
ax2.set_title('模拟退火温度衰减')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{proj}/scripts/convergence_comparison.png", dpi=150)
print(f"\n图片保存至: {proj}/scripts/convergence_comparison.png")
