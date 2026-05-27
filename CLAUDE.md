# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目背景

《地质学大数据前沿与应用》课程作业（南京大学，樊隽轩教授，2026 春）。使用 CONOP（模拟退火约束最优化）方法对南极 Seymour Island 白垩纪-古近纪菊石地层（12 剖面、49 分类单元）进行定量对比，分析退火参数对收敛效果的影响。

## 运行 CONOP

**CONOP 是 Windows 二进制程序，只能在 Windows 上手动运行。**

工作目录：`CONOP-run/`
可执行文件：`CONOP64ver8p621.exe`（64位）或 `CONOP32.exe`（32位）
配置文件：`CONOP-run/conop9.cfg`

关键参数（`&getrun` 段）：
- `RATIO`：降温速率（0.98 为基准，越小降温越快）
- `STARTEMP`：初始温度（默认 250）
- `STEPS`：每温度步的扰动次数（默认 600）
- `SOLVER=anneal`：模拟退火求解器

批量参数扫描脚本为 `scripts/batch_run.bat`（Windows，修改 `CONOP_DIR` 路径后运行），每组参数重复 3 次，手动确认后自动保存结果。

## 数据流

```
CONOP-run/
├── conop9.cfg          # 配置（运行前修改参数）
├── loadfile.dat        # 主数据（各剖面化石记录）
├── events.txt          # 事件定义（FAD/LAD/年代）
└── sections.txt        # 剖面列表

运行后输出：
├── trajectory.txt      # 收敛轨迹（温度 + 代价，每步记录）
├── bestsoln.dat        # 最优事件排列
├── outmain.txt         # 主输出（各剖面惩罚详情）
├── ordr.dat            # 事件顺序
└── runlog.txt          # 运行日志
```

结果按参数组归档：`results/<tag>/run_N/`（tag 如 `baseline`、`ratio_099`、`steps_1200`）。

## Python 分析脚本

```bash
# 收敛曲线对比分析（需 git 历史中有对照数据）
uv run scripts/analyze_convergence.py

# 或使用项目 conda 环境
conda run -n <env> python scripts/analyze_convergence.py
```

依赖：`numpy`, `pandas`, `matplotlib`, `scipy`（见 `requirements.txt`）。`torch` 和 `pymupdf` 按需使用。

`scripts/summary.csv` 是各次运行 best_fit 汇总表，字段：`实验组, run_id, RATIO, STARTEMP, STEPS, best_fit`。

## 性能优化的 SA（B 区：增量 cost + NumPy + numba + 多进程）

完整 baseline SA（180k iters）从 16 秒 → 1 秒，50 次并行重启从 13 分钟 → 10 秒。

```bash
# 单次 SA 基准
uv run --with numpy --with numba --with pandas --with matplotlib --with scipy \
  python scripts/benchmark_sa.py --steps 600 --trials 300

# 50 次并行多重启（解的不确定性 / consensus / rank 分布 / jackknife）
uv run --with numpy --with numba --with pandas --with matplotlib --with scipy \
  python scripts/run_multistart.py --n 50 --tag <tag> --workers 8
# 输出: results_py/multistart/<timestamp>_<tag>/{summary.csv, bestsoln_s*.dat, manifest.json}

# 回归测试（任何 cost 重构后必跑）
uv run --with numpy --with numba --with pandas --with matplotlib --with scipy --with pytest \
  python -m pytest tests/test_regression.py -v
```

关键 ConfigKnobs（`AnnealConfig`）：
- `use_fast_ordinal=True`：增量 ordinal + numba 路径（仅 ordinal 模式有效，默认开）
- `early_stop_patience=80`：连续 80 步 best_fit 无改善则停（0 = 关闭）
- `early_stop_min_step=50`：至少跑这么多步才允许早停

`FastOrdinalState`（`conop_py/cost.py`）的关键：差分公式 O(n_s) 计算 Δordinal，避开 sort+merge；维护 `_pos_arr` numpy 数组 + numba JIT 内层循环。降级路径在无 numba 时自动启用纯 Python 版本。

## 实验结论速查

- **RATIO=0.98, STARTEMP=250, STEPS=600** 为最优基准配置（best_fit ≈ 220.09，波动 ±0.01）
- RATIO 是最敏感参数：0.95 导致结果不稳定，0.99 收敛不充分
- STEPS=600 足够，翻倍至 1200 无显著提升
- 初始温度（100/250/500）对结果影响很小
- 详细实验报告见 `实验报告.md`

## 输出物

- `论文/`：小论文草稿（格式参考《高校地质学报》）
- `PPT/`：课堂汇报幻灯片
- `slides/`：课程讲义 PDF
- `references/`：参考文献 PDF
