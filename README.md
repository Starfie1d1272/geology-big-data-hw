# 地质学大数据前沿与应用 —— 任务2：CONOP 约束最优化

《地质学大数据前沿与应用》课程作业项目。樊隽轩教授，南京大学地球科学与工程学院，2026 年春。

## 任务概览

利用 CONOP（Constrained Optimization，约束最优化）方法进行定量地层对比。

个人任务：每人撰写 4-8 页小论文（格式参考《高校地质学报》）
小组任务：15 分钟 PPT 课堂汇报

## 项目分工

四人小组，分工如下：

- **成员 A（你）**：CONOP 原理 + 论文方法部分（含深度学习辅助优化探索）
- **成员 B（李彦谷）**：数据分析 + 可视化
- **成员 C**：算法实现 / 参数对比实验
- **成员 D**：论文撰写 + PPT 整合

## 项目结构

```
├── CONOP-run/          # CONOP 已有的运行数据（南极 Seymour Island 菊石，12 剖面）
├── references/         # 参考文献 PDF
├── scripts/            # 分析 & 可视化脚本
├── 论文/               # 小论文草稿
└── PPT/                # 汇报 PPT
```

## 数据集

南极 Seymour Island 白垩纪-古近纪菊石地层对比数据：

- 12 个剖面（Seymour Island 露头 + 智利 Quiriquina + 同位素年代参考 O97）
- 49 个分类单元（主要为菊石，含锶同位素、铱异常等标志层）
- 120 个事件（FAD/LAD + 同位素年龄事件）
- 193 对共现对，593 对 FAD-LAD 对

## 依赖

见 `requirements.txt`。核心依赖：

- Python 3.11+
- numpy / pandas / matplotlib / scipy（数据分析与可视化）
- torch（深度学习辅助建模）

## 运行 CONOP

CONOP 为 Windows 程序（`CONOP32.exe` / `CONOP64`），尚未开源。
目前已有一次运行结果在 `CONOP-run/` 下。

在台式机（Windows）上修改 `conop9.cfg` 后双击 exe 即可重新运行。
