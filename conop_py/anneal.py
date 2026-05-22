"""模拟退火搜索（Sadler & Cooper 2003 §3.3）。

算法要点：
- 初始化：FAD 随机排入序列前半段、LAD 随机排入后半段，自动满足共存约束
- 扰动：随机选 1 个事件，从当前位置抽出，插入随机新位置
- 接受：ΔE < 0 必接受；ΔE > 0 以 exp(-ΔE/T) 概率接受
- 降温：每 trials 次扰动后 T *= ratio
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from functools import partial

from conop_py.cost import (
    EventKey, Sequence, build_section_observations, ordinal_misfit,
    build_pairwise_support, weighted_ordinal_misfit,
)
from conop_py.io import Entity, Observation


@dataclass
class AnnealConfig:
    startemp: float = 250.0
    ratio: float = 0.980
    steps: int = 600          # 降温次数（与原版 batch_run.bat baseline 一致）
    trials: int = 300         # 每温度阶段扰动次数
    seed: int | None = None
    force_fad_before_lad: bool = True  # 对应原版 FORCEFb4L='ON'


@dataclass
class TrajectoryPoint:
    cooling_step: int
    temperature: float
    current_fit: float
    best_fit: float
    accepted: int = 0
    proposed: int = 0


@dataclass
class AnnealResult:
    best_sequence: Sequence
    best_fit: float
    trajectory: list[TrajectoryPoint] = field(default_factory=list)


def initial_sequence(entities: list[Entity], rng: random.Random) -> Sequence:
    """生成满足共存约束的初始序列：FAD 随机入前半段，LAD 随机入后半段，
    markers 随机插入（可以位于序列任何位置）。"""
    taxa = [e for e in entities if e.is_taxon]
    markers = [e for e in entities if not e.is_taxon]

    fads = [(e.id, 1) for e in taxa]
    lads = [(e.id, 2) for e in taxa]
    rng.shuffle(fads)
    rng.shuffle(lads)
    seq: Sequence = fads + lads

    # markers 按观测的 event_type 插入到随机位置
    # 注意 markers 在 events.txt 中 entity_id 已分配，event_type 由 outmain 给出
    # 由于这里没有 obs 信息，markers 默认 event_type=5 (AGE)；运行时由调用方覆盖
    return seq


def build_anchor_order(observations: list[Observation]) -> list[EventKey]:
    """从观测中提取有绝对年龄约束的锚点事件，按从老到新排序。

    识别方式：w1 > 10 → 绝对同位素年龄约束（Ma），普通观测 w1=1.0。
    这些是 AGE(5) / ASH(4) marker，如 K-Pg 界线（~65 Ma）、晚白垩世火山灰层。

    年龄越大 = 越老 = 在地层中位置越低 = 在模型序列中排位越靠前。
    返回的列表就是锚点在模型序列中应维持的相对顺序。
    """
    anchor_ages: dict[EventKey, float] = {}
    for o in observations:
        if o.weight > 10:  # w1 > 10 → 绝对年龄约束（Ma）
            key = (o.entity_id, o.event_type)
            age_mid = (o.weight + o.weight2) / 2
            # 同一锚点可能在多个剖面出现，取最大值（最老的估计，保守）
            if key not in anchor_ages or age_mid > anchor_ages[key]:
                anchor_ages[key] = age_mid
    # 按年龄降序：最老的排在序列最前面
    return sorted(anchor_ages.keys(), key=lambda k: anchor_ages[k], reverse=True)


def _anchor_order_valid(
    seq: Sequence,
    anchor_order: list[EventKey],
    anchor_keys: set[EventKey],
) -> bool:
    """检查序列中锚点事件的相对顺序是否符合绝对年龄约束。O(n)。"""
    actual = [ev for ev in seq if ev in anchor_keys]
    return actual == anchor_order


def build_marker_keys(observations: list[Observation], taxa_ids: set[int]) -> list[EventKey]:
    """从观测中提取 marker 的 (entity_id, event_type) 唯一组合。"""
    seen = set()
    for o in observations:
        if o.entity_id not in taxa_ids:
            seen.add((o.entity_id, o.event_type))
    return list(seen)


def _build_fad_lad_map(entities: list[Entity]) -> dict[int, tuple[EventKey, EventKey]]:
    """构建 taxon entity_id → (FAD_key, LAD_key) 的映射，用于 FORCEFb4L 检查。"""
    return {
        e.id: ((e.id, 1), (e.id, 2))
        for e in entities if e.is_taxon
    }


def build_initial(
    entities: list[Entity],
    observations: list[Observation],
    rng: random.Random,
    anchor_order: list[EventKey] | None = None,
) -> Sequence:
    """构建包含所有 120 个事件的初始序列。

    锚点 markers（有绝对年龄约束）按正确年龄顺序放置，
    非锚点 markers 随机分布，FAD 入前半段、LAD 入后半段。
    """
    taxa = [e for e in entities if e.is_taxon]
    taxa_ids = {e.id for e in taxa}

    fads = [(e.id, 1) for e in taxa]
    lads = [(e.id, 2) for e in taxa]
    all_markers = build_marker_keys(observations, taxa_ids)

    anchor_keys: set[EventKey] = set(anchor_order) if anchor_order else set()
    non_anchor = [m for m in all_markers if m not in anchor_keys]
    ordered_anchors: list[EventKey] = list(anchor_order) if anchor_order else []

    rng.shuffle(fads)
    rng.shuffle(lads)
    rng.shuffle(non_anchor)

    # 锚点按年龄顺序分散插入：前半在 FAD 段末，后半在 LAD 段末
    half_a = len(ordered_anchors) // 2
    half_m = len(non_anchor) // 2
    seq: Sequence = (
        fads
        + non_anchor[:half_m]
        + ordered_anchors[:half_a]
        + lads
        + non_anchor[half_m:]
        + ordered_anchors[half_a:]
    )
    return seq


def anneal(
    entities: list[Entity],
    observations: list[Observation],
    cfg: AnnealConfig,
    misfit_fn=ordinal_misfit,
    use_weighted: bool = False,
    use_anchors: bool = True,
    verbose: bool = True,
) -> AnnealResult:
    """模拟退火主循环。

    Args:
        use_weighted: 若 True，使用多剖面支持度加权的 Ordinal penalty（忽略 misfit_fn）。
        use_anchors:  若 True，自动识别 AGE/ASH 锚点，SA 过程中强制维持其绝对年龄顺序。
    """
    rng = random.Random(cfg.seed)
    section_obs = build_section_observations(observations)

    # 确定实际使用的 misfit 函数
    if use_weighted:
        if verbose:
            print("预计算多剖面支持度权重…")
        ps = build_pairwise_support(observations)
        actual_misfit = partial(weighted_ordinal_misfit, pairwise_support=ps)
    else:
        actual_misfit = misfit_fn

    # 锚点设置
    anchor_order: list[EventKey] = []
    anchor_keys: set[EventKey] = set()
    if use_anchors:
        anchor_order = build_anchor_order(observations)
        anchor_keys = set(anchor_order)
        if verbose and anchor_order:
            print(f"锚点事件 {len(anchor_order)} 个（AGE/ASH markers，按绝对年龄固定相对顺序）")

    seq = build_initial(entities, observations, rng, anchor_order=anchor_order or None)
    n = len(seq)

    # FORCEFb4L：构建 FAD↔LAD 配对映射
    fad_lad_map = _build_fad_lad_map(entities) if cfg.force_fad_before_lad else {}

    current_fit = actual_misfit(seq, section_obs)
    best_seq = seq[:]
    best_fit = current_fit

    T = cfg.startemp
    trajectory: list[TrajectoryPoint] = []

    if verbose:
        mode = "weighted-ordinal" if use_weighted else "ordinal"
        anchor_info = f", {len(anchor_order)} anchors" if anchor_keys else ""
        print(f"初始 misfit = {current_fit:.2f}  (n={n} events, mode={mode}{anchor_info})")

    for step in range(cfg.steps):
        accepted = 0
        for _ in range(cfg.trials):
            # 扰动：抽出一个事件，插到新位置
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            ev = seq.pop(i)
            seq.insert(j, ev)

            # 锚点约束：若移动了锚点且破坏了年龄顺序，直接撤销
            if anchor_keys and ev in anchor_keys:
                if not _anchor_order_valid(seq, anchor_order, anchor_keys):
                    seq.pop(j)
                    seq.insert(i, ev)
                    continue

            # FORCEFb4L：FAD 必须在对应 LAD 之前（地质约束）
            if fad_lad_map and ev[0] in fad_lad_map:
                fad_key, lad_key = fad_lad_map[ev[0]]
                # 在新序列中找到两者的位置
                fad_pos = seq.index(fad_key)
                lad_pos = seq.index(lad_key)
                if fad_pos >= lad_pos:
                    seq.pop(j)
                    seq.insert(i, ev)
                    continue

            new_fit = actual_misfit(seq, section_obs)
            delta = new_fit - current_fit

            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                current_fit = new_fit
                accepted += 1
                if new_fit < best_fit:
                    best_fit = new_fit
                    best_seq = seq[:]
            else:
                # 撤销
                seq.pop(j)
                seq.insert(i, ev)

        point = TrajectoryPoint(
            cooling_step=step,
            temperature=T,
            current_fit=current_fit,
            best_fit=best_fit,
            accepted=accepted,
            proposed=cfg.trials,
        )
        trajectory.append(point)

        if verbose and (step % 20 == 0 or step == cfg.steps - 1):
            print(f"  step {step:3d}/{cfg.steps}  T={T:8.3f}  "
                  f"cur={current_fit:7.2f}  best={best_fit:7.2f}  "
                  f"accept={accepted}/{cfg.trials}")

        T *= cfg.ratio
        if T < 1e-3:
            break

    if verbose:
        print(f"最终 best_fit = {best_fit:.2f}")

    return AnnealResult(best_sequence=best_seq, best_fit=best_fit, trajectory=trajectory)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from conop_py.io import parse_cfg, parse_events, parse_loadfile

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="CONOP-run")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--startemp", type=float, default=None)
    ap.add_argument("--ratio", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-traj", default=None, help="trajectory.csv 输出路径")
    ap.add_argument("--out-soln", default=None, help="bestsoln.dat 输出路径")
    ap.add_argument("--weighted", action="store_true",
                    help="使用多剖面支持度加权的 Ordinal penalty")
    ap.add_argument("--no-anchors", action="store_true",
                    help="禁用 AGE/ASH marker 锚点约束（默认开启）")
    args = ap.parse_args()

    data = Path(args.data_dir)
    raw_cfg = parse_cfg(data / "conop9.cfg")

    cfg = AnnealConfig(
        startemp=args.startemp if args.startemp is not None else float(raw_cfg.get("STARTEMP", 250)),
        ratio=args.ratio if args.ratio is not None else float(raw_cfg.get("RATIO", 0.98)),
        steps=args.steps if args.steps is not None else int(raw_cfg.get("STEPS", 600)),
        trials=args.trials if args.trials is not None else int(raw_cfg.get("TRIALS", 300)),
        seed=args.seed,
    )
    obs = parse_loadfile(data / "loadfile.dat")
    from conop_py.io import infer_taxa_from_observations
    entities = parse_events(data / "events.txt", taxon_ids=infer_taxa_from_observations(obs))

    print(f"配置：STARTEMP={cfg.startemp} RATIO={cfg.ratio} STEPS={cfg.steps} TRIALS={cfg.trials} seed={cfg.seed}")
    print(f"模式：{'加权Ordinal' if args.weighted else 'Ordinal'}  锚点约束：{'关' if args.no_anchors else '开'}")
    res = anneal(entities, obs, cfg,
                 use_weighted=args.weighted,
                 use_anchors=not args.no_anchors)

    if args.out_traj:
        with open(args.out_traj, "w") as f:
            f.write("step,temperature,current,best,accepted,proposed\n")
            for p in res.trajectory:
                f.write(f"{p.cooling_step},{p.temperature:.6f},{p.current_fit:.4f},"
                        f"{p.best_fit:.4f},{p.accepted},{p.proposed}\n")
        print(f"trajectory 已保存: {args.out_traj}")

    if args.out_soln:
        with open(args.out_soln, "w") as f:
            for pos, (eid, etype) in enumerate(res.best_sequence, 1):
                f.write(f"{eid:>6d}{etype:>6d}{pos:>6d}\n")
        print(f"bestsoln 已保存: {args.out_soln}")
