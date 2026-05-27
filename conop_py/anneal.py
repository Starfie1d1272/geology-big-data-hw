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

from conop_py.cost import (
    ConopContext, EventKey, Sequence, build_section_observations,
    ordinal_misfit, level_misfit, eventual_misfit, combined_misfit,
    build_pairwise_support, weighted_ordinal_misfit,
    FastOrdinalState,
)
from conop_py.io import Entity, Observation
from collections import defaultdict


@dataclass
class AnnealConfig:
    startemp: float = 250.0
    ratio: float = 0.980
    steps: int = 600          # 降温次数（与原版 batch_run.bat baseline 一致）
    trials: int = 300         # 每温度阶段扰动次数
    seed: int | None = None
    force_fad_before_lad: bool = True  # 对应原版 FORCEFb4L='ON'
    # 早停（连续 patience 个温度阶段 best_fit 无改善则退出，0 = 关闭）
    early_stop_patience: int = 0
    early_stop_min_step: int = 50  # 至少跑这么多温度阶段才允许早停
    # 走 FastOrdinalState 增量路径（仅 ordinal 模式有效；fallback 慢路径自动保留）
    use_fast_ordinal: bool = True
    # 共存约束惩罚权重（对应 CONOP FORCECOEX=SS）
    # 设为 0 禁用；越大越严格。数据集有 24 个固有冲突，建议 >= 100
    coex_penalty: float = 0.0


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


def _anneal_fast_ordinal(
    seq: Sequence,
    section_obs: dict[int, list[tuple[float, EventKey]]],
    cfg: AnnealConfig,
    rng: random.Random,
    anchor_order: list[EventKey],
    anchor_keys: set[EventKey],
    fad_lad_map: dict[int, tuple[EventKey, EventKey]],
    verbose: bool,
) -> AnnealResult:
    """ordinal-only 快速路径：FastOrdinalState 增量 + 早停。

    与 anneal() 的慢路径相比：
      - 每步扰动不调 ordinal_misfit() 全量，只重算 |affected sections| 个 section
      - 用 pos_arr O(1) 查 FAD/LAD 位置，省掉 seq.index()
      - 早停：连续 patience 个温度阶段 best_fit 无改善则提前退出
    """
    state = FastOrdinalState(seq, section_obs)
    n = state.n
    pd = state._pos_dict   # 直接引用以省一层属性访问
    current_fit = state.total
    best_fit = current_fit
    best_seq = seq[:]

    def anchor_valid_fast() -> bool:
        """检查 anchor 在 pos_dict 中的相对顺序是否仍按 anchor_order 升序。"""
        last = -1
        for a in anchor_order:
            p = pd[a]
            if p <= last:
                return False
            last = p
        return True

    T = cfg.startemp
    trajectory: list[TrajectoryPoint] = []

    if verbose:
        anchor_info = f", {len(anchor_order)} anchors" if anchor_keys else ""
        print(f"初始 misfit = {current_fit:.2f}  (n={n} events, mode=ordinal-fast{anchor_info})")

    stagnation = 0  # 连续 best 无改善的温度阶段数

    for step in range(cfg.steps):
        accepted = 0
        improved_this_step = False
        for _ in range(cfg.trials):
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            ev = seq[i]
            # 同步：维护 seq 列表（anchor / FORCEFb4L 检查需要）和 state
            seq.pop(i); seq.insert(j, ev)
            delta, undo = state.trial_move(seq, ev, j, i)

            # 约束检查
            if anchor_keys and ev in anchor_keys:
                if not anchor_valid_fast():
                    seq.pop(j); seq.insert(i, ev)
                    state.revert(seq, undo)
                    continue
            if fad_lad_map and ev[0] in fad_lad_map:
                fad_key, lad_key = fad_lad_map[ev[0]]
                if pd[fad_key] >= pd[lad_key]:
                    seq.pop(j); seq.insert(i, ev)
                    state.revert(seq, undo)
                    continue

            # Metropolis 接受准则
            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                current_fit += delta
                accepted += 1
                if current_fit < best_fit:
                    best_fit = current_fit
                    best_seq = seq[:]
                    improved_this_step = True
            else:
                seq.pop(j); seq.insert(i, ev)
                state.revert(seq, undo)

        trajectory.append(TrajectoryPoint(
            cooling_step=step, temperature=T, current_fit=current_fit,
            best_fit=best_fit, accepted=accepted, proposed=cfg.trials,
        ))
        if verbose and (step % 20 == 0 or step == cfg.steps - 1):
            print(f"  step {step:3d}/{cfg.steps}  T={T:8.3f}  "
                  f"cur={current_fit:7.2f}  best={best_fit:7.2f}  "
                  f"accept={accepted}/{cfg.trials}")

        # 早停判定：连续 patience 步 best 无改善 且 已跑过 min_step → 退出
        if cfg.early_stop_patience > 0:
            stagnation = 0 if improved_this_step else stagnation + 1
            if (step >= cfg.early_stop_min_step
                    and stagnation >= cfg.early_stop_patience):
                if verbose:
                    print(f"  早停于 step {step}: best_fit 连续 {stagnation} 步无改善")
                break

        T *= cfg.ratio
        if T < 1e-3:
            break

    if verbose:
        print(f"最终 best_fit = {best_fit:.2f}")
    return AnnealResult(best_sequence=best_seq, best_fit=best_fit, trajectory=trajectory)


def anneal(
    entities: list[Entity],
    observations: list[Observation],
    cfg: AnnealConfig,
    misfit_fn=None,          # fn(ctx: ConopContext) -> float，默认 ordinal_misfit
    misfit_weights: dict[str, float] | None = None,  # 传给 combined_misfit 的权重
    use_weighted: bool = False,
    use_anchors: bool = True,
    verbose: bool = True,
) -> AnnealResult:
    """模拟退火主循环。

    Args:
        misfit_fn:      接受 ConopContext 的代价函数。若为 None，使用单模式默认值。
        misfit_weights: 多目标权重 dict，如 {'ordinal': 1.0, 'level': 0.5, 'eventual': 0.3}。
                        有值时忽略 misfit_fn，使用 combined_misfit。
        use_weighted:   若 True，Ordinal 组件使用多剖面支持度加权。
        use_anchors:    若 True，强制维持 AGE/ASH 锚点的绝对年龄顺序。
    """
    rng = random.Random(cfg.seed)
    section_obs = build_section_observations(observations)

    # 确定 misfit 函数
    if misfit_weights:
        # 多目标模式
        if verbose:
            print(f"多目标权重: {misfit_weights}")
        ps = build_pairwise_support(observations) if use_weighted else None
        def actual_misfit(ctx: ConopContext) -> float:
            return combined_misfit(ctx, misfit_weights, ps)
        mode = f"combined({','.join(f'{k}={v}' for k,v in misfit_weights.items())})"
    elif misfit_fn is not None:
        actual_misfit = misfit_fn
        mode = misfit_fn.__name__
    elif use_weighted:
        if verbose:
            print("预计算多剖面支持度权重…")
        ps = build_pairwise_support(observations)
        actual_misfit = lambda ctx: weighted_ordinal_misfit(ctx, ps)
        mode = "weighted-ordinal"
    else:
        actual_misfit = ordinal_misfit
        mode = "ordinal"

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

    # FORCEFb4L
    fad_lad_map = _build_fad_lad_map(entities) if cfg.force_fad_before_lad else {}

    # Coexistence 预计算（只在启用时）
    taxon_coex: dict[int, dict[int, set[int]]] | None = None
    if cfg.coex_penalty > 0:
        taxon_coex = _build_taxon_coex(section_obs)

    def _coex_delta(eid: int, pos: dict) -> int:
        """taxon eid 在当前状态下涉及的共存违反数。只检查 taxon_coex 中标记的必须共存对。"""
        if taxon_coex is None:
            return 0
        v = 0
        if (eid, 1) not in pos or (eid, 2) not in pos:
            return 0
        fa, la = pos[(eid, 1)], pos[(eid, 2)]
        for sec_id, per_sec in taxon_coex.items():
            others = per_sec.get(eid)
            if not others:
                continue
            for oeid in others:
                fb, lb = pos[(oeid, 1)], pos[(oeid, 2)]
                if not (fa < lb and fb < la):
                    v += 1
        return v

    # ------ 快速路径分派：FastOrdinalState 增量 + 早停，仅 ordinal 模式 ------
    is_fast_path = (
        getattr(cfg, "use_fast_ordinal", False)
        and misfit_weights is None
        and not use_weighted
        and (misfit_fn is None or misfit_fn is ordinal_misfit)
    )
    if is_fast_path:
        if verbose:
            print(f"启用快速路径 (FastOrdinalState 增量 + 早停)")
        return _anneal_fast_ordinal(
            seq=seq, section_obs=section_obs, cfg=cfg, rng=rng,
            anchor_order=anchor_order, anchor_keys=anchor_keys,
            fad_lad_map=fad_lad_map, verbose=verbose,
        )

    # 构建 ConopContext（观测数据部分不变，sequence 变化时只 rebuild pos）
    base_ctx = ConopContext.build(seq, section_obs)
    raw_current = actual_misfit(base_ctx)          # 纯 misfit（Level / Ordinal / …）
    tol_coex_current = 0                           # 全局共存违反总数（增量维护）

    # Section 过滤预计算（Level 增量：只重算 move ev 涉及的 2-3 个剖面）
    from conop_py.cost import _sec_level
    use_inc_sec = misfit_fn is level_misfit and cfg.coex_penalty > 0
    if use_inc_sec:
        event_to_secs: dict[EventKey, set[int]] = {}
        for sec_id, evs in section_obs.items():
            for _level, ev in evs:
                event_to_secs.setdefault(ev, set()).add(sec_id)
        sec_level_cache: dict[int, float] = {}
        for sec_id in section_obs:
            sec_level_cache[sec_id] = _sec_level(base_ctx, sec_id)
    best_seq = seq[:]
    best_fit = raw_current

    T = cfg.startemp
    trajectory: list[TrajectoryPoint] = []

    if verbose:
        anchor_info = f", {len(anchor_order)} anchors" if anchor_keys else ""
        coex_info = f", coex_w={cfg.coex_penalty}" if cfg.coex_penalty > 0 else ""
        print(f"初始 misfit = {raw_current:.2f}  (n={n} events, mode={mode}{anchor_info}{coex_info})")

    for step in range(cfg.steps):
        accepted = 0
        for _ in range(cfg.trials):
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            ev = seq.pop(i)
            seq.insert(j, ev)

            if anchor_keys and ev in anchor_keys:
                if not _anchor_order_valid(seq, anchor_order, anchor_keys):
                    seq.pop(j); seq.insert(i, ev)
                    continue

            if fad_lad_map and ev[0] in fad_lad_map:
                fad_key, lad_key = fad_lad_map[ev[0]]
                fad_pos = seq.index(fad_key)
                lad_pos = seq.index(lad_key)
                if fad_pos >= lad_pos:
                    seq.pop(j); seq.insert(i, ev)
                    continue

            # 记下移动前的位置字典，后面对比共存变化
            prev_pos = base_ctx.pos

            ctx = base_ctx.rebuild_pos(seq)

            # Level 增量：只重算 move ev 涉及的剖面（典型 2-3 个 vs 全 12 个）
            if use_inc_sec:
                affected = event_to_secs.get(ev, set(section_obs.keys()))
                raw_new = raw_current
                saved_secs: dict[int, float] = {}
                for s in affected:
                    old = sec_level_cache[s]
                    saved_secs[s] = old
                    new_val = _sec_level(ctx, s)
                    sec_level_cache[s] = new_val
                    raw_new += new_val - old
            else:
                raw_new = actual_misfit(ctx)

            # 共存约束（增量式：只算 moved ev 涉及的变化 → 全局差）
            if cfg.coex_penalty > 0:
                coex_before = _coex_delta(ev[0], prev_pos)
                coex_after = _coex_delta(ev[0], ctx.pos)
                delta_coex = coex_after - coex_before  # 全局共存变化量
                delta = (raw_new - raw_current) + cfg.coex_penalty * delta_coex
            else:
                delta = raw_new - raw_current
                delta_coex = 0

            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                raw_current = raw_new
                tol_coex_current += delta_coex
                accepted += 1
                if raw_new < best_fit:
                    best_fit = raw_new
                    best_seq = seq[:]
            else:
                # 回滚 section 缓存
                if use_inc_sec:
                    for s, old_val in saved_secs.items():
                        sec_level_cache[s] = old_val
                seq.pop(j); seq.insert(i, ev)

        point = TrajectoryPoint(
            cooling_step=step, temperature=T, current_fit=raw_current,
            best_fit=best_fit, accepted=accepted, proposed=cfg.trials,
        )
        trajectory.append(point)

        if verbose and (step % 20 == 0 or step == cfg.steps - 1):
            if cfg.coex_penalty > 0:
                from conop_py.cost import coexistence_violations
                coex_real = coexistence_violations(base_ctx.rebuild_pos(seq))
                print(f"  step {step:3d}/{cfg.steps}  T={T:8.3f}  "
                      f"cur={raw_current:7.2f}  best={best_fit:7.2f}  "
                      f"accept={accepted}/{cfg.trials}  coex={coex_real}")
            else:
                print(f"  step {step:3d}/{cfg.steps}  T={T:8.3f}  "
                      f"cur={raw_current:7.2f}  best={best_fit:7.2f}  "
                      f"accept={accepted}/{cfg.trials}")

        T *= cfg.ratio
        if T < 1e-3:
            break

    if verbose:
        print(f"最终 best_fit = {best_fit:.2f}")

    if verbose and cfg.coex_penalty > 0:
        from conop_py.cost import coexistence_violations
        ctx_final = base_ctx.rebuild_pos(best_seq)
        coex_final = coexistence_violations(ctx_final)
        print(f"  共存违反: {coex_final} 次（weight={cfg.coex_penalty}）")

    return AnnealResult(best_sequence=best_seq, best_fit=best_fit, trajectory=trajectory)


# ---------------------------------------------------------------------------
# 共存辅助
# ---------------------------------------------------------------------------


def _build_taxon_coex(
    section_obs: dict[int, list[tuple[float, EventKey]]],
) -> dict[int, dict[int, set[int]]]:
    """预计算每 section 中必须共存的 taxon 对。

    对每个剖面：taxa A 和 B 在观测中 range 重叠 → 在模型中 range 也必须重叠。
    返回 {sec_id: {eid: set[other_eid]}}，用于 _coex_delta 快速查表。
    """
    out: dict[int, dict[int, set[int]]] = {}
    for sec_id, evs in section_obs.items():
        fad: dict[int, float] = {}
        lad: dict[int, float] = {}
        for level, (eid, etype) in evs:
            if etype == 1:
                fad[eid] = level
            elif etype == 2:
                lad[eid] = level
        taxa_with_both = sorted(set(fad) & set(lad))
        per_eid: dict[int, set[int]] = defaultdict(set)
        for i in range(len(taxa_with_both)):
            a = taxa_with_both[i]
            for j in range(i + 1, len(taxa_with_both)):
                b = taxa_with_both[j]
                # 观测 range 不重叠 → 没有共存要求
                if lad[a] < fad[b] or lad[b] < fad[a]:
                    continue
                per_eid[a].add(b)
                per_eid[b].add(a)
        if per_eid:
            out[sec_id] = dict(per_eid)
    return out


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
                 misfit_fn=ordinal_misfit,
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
