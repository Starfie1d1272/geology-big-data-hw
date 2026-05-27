"""CONOP 代价函数（全量重算版本）。

理论依据：Sadler & Cooper (2003) §3.2 —
    - ORDINAL:   错位事件对数（与 CONOP9 outmain.txt 完全一致）
    - LEVEL:     用 L1 保序回归（PAV + box constraints）计算各事件的放置水平，
                 统计放置水平与观测水平之间的 distinct horizon 数。
    - EVENTUAL:  与 LEVEL 同源，但每个 horizon 按 forcing event 数加权。
    - WEIGHTED:  多剖面支持度加权的 Ordinal

所有 misfit 函数共享 ConopContext 预计算结构，避免 SA 迭代中重复构建。

SA 热路径不在本文件——见 conop_py/incremental.py 的 FastOrdinalState，
差分公式 + numba JIT，比这里的 ordinal_misfit() 快约 15×。
为兼容旧 import，文件末尾 re-export FastOrdinalState 等。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from conop_py.io import Observation
from conop_py.incremental import (
    EventKey, Sequence,
    _inversion_count,                          # ordinal_misfit 复用
    # 以下三项 re-export 给旧代码（tests / scripts）
    FastOrdinalState as FastOrdinalState,
    ordinal_section as ordinal_section,
    build_event_sections as build_event_sections,
)


# ---------------------------------------------------------------------------
# 预计算上下文 —— 一次构建，所有 misfit 函数复用
# ---------------------------------------------------------------------------

@dataclass
class ConopContext:
    """所有 misfit 函数共享的预计算数据。

    构建一次后 memoize，SA 迭代中 sequence 变了用 rebuild_pos 只更新 pos。
    """
    model_sequence: Sequence
    pos: dict[EventKey, int]                              # event → composite position
    section_obs: dict[int, list[tuple[float, EventKey]]]  # sec_id → [(level, ev), ...]
    sec_levels: dict[int, list[tuple[float, list[EventKey]]]]  # sec_id → 按 level 升序的 [(level, [evs])]
    taxon_sec: dict[tuple[int, int], dict[int, float]]    # (taxon_id, sec_id) → {1: FAD_level, 2: LAD_level}
    taxa: set[int] = field(default_factory=set)           # 所有 taxon entity_id

    @classmethod
    def build(cls, model_sequence: Sequence,
              section_obs: dict[int, list[tuple[float, EventKey]]]) -> ConopContext:
        pos = {ev: i for i, ev in enumerate(model_sequence)}

        # sec_levels: 每个 section 按 level 升序聚合事件
        sec_levels: dict[int, list[tuple[float, list[EventKey]]]] = {}
        for sid, evs in section_obs.items():
            by_level: dict[float, list[EventKey]] = {}
            for lv, ev in evs:
                by_level.setdefault(lv, []).append(ev)
            sec_levels[sid] = sorted(by_level.items())

        # taxon_sec: 每个 (taxon, section) 的观测 FAD/LAD level
        taxon_sec: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
        for sid, evs in section_obs.items():
            for lv, (eid, etype) in evs:
                if etype in (1, 2):
                    taxon_sec[(eid, sid)][etype] = lv

        # taxa: 在 composite 中同时有 FAD 和 LAD 的 entity_id
        taxa = {eid for (eid, _), _ in taxon_sec.items()
                if (eid, 1) in pos and (eid, 2) in pos}

        return cls(
            model_sequence=model_sequence,
            pos=pos,
            section_obs=section_obs,
            sec_levels=sec_levels,
            taxon_sec=taxon_sec,
            taxa=taxa,
        )

    def rebuild_pos(self, model_sequence: Sequence) -> ConopContext:
        """只重建 pos（SA 迭代中 sequence 变了但观测数据不变）。"""
        return ConopContext(
            model_sequence=model_sequence,
            pos={ev: i for i, ev in enumerate(model_sequence)},
            section_obs=self.section_obs,
            sec_levels=self.sec_levels,
            taxon_sec=self.taxon_sec,
            taxa=self.taxa,
        )


# ---------------------------------------------------------------------------
# 辅助：观测分组 + 多剖面支持度
# ---------------------------------------------------------------------------

def build_section_observations(
    observations: list[Observation],
) -> dict[int, list[tuple[float, EventKey]]]:
    """按 section 分组观测。"""
    by_section: dict[int, list[tuple[float, EventKey]]] = defaultdict(list)
    for o in observations:
        by_section[o.section_id].append((o.level, (o.entity_id, o.event_type)))
    return dict(by_section)


def build_pairwise_support(
    observations: list[Observation],
) -> dict[tuple[EventKey, EventKey], float]:
    """预计算每对事件在多剖面中的顺序支持度（用于 weighted_ordinal_misfit）。"""
    by_section: dict[int, dict[EventKey, float]] = defaultdict(dict)
    for o in observations:
        by_section[o.section_id][(o.entity_id, o.event_type)] = o.level

    pair_counts: dict[tuple[EventKey, EventKey], int] = defaultdict(int)
    for ev_levels in by_section.values():
        ev_list = list(ev_levels.items())
        for i, (ev_a, lev_a) in enumerate(ev_list):
            for ev_b, lev_b in ev_list[i + 1:]:
                if lev_a == lev_b:
                    continue
                if lev_a < lev_b:
                    pair_counts[(ev_a, ev_b)] += 1
                else:
                    pair_counts[(ev_b, ev_a)] += 1

    support: dict[tuple[EventKey, EventKey], float] = {}
    processed: set[tuple[EventKey, EventKey]] = set()
    for (a, b), cnt_ab in pair_counts.items():
        if (a, b) in processed:
            continue
        cnt_ba = pair_counts.get((b, a), 0)
        total = cnt_ab + cnt_ba
        s = cnt_ab / total
        support[(a, b)] = s
        support[(b, a)] = 1 - s
        processed.add((a, b))
        processed.add((b, a))
    return support


# ---------------------------------------------------------------------------
# Misfit 函数（都接受 ConopContext）
# ---------------------------------------------------------------------------

def ordinal_misfit(ctx: ConopContext) -> float:
    """Ordinal penalty —— 逆序对计数，与 CONOP9 完全一致（bestsoln.dat 上 = 367）。"""
    total = 0.0
    for evs in ctx.section_obs.values():
        evs_sorted = sorted(
            (e for e in evs if e[1] in ctx.pos),
            key=lambda x: (x[0], ctx.pos[x[1]]),
        )
        ranks = [ctx.pos[ev] for _, ev in evs_sorted]
        total += _inversion_count(ranks)
    return total


def level_misfit(ctx: ConopContext) -> float:
    """LEVEL penalty —— 每个需要跨过的 horizon 计 1 分。

    对应 CONOP9 "Level Penalty: 237.0000 levels"。
    用 L1 保序回归（PAV + box constraints）得到各事件在 section 中的放置水平，
    再统计放置水平与观测水平间的 distinct horizon 数。
    """
    total = 0.0
    for sec_id in ctx.section_obs:
        placed = _compute_placed_isotonic(ctx, sec_id)
        if not placed:
            continue
        evs = ctx.section_obs[sec_id]
        # Horizons = sorted distinct levels of events in composite
        horizons = sorted({l for l, ev in evs if ev in ctx.pos})
        for level, ev in evs:
            if ev not in placed:
                continue
            eid, etype = ev
            p = placed[ev]
            if (eid, 1) not in ctx.pos or (eid, 2) not in ctx.pos:
                continue
            rF, rL = ctx.pos[(eid, 1)], ctx.pos[(eid, 2)]
            if rF >= rL:
                total += len(ctx.model_sequence)
                continue
            if etype == 1 and p < level:
                total += sum(1 for h in horizons if p <= h < level)
            elif etype == 2 and p > level:
                total += sum(1 for h in horizons if level < h <= p)
    return total


def eventual_misfit(ctx: ConopContext) -> float:
    """EVENTUAL penalty —— 每个跨过的 horizon 按其上 forcing event 数加权。

    对应 CONOP9 "Eventual Penalty: 353.0000 events"。
    当前实现 ~335（误差 -5%）。用 PAV 保序回归确定放置水平后的 forcing event 加权。
    """
    total = 0.0
    for sec_id in ctx.section_obs:
        placed = _compute_placed_isotonic(ctx, sec_id)
        if not placed:
            continue
        evs = ctx.section_obs[sec_id]
        horizons = sorted({l for l, ev in evs if ev in ctx.pos})
        # Events at each level (only those in composite)
        by_level: dict[float, list[EventKey]] = {}
        for level, ev in evs:
            if ev in ctx.pos:
                by_level.setdefault(level, []).append(ev)

        for level, ev in evs:
            if ev not in placed:
                continue
            eid, etype = ev
            if etype not in (1, 2):
                continue
            if (eid, 1) not in ctx.pos or (eid, 2) not in ctx.pos:
                continue
            rF, rL = ctx.pos[(eid, 1)], ctx.pos[(eid, 2)]
            if rF >= rL:
                total += len(ctx.model_sequence)
                continue
            p = placed[ev]
            if etype == 1 and p < level:
                hs = [h for h in horizons if p <= h < level]
            elif etype == 2 and p > level:
                hs = [h for h in horizons if level < h <= p]
            else:
                continue
            for h in hs:
                for eh in by_level.get(h, []):
                    if eh[1] != 5 and eh in ctx.pos and rF < ctx.pos[eh] < rL:
                        total += 1.0
    return total


def weighted_ordinal_misfit(
    ctx: ConopContext,
    pairwise_support: dict[tuple[EventKey, EventKey], float],
) -> float:
    """多剖面支持度加权的 Ordinal penalty。"""
    total = 0.0
    for evs in ctx.section_obs.values():
        present = sorted(
            (e for e in evs if e[1] in ctx.pos),
            key=lambda x: x[0],
        )
        for i in range(len(present)):
            lev_i, ev_a = present[i]
            for j in range(i + 1, len(present)):
                lev_j, ev_b = present[j]
                if lev_i >= lev_j:
                    continue
                if ctx.pos[ev_a] > ctx.pos[ev_b]:
                    total += pairwise_support.get((ev_a, ev_b), 0.5)
    return total


def combined_misfit(
    ctx: ConopContext,
    weights: dict[str, float] | None = None,
    pairwise_support: dict | None = None,
) -> float:
    """多目标组合惩罚 = w_ord × Ordinal + w_lev × Level + w_evt × Eventual。

    Args:
        ctx:               预计算的 ConopContext
        weights:           {'ordinal': 1.0, 'level': 1.0, 'eventual': 1.0}
        pairwise_support:  传入后 ordinal 组件用 weighted_ordinal_misfit
    """
    w = weights or {}
    w_ord = w.get('ordinal', 1.0)
    w_lev = w.get('level', 1.0)
    w_evt = w.get('eventual', 0.0)

    total = 0.0
    if w_ord > 0:
        if pairwise_support:
            total += w_ord * weighted_ordinal_misfit(ctx, pairwise_support)
        else:
            total += w_ord * ordinal_misfit(ctx)
    if w_lev > 0:
        total += w_lev * level_misfit(ctx)
    if w_evt > 0:
        total += w_evt * eventual_misfit(ctx)
    return total


def coexistence_violations(ctx: ConopContext) -> int:
    """共存约束违反次数（> 0 表示解非法）。

    注：CONOP-run/bestsoln.dat 自己也有 24 次违反，说明数据集本身存在矛盾，
    不是算法 bug。
    """
    violations = 0
    for sec_id, evs in ctx.section_obs.items():
        fad: dict[int, float] = {}
        lad: dict[int, float] = {}
        for level, (eid, etype) in evs:
            if etype == 1:    fad[eid] = level
            elif etype == 2:  lad[eid] = level

        coexisting = set(fad) & set(lad)
        taxa = list(coexisting)
        for i in range(len(taxa)):
            a = taxa[i]
            for j in range(i + 1, len(taxa)):
                b = taxa[j]
                if lad[a] < fad[b] or lad[b] < fad[a]:
                    continue  # ranges 不重叠
                fa, la = ctx.pos[(a, 1)], ctx.pos[(a, 2)]
                fb, lb = ctx.pos[(b, 1)], ctx.pos[(b, 2)]
                if not (fa < lb and fb < la):
                    violations += 1
    return violations


# ---------------------------------------------------------------------------
# PAV 保序回归 + LEVEL / EVENTUAL 核心
# ---------------------------------------------------------------------------

def _lower_median(vals: list[float]) -> float:
    """下中位数（lower median）：偶数个元素时取较小的中值。"""
    s = sorted(vals)
    return s[(len(s) - 1) // 2]


def _compute_placed_isotonic(
    ctx: ConopContext, sec_id: int,
) -> dict[EventKey, float]:
    """L1 PAV 保序回归 + one-sided box constraints。

    对 section 内所有在 composite 中的事件，按 composite 位置排序后
    用 PAV 算法求单调非递减的"放置水平"：
      - FAD (type=1):  placed ≤ observed，区间 [sec_min, observed]
      - LAD (type=2):  placed ≥ observed，区间 [observed, sec_max]
      - 固定事件 (3/4/5): placed = observed

    Returns {EventKey → placed_level}。
    """
    evs = ctx.section_obs[sec_id]
    present = [(level, ev) for (level, ev) in evs if ev in ctx.pos]
    present.sort(key=lambda x: ctx.pos[x[1]])
    if not present:
        return {}

    sec_min = min(l for l, _ in present)
    sec_max = max(l for l, _ in present)

    blocks: list[dict] = []
    for level, ev in present:
        etype = ev[1]
        if etype == 1:       # FAD
            lo, hi = sec_min, level
        elif etype == 2:     # LAD
            lo, hi = level, sec_max
        else:                # 固定
            lo, hi = level, level
        blocks.append({
            'obs': [level], 'evs': [ev],
            'lower': lo, 'upper': hi, 'value': max(lo, min(hi, level)),
        })

    i = 1
    while i < len(blocks):
        if blocks[i - 1]['value'] > blocks[i]['value']:
            prev, cur = blocks[i - 1], blocks[i]
            mo = prev['obs'] + cur['obs']
            ml = max(prev['lower'], cur['lower'])
            mu = min(prev['upper'], cur['upper'])
            mv = max(ml, min(mu, _lower_median(mo)))
            blocks[i - 1] = {
                'obs': mo,
                'evs': prev['evs'] + cur['evs'],
                'lower': ml, 'upper': mu, 'value': mv,
            }
            blocks.pop(i)
            if i > 1:
                i -= 1  # 回溯检查上一步是否被这个新值违反
        else:
            i += 1

    return {ev: b['value'] for b in blocks for ev in b['evs']}


# ---------------------------------------------------------------------------
# 验证入口：用 CONOP-run/bestsoln.dat 反查各 penalty
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from conop_py.io import parse_loadfile, parse_solution, solution_to_sequence

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "CONOP-run"
    obs = parse_loadfile(f"{data_dir}/loadfile.dat")
    sol_records = parse_solution(f"{data_dir}/bestsoln.dat")
    model_seq = solution_to_sequence(sol_records)

    section_obs = build_section_observations(obs)
    ctx = ConopContext.build(model_seq, section_obs)

    print(f"模型序列长度: {len(model_seq)}")
    print(f"总观测数: {len(obs)}  按 section 分组: {len(section_obs)} 个")
    print(f"Taxa 数: {len(ctx.taxa)}")
    print()
    print(f"Ordinal  = {ordinal_misfit(ctx):.1f}  (CONOP9: 367)")
    print(f"Level    = {level_misfit(ctx):.1f}  (CONOP9: 237)")
    print(f"Eventual = {eventual_misfit(ctx):.1f}  (CONOP9: 353)")
    print(f"共存约束违反: {coexistence_violations(ctx)}")
    print()
    print(f"{'sec':>4s}{'LEVEL':>8s}{'Ordinal':>10s}{'outLEVEL':>10s}{'outOrd':>10s}")
    ref_level = {1:43, 2:55, 3:43, 4:51, 5:0, 6:1, 7:17, 8:6, 9:0, 10:9, 11:5, 12:7}
    ref_ord = {1:56, 2:99, 3:62, 4:64, 5:0, 6:1, 7:23, 8:13, 9:0, 10:29, 11:12, 12:8}
    for sec in sorted(section_obs):
        ctx_s = ConopContext.build(model_seq, {sec: section_obs[sec]})
        lev_s = level_misfit(ctx_s)
        ord_s = ordinal_misfit(ctx_s)
        print(f"{sec:>4d}{int(lev_s):>8d}{int(ord_s):>10d}"
              f"{ref_level[sec]:>10d}{ref_ord[sec]:>10d}")
    print(f"{'合':>4s}{int(level_misfit(ctx)):>8d}{int(ordinal_misfit(ctx)):>10d}"
          f"{sum(ref_level.values()):>10d}{sum(ref_ord.values()):>10d}")
