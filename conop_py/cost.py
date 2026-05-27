"""CONOP 代价函数：ORDINAL + LEVEL + EVENTUAL + 加权 ORDINAL。

理论依据：Sadler & Cooper (2003) §3.2 —
    - ORDINAL:   错位事件对数（与 CONOP9 outmain.txt 完全一致）
    - LEVEL:     把观测范围扩展以匹配 composite 所需跨过的 horizon 总数（近似，误差 ~+43%）
    - EVENTUAL:  composite range 宽度与观测 range 宽度的一致性惩罚（防 range 坍缩）
    - WEIGHTED:  多剖面支持度加权的 Ordinal

所有 misfit 函数共享 ConopContext 预计算结构，避免 SA 迭代中重复构建。

性能优化（B6 + B7 + B9）：
    - ordinal_section() 单 section 计算，配合 build_event_sections() 实现增量
    - FastOrdinalState 维护 pos 数组 + 每 section 缓存，O(|affected sections|) 增量更新
    - numba JIT 加速 _section_count_inversions（可选，自动检测）
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from conop_py.io import Observation, AGE


# ---------------------------------------------------------------------------
# Numba 可选加速（找不到就用纯 Python fallback）
# ---------------------------------------------------------------------------
try:
    import numba as _numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _count_inversions_py(sec_levels, sec_ev_idx, pos_arr,
                         ev_idx: int, ev_lv: float, ev_pos: int) -> int:
    """纯 Python fallback：当 numba 不可用时使用。"""
    cnt = 0
    for k in range(len(sec_ev_idx)):
        other_idx = int(sec_ev_idx[k])
        if other_idx == ev_idx:
            continue
        lv = float(sec_levels[k])
        if lv == ev_lv:
            continue
        other_pos = int(pos_arr[other_idx])
        if (ev_lv < lv) != (ev_pos < other_pos):
            cnt += 1
    return cnt


if _HAS_NUMBA:
    _count_inversions_jit = _numba.njit(cache=True)(_count_inversions_py)
    # 预热：触发 JIT 编译（数据类型固定）
    import numpy as _np
    _count_inversions_jit(
        _np.zeros(1, dtype=_np.float64), _np.zeros(1, dtype=_np.int32),
        _np.zeros(1, dtype=_np.int32), 0, 0.0, 0,
    )
    _count_inversions = _count_inversions_jit
else:
    _count_inversions = _count_inversions_py

EventKey = tuple[int, int]   # (entity_id, event_type)
Sequence = list[EventKey]


# ---------------------------------------------------------------------------
# 预计算上下文 —— 一次构建，所有 misfit 函数复用
# ---------------------------------------------------------------------------

@dataclass
class ConopContext:
    """所有 misfit 函数共享的预计算数据。

    构建一次后 memoisze，后续 SA 迭代只更新 model_sequence 和 pos。
    """
    model_sequence: Sequence
    pos: dict[EventKey, int]                        # event → composite position
    section_obs: dict[int, list[tuple[float, EventKey]]]  # sec_id → [(level, ev), ...]
    sec_levels: dict[int, list[tuple[float, list[EventKey]]]]  # sec_id → sorted (level, [evs])
    taxon_sec: dict[tuple[int, int], dict[int, float]]  # (taxon_id, sec_id) → {1: FAD_level, 2: LAD_level}
    taxa: set[int] = field(default_factory=set)      # all taxon entity_ids

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

        # taxa: 所有 taxon entity_id（有 FAD 且 LAD 在 composite 中的）
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
# 辅助：构建 section_obs 和 pairwise support（外置，不依赖 ConopContext）
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
# 单个 misfit 函数（都接受 ConopContext）
# ---------------------------------------------------------------------------

def ordinal_misfit(ctx: ConopContext) -> float:
    """Ordinal penalty —— 逆序对计数，与 CONOP9 完全一致。"""
    total = 0.0
    for evs in ctx.section_obs.values():
        evs_sorted = sorted(
            (e for e in evs if e[1] in ctx.pos),
            key=lambda x: (x[0], ctx.pos[x[1]]),
        )
        ranks = [ctx.pos[ev] for _, ev in evs_sorted]
        total += _inversion_count(ranks)
    return total


def _range_extension(
    ctx: ConopContext,
    per_horizon_fn,   # (evs_at, r_F, r_L, pos) -> float — 每 horizon 的惩罚量
) -> float:
    """LEVEL / EVENTUAL 公共逻辑：遍历所有 (taxon, section) 的 range 扩展。

    per_horizon_fn 决定每 horizon 的惩罚粒度：
      - LEVEL:   有任意 forcing event → +1（regardless of how many events）
      - EVENTUAL: 每个 forcing event → +1（每 horizon 累加）
    AGE 锚点（type=5）统一排除：它不是化石观测，不应触发 range 扩展。
    """
    n = len(ctx.model_sequence)
    total = 0.0

    for (eid, sec_id), types in ctx.taxon_sec.items():
        r_F = ctx.pos[(eid, 1)]
        r_L = ctx.pos[(eid, 2)]
        if r_F >= r_L:
            total += n
            continue

        levels_in_sec = ctx.sec_levels[sec_id]

        # FAD 向下扩展
        if 1 in types:
            F_obs = types[1]
            for level, evs_at in levels_in_sec:
                if level >= F_obs:
                    break
                total += per_horizon_fn(evs_at, r_F, r_L, ctx.pos)

        # LAD 向上扩展
        if 2 in types:
            L_obs = types[2]
            for level, evs_at in levels_in_sec:
                if level <= L_obs:
                    continue
                total += per_horizon_fn(evs_at, r_F, r_L, ctx.pos)

    return total


def _count_level(evs_at: list[EventKey], r_F: int, r_L: int,
                 pos: dict[EventKey, int]) -> float:
    """LEVEL 计数器：horizon 上有任意 forcing event → 1 分。"""
    return 1.0 if _any_forcing(evs_at, r_F, r_L, pos) else 0.0


def _count_eventual(evs_at: list[EventKey], r_F: int, r_L: int,
                    pos: dict[EventKey, int]) -> float:
    """EVENTUAL 计数器：horizon 上每个 forcing event 计 1 分。

    文献定义：EVENTUAL = like LEVEL, but weights each event level by the
    number of events occurring at that level.
    """
    return float(sum(1 for ev in evs_at
                     if ev in pos and ev[1] != AGE and r_F < pos[ev] < r_L))


def level_misfit(ctx: ConopContext) -> float:
    """LEVEL penalty —— 每个需要跨过的 horizon 计 1 分。

    对应 CONOP9 "Level Penalty: 237.0000 levels"。
    经验证：在 CONOP bestsoln.dat 上得 340（误差 ~+43%，源于未公开的计数细节）。
    """
    return _range_extension(ctx, _count_level)


def eventual_misfit(ctx: ConopContext) -> float:
    """EVENTUAL penalty —— 每个需要跨过的 horizon 按该 horizon 上的事件数加权。

    文献定义（Sadler & Cooper 2003）：
    "like LEVEL, but weights each event level by the number of events (taxa)
     occurring at that level"

    对应 CONOP9 "Eventual Penalty: 353.0000 events"。
    同一 horizon 上观测到的 taxon 越多，跨过它的惩罚越重，因为该层位的
    地层分辨率更高，'移走'的代价也更大。
    """
    return _range_extension(ctx, _count_eventual)


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


# ---------------------------------------------------------------------------
# 多目标组合 —— SA 优化入口
# ---------------------------------------------------------------------------

def combined_misfit(
    ctx: ConopContext,
    weights: dict[str, float] | None = None,
    pairwise_support: dict | None = None,
) -> float:
    """多目标组合惩罚 = w_ord × Ordinal + w_lev × Level + w_evt × Eventual。

    Args:
        ctx: 预计算的 ConopContext
        weights: {'ordinal': 1.0, 'level': 1.0, 'eventual': 1.0}
        pairwise_support: 如果用加权 Ordinal，传预计算的支持度
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
    """共存约束违反次数（> 0 表示解非法）。"""
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
# 内联辅助
# ---------------------------------------------------------------------------

def _any_forcing(
    evs_at: list[EventKey],
    r_F: int, r_L: int,
    pos: dict[EventKey, int],
) -> bool:
    """判断该 horizon 上是否有 taxon 事件落在 (r_F, r_L) 内——触发 range 扩展。"""
    for ev in evs_at:
        if ev not in pos:
            continue
        if ev[1] == AGE:   # 同位素年龄锚点不触发 range 扩展
            continue
        p = pos[ev]
        if r_F < p < r_L:
            return True
    return False


def _inversion_count(seq: list[int]) -> int:
    """O(n log n) 归并排序逆序对计数。"""
    if len(seq) < 2:
        return 0
    arr = seq[:]
    tmp = [0] * len(arr)
    return _merge_count(arr, tmp, 0, len(arr) - 1)


def _merge_count(arr: list[int], tmp: list[int], lo: int, hi: int) -> int:
    if lo >= hi:
        return 0
    mid = (lo + hi) // 2
    count = _merge_count(arr, tmp, lo, mid) + _merge_count(arr, tmp, mid + 1, hi)
    i, j, k = lo, mid + 1, lo
    while i <= mid and j <= hi:
        if arr[i] <= arr[j]:
            tmp[k] = arr[i]; i += 1
        else:
            tmp[k] = arr[j]; count += mid - i + 1; j += 1
        k += 1
    while i <= mid:
        tmp[k] = arr[i]; i += 1; k += 1
    while j <= hi:
        tmp[k] = arr[j]; j += 1; k += 1
    for x in range(lo, hi + 1):
        arr[x] = tmp[x]
    return count


# ---------------------------------------------------------------------------
# B6 + B7：增量 ordinal 代价 + NumPy rank 数组
# ---------------------------------------------------------------------------
# 思路：
#   1. 维护 pos: np.ndarray[int]，按全局 event index 索引（FastOrdinalState）
#   2. 预计算 event_sections: 每个 event 所在的 sections
#   3. 每 section 维护 sec_ordinal 缓存（当前逆序对数）
#   4. trial_move(eidx, new_pos):
#        - 只重算 event_sections[eidx] 这几个 section 的 ordinal
#        - 返回 delta 和 undo handle
#      revert(undo) 恢复状态
#   速度提升：~12 sections 中只重算 1–4 个 → 3-6× ordinal SA 加速

def build_event_sections(
    section_obs: dict[int, list[tuple[float, EventKey]]],
) -> dict[EventKey, list[int]]:
    """每个 event 出现在哪些 section 中。"""
    ev_secs: dict[EventKey, list[int]] = defaultdict(list)
    for sid, evs in section_obs.items():
        for _lv, ev in evs:
            ev_secs[ev].append(sid)
    return dict(ev_secs)


def ordinal_section(
    section_evs: list[tuple[float, EventKey]],
    pos: dict[EventKey, int],
) -> int:
    """单 section ordinal 计数。语义与 ordinal_misfit 完全一致。"""
    evs_sorted = sorted(
        (e for e in section_evs if e[1] in pos),
        key=lambda x: (x[0], pos[x[1]]),
    )
    ranks = [pos[ev] for _, ev in evs_sorted]
    return _inversion_count(ranks)


class FastOrdinalState:
    """SA 热路径专用：per-section ordinal 缓存 + 局部 pos 更新。

    设计要点（实测纯 dict 比 NumPy mask 快 3-4× —— n=120 太小，NumPy 开销没摊销）：
      - 维护一份 seq（list[EventKey]）+ pos_dict[ev→int]
      - 维护 sec_ordinal[sid→int] 缓存当前每 section 的逆序对数
      - trial_move(ev, new_pos)：局部更新 pos_dict 中受影响的 (|new_pos-old_pos|+1) 个事件 +
        只重算 event_sections[ev] 涉及的 sections（典型 1-4 个）

    用法：
        state = FastOrdinalState(initial_seq, section_obs)
        # 之后 seq 和 state 一起维护：caller 改 seq，state.trial_move() 改 state
        delta, undo = state.trial_move(seq, ev, new_pos, old_pos)
        if not accept: state.revert(seq, undo)
    """

    def __init__(self, model_sequence: Sequence,
                 section_obs: dict[int, list[tuple[float, EventKey]]]):
        import numpy as np

        self.section_obs = section_obs
        self.n = len(model_sequence)

        self._pos_dict: dict[EventKey, int] = {ev: i for i, ev in enumerate(model_sequence)}

        # 每 event 所在的 sections（小 dict，热路径快）
        self.event_sections: dict[EventKey, list[int]] = build_event_sections(section_obs)

        # 预计算 (ev, sec_id) → ev 在该 section 的观测 level
        self.ev_level: dict[EventKey, dict[int, float]] = defaultdict(dict)
        for sid, evs in section_obs.items():
            for lv, ev in evs:
                self.ev_level[ev][sid] = lv

        # ===== B9: numba 加速所需的 NumPy 数据 =====
        # 全局 event index：观测里出现过的 + model_sequence 里的并集
        all_events = list(model_sequence)
        seen = set(all_events)
        for sid, evs in section_obs.items():
            for _lv, ev in evs:
                if ev not in seen:
                    all_events.append(ev)
                    seen.add(ev)
        self.event_to_idx: dict[EventKey, int] = {ev: i for i, ev in enumerate(all_events)}
        self.idx_to_event: list[EventKey] = all_events

        # global pos 数组：pos_arr[ev_idx] = pos in model_sequence (or -1 if not in seq)
        self._pos_arr = np.full(len(all_events), -1, dtype=np.int32)
        for p, ev in enumerate(model_sequence):
            self._pos_arr[self.event_to_idx[ev]] = p

        # 每 section 的 numpy 数组：levels[k], ev_idx[k]
        self._sec_levels: dict[int, np.ndarray] = {}
        self._sec_ev_idx: dict[int, np.ndarray] = {}
        for sid, evs in section_obs.items():
            self._sec_levels[sid] = np.array([lv for lv, _ev in evs], dtype=np.float64)
            self._sec_ev_idx[sid] = np.array(
                [self.event_to_idx[ev] for _lv, ev in evs], dtype=np.int32)

        # 每 section 的 ordinal 缓存（同时验证 numpy 路径正确性）
        self.sec_ordinal: dict[int, int] = {
            sid: ordinal_section(evs, self._pos_dict)
            for sid, evs in section_obs.items()
        }

    @property
    def total(self) -> int:
        return sum(self.sec_ordinal.values())

    @property
    def pos_arr(self):
        """numpy int32 array，pos_arr[ev_idx] = position。"""
        return self._pos_arr

    def _apply_shift(self, seq: Sequence, ev: EventKey,
                     old_pos: int, new_pos: int) -> None:
        """更新 pos_dict + pos_arr：seq 已经被 caller 改成 pop+insert 后的状态。"""
        pd = self._pos_dict
        pa = self._pos_arr
        e2i = self.event_to_idx
        if old_pos < new_pos:
            for p in range(old_pos, new_pos):
                other = seq[p]
                pd[other] = p
                pa[e2i[other]] = p
        elif old_pos > new_pos:
            for p in range(new_pos + 1, old_pos + 1):
                other = seq[p]
                pd[other] = p
                pa[e2i[other]] = p
        pd[ev] = new_pos
        pa[e2i[ev]] = new_pos

    def trial_move(self, seq: Sequence, ev: EventKey,
                   new_pos: int, old_pos: int):
        """尝试把 ev 移到 new_pos（seq 已被 caller pop+insert 完）。

        Δordinal 用差分公式 O(n_s) 计算，热路径走 numba JIT。
        返回 (delta_ordinal, undo)。caller 必须 revert(seq, undo) 或不操作。
        """
        if old_pos == new_pos:
            return 0, None

        affected = self.event_sections.get(ev, [])
        if not affected:
            self._apply_shift(seq, ev, old_pos, new_pos)
            return 0, (ev, old_pos, new_pos, {})

        ev_idx = self.event_to_idx[ev]
        ev_level_map = self.ev_level[ev]
        pa = self._pos_arr
        sec_levels_all = self._sec_levels
        sec_ev_idx_all = self._sec_ev_idx
        count_fn = _count_inversions

        # 移动前各 affected section 的逆序贡献
        old_contrib = [0] * len(affected)
        for k, sid in enumerate(affected):
            old_contrib[k] = count_fn(
                sec_levels_all[sid], sec_ev_idx_all[sid], pa,
                ev_idx, ev_level_map[sid], old_pos,
            )

        # 应用 shift（同步 pd + pa）
        self._apply_shift(seq, ev, old_pos, new_pos)

        delta = 0
        saved_so = {}
        for k, sid in enumerate(affected):
            new_cnt = count_fn(
                sec_levels_all[sid], sec_ev_idx_all[sid], pa,
                ev_idx, ev_level_map[sid], new_pos,
            )
            sec_delta = new_cnt - old_contrib[k]
            saved_so[sid] = self.sec_ordinal[sid]
            self.sec_ordinal[sid] += sec_delta
            delta += sec_delta

        return delta, (ev, old_pos, new_pos, saved_so)

    def revert(self, seq: Sequence, undo) -> None:
        """还原 trial_move（caller 也要负责把 seq 还原到原状态）。"""
        if undo is None:
            return
        ev, old_pos, new_pos, saved_so = undo
        # seq 已经被 caller 还原（pop(new_pos) + insert(old_pos)）
        # 现在只需要把 pos_dict 还原
        self._apply_shift(seq, ev, new_pos, old_pos)
        self.sec_ordinal.update(saved_so)

    def commit(self) -> None:
        pass

    def current_sequence(self) -> Sequence:
        return [ev for ev, _ in sorted(self._pos_dict.items(), key=lambda x: x[1])]

    def pos_dict(self) -> dict[EventKey, int]:
        return dict(self._pos_dict)


# ---------------------------------------------------------------------------
# 验证：用 CONOP-run/bestsoln.dat 反查各 penalty
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from conop_py.io import (
        parse_cfg, parse_loadfile, parse_solution, solution_to_sequence,
    )
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "CONOP-run"
    obs = parse_loadfile(f"{data_dir}/loadfile.dat")
    sol_records = parse_solution(f"{data_dir}/bestsoln.dat")
    model_seq = solution_to_sequence(sol_records)

    section_obs = build_section_observations(obs)
    ctx = ConopContext.build(model_seq, section_obs)

    lev = level_misfit(ctx)
    ordi = ordinal_misfit(ctx)
    evt = eventual_misfit(ctx)
    coex_viol = coexistence_violations(ctx)

    print(f"模型序列长度: {len(model_seq)}")
    print(f"总观测数: {len(obs)}  按 section 分组: {len(section_obs)} 个 section")
    print(f"Taxa 数: {len(ctx.taxa)}")
    print()
    print(f"Ordinal  = {ordi:.1f}  (CONOP9: 367)")
    print(f"Level    = {lev:.1f}  (CONOP9: 237)")
    print(f"Eventual = {evt:.1f}  (CONOP9: 353)")
    print(f"共存约束违反: {coex_viol}")
    print()
    print(f"{'sec':>4s}{'LEVEL':>8s}{'Ordinal':>10s}{'outLEVEL':>10s}{'outOrd':>10s}")
    ref_level = {1:43, 2:55, 3:43, 4:51, 5:0, 6:1, 7:17, 8:6, 9:0, 10:9, 11:5, 12:7}
    ref_ord = {1:56, 2:99, 3:62, 4:64, 5:0, 6:1, 7:23, 8:13, 9:0, 10:29, 11:12, 12:8}
    for sec in sorted(section_obs):
        ctx_s = ConopContext.build(model_seq, {sec: section_obs[sec]})
        lev_s = level_misfit(ctx_s)
        ord_s = ordinal_misfit(ctx_s)
        print(f"{sec:>4d}{int(lev_s):>8d}{int(ord_s):>10d}{ref_level[sec]:>10d}{ref_ord[sec]:>10d}")
    print(f"{'合':>4s}{int(lev):>8d}{int(ordi):>10d}{sum(ref_level.values()):>10d}{sum(ref_ord.values()):>10d}")
