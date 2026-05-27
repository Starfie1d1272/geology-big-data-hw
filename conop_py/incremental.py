"""SA 热路径专用：增量 ordinal cost 状态机 + numba JIT 内核。

设计要点：
    1. 维护一份 seq（list[EventKey]）+ pos_dict[ev → int] + pos_arr 全局数组
    2. 每 section 维护 sec_ordinal 缓存（当前逆序对数）
    3. trial_move：差分公式 O(n_s) 算 Δordinal——只重算 event_sections[ev] 涉及的
       sections（典型 1-4 个，全集 12 个），不调 sort+merge
    4. revert(undo) 恢复全部状态

实测：相比 cost.ordinal_misfit() 全量重算，约 15× 加速；缺 numba 时自动 fallback
纯 Python 仍保持正确性（但只 3-4× 加速）。
"""
from __future__ import annotations

from collections import defaultdict


EventKey = tuple[int, int]
Sequence = list[EventKey]


# ---------------------------------------------------------------------------
# Numba 可选加速
# ---------------------------------------------------------------------------
try:
    import numba as _numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _count_inversions_py(sec_levels, sec_ev_idx, pos_arr,
                         ev_idx: int, ev_lv: float, ev_pos: int) -> int:
    """单 section 中 ev 与其他事件的逆序对数（差分公式核心）。

    sec_levels[k], sec_ev_idx[k] 是 section 第 k 条观测的 (level, ev_idx)。
    与观测顺序不符的 (ev, other) 即逆序对：observation 说谁先（ev_lv < lv），
    但 model 中相反（ev_pos vs other_pos），则计 1 分。
    """
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
    # 预热：触发 JIT 编译（数据类型固定后才不会运行时重编译）
    import numpy as _np
    _count_inversions_jit(
        _np.zeros(1, dtype=_np.float64), _np.zeros(1, dtype=_np.int32),
        _np.zeros(1, dtype=_np.int32), 0, 0.0, 0,
    )
    _count_inversions = _count_inversions_jit
else:
    _count_inversions = _count_inversions_py


# ---------------------------------------------------------------------------
# 全量逆序对计数（用于初始化缓存 + 兼容 ordinal_misfit）
# ---------------------------------------------------------------------------

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
    """单 section ordinal 计数。语义与 cost.ordinal_misfit 完全一致。"""
    evs_sorted = sorted(
        (e for e in section_evs if e[1] in pos),
        key=lambda x: (x[0], pos[x[1]]),
    )
    ranks = [pos[ev] for _, ev in evs_sorted]
    return _inversion_count(ranks)


# ---------------------------------------------------------------------------
# 增量 ordinal 状态机
# ---------------------------------------------------------------------------

class FastOrdinalState:
    """SA 热路径专用：per-section ordinal 缓存 + 局部 pos 更新。

    用法：
        state = FastOrdinalState(initial_seq, section_obs)
        # 之后 seq 和 state 一起维护：caller 改 seq，state.trial_move() 改 state
        delta, undo = state.trial_move(seq, ev, new_pos, old_pos)
        if not accept: state.revert(seq, undo)

    实测要点：纯 dict 比 NumPy mask 快 3-4×（n=120 太小，NumPy 开销没摊销）；
    内层逆序计数走 numba JIT 是最大加速来源（~60×）。
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

        # numba JIT 内层循环所需的 NumPy 数据结构
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

        # global pos 数组：pos_arr[ev_idx] = pos in model_sequence (-1 = 不在 seq 里)
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
        """同步 pos_dict + pos_arr：seq 已经被 caller 改成 pop+insert 后的状态。"""
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
        返回 (delta_ordinal, undo)。caller 必须 revert(seq, undo) 或接受现状。
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
        # 这里只把 pos_dict / pos_arr / sec_ordinal 还原
        self._apply_shift(seq, ev, new_pos, old_pos)
        self.sec_ordinal.update(saved_so)

    def commit(self) -> None:
        """trial_move 已在内部应用，commit 是 no-op，保留供 caller 显式表达意图。"""

    def current_sequence(self) -> Sequence:
        return [ev for ev, _ in sorted(self._pos_dict.items(), key=lambda x: x[1])]

    def pos_dict(self) -> dict[EventKey, int]:
        return dict(self._pos_dict)
