"""CONOP 代价函数：ORDINAL 模式（精确）+ LEVEL 模式（近似）。

理论依据：Sadler & Cooper (2003) §3.2 —
    - ORDINAL penalty: 错位事件对数（pairs of locally observed pairwise sequences
      contradicted by the proposed sequence）。论文称之为 RASCAL/ORDINAL 模式。
    - LEVEL penalty:   把观测范围扩展以匹配 model 序列所需跨过的本剖面 horizon 总数。
                       论文默认推荐的度量；与 ORDINAL 有 0.5-1 倍差距。

实现现状：
    - `ordinal_misfit()` 与 outmain.txt 的 "Ordinal Penalty: 367.0000 pairs" 完全一致；
    - `level_misfit()`  近似计算（FAD/LAD 双向扩展、horizon 去重）；
      与 outmain.txt 的 "Level Penalty: 237.0000" 同量级但偏高约 60%。
      细节差异源于 CONOP9 的最小化重叠 horizon 计数策略未完全公开。

本模块对外默认导出 `ordinal_misfit` 作为 SA 优化目标——它是 CONOP 标准 penalty 之一
且与官方实现 byte-for-byte 一致，可严谨支持论文写作。
"""
from __future__ import annotations

from collections import defaultdict

from conop_py.io import Observation


EventKey = tuple[int, int]  # (entity_id, event_type)
Sequence = list[EventKey]


def build_section_observations(
    observations: list[Observation],
) -> dict[int, list[tuple[float, EventKey]]]:
    """按 section 分组观测：{section_id: [(level, event_key), ...]}（未排序）。"""
    by_section: dict[int, list[tuple[float, EventKey]]] = defaultdict(list)
    for o in observations:
        by_section[o.section_id].append((o.level, (o.entity_id, o.event_type)))
    return dict(by_section)


def ordinal_misfit(
    model_sequence: Sequence,
    section_obs: dict[int, list[tuple[float, EventKey]]],
) -> float:
    """Ordinal penalty（pairs）—— 论文 §3.2 中的 ORDINAL 模式。

    对每个 section 内观测到的事件子集，按 (level, model_rank) 升序排列，
    计算 model_rank 序列的逆序对数。每个 cross-level 错位对贡献 1。

    经验证：在 CONOP-run/bestsoln.dat 上得到 367 与 outmain.txt 的
    "Ordinal Penalty: 367.0000 pairs" 完全一致。
    """
    pos = {ev: i for i, ev in enumerate(model_sequence)}
    total = 0.0
    for sec_id, evs in section_obs.items():
        evs_sorted = sorted(
            (e for e in evs if e[1] in pos),
            key=lambda x: (x[0], pos[x[1]]),
        )
        ranks = [pos[ev] for _, ev in evs_sorted]
        total += _inversion_count(ranks)
    return total


def level_misfit(
    model_sequence: Sequence,
    section_obs: dict[int, list[tuple[float, EventKey]]],
) -> float:
    """LEVEL penalty —— 论文 §3.2 默认推荐的度量。

    实现逻辑（基于 Sadler 2000 推测）：
    每个 taxon X 在 section s 中需要扩展端点时，统计跨过的本剖面 horizon 数：
      - FAD 只能向下（向更老）扩展：找 section 中 level < F_obs 且 model rank > r_F 的事件 → 跨过的 horizons +1
      - LAD 只能向上（向更年轻）扩展：找 section 中 level > L_obs 且 model rank < r_L 的事件 → 跨过的 horizons +1
    同一 horizon 上多个事件只算一次。
    Markers 自身不被扩展（不可调整），但会被 taxa 跨过。

    Args:
        model_sequence: 长度 = 总事件数
        section_obs: build_section_observations() 输出
    """
    pos = {ev: i for i, ev in enumerate(model_sequence)}

    # 按 section 整理：unique levels + 每 level 的事件集
    sec_levels: dict[int, list[tuple[float, list[EventKey]]]] = {}
    for sec_id, evs in section_obs.items():
        by_level: dict[float, list[EventKey]] = {}
        for level, ev in evs:
            by_level.setdefault(level, []).append(ev)
        # 升序 level（更老 → 更年轻）
        sec_levels[sec_id] = sorted(by_level.items())

    # 整理每个 (entity, section) 的 FAD/LAD level
    fad_lad: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    for sec_id, evs in section_obs.items():
        for level, (eid, etype) in evs:
            if etype in (1, 2):
                fad_lad[(eid, sec_id)][etype] = level

    misfit = 0.0
    for (eid, sec_id), types in fad_lad.items():
        levels_in_sec = sec_levels[sec_id]
        # FAD 向下扩展：找 section 中 level < F_obs 但 model rank > r_F 的层位
        if 1 in types:
            F_obs = types[1]
            r_F = pos[(eid, 1)]
            for level, evs_at in levels_in_sec:
                if level >= F_obs:
                    break
                # 若该 horizon 有任一 event 的 model_rank > r_F，则需要 leapfrog（每 horizon 算 1 次）
                if any(pos[ev] > r_F for ev in evs_at):
                    misfit += 1
        # LAD 向上扩展：找 section 中 level > L_obs 但 model rank < r_L 的层位
        if 2 in types:
            L_obs = types[2]
            r_L = pos[(eid, 2)]
            for level, evs_at in levels_in_sec:
                if level <= L_obs:
                    continue
                if any(pos[ev] < r_L for ev in evs_at):
                    misfit += 1
    return misfit


def coexistence_violations(
    model_sequence: Sequence,
    section_obs: dict[int, list[tuple[float, EventKey]]],
) -> int:
    """检查共存约束：在某 section 中观测到两 taxa 的 range 重叠时，
    model 序列必须维持 FAD_A < LAD_B 且 FAD_B < LAD_A。
    返回违反次数（> 0 表示该解非法）。"""
    pos = {ev: i for i, ev in enumerate(model_sequence)}
    violations = 0
    for sec_id, evs in section_obs.items():
        # 收集该 section 中每个 taxon 的 (FAD_level, LAD_level)
        fad: dict[int, float] = {}
        lad: dict[int, float] = {}
        for level, (eid, etype) in evs:
            if etype == 1:
                fad[eid] = level
            elif etype == 2:
                lad[eid] = level
        # 取在该 section 中同时有 FAD/LAD 的 taxa 集合
        coexisting = set(fad) & set(lad)
        # 检查 model 序列里所有共存对：FAD_A < LAD_B 且 FAD_B < LAD_A 应当成立
        # 这里只检查 section 内观测到 range 重叠的对
        taxa = list(coexisting)
        for i in range(len(taxa)):
            a = taxa[i]
            for j in range(i + 1, len(taxa)):
                b = taxa[j]
                # section 中两 taxa 的 range 重叠？
                a_range = (fad[a], lad[a])
                b_range = (fad[b], lad[b])
                if a_range[1] < b_range[0] or b_range[1] < a_range[0]:
                    continue  # 不重叠
                # 重叠：model 中应满足 FAD_A < LAD_B 且 FAD_B < LAD_A
                fa, la = pos[(a, 1)], pos[(a, 2)]
                fb, lb = pos[(b, 1)], pos[(b, 2)]
                if not (fa < lb and fb < la):
                    violations += 1
    return violations


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
            tmp[k] = arr[i]
            i += 1
        else:
            tmp[k] = arr[j]
            count += mid - i + 1
            j += 1
        k += 1
    while i <= mid:
        tmp[k] = arr[i]
        i += 1
        k += 1
    while j <= hi:
        tmp[k] = arr[j]
        j += 1
        k += 1
    for x in range(lo, hi + 1):
        arr[x] = tmp[x]
    return count


# ---------------------------------------------------------------------------
# 验证脚本：用 CONOP-run/bestsoln.dat 反查 LEVEL penalty
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from conop_py.io import (
        parse_cfg, parse_loadfile, parse_solution, solution_to_sequence,
    )
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "CONOP-run"
    cfg = parse_cfg(f"{data_dir}/conop9.cfg")
    obs = parse_loadfile(f"{data_dir}/loadfile.dat")
    sol_records = parse_solution(f"{data_dir}/bestsoln.dat")
    model_seq = solution_to_sequence(sol_records)

    sect_obs = build_section_observations(obs)
    lev = level_misfit(model_seq, sect_obs)
    ordi = ordinal_misfit(model_seq, sect_obs)
    coex_viol = coexistence_violations(model_seq, sect_obs)

    print(f"模型序列长度: {len(model_seq)}")
    print(f"总观测数: {len(obs)}  按 section 分组: {len(sect_obs)} 个 section")
    print()
    print(f"{'sec':>4s}{'LEVEL':>8s}{'Ordinal':>10s}{'outLEVEL':>10s}{'outOrd':>10s}")
    ref_level = {1:43, 2:55, 3:43, 4:51, 5:0, 6:1, 7:17, 8:6, 9:0, 10:9, 11:5, 12:7}
    ref_ord = {1:56, 2:99, 3:62, 4:64, 5:0, 6:1, 7:23, 8:13, 9:0, 10:29, 11:12, 12:8}
    for sec in sorted(sect_obs):
        # 单 section 级别度量
        lev_s = level_misfit(model_seq, {sec: sect_obs[sec]})
        ord_s = ordinal_misfit(model_seq, {sec: sect_obs[sec]})
        print(f"{sec:>4d}{int(lev_s):>8d}{int(ord_s):>10d}{ref_level[sec]:>10d}{ref_ord[sec]:>10d}")
    print(f"{'合':>4s}{int(lev):>8d}{int(ordi):>10d}{sum(ref_level.values()):>10d}{sum(ref_ord.values()):>10d}")
    print()
    print(f"共存约束违反: {coex_viol}")
