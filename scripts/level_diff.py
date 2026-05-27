"""Level 误差反向工程：逐 event 对比 CONOP9 outsect.txt 真值与我们的实现。

CONOP9 outsect.txt 每个 event 一行：
    <id> - <type>  <observed_m>  <placed_m>  <ext1_m>  <ext2_m>  <ext_levels>  {<code> <NAME> {<species>

第 7 个数值（ext_levels）即该 event 在该 section 的 LEVEL 贡献。
本脚本：
    1. 解析 outsect.txt → dict[(sec_id, ev_id, ev_type)] = level_truth
    2. 调用我们的 level_misfit 重写版返回 per-event 贡献
    3. 输出每个 event 的 (truth, ours, diff) 表，按 |diff| 排序

输出：results_py/level_diff.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conop_py.io import (
    parse_loadfile, parse_solution, solution_to_sequence, AGE,
)
from conop_py.cost import ConopContext, build_section_observations


# ---------------------------------------------------------------------------
# 1. 解析 outsect.txt
# ---------------------------------------------------------------------------

# 形如:  11 - 2    1135.000  1135.000     0.000     0.000      0  {1192     LAD {...
EVENT_LINE = re.compile(
    r"^\s+(\d+)\s*-\s*(\d+)\s+"          # event_id - type
    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+"    # observed_m  placed_m
    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+"    # ext1_m  ext2_m
    r"(\d+)\s+"                          # ext_levels  ← 真值
    r"\{"                                # 后面是 {code ... 注释
)
SECTION_HEADER = re.compile(r"SECTION\s*-\s*(\d+)")


def parse_outsect(path: Path) -> dict[tuple[int, int, int], int]:
    """返回 {(sec_id, event_id, event_type) → level_contribution_truth}。"""
    out: dict[tuple[int, int, int], int] = {}
    cur_sec: int | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = SECTION_HEADER.search(line)
        if m:
            cur_sec = int(m.group(1))
            continue
        m = EVENT_LINE.match(line)
        if m and cur_sec is not None:
            eid, etype = int(m.group(1)), int(m.group(2))
            levels = int(m.group(7))
            out[(cur_sec, eid, etype)] = levels
    return out


# ---------------------------------------------------------------------------
# 2. 我们的 level_misfit 逐 event 拆分
# ---------------------------------------------------------------------------

def our_level_breakdown(ctx: ConopContext) -> dict[tuple[int, int, int], int]:
    """对每个 (taxon, section) 拆开 range extension 的 LEVEL 贡献。

    返回 dict 键 (sec_id, eid, etype)，etype=1 表示 FAD 端的贡献，etype=2 表示 LAD 端。
    与 outsect.txt 的格式（每行一个 event 类型）一一对应。
    """
    n = len(ctx.model_sequence)
    out: dict[tuple[int, int, int], int] = {}

    for (eid, sec_id), types in ctx.taxon_sec.items():
        r_F = ctx.pos.get((eid, 1))
        r_L = ctx.pos.get((eid, 2))
        if r_F is None or r_L is None:
            continue
        if r_F >= r_L:
            # 当前实现：composite 中 FAD >= LAD 算 n 分。CONOP9 不会出这种解，
            # 暂记到 FAD 名下
            out[(sec_id, eid, 1)] = out.get((sec_id, eid, 1), 0) + n
            continue

        levels_in_sec = ctx.sec_levels[sec_id]

        if 1 in types:
            F_obs = types[1]
            cnt = 0
            for level, evs_at in levels_in_sec:
                if level >= F_obs:
                    break
                if _any_forcing(evs_at, r_F, r_L, ctx.pos):
                    cnt += 1
            if cnt:
                out[(sec_id, eid, 1)] = cnt

        if 2 in types:
            L_obs = types[2]
            cnt = 0
            for level, evs_at in levels_in_sec:
                if level <= L_obs:
                    continue
                if _any_forcing(evs_at, r_F, r_L, ctx.pos):
                    cnt += 1
            if cnt:
                out[(sec_id, eid, 2)] = cnt

    return out


def _any_forcing(evs_at, r_F, r_L, pos) -> bool:
    """复制 cost.py._any_forcing — 让本脚本独立调试。"""
    for ev in evs_at:
        if ev not in pos:
            continue
        if ev[1] == AGE:
            continue
        p = pos[ev]
        if r_F < p < r_L:
            return True
    return False


# ---------------------------------------------------------------------------
# 3. 主入口
# ---------------------------------------------------------------------------

def main():
    data = ROOT / "CONOP-run"
    truth = parse_outsect(data / "outsect.txt")
    print(f"outsect.txt 解析到 {len(truth)} 个 event level 真值")
    print(f"真值总和: {sum(truth.values())}  (期望 237)")

    sec_truth: dict[int, int] = {}
    for (sec, _, _), v in truth.items():
        sec_truth[sec] = sec_truth.get(sec, 0) + v
    print(f"per-section 真值: {dict(sorted(sec_truth.items()))}")

    # 我们的实现
    obs = parse_loadfile(data / "loadfile.dat")
    seq = solution_to_sequence(parse_solution(data / "bestsoln.dat"))
    section_obs = build_section_observations(obs)
    ctx = ConopContext.build(seq, section_obs)
    ours = our_level_breakdown(ctx)
    print(f"\n我们的 per-event 拆分: {len(ours)} 项")
    sec_ours: dict[int, int] = {}
    for (sec, _, _), v in ours.items():
        sec_ours[sec] = sec_ours.get(sec, 0) + v
    print(f"per-section 我们的: {dict(sorted(sec_ours.items()))}")
    print(f"我们的总和: {sum(ours.values())}  (vs 真值 237，差 {sum(ours.values())-237:+d})")

    # 对齐到 (sec, eid, etype)，按 |diff| 排序
    all_keys = set(truth) | set(ours)
    rows = []
    for k in all_keys:
        t = truth.get(k, 0)
        o = ours.get(k, 0)
        if t != o:
            rows.append((k, t, o, o - t))
    rows.sort(key=lambda r: -abs(r[3]))
    print(f"\n差异条目数: {len(rows)}")
    print(f"{'sec':>4s} {'eid':>4s} {'typ':>3s}  {'truth':>5s}  {'ours':>5s}  {'diff':>5s}")
    print("-" * 45)
    for (sec, eid, etype), t, o, d in rows[:30]:
        print(f"{sec:>4d} {eid:>4d} {etype:>3d}  {t:>5d}  {o:>5d}  {d:>+5d}")
    if len(rows) > 30:
        print(f"... 还有 {len(rows)-30} 条")

    # 落盘
    out_path = ROOT / "results_py" / "level_diff.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("sec_id,event_id,event_type,truth,ours,diff\n")
        for k in sorted(all_keys):
            sec, eid, etype = k
            t = truth.get(k, 0)
            o = ours.get(k, 0)
            f.write(f"{sec},{eid},{etype},{t},{o},{o-t}\n")
    print(f"\n完整 CSV → {out_path}")


if __name__ == "__main__":
    main()
