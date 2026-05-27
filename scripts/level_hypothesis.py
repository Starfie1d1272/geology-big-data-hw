"""验证 LEVEL 的真定义：section 中横跨 [placed, observed) 或 (observed, placed] 区间的本地 horizon 数。

直接用 outsect.txt 提供的 observed / placed 算出预测的 ext_levels，
和 outsect.txt 第 7 列对比。若全部命中说明这个公式就是 CONOP9 的 LEVEL 定义。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# outsect.txt event 行：id - type  observed_m  placed_m  ext1_m  ext2_m  ext_levels  {code  TYPE  {name
EVENT_LINE = re.compile(
    r"^\s+(\d+)\s*-\s*(\d+)\s+"
    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+"
    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+"
    r"(\d+)\s+"
    r"\{"
)
SECTION_HEADER = re.compile(r"SECTION\s*-\s*(\d+)")


def parse(path: Path):
    """返回 list of dict, 每个 event 一条记录。"""
    rows = []
    cur_sec = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = SECTION_HEADER.search(line)
        if m:
            cur_sec = int(m.group(1))
            continue
        m = EVENT_LINE.match(line)
        if m and cur_sec is not None:
            rows.append(dict(
                sec=cur_sec,
                eid=int(m.group(1)),
                etype=int(m.group(2)),
                observed=float(m.group(3)),
                placed=float(m.group(4)),
                truth=int(m.group(7)),
            ))
    return rows


def main():
    rows = parse(ROOT / "CONOP-run" / "outsect.txt")
    print(f"解析 {len(rows)} 条 event 记录")

    # 每个 section 的 distinct 观测 horizon 列表
    sec_horizons: dict[int, list[float]] = {}
    for r in rows:
        sec_horizons.setdefault(r["sec"], []).append(r["observed"])
    for sec in sec_horizons:
        sec_horizons[sec] = sorted(set(sec_horizons[sec]))

    # 用公式预测：FAD 端 ext = #{h: placed ≤ h < observed}; LAD 端 ext = #{h: observed < h ≤ placed}
    mismatches = 0
    total_pred = 0
    for r in rows:
        horizons = sec_horizons[r["sec"]]
        obs, plc = r["observed"], r["placed"]
        if r["etype"] == 1:  # FAD：placed ≤ obs, 区间 [placed, obs)
            pred = sum(1 for h in horizons if plc <= h < obs)
        elif r["etype"] == 2:  # LAD：placed ≥ obs, 区间 (obs, placed]
            pred = sum(1 for h in horizons if obs < h <= plc)
        else:  # AGE / ASH marker：obs 应等于 placed，ext = 0
            pred = 0
            if obs != plc:
                # marker 也可能被移动，按 FAD/LAD 通用规则
                if plc < obs:
                    pred = sum(1 for h in horizons if plc <= h < obs)
                else:
                    pred = sum(1 for h in horizons if obs < h <= plc)
        total_pred += pred
        if pred != r["truth"]:
            mismatches += 1
            if mismatches <= 15:
                print(f"  不一致 sec={r['sec']} eid={r['eid']} type={r['etype']}  "
                      f"obs={obs:.1f} plc={plc:.1f}  truth={r['truth']} pred={pred}")

    print(f"\n总预测 LEVEL: {total_pred}  (真值 237，差 {total_pred-237:+d})")
    print(f"不一致: {mismatches}/{len(rows)} 条")
    if mismatches == 0:
        print("\n✓ 假设成立：LEVEL = section 横跨区间的 horizon 数")


if __name__ == "__main__":
    main()
