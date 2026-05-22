"""CONOP 输入文件解析。

文件格式（基于 CONOP-run/ 实测）：
- sections.txt:  id 'short' id 'name' flag
- events.txt:    id 'code' 'name'   前 N_TAXA 行是 taxon，后续是 marker
- loadfile.dat:  entity_id event_type section_id level horizon_idx section_id_dup w1 w2
- conop9.cfg:    Fortran namelist (&getinn &getans &getrun &getout)
- bestsoln.dat:  entity_id event_type position   (120 行)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# event_type 编码（基于 outmain.txt BEST PERMUTATION 字段确认）
FAD = 1   # First Appearance Datum (taxon)
LAD = 2   # Last Appearance Datum (taxon)
ASH = 4   # Ash bed / 火山灰层 marker
AGE = 5   # Age constraint marker (同位素年龄)
MARKER_TYPES = {ASH, AGE}


@dataclass(frozen=True)
class Section:
    id: int          # 1-based
    short: str       # e.g. 'SyA'
    name: str        # 完整名称
    is_reference: bool  # cfg 中 flag=0 (如 O97 时间参考)


@dataclass(frozen=True)
class Entity:
    """events.txt 中的一行，可以是 taxon 也可以是 marker。"""
    id: int       # 1-based
    code: str     # 数据库代号，如 '1051'
    name: str     # 完整名称
    is_taxon: bool  # True=taxon (有 FAD+LAD), False=marker (单点事件)


@dataclass(frozen=True)
class Observation:
    """loadfile.dat 中的一行观测。"""
    section_id: int
    entity_id: int
    event_type: int  # 1=FAD, 2=LAD, 其他=marker 子类型
    level: float     # 剖面内层位（米/英尺）
    weight: float = 1.0   # w1：普通观测=1.0；AGE/ASH marker=同位素年龄下界（Ma）
    weight2: float = 1.0  # w2：普通观测=1.0；AGE/ASH marker=同位素年龄上界（Ma）


@dataclass(frozen=True)
class SolutionRecord:
    """bestsoln.dat / soln.dat 中的一行。"""
    entity_id: int
    event_type: int
    position: int  # 1-based 在模型序列中的位置


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_QUOTED = re.compile(r"'([^']*)'")


def parse_sections(path: str | Path) -> list[Section]:
    sections = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        quoted = _QUOTED.findall(line)
        # 去掉引号后剩下的数字串
        nums = re.findall(r"-?\d+", _QUOTED.sub(" ", line))
        # 格式: id 'short' id 'name' flag
        sec_id = int(nums[0])
        short = quoted[0] if quoted else ""
        name = quoted[1] if len(quoted) > 1 else ""
        flag = int(nums[-1])
        sections.append(Section(id=sec_id, short=short, name=name, is_reference=(flag == 0)))
    return sections


def parse_events(
    path: str | Path,
    n_taxa: int | None = None,
    taxon_ids: set[int] | None = None,
) -> list[Entity]:
    """events.txt 总共 N 行 = TAXA + EVENTS（cfg 中给定）。

    taxa（有 FAD+LAD）和 markers（单点事件）的 entity_id 在 events.txt
    中是稀疏分布的，不能简单按 id ≤ n_taxa 判断。

    Args:
        path: events.txt 路径
        n_taxa: cfg 中的 TAXA 字段（用于校验数量）
        taxon_ids: 可选，从 loadfile.dat 推断出的 taxa entity_id 集合。
                   未提供时退化为按 id 顺序假设（不可靠）。
    """
    entities = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        quoted = _QUOTED.findall(line)
        nums = re.findall(r"-?\d+", _QUOTED.sub(" ", line))
        ent_id = int(nums[0])
        code = quoted[0] if quoted else ""
        name = quoted[1] if len(quoted) > 1 else ""
        if taxon_ids is not None:
            is_taxon = ent_id in taxon_ids
        else:
            is_taxon = (n_taxa is not None and ent_id <= n_taxa)
        entities.append(Entity(id=ent_id, code=code, name=name, is_taxon=is_taxon))
    return entities


def infer_taxa_from_observations(observations: list[Observation]) -> set[int]:
    """从 loadfile.dat 观测中识别 taxa：有 FAD(1) 或 LAD(2) 观测的 entity。"""
    return {o.entity_id for o in observations if o.event_type in (1, 2)}


def parse_loadfile(path: str | Path) -> list[Observation]:
    """CONOP9 loadfile.dat: entity_id event_type section_id level ..."""
    obs = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        entity_id = int(parts[0])
        event_type = int(parts[1])
        section_id = int(parts[2])
        level = float(parts[3])
        weight = float(parts[6]) if len(parts) >= 7 else 1.0
        weight2 = float(parts[7]) if len(parts) >= 8 else weight
        obs.append(Observation(
            section_id=section_id,
            entity_id=entity_id,
            event_type=event_type,
            level=level,
            weight=weight,
            weight2=weight2,
        ))
    return obs


def parse_cfg(path: str | Path) -> dict[str, str]:
    """解析 Fortran namelist 风格的 conop9.cfg。返回扁平 dict (key -> str)。

    只关心算法相关字段：PENALTY/STEPS/STARTEMP/RATIO/TRIALS/SECTIONS/TAXA/EVENTS/TEASER/HOODSIZE
    """
    cfg = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(("&", "/", "!")):
            continue
        # 形如 KEY=VALUE 或 KEY='VALUE'
        m = re.match(r"([A-Z_][A-Z0-9_]*)\s*=\s*'?([^'\s]+)'?", line, re.IGNORECASE)
        if m:
            cfg[m.group(1).upper()] = m.group(2)
    return cfg


def parse_solution(path: str | Path) -> list[SolutionRecord]:
    """读取 bestsoln.dat / soln.dat。每行 3 列: entity_id event_type position。"""
    records = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        records.append(SolutionRecord(
            entity_id=int(parts[0]),
            event_type=int(parts[1]),
            position=int(parts[2]),
        ))
    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def event_key(entity_id: int, event_type: int) -> tuple[int, int]:
    """统一的 event 标识键。模型序列中每个 event 由 (entity_id, event_type) 唯一确定。"""
    return (entity_id, event_type)


def solution_to_sequence(records: list[SolutionRecord]) -> list[tuple[int, int]]:
    """把 bestsoln.dat 解析结果转成 [event_key] 列表，按 position 升序。"""
    ordered = sorted(records, key=lambda r: r.position)
    return [event_key(r.entity_id, r.event_type) for r in ordered]


def summarize(data_dir: str | Path, cfg: dict[str, str]) -> dict:
    """打印 / 返回解析后的数据集统计。"""
    data_dir = Path(data_dir)
    sections = parse_sections(data_dir / "sections.txt")
    obs = parse_loadfile(data_dir / "loadfile.dat")
    taxon_ids = infer_taxa_from_observations(obs)
    entities = parse_events(data_dir / "events.txt", taxon_ids=taxon_ids)
    n_taxa = sum(1 for e in entities if e.is_taxon)
    n_markers = sum(1 for e in entities if not e.is_taxon)
    return {
        "sections": len(sections),
        "entities": len(entities),
        "taxa": n_taxa,
        "markers": n_markers,
        "total_events": n_taxa * 2 + n_markers,
        "observations": len(obs),
    }


if __name__ == "__main__":
    import sys

    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "CONOP-run")
    cfg = parse_cfg(data_dir / "conop9.cfg")
    stats = summarize(data_dir, cfg)
    print(f"配置：{cfg}")
    print(f"统计：{stats}")
