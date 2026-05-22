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
    EventKey, Sequence, build_section_observations, ordinal_misfit,
)
from conop_py.io import Entity, Observation


@dataclass
class AnnealConfig:
    startemp: float = 250.0
    ratio: float = 0.980
    steps: int = 300          # 降温次数
    trials: int = 300         # 每温度阶段扰动次数
    seed: int | None = None


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


def build_marker_keys(observations: list[Observation], taxa_ids: set[int]) -> list[EventKey]:
    """从观测中提取 marker 的 (entity_id, event_type) 唯一组合。"""
    seen = set()
    for o in observations:
        if o.entity_id not in taxa_ids:
            seen.add((o.entity_id, o.event_type))
    return list(seen)


def build_initial(
    entities: list[Entity],
    observations: list[Observation],
    rng: random.Random,
) -> Sequence:
    """构建包含所有 120 个事件的初始序列。"""
    taxa = [e for e in entities if e.is_taxon]
    taxa_ids = {e.id for e in taxa}

    fads = [(e.id, 1) for e in taxa]
    lads = [(e.id, 2) for e in taxa]
    markers = build_marker_keys(observations, taxa_ids)

    rng.shuffle(fads)
    rng.shuffle(lads)
    rng.shuffle(markers)

    # FAD + 一半 markers + LAD + 另一半 markers
    half = len(markers) // 2
    seq: Sequence = fads + markers[:half] + lads + markers[half:]
    return seq


def anneal(
    entities: list[Entity],
    observations: list[Observation],
    cfg: AnnealConfig,
    misfit_fn=ordinal_misfit,
    verbose: bool = True,
) -> AnnealResult:
    """模拟退火主循环。"""
    rng = random.Random(cfg.seed)
    section_obs = build_section_observations(observations)

    seq = build_initial(entities, observations, rng)
    n = len(seq)
    current_fit = misfit_fn(seq, section_obs)
    best_seq = seq[:]
    best_fit = current_fit

    T = cfg.startemp
    trajectory: list[TrajectoryPoint] = []

    if verbose:
        print(f"初始 misfit = {current_fit:.2f}  (n={n} events)")

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

            new_fit = misfit_fn(seq, section_obs)
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
    args = ap.parse_args()

    data = Path(args.data_dir)
    raw_cfg = parse_cfg(data / "conop9.cfg")

    cfg = AnnealConfig(
        startemp=args.startemp if args.startemp is not None else float(raw_cfg.get("STARTEMP", 250)),
        ratio=args.ratio if args.ratio is not None else float(raw_cfg.get("RATIO", 0.98)),
        steps=args.steps if args.steps is not None else int(raw_cfg.get("STEPS", 300)),
        trials=args.trials if args.trials is not None else int(raw_cfg.get("TRIALS", 300)),
        seed=args.seed,
    )
    obs = parse_loadfile(data / "loadfile.dat")
    from conop_py.io import infer_taxa_from_observations
    entities = parse_events(data / "events.txt", taxon_ids=infer_taxa_from_observations(obs))

    print(f"配置：STARTEMP={cfg.startemp} RATIO={cfg.ratio} STEPS={cfg.steps} TRIALS={cfg.trials} seed={cfg.seed}")
    res = anneal(entities, obs, cfg)

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
