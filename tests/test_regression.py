"""回归测试：固定 cost 函数在 CONOP-run/bestsoln.dat 上的输出。

任何重构都不应改变这些数值。运行：
    uv run --with-requirements requirements.txt python -m pytest tests/test_regression.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conop_py.cost import (
    ConopContext, build_section_observations,
    ordinal_misfit, level_misfit, eventual_misfit, coexistence_violations,
)
from conop_py.io import parse_loadfile, parse_solution, solution_to_sequence


DATA_DIR = Path(__file__).resolve().parent.parent / "CONOP-run"


def _build_ctx():
    obs = parse_loadfile(DATA_DIR / "loadfile.dat")
    sol = parse_solution(DATA_DIR / "bestsoln.dat")
    seq = solution_to_sequence(sol)
    section_obs = build_section_observations(obs)
    return ConopContext.build(seq, section_obs)


def test_ordinal_matches_conop9():
    """Ordinal 必须 100% 复现 CONOP9 outmain.txt 的 367。"""
    ctx = _build_ctx()
    assert int(ordinal_misfit(ctx)) == 367


def test_level_current_implementation():
    """Level 当前实现给 340（CONOP9 真值 237，已知系统偏差 +43%）。"""
    ctx = _build_ctx()
    assert int(level_misfit(ctx)) == 340


def test_eventual_current_implementation():
    """Eventual 当前实现给 491（CONOP9 真值 353）。"""
    ctx = _build_ctx()
    assert int(eventual_misfit(ctx)) == 491


def test_coexistence_pinned():
    """共存约束在 CONOP9 bestsoln 上的违反数：固定当前值（数据本身存在冲突）。"""
    ctx = _build_ctx()
    assert coexistence_violations(ctx) == 24


def test_fast_state_initial_total_matches():
    """FastOrdinalState.total == ordinal_misfit."""
    from conop_py.cost import FastOrdinalState
    obs = parse_loadfile(DATA_DIR / "loadfile.dat")
    sol = parse_solution(DATA_DIR / "bestsoln.dat")
    seq = solution_to_sequence(sol)
    section_obs = build_section_observations(obs)
    ctx = ConopContext.build(seq, section_obs)
    state = FastOrdinalState(seq, section_obs)
    assert state.total == int(ordinal_misfit(ctx)) == 367


def test_fast_state_random_moves_consistent():
    """对随机扰动后的状态，FastOrdinalState.total 必须等于 ordinal_misfit。

    这是增量更新 + revert 逻辑的核心回归。
    """
    import random as _random
    from conop_py.cost import FastOrdinalState
    obs = parse_loadfile(DATA_DIR / "loadfile.dat")
    sol = parse_solution(DATA_DIR / "bestsoln.dat")
    seq = solution_to_sequence(sol)
    section_obs = build_section_observations(obs)

    seq = list(seq)
    state = FastOrdinalState(seq, section_obs)
    rng = _random.Random(123)
    n = len(seq)

    # Reject 全部 → 状态应保持初始 367
    for _ in range(200):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        ev = seq[i]
        seq.pop(i); seq.insert(j, ev)
        delta, undo = state.trial_move(seq, ev, j, i)
        # 还原
        seq.pop(j); seq.insert(i, ev)
        state.revert(seq, undo)
        ctx_check = ConopContext.build(seq, section_obs)
        assert state.total == int(ordinal_misfit(ctx_check)), (
            f"revert 后不一致: state.total={state.total} vs full={int(ordinal_misfit(ctx_check))}"
        )

    # Accept 全部 → 仍然一致
    for _ in range(100):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        ev = seq[i]
        seq.pop(i); seq.insert(j, ev)
        delta, undo = state.trial_move(seq, ev, j, i)
        ctx_check = ConopContext.build(seq, section_obs)
        assert state.total == int(ordinal_misfit(ctx_check)), (
            f"accept 后不一致: state.total={state.total} vs full={int(ordinal_misfit(ctx_check))}"
        )


def test_per_section_ordinal():
    """逐 section Ordinal 与 outmain.txt 一致。"""
    obs = parse_loadfile(DATA_DIR / "loadfile.dat")
    sol = parse_solution(DATA_DIR / "bestsoln.dat")
    seq = solution_to_sequence(sol)
    section_obs = build_section_observations(obs)

    ref = {1: 56, 2: 99, 3: 62, 4: 64, 5: 0, 6: 1,
           7: 23, 8: 13, 9: 0, 10: 29, 11: 12, 12: 8}
    for sec in sorted(section_obs):
        ctx_s = ConopContext.build(seq, {sec: section_obs[sec]})
        assert int(ordinal_misfit(ctx_s)) == ref[sec], (
            f"section {sec}: got {int(ordinal_misfit(ctx_s))} expected {ref[sec]}"
        )
