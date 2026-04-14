"""Smoke tests for LesionPathDomain (HW4).

The domain is not registered with `domain_factory`, so these tests exercise it
directly. They are skipped if the MNI152 T1 file isn't available locally.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from deepxube_hw4.domain import (
    DEFAULT_T1,
    LesionAction,
    LesionGoal,
    LesionPathDomain,
    LesionState,
)

pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_T1).exists(),
    reason=f"MNI152 T1 not available at {DEFAULT_T1}",
)


@pytest.fixture(scope="module")
def domain() -> LesionPathDomain:
    return LesionPathDomain(downsample=4)


def test_instantiation(domain: LesionPathDomain) -> None:
    assert domain.t1.ndim == 3
    assert domain.shape == domain.t1.shape
    assert domain.lesion.sum() > 0
    assert len(domain.actions_fixed) == 6


def test_sample_problem_instances(domain: LesionPathDomain) -> None:
    starts, goals = domain.sample_problem_instances([0, 0, 0])
    assert len(starts) == 3 and len(goals) == 3
    assert all(isinstance(s, LesionState) for s in starts)
    assert all(isinstance(g, LesionGoal) for g in goals)


def test_start_not_solved(domain: LesionPathDomain) -> None:
    starts, goals = domain.sample_problem_instances([0])
    # Start is chosen opposite the lesion centroid — must be outside the mask.
    assert not any(domain.is_solved(starts, goals))


def test_next_state_bounds_and_cost(domain: LesionPathDomain) -> None:
    s = LesionState(0, 0, 0)
    # All six moves from the origin corner: -i/-j/-k clamp in place.
    actions = [LesionAction(i) for i in range(6)]
    nexts, costs = domain.next_state([s] * 6, actions)
    assert len(nexts) == 6 and costs == [1.0] * 6
    for ns in nexts:
        assert 0 <= ns.i < domain.shape[0]
        assert 0 <= ns.j < domain.shape[1]
        assert 0 <= ns.k < domain.shape[2]


def test_straight_line_path_reaches_lesion(domain: LesionPathDomain) -> None:
    start = domain.sample_start_state()
    states, actions = domain.straight_line_path(start, domain.lesion_centroid)
    assert len(states) == len(actions) + 1
    assert domain.is_solved([states[-1]], [LesionGoal()])[0]


def test_string_to_action(domain: LesionPathDomain) -> None:
    for i in range(6):
        a = domain.string_to_action(str(i))
        assert a is not None and a.idx == i
    assert domain.string_to_action("bogus") is None
