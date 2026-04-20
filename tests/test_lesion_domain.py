"""Smoke tests for LesionEvolutionDomain (HW4).

Skipped if the SOOP `sub-1` data isn't present locally.
"""
from __future__ import annotations

import pytest

from deepxube_hw4.soop import DEFAULT_ROOT

pytestmark = pytest.mark.skipif(
    not (DEFAULT_ROOT / "sub-1" / "dwi" / "sub-1_rec-TRACE_dwi.nii.gz").exists(),
    reason="SOOP sub-1 not downloaded; see deepxube_hw4/README.md",
)


@pytest.fixture(scope="module")
def domain():
    from deepxube_hw4.train_evolution import TrainableLesionEvo
    return TrainableLesionEvo(subject_id="sub-1", n_parcels=300)


def test_instantiation(domain):
    assert domain.K > 0
    assert len(domain.acute) > 0


def test_legal_actions_includes_stop_and_in_range(domain):
    from deepxube_hw4.evolution import STOP
    s = domain.start_state()
    acts = domain.legal_actions(s)
    assert any(a.kind == STOP for a in acts)
    for a in acts:
        if a.kind != STOP:
            assert 1 <= a.parcel <= domain.K


def test_next_state_and_solved(domain):
    from deepxube_hw4.evolution import EvolutionGoal
    s = domain.start_state()
    goal = EvolutionGoal(s.active)
    assert domain.is_solved([s], [goal]) == [True]
    acts = [a for a in domain.legal_actions(s) if a.kind != 0]
    if acts:
        s2, _ = domain.next_state([s], [acts[0]])
        assert domain.is_solved(s2, [goal]) == [False]


def test_sample_problem_instances(domain):
    starts, goals = domain.sample_problem_instances([0, 3, 6])
    assert len(starts) == len(goals) == 3
    assert domain.is_solved([starts[0]], [goals[0]]) == [True]


def test_nn_input_shape(domain):
    from deepxube.factories.nnet_input_factory import get_nnet_input_t
    from deepxube_hw4.evolution import EvolutionGoal
    nin = get_nnet_input_t(("lesion_evo", "lesion_evo_sga"))(domain=domain)
    s = domain.start_state()
    g = EvolutionGoal(s.active)
    acts = domain.legal_actions(s)
    feats = nin.to_np([s] * len(acts), [g] * len(acts), acts)
    assert feats[0].shape == (len(acts), nin.get_feat_dim())


def test_simulator_rollout(domain):
    import numpy as np
    from deepxube_hw4.simulator import simulate
    traj = simulate(domain, np.random.default_rng(0), max_steps=10)
    assert traj.length >= 0
    assert len(traj.states) == traj.length + 1


def test_action_cost_range_and_determinism(domain):
    from deepxube_hw4.evolution import (
        COST_MAX, COST_MIN, COST_STOP, EXPAND, SHRINK, STOP,
    )

    assert domain.action_cost(STOP) == COST_STOP
    seen = set()
    for kind in (EXPAND, SHRINK):
        for p in range(1, min(domain.K, 30) + 1):
            c1 = domain.action_cost(kind, p)
            c2 = domain.action_cost(kind, p)
            assert c1 == c2, "action_cost must be deterministic"
            assert COST_MIN <= c1 <= COST_MAX, f"{kind} p={p} cost={c1} out of range"
            seen.add(round(c1, 4))
    # Bio-costs should vary across parcels (not collapse to a single value).
    assert len(seen) > 1, "action_cost should differ by parcel"


def test_next_state_returns_bio_costs(domain):
    from deepxube_hw4.evolution import COST_MAX, COST_MIN
    s = domain.start_state()
    acts = [a for a in domain.legal_actions(s) if a.kind != 0][:5]
    if not acts:
        return
    _, costs = domain.next_state([s] * len(acts), acts)
    assert all(COST_MIN <= c <= COST_MAX for c in costs)
    for a, c in zip(acts, costs):
        assert c == domain.action_cost(a.kind, a.parcel)


def test_dual_head_forward(domain):
    import torch
    from deepxube.factories.nnet_input_factory import get_nnet_input_t
    from deepxube_hw4.evolution import EvolutionGoal
    from deepxube_hw4.train_evolution import LesionEvoMLP
    nin = get_nnet_input_t(("lesion_evo", "lesion_evo_sga"))(domain=domain)
    heur = LesionEvoMLP(nin, out_dim=2, q_fix=False, hidden=32, n_layers=2)
    heur.eval()
    s = domain.start_state()
    g = EvolutionGoal(s.active)
    acts = domain.legal_actions(s)[:4]
    feats = nin.to_np([s] * len(acts), [g] * len(acts), acts)
    with torch.no_grad():
        out = heur([torch.from_numpy(x) for x in feats])[0]
    assert out.shape == (len(acts), 2)
