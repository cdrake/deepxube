"""Biologically-motivated forward evolution simulator for acute -> chronic.

Models three stroke-evolution dynamics at the parcel level:
- edema resolution: marginal parcels with partial lesion coverage may shrink
- penumbral completion: frontier parcels with bright DWI may get recruited
- core persistence: parcels with high coverage + bright DWI rarely resolve

No vascular atlas yet — DWI signal intensity and spatial contiguity stand in
for territory/severity priors. Swap these heuristics for atlas-derived scores
once MNI registration lands.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from deepxube_hw4.evolution import (
    EXPAND, SHRINK, STOP,
    EvolutionAction, EvolutionGoal, EvolutionState, LesionEvolutionDomain,
)


@dataclass
class Trajectory:
    states: List[EvolutionState]
    actions: List[EvolutionAction]

    @property
    def length(self) -> int:
        return len(self.actions)


def parcel_dwi_mean(labels: NDArray, dwi: NDArray) -> NDArray:
    """(K+1,) mean DWI signal per parcel (index 0 unused)."""
    K = int(labels.max())
    flat_lab = labels.ravel()
    flat_dwi = dwi.ravel().astype(np.float32)
    sizes = np.bincount(flat_lab, minlength=K + 1)
    sums = np.bincount(flat_lab, weights=flat_dwi, minlength=K + 1)
    return np.where(sizes > 0, sums / np.maximum(sizes, 1), 0.0)


def simulate(
    domain: LesionEvolutionDomain,
    rng: np.random.Generator,
    max_steps: int = 40,
    shrink_bias: float = 3.0,
    expand_bias: float = 0.3,
    p_stop_step: float = 0.05,
) -> Trajectory:
    """One stochastic acute -> chronic trajectory.

    Each step: Bernoulli(p_stop_step) terminates; otherwise sample a shrink or
    expand action weighted by biology-motivated scores. `shrink_bias` /
    `expand_bias` shift the grow-vs-shrink balance during calibration.
    """
    parc = domain.parc
    brightness = parcel_dwi_mean(parc.labels, domain.subject.dwi)
    b_norm = brightness / max(np.percentile(brightness[1:], 99), 1e-6)
    b_norm = np.clip(b_norm, 0.0, 1.0)
    coverage_full = parc.parcel_coverage(domain.subject.mask)  # (K,)

    state = domain.start_state()
    states = [state]
    actions: List[EvolutionAction] = []

    for _ in range(max_steps):
        if rng.random() < p_stop_step:
            break
        acts = [a for a in domain.legal_actions(state) if a.kind != STOP]
        if not acts:
            break
        weights = np.zeros(len(acts), dtype=np.float64)
        for i, a in enumerate(acts):
            if a.kind == SHRINK:
                cov = coverage_full[a.parcel - 1]
                b = b_norm[a.parcel]
                weights[i] = shrink_bias * (1.0 - cov) * (1.0 - 0.5 * b)
            elif a.kind == EXPAND:
                b = b_norm[a.parcel]
                weights[i] = expand_bias * b
        total = weights.sum()
        if total <= 0:
            break
        probs = weights / total
        chosen = acts[int(rng.choice(len(acts), p=probs))]
        states_next, _ = domain.next_state([state], [chosen])
        state = states_next[0]
        states.append(state)
        actions.append(chosen)

    return Trajectory(states, actions)


@dataclass
class CalibrationResult:
    shrink_bias: float
    expand_bias: float
    p_stop_step: float
    mean_ratio: float
    mean_length: float
    score: float


def calibrate(
    domains: List[LesionEvolutionDomain],
    target_ratio: float = 0.85,
    target_length: float = 15.0,
    n_rollouts_per_subject: int = 3,
    seed: int = 0,
) -> CalibrationResult:
    """Grid-search over (shrink_bias, expand_bias, p_stop_step) to hit targets.

    Score = |ratio - target_ratio| + 0.02 * |length - target_length|.
    """
    rng = np.random.default_rng(seed)
    grid = [
        (sb, eb, ps)
        for sb in (1.0, 2.0, 3.0, 5.0)
        for eb in (0.3, 0.6, 1.0)
        for ps in (0.05, 0.08, 0.12, 0.18)
    ]
    best: Optional[CalibrationResult] = None
    for sb, eb, ps in grid:
        ratios: List[float] = []
        lengths: List[float] = []
        for dom in domains:
            start_sz = max(len(dom.acute), 1)
            for _ in range(n_rollouts_per_subject):
                t = simulate(dom, rng, shrink_bias=sb, expand_bias=eb, p_stop_step=ps)
                ratios.append(len(t.states[-1].active) / start_sz)
                lengths.append(float(t.length))
        mr = float(np.mean(ratios))
        ml = float(np.mean(lengths))
        score = abs(mr - target_ratio) + 0.02 * abs(ml - target_length)
        if best is None or score < best.score:
            best = CalibrationResult(sb, eb, ps, mr, ml, score)
    assert best is not None
    return best


def generate_training_pairs(
    domain: LesionEvolutionDomain,
    n_trajectories: int,
    max_steps: int = 40,
    seed: Optional[int] = None,
) -> List[Tuple[EvolutionState, EvolutionGoal, int]]:
    """Roll out `n_trajectories` simulations; return (start, goal, length) tuples."""
    rng = np.random.default_rng(seed)
    out: List[Tuple[EvolutionState, EvolutionGoal, int]] = []
    start = domain.start_state()
    for _ in range(n_trajectories):
        traj = simulate(domain, rng, max_steps=max_steps)
        out.append((start, EvolutionGoal(traj.states[-1].active), traj.length))
    return out
