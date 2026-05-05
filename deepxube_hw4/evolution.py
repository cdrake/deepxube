"""Forward lesion-evolution domain.

State: set of active (infarcted) parcel IDs.
Start: parcels covered by the acute mask.
Goal: a target parcel set (during training — e.g. a simulated or paired chronic mask).
Actions: expand into an adjacent inactive parcel, shrink an active parcel, or stop.
Step cost: biologically-weighted — cheap for plausible evolutions (edema
resolution on dim, low-coverage parcels; penumbra completion on bright
parcels) and expensive for implausible ones (un-infarcting the core, seeding
new infarct in dim cortex). STOP keeps unit cost. See `action_cost`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from deepxube.base.domain import (
    Action, Goal, State, StateGoalVizable, StringToAct,
)
from deepxube.utils.timing_utils import Times
from deepxube_hw4.parcels import Parcellation, brain_mask, parcel_adjacency, parcellate_dwi
from deepxube_hw4.soop import SOOPSubject, load_subject

STOP = 0
EXPAND = 1
SHRINK = 2

# Bio-cost range. c_min is the admissible lower bound for any toggling action;
# c_max is the cost of the most biologically implausible toggle.
COST_MIN = 0.5
COST_MAX = 2.0
COST_STOP = 1.0


class EvolutionState(State):
    __slots__ = ("active",)

    def __init__(self, active: FrozenSet[int]):
        self.active = active

    def __hash__(self) -> int:
        return hash(self.active)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EvolutionState):
            return self.active == other.active
        return NotImplemented

    def __repr__(self) -> str:
        return f"Evo(|active|={len(self.active)})"


class EvolutionAction(Action):
    __slots__ = ("kind", "parcel")

    def __init__(self, kind: int, parcel: int = 0):
        self.kind = int(kind)
        self.parcel = int(parcel)

    def __hash__(self) -> int:
        return hash((self.kind, self.parcel))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EvolutionAction):
            return self.kind == other.kind and self.parcel == other.parcel
        return NotImplemented

    def __repr__(self) -> str:
        name = {STOP: "stop", EXPAND: "expand", SHRINK: "shrink"}[self.kind]
        return f"{name}({self.parcel})" if self.kind != STOP else "stop"


@dataclass
class EvolutionGoal(Goal):
    target: FrozenSet[int]


class LesionEvolutionDomain(
    StateGoalVizable[EvolutionState, EvolutionAction, EvolutionGoal],
    StringToAct[EvolutionState, EvolutionAction, EvolutionGoal],
):
    def __init__(
        self,
        subject: SOOPSubject,
        parc: Parcellation,
        coverage: float = 0.3,
        target: Optional[FrozenSet[int]] = None,
    ):
        super().__init__()
        self.subject = subject
        self.parc = parc
        self.adj = parcel_adjacency(parc.labels)
        self.acute: FrozenSet[int] = frozenset(
            int(p) for p in parc.lesion_parcel_set(subject.mask, coverage)
        )
        self.target: Optional[FrozenSet[int]] = target
        self._b_norm, self._coverage = self._precompute_bio_priors()

    # --- Biological priors (per-parcel) used by action_cost ---
    def _precompute_bio_priors(self) -> Tuple[NDArray, NDArray]:
        """Return (brightness[K+1], coverage[K]) arrays used for bio-cost.

        `brightness[p]` is the mean DWI signal of parcel p, normalized to
        [0, 1] by the 99th percentile over parcels (index 0 unused, matches
        the 1..K parcel indexing used elsewhere).

        `coverage[p-1]` is the fraction of parcel p's voxels that fall
        inside the acute lesion mask. 0-indexed to match
        `Parcellation.parcel_coverage`.
        """
        labels = self.parc.labels
        dwi = self.subject.dwi.astype(np.float32)
        flat_lab = labels.ravel()
        flat_dwi = dwi.ravel()
        K = self.parc.n_parcels
        sizes = np.bincount(flat_lab, minlength=K + 1)
        sums = np.bincount(flat_lab, weights=flat_dwi, minlength=K + 1)
        brightness = np.where(sizes > 0, sums / np.maximum(sizes, 1), 0.0)
        norm = float(max(np.percentile(brightness[1:], 99), 1e-6))
        b_norm = np.clip(brightness / norm, 0.0, 1.0).astype(np.float32)
        coverage = self.parc.parcel_coverage(self.subject.mask).astype(np.float32)
        return b_norm, coverage

    def action_cost(self, kind: int, parcel: int = 0) -> float:
        """Biologically-weighted step cost in [COST_MIN, COST_MAX].

        - SHRINK: cheap on dim, low-coverage parcels (edema resolution);
          expensive on bright, high-coverage parcels (would un-infarct core).
        - EXPAND: cheap on bright parcels (penumbral completion); expensive
          on dim parcels (would seed new infarct in uninjured tissue).
        - STOP: COST_STOP (unit).
        """
        if kind == STOP:
            return COST_STOP
        b = float(self._b_norm[parcel])
        if kind == SHRINK:
            cov = float(self._coverage[parcel - 1])
            plaus = (1.0 - cov) * (1.0 - 0.5 * b)
        else:  # EXPAND
            plaus = b
        plaus = max(0.0, min(1.0, plaus))
        return COST_MIN + (COST_MAX - COST_MIN) * (1.0 - plaus)

    @classmethod
    def from_subject_id(
        cls, sid: str, n_parcels: int = 300, coverage: float = 0.3,
    ) -> "LesionEvolutionDomain":
        subj = load_subject(sid)
        brain = brain_mask(subj.dwi)
        parc = parcellate_dwi(subj.dwi, brain, n_parcels=n_parcels)
        return cls(subj, parc, coverage=coverage)

    # --- Action generation ---
    def legal_actions(self, state: EvolutionState) -> List[EvolutionAction]:
        acts: List[EvolutionAction] = [EvolutionAction(STOP)]
        active = state.active
        K = self.parc.n_parcels
        if not active:
            return acts
        frontier: set[int] = set()
        boundary: set[int] = set()  # active parcels with at least one inactive neighbor
        for p in active:
            neigh = [int(q) for q in np.where(self.adj[p])[0] if 1 <= int(q) <= K]
            has_inactive = False
            for q in neigh:
                if q in active:
                    continue
                frontier.add(q)
                has_inactive = True
            if has_inactive:
                boundary.add(int(p))
        acts.extend(EvolutionAction(EXPAND, p) for p in sorted(frontier))
        shrink_set = boundary if boundary else active
        acts.extend(EvolutionAction(SHRINK, p) for p in sorted(shrink_set))
        return acts

    # --- Core Domain API ---
    def start_state(self) -> EvolutionState:
        return EvolutionState(self.acute)

    def is_solved(
        self, states: List[EvolutionState], goals: List[EvolutionGoal],
    ) -> List[bool]:
        return [s.active == g.target for s, g in zip(states, goals)]

    def next_state(
        self, states: List[EvolutionState], actions: List[EvolutionAction],
    ) -> Tuple[List[EvolutionState], List[float]]:
        out_s: List[EvolutionState] = []
        out_c: List[float] = []
        for s, a in zip(states, actions):
            if a.kind == STOP:
                out_s.append(s)
            elif a.kind == EXPAND:
                out_s.append(EvolutionState(s.active | {a.parcel}))
            elif a.kind == SHRINK:
                out_s.append(EvolutionState(s.active - {a.parcel}))
            else:
                raise ValueError(f"bad action kind {a.kind}")
            out_c.append(self.action_cost(a.kind, a.parcel))
        return out_s, out_c

    def sample_state_action(
        self, states: List[EvolutionState],
    ) -> List[EvolutionAction]:
        rng = np.random.default_rng()
        out: List[EvolutionAction] = []
        for s in states:
            acts = self.legal_actions(s)
            out.append(acts[int(rng.integers(len(acts)))])
        return out

    def sample_problem_instances(
        self, num_steps_l: List[int], times: Optional[Times] = None,
    ) -> Tuple[List[EvolutionState], List[EvolutionGoal]]:
        """Generate (start, goal) pairs by random-walking from the acute state.

        The goal is the final state after a random walk of `num_steps` legal actions;
        this is a placeholder supervision source until the biological simulator lands.
        """
        rng = np.random.default_rng()
        starts: List[EvolutionState] = []
        goals: List[EvolutionGoal] = []
        for n in num_steps_l:
            s = self.start_state()
            for _ in range(max(0, int(n))):
                acts = [a for a in self.legal_actions(s) if a.kind != STOP]
                if not acts:
                    break
                a = acts[int(rng.integers(len(acts)))]
                s, _ = self.next_state([s], [a])
                s = s[0]
            starts.append(self.start_state())
            goals.append(EvolutionGoal(s.active))
        return starts, goals

    # --- REPL / viz ---
    def string_to_action(self, act_str: str) -> Optional[EvolutionAction]:
        t = act_str.strip().lower().split()
        if t == ["stop"]:
            return EvolutionAction(STOP)
        if len(t) == 2 and t[0] in {"expand", "shrink"}:
            kind = EXPAND if t[0] == "expand" else SHRINK
            try:
                return EvolutionAction(kind, int(t[1]))
            except ValueError:
                return None
        return None

    def string_to_action_help(self) -> str:
        return "expand <parcel> | shrink <parcel> | stop"

    def state_mask(self, state: EvolutionState) -> NDArray:
        """Reconstruct a binary voxel mask from the active parcel set."""
        m = np.zeros_like(self.parc.labels, dtype=np.uint8)
        if state.active:
            active_arr = np.fromiter(state.active, dtype=np.int32)
            m = np.isin(self.parc.labels, active_arr).astype(np.uint8)
        return m

    def visualize_state_goal(
        self,
        state: EvolutionState,
        goal: Optional[EvolutionGoal],
        fig: Figure,
        path: Optional[List[EvolutionState]] = None,
    ) -> None:
        mask = self.state_mask(state)
        target_mask = (
            np.isin(self.parc.labels, np.fromiter(goal.target, dtype=np.int32))
            if (goal is not None and goal.target)
            else np.zeros_like(self.parc.labels)
        )
        acute_mask = np.isin(self.parc.labels, np.fromiter(self.acute, dtype=np.int32))
        zs = np.where((mask | target_mask | acute_mask).any(axis=(0, 1)))[0]
        if len(zs) == 0:
            zs = [self.parc.labels.shape[2] // 2]
        slices = np.linspace(zs.min(), zs.max(), min(4, len(zs))).astype(int)
        axes = fig.subplots(1, len(slices))
        if len(slices) == 1:
            axes = [axes]
        dwi = self.subject.dwi
        vmax = np.percentile(dwi, 99)
        for ax, z in zip(axes, slices):
            ax.imshow(np.clip(dwi[:, :, z].T / max(vmax, 1e-6), 0, 1),
                      cmap="gray", origin="lower")
            if acute_mask[:, :, z].any():
                ax.contour(acute_mask[:, :, z].T, levels=[0.5], colors="cyan", linewidths=0.8)
            if target_mask[:, :, z].any():
                ax.contour(target_mask[:, :, z].T, levels=[0.5], colors="lime", linewidths=0.8)
            if mask[:, :, z].any():
                overlay = np.zeros(mask[:, :, z].T.shape + (4,), dtype=np.float32)
                overlay[..., 0] = 1.0
                overlay[..., 3] = (mask[:, :, z].T > 0) * 0.5
                ax.imshow(overlay, origin="lower")
            ax.set_title(f"z={z}")
            ax.axis("off")
        fig.suptitle(
            f"state |active|={len(state.active)}  acute=cyan  target=lime  current=red"
        )
        fig.tight_layout()
