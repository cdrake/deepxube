"""Lesion-as-goal voxel pathfinding domain.

State: (i, j, k) voxel in a downsampled MNI152 grid.
Action: one of 6 face-neighbor steps.
Goal: any voxel inside the lesion mask.

Only implements what's needed for viz + future solver wiring. Deliberately
skips nnet input / pathfinding mixins until the training slice lands.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from deepxube.base.domain import (
    Action, Domain, Goal, State, StateGoalVizable, StringToAct,
)
from deepxube.utils.timing_utils import Times

DEFAULT_T1 = Path(
    "/Users/chrisdrake/Dev/niivue/niivue/packages/niivue/demos/images/mni152.nii.gz"
)
LESION_CENTER_FRAC = (0.65, 0.55, 0.55)
LESION_RADIUS_MM = 18.0
BRAIN_PERCENTILE = 40.0

# 6-connected moves: +i, -i, +j, -j, +k, -k
_DELTAS: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
)


class LesionState(State):
    __slots__ = ("i", "j", "k")

    def __init__(self, i: int, j: int, k: int):
        self.i, self.j, self.k = int(i), int(j), int(k)

    def __hash__(self) -> int:
        return hash((self.i, self.j, self.k))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LesionState):
            return (self.i, self.j, self.k) == (other.i, other.j, other.k)
        return NotImplemented

    def __repr__(self) -> str:
        return f"({self.i},{self.j},{self.k})"


class LesionAction(Action):
    __slots__ = ("idx",)

    def __init__(self, idx: int):
        self.idx = int(idx)

    def __hash__(self) -> int:
        return self.idx

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LesionAction):
            return self.idx == other.idx
        return NotImplemented

    def __repr__(self) -> str:
        return f"{'+i -i +j -j +k -k'.split()[self.idx]}"


@dataclass
class LesionGoal(Goal):
    """Trivial marker — goal membership is decided by the domain's lesion mask."""
    label: str = "lesion"


class LesionPathDomain(
    StateGoalVizable[LesionState, LesionAction, LesionGoal],
    StringToAct[LesionState, LesionAction, LesionGoal],
):
    def __init__(
        self,
        t1_path: Path = DEFAULT_T1,
        downsample: int = 4,
        lesion_center_frac: Tuple[float, float, float] = LESION_CENTER_FRAC,
        lesion_radius_mm: float = LESION_RADIUS_MM,
    ):
        super().__init__()
        img = nib.load(str(t1_path))
        t1 = np.asarray(img.get_fdata(), dtype=np.float32)
        voxel_mm = np.abs(np.diag(img.affine)[:3]).astype(np.float32)

        f = max(1, int(downsample))
        self.t1: NDArray = t1[::f, ::f, ::f]
        self.voxel_mm: NDArray = voxel_mm * f
        self.shape: Tuple[int, int, int] = self.t1.shape  # type: ignore[assignment]
        self.downsample: int = f

        self.brain: NDArray = self.t1 > np.percentile(self.t1, BRAIN_PERCENTILE)
        self.lesion: NDArray = self._make_lesion(lesion_center_frac, lesion_radius_mm)
        if self.lesion.sum() == 0:
            raise RuntimeError("Lesion mask is empty after brain masking.")
        self.lesion_centroid: NDArray = np.argwhere(self.lesion).mean(axis=0).round().astype(int)

        self.actions_fixed: List[LesionAction] = [LesionAction(i) for i in range(6)]

    def _make_lesion(
        self, frac: Tuple[float, float, float], radius_mm: float,
    ) -> NDArray:
        center = np.array([int(f * s) for f, s in zip(frac, self.shape)])
        radii = radius_mm / self.voxel_mm
        ii, jj, kk = np.indices(self.shape)
        d2 = (
            ((ii - center[0]) / radii[0]) ** 2
            + ((jj - center[1]) / radii[1]) ** 2
            + ((kk - center[2]) / radii[2]) ** 2
        )
        return ((d2 <= 1.0) & self.brain).astype(np.uint8)

    def sample_start_state(self) -> LesionState:
        """Pick a brain-surface voxel roughly opposite the lesion centroid."""
        idx = np.argwhere(self.brain)
        direction = idx.mean(axis=0) - self.lesion_centroid
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        scores = (idx - self.lesion_centroid) @ direction
        p = idx[int(np.argmax(scores))]
        return LesionState(*p)

    def straight_line_path(
        self, start: LesionState, goal_vox: NDArray,
    ) -> Tuple[List[LesionState], List[LesionAction]]:
        """Placeholder solver: greedily step toward lesion centroid on the 6-grid."""
        states: List[LesionState] = [start]
        actions: List[LesionAction] = []
        cur = np.array([start.i, start.j, start.k])
        while not self.lesion[cur[0], cur[1], cur[2]]:
            diff = goal_vox - cur
            axis = int(np.argmax(np.abs(diff)))
            step = int(np.sign(diff[axis]))
            if step == 0:
                break
            delta_idx = {(1, 0, 0): 0, (-1, 0, 0): 1, (0, 1, 0): 2,
                         (0, -1, 0): 3, (0, 0, 1): 4, (0, 0, -1): 5}
            delta = [0, 0, 0]
            delta[axis] = step
            aidx = delta_idx[tuple(delta)]  # type: ignore[arg-type]
            actions.append(LesionAction(aidx))
            cur = cur + np.array(delta)
            states.append(LesionState(*cur))
        return states, actions

    # --- Domain API ---
    def is_solved(
        self, states: List[LesionState], goals: List[LesionGoal],
    ) -> List[bool]:
        return [bool(self.lesion[s.i, s.j, s.k]) for s in states]

    def next_state(
        self, states: List[LesionState], actions: List[LesionAction],
    ) -> Tuple[List[LesionState], List[float]]:
        out: List[LesionState] = []
        for s, a in zip(states, actions):
            di, dj, dk = _DELTAS[a.idx]
            ni = int(np.clip(s.i + di, 0, self.shape[0] - 1))
            nj = int(np.clip(s.j + dj, 0, self.shape[1] - 1))
            nk = int(np.clip(s.k + dk, 0, self.shape[2] - 1))
            out.append(LesionState(ni, nj, nk))
        return out, [1.0] * len(out)

    def sample_state_action(self, states: List[LesionState]) -> List[LesionAction]:
        rng = np.random.default_rng()
        return [LesionAction(int(rng.integers(6))) for _ in states]

    def sample_problem_instances(
        self, num_steps_l: List[int], times: Optional[Times] = None,
    ) -> Tuple[List[LesionState], List[LesionGoal]]:
        starts = [self.sample_start_state() for _ in num_steps_l]
        return starts, [LesionGoal() for _ in num_steps_l]

    # --- Viz / REPL ---
    def string_to_action(self, act_str: str) -> Optional[LesionAction]:
        if act_str in {str(i) for i in range(6)}:
            return LesionAction(int(act_str))
        return None

    def string_to_action_help(self) -> str:
        return "0:+i 1:-i 2:+j 3:-j 4:+k 5:-k"

    def visualize_state_goal(
        self,
        state: LesionState,
        goal: LesionGoal,
        fig: Figure,
        path: Optional[List[LesionState]] = None,
    ) -> None:
        """Triplanar view through the current state's voxel, with lesion + path."""
        i, j, k = state.i, state.j, state.k
        axes = fig.subplots(1, 3)

        def _norm(a: NDArray) -> NDArray:
            lo, hi = np.percentile(a, [1, 99])
            return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

        slices = [
            ("sagittal i=%d" % i, self.t1[i, :, :].T, self.lesion[i, :, :].T, (j, k)),
            ("coronal j=%d"  % j, self.t1[:, j, :].T, self.lesion[:, j, :].T, (i, k)),
            ("axial k=%d"    % k, self.t1[:, :, k].T, self.lesion[:, :, k].T, (i, j)),
        ]
        path_pts = np.array([[p.i, p.j, p.k] for p in (path or [])])

        for ax, (title, t1_sl, les_sl, (x, y)) in zip(axes, slices):
            ax.imshow(_norm(t1_sl), cmap="gray", origin="lower")
            if les_sl.any():
                overlay = np.zeros(les_sl.shape + (4,), dtype=np.float32)
                overlay[..., 0] = 1.0
                overlay[..., 3] = (les_sl > 0).astype(np.float32) * 0.45
                ax.imshow(overlay, origin="lower")
            if len(path_pts):
                if "sagittal" in title:
                    px, py = path_pts[:, 1], path_pts[:, 2]
                elif "coronal" in title:
                    px, py = path_pts[:, 0], path_pts[:, 2]
                else:
                    px, py = path_pts[:, 0], path_pts[:, 1]
                ax.plot(px, py, "-", color="yellow", lw=1.2, alpha=0.9)
            ax.scatter([x], [y], c="lime", s=60, edgecolor="k", zorder=5)
            ax.set_title(title)
            ax.set_axis_off()
        fig.suptitle(f"state={state}  solved={self.lesion[i, j, k] > 0}")
        fig.tight_layout()
