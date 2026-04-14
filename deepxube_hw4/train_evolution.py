"""Trainable wrapper around LesionEvolutionDomain for DeepXube DAVI.

Single-subject MVP: one subject => fixed K parcels => fixed-size NN input.
Cross-subject generalization is out of scope for the first pass.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from deepxube.base.domain import ActsEnum, StartGoalWalkable
from deepxube.base.factory import Parser
from deepxube.base.heuristic import HeurNNet
from deepxube.base.nnet_input import StateGoalActIn
from deepxube.factories.domain_factory import domain_factory
from deepxube.factories.heuristic_factory import heuristic_factory
from deepxube.factories.nnet_input_factory import register_nnet_input
from deepxube.nnet.pytorch_models import FullyConnectedModel
from deepxube_hw4.evolution import (
    EXPAND, SHRINK, STOP,
    EvolutionAction, EvolutionGoal, EvolutionState, LesionEvolutionDomain,
)
from deepxube_hw4.parcels import parcellate_dwi
from deepxube_hw4.soop import load_subject


@domain_factory.register_class("lesion_evo")
class TrainableLesionEvo(
    LesionEvolutionDomain,
    ActsEnum[EvolutionState, EvolutionAction, EvolutionGoal],
    StartGoalWalkable[EvolutionState, EvolutionAction, EvolutionGoal],
):
    def __init__(self, subject_id: str = "sub-1", n_parcels: int = 300, coverage: float = 0.3):
        subj = load_subject(subject_id)
        brain = subj.dwi > np.percentile(subj.dwi, 40)
        parc = parcellate_dwi(subj.dwi, brain, n_parcels=n_parcels)
        LesionEvolutionDomain.__init__(self, subj, parc, coverage=coverage)
        self.subject_id: str = subject_id
        self.K: int = parc.n_parcels

    def get_state_actions(self, states: List[EvolutionState]) -> List[List[EvolutionAction]]:
        return [self.legal_actions(s) for s in states]

    def sample_start_states(self, num_states: int) -> List[EvolutionState]:
        rng = np.random.default_rng()
        base = self.acute
        out: List[EvolutionState] = []
        for _ in range(num_states):
            active = set(base)
            # Mild diversity: drop up to 20% of acute parcels or add one frontier parcel.
            if active and rng.random() < 0.4:
                n_drop = int(rng.integers(1, max(2, len(active) // 5 + 1)))
                for _ in range(n_drop):
                    if active:
                        active.discard(int(rng.choice(list(active))))
            out.append(EvolutionState(frozenset(active)))
        return out

    def sample_goal_from_state(
        self,
        states_start: Optional[List[EvolutionState]],
        states_goal: List[EvolutionState],
    ) -> List[EvolutionGoal]:
        return [EvolutionGoal(s.active) for s in states_goal]

    def __repr__(self) -> str:
        return f"TrainableLesionEvo({self.subject_id}, K={self.K})"


@domain_factory.register_parser("lesion_evo")
class TrainableLesionEvoParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        parts = args_str.split(".")
        sid = parts[0]
        n_parcels = int(parts[1]) if len(parts) > 1 else 300
        return {"subject_id": sid, "n_parcels": n_parcels}

    def help(self) -> str:
        return "Format: <subject_id>[.<n_parcels>]  e.g. 'sub-1.300'"


# --- NN input: state, goal, goal\state (needs-expand), state\goal (needs-shrink),
#     kind one-hot, parcel one-hot, and a scalar "action parcel is in s△g" bit.
@register_nnet_input("lesion_evo", "lesion_evo_sga")
class LesionEvoSGAIn(StateGoalActIn[TrainableLesionEvo, EvolutionState, EvolutionGoal, EvolutionAction]):
    def get_feat_dim(self) -> int:
        K = self.domain.K
        return 4 * K + 3 + (K + 1) + 1

    def get_input_info(self) -> int:
        return self.get_feat_dim()

    def to_np(
        self,
        states: List[EvolutionState],
        goals: List[EvolutionGoal],
        actions: List[EvolutionAction],
    ) -> List[NDArray]:
        K = self.domain.K
        N = len(states)
        D = 4 * K + 3 + (K + 1) + 1
        feat = np.zeros((N, D), dtype=np.float32)
        for i, (s, g, a) in enumerate(zip(states, goals, actions)):
            s_arr = np.fromiter(s.active, dtype=np.int64) - 1 if s.active else np.empty(0, dtype=np.int64)
            g_arr = np.fromiter(g.target, dtype=np.int64) - 1 if g.target else np.empty(0, dtype=np.int64)
            needs_expand = g.target - s.active  # in goal, not in state
            needs_shrink = s.active - g.target  # in state, not in goal
            sym_diff = needs_expand | needs_shrink
            if s_arr.size:
                feat[i, s_arr] = 1.0
            if g_arr.size:
                feat[i, K + g_arr] = 1.0
            if needs_expand:
                idx = np.fromiter(needs_expand, dtype=np.int64) - 1
                feat[i, 2 * K + idx] = 1.0
            if needs_shrink:
                idx = np.fromiter(needs_shrink, dtype=np.int64) - 1
                feat[i, 3 * K + idx] = 1.0
            feat[i, 4 * K + int(a.kind)] = 1.0
            feat[i, 4 * K + 3 + int(a.parcel)] = 1.0
            feat[i, D - 1] = 1.0 if (a.kind != 0 and a.parcel in sym_diff) else 0.0
        return [feat]


@heuristic_factory.register_class("lesion_evo_mlp")
class LesionEvoMLP(HeurNNet[LesionEvoSGAIn]):
    @staticmethod
    def nnet_input_type() -> Type[LesionEvoSGAIn]:
        return LesionEvoSGAIn

    def __init__(
        self,
        nnet_input: LesionEvoSGAIn,
        out_dim: int,
        q_fix: bool,
        hidden: int = 512,
        n_layers: int = 3,
    ):
        super().__init__(nnet_input, out_dim, q_fix)
        in_dim = nnet_input.get_feat_dim()
        self.net = nn.Sequential(
            FullyConnectedModel(
                in_dim,
                [hidden] * n_layers,
                ["RELU"] * n_layers,
                batch_norms=[True] * n_layers,
            ),
            nn.Linear(hidden, out_dim),
        )

    def _forward(self, inputs: List[Tensor]) -> Tensor:
        return self.net(inputs[0])


@heuristic_factory.register_parser("lesion_evo_mlp")
class LesionEvoMLPParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if args_str:
            for tok in args_str.split("_"):
                if tok.endswith("H"):
                    kwargs["hidden"] = int(tok[:-1])
                elif tok.endswith("L"):
                    kwargs["n_layers"] = int(tok[:-1])
        return kwargs

    def help(self) -> str:
        return "Format: <hidden>H_<n_layers>L  e.g. '512H_3L'"
