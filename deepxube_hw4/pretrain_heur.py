"""Supervised warm-start for the lesion-evolution heuristic.

Rolls calibrated-simulator trajectories, emits (state, goal, action, cost-to-go)
tuples, and trains `LesionEvoMLP` with MSE on cost-to-go. Saves `heur.pt` +
`heur_targ.pt` into the output dir so DAVI can refine on top.

Usage:
  python -m deepxube_hw4.pretrain_heur --domain lesion_evo.sub-1.300 \
      --out deepxube_hw4/output_warm --n_traj 200 --epochs 8
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

import deepxube_hw4.train_evolution  # noqa: F401  (registers factory entries)
from deepxube.factories.domain_factory import domain_factory
from deepxube.factories.nnet_input_factory import get_nnet_input_t
from deepxube_hw4.evolution import (
    EXPAND, SHRINK, STOP,
    EvolutionAction, EvolutionGoal, EvolutionState,
)
from deepxube_hw4.simulator import simulate
from deepxube_hw4.train_evolution import LesionEvoMLP, TrainableLesionEvo


def build_dataset(
    domain: TrainableLesionEvo, n_traj: int, max_steps: int, seed: int,
) -> Tuple[List[EvolutionState], List[EvolutionGoal], List[EvolutionAction], List[float]]:
    rng = np.random.default_rng(seed)
    states: List[EvolutionState] = []
    goals: List[EvolutionGoal] = []
    actions: List[EvolutionAction] = []
    ctgs: List[float] = []
    for _ in range(n_traj):
        traj = simulate(domain, rng, max_steps=max_steps)
        goal = EvolutionGoal(traj.states[-1].active)
        T = len(traj.actions)
        # Greedy path samples: at step t, taking traj.actions[t] leads to a
        # state that is (T-t-1) steps from goal, so Q(s,g,a) = 1 + (T-t-1) = T-t.
        for t in range(T):
            states.append(traj.states[t])
            goals.append(goal)
            actions.append(traj.actions[t])
            ctgs.append(float(T - t))
        # Goal state: STOP action has cost-to-go 0.
        states.append(traj.states[T])
        goals.append(goal)
        actions.append(EvolutionAction(STOP))
        ctgs.append(0.0)
        # Negative samples: at step t, a random *different* legal action should
        # be no better (penalized to 1 + T-t, a mild upper bound).
        for t in range(T):
            s = traj.states[t]
            legal = [a for a in domain.legal_actions(s) if a != traj.actions[t]]
            if not legal:
                continue
            a_neg = legal[int(rng.integers(len(legal)))]
            states.append(s)
            goals.append(goal)
            actions.append(a_neg)
            ctgs.append(float(T - t + 1))
    return states, goals, actions, ctgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_traj", type=int, default=400)
    p.add_argument("--max_steps", type=int, default=25)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    name, args_str = args.domain.split(".", 1)
    kwargs = domain_factory.get_kwargs(name, args_str)
    domain: TrainableLesionEvo = domain_factory.build_class(name, kwargs)
    print(f"Domain: {domain}  K={domain.K}")

    states, goals, actions, ctgs = build_dataset(
        domain, args.n_traj, args.max_steps, args.seed,
    )
    N = len(states)
    print(f"Dataset: N={N}  mean_ctg={np.mean(ctgs):.2f}  max_ctg={max(ctgs):.0f}")

    nnet_input_t = get_nnet_input_t(("lesion_evo", "lesion_evo_sga"))
    nnet_input = nnet_input_t(domain=domain)
    heur = LesionEvoMLP(
        nnet_input, out_dim=1, q_fix=False,
        hidden=args.hidden, n_layers=args.n_layers,
    )
    opt = torch.optim.Adam(heur.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    feats = nnet_input.to_np(states, goals, actions)[0]
    feats_t = torch.from_numpy(feats)
    targ_t = torch.tensor(ctgs, dtype=torch.float32)

    rng = np.random.default_rng(args.seed)
    heur.train()
    for ep in range(args.epochs):
        order = rng.permutation(N)
        losses = []
        for i in range(0, N, args.batch_size):
            idx = order[i : i + args.batch_size]
            x = feats_t[idx]
            y = targ_t[idx]
            pred = heur([x])[0].squeeze(-1)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        heur.eval()
        with torch.no_grad():
            pred_all = heur([feats_t])[0].squeeze(-1).numpy()
            mae = float(np.mean(np.abs(pred_all - np.asarray(ctgs))))
        heur.train()
        print(f"epoch {ep + 1}/{args.epochs}  loss={np.mean(losses):.4f}  mae={mae:.3f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(heur.state_dict(), out_dir / "heur.pt")
    shutil.copy2(out_dir / "heur.pt", out_dir / "heur_targ.pt")
    print(f"Wrote {out_dir}/heur.pt and heur_targ.pt")


if __name__ == "__main__":
    main()
