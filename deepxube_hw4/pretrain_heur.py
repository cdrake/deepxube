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
    lambda_len: float = 0.0, dual_head: bool = False,
) -> Tuple[List[EvolutionState], List[EvolutionGoal], List[EvolutionAction], np.ndarray]:
    """Build (state, goal, action, target) tuples from simulator rollouts.

    Scalar mode (dual_head=False): target is the weighted combination
        target = λ · len_remaining + (1 − λ) · cost_remaining
    λ = 0 → pure bio-cost; λ = 1 → pure length; λ ∈ (0, 1) → multi-objective.

    Dual-head mode (dual_head=True): target is a (2,)-vector per row —
    [length_to_go, cost_to_go] — for a net that predicts both heads and
    combines them at inference. `lambda_len` is ignored (each head gets
    clean scalar supervision).

    Returns targets as an np.ndarray of shape (N,) or (N, 2).
    """
    rng = np.random.default_rng(seed)
    states: List[EvolutionState] = []
    goals: List[EvolutionGoal] = []
    actions: List[EvolutionAction] = []
    len_ctgs: List[float] = []
    cost_ctgs: List[float] = []
    for _ in range(n_traj):
        traj = simulate(domain, rng, max_steps=max_steps)
        goal = EvolutionGoal(traj.states[-1].active)
        T = len(traj.actions)
        step_costs = np.array(
            [domain.action_cost(a.kind, a.parcel) for a in traj.actions],
            dtype=np.float32,
        )
        cost_remaining = (
            np.cumsum(step_costs[::-1])[::-1] if T > 0
            else np.zeros(0, dtype=np.float32)
        )
        len_remaining = np.arange(T, 0, -1, dtype=np.float32)
        for t in range(T):
            states.append(traj.states[t])
            goals.append(goal)
            actions.append(traj.actions[t])
            len_ctgs.append(float(len_remaining[t]))
            cost_ctgs.append(float(cost_remaining[t]))
        # Goal state: STOP has zero cost-to-go under both.
        states.append(traj.states[T])
        goals.append(goal)
        actions.append(EvolutionAction(STOP))
        len_ctgs.append(0.0); cost_ctgs.append(0.0)
        # Negative: random different legal action at step t costs one more
        # step + its bio-cost, then still needs remaining to hit the goal.
        for t in range(T):
            s = traj.states[t]
            legal = [a for a in domain.legal_actions(s) if a != traj.actions[t]]
            if not legal:
                continue
            a_neg = legal[int(rng.integers(len(legal)))]
            states.append(s)
            goals.append(goal)
            actions.append(a_neg)
            len_ctgs.append(float(len_remaining[t]) + 1.0)
            cost_ctgs.append(
                float(cost_remaining[t]) + domain.action_cost(a_neg.kind, a_neg.parcel)
            )

    len_arr = np.asarray(len_ctgs, dtype=np.float32)
    cost_arr = np.asarray(cost_ctgs, dtype=np.float32)
    if dual_head:
        targets = np.stack([len_arr, cost_arr], axis=1)  # (N, 2)
    else:
        targets = lambda_len * len_arr + (1.0 - lambda_len) * cost_arr  # (N,)
    return states, goals, actions, targets


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
    p.add_argument(
        "--lambda_len", type=float, default=0.0,
        help="Mix factor: target = lambda_len·len + (1-lambda_len)·bio_cost. "
             "0.0 = pure bio-cost; 1.0 = pure length (unit-cost baseline). "
             "Ignored if --dual_head is set.",
    )
    p.add_argument(
        "--dual_head", action="store_true",
        help="Train a 2-head net predicting [length_to_go, cost_to_go]; "
             "heads are combined at inference time.",
    )
    args = p.parse_args()

    name, args_str = args.domain.split(".", 1)
    kwargs = domain_factory.get_kwargs(name, args_str)
    domain: TrainableLesionEvo = domain_factory.build_class(name, kwargs)
    print(f"Domain: {domain}  K={domain.K}")

    states, goals, actions, ctgs = build_dataset(
        domain, args.n_traj, args.max_steps, args.seed,
        lambda_len=args.lambda_len, dual_head=args.dual_head,
    )
    N = len(states)
    if args.dual_head:
        print(f"Dataset: N={N}  dual_head=True  "
              f"mean_len={ctgs[:, 0].mean():.2f}  mean_cost={ctgs[:, 1].mean():.2f}  "
              f"max_len={ctgs[:, 0].max():.2f}  max_cost={ctgs[:, 1].max():.2f}")
    else:
        print(f"Dataset: N={N}  lambda_len={args.lambda_len}  "
              f"mean_ctg={np.mean(ctgs):.2f}  max_ctg={ctgs.max():.2f}")

    nnet_input_t = get_nnet_input_t(("lesion_evo", "lesion_evo_sga"))
    nnet_input = nnet_input_t(domain=domain)
    out_dim = 2 if args.dual_head else 1
    heur = LesionEvoMLP(
        nnet_input, out_dim=out_dim, q_fix=False,
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
            pred = heur([x])[0]
            if not args.dual_head:
                pred = pred.squeeze(-1)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        heur.eval()
        with torch.no_grad():
            pred_all = heur([feats_t])[0]
            if not args.dual_head:
                pred_all = pred_all.squeeze(-1)
            pred_np = pred_all.numpy()
        if args.dual_head:
            mae_len = float(np.mean(np.abs(pred_np[:, 0] - ctgs[:, 0])))
            mae_cost = float(np.mean(np.abs(pred_np[:, 1] - ctgs[:, 1])))
            print(f"epoch {ep + 1}/{args.epochs}  loss={np.mean(losses):.4f}  "
                  f"mae_len={mae_len:.3f}  mae_cost={mae_cost:.3f}")
        else:
            mae = float(np.mean(np.abs(pred_np - np.asarray(ctgs))))
            print(f"epoch {ep + 1}/{args.epochs}  loss={np.mean(losses):.4f}  mae={mae:.3f}")
        heur.train()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(heur.state_dict(), out_dir / "heur.pt")
    shutil.copy2(out_dir / "heur.pt", out_dir / "heur_targ.pt")
    print(f"Wrote {out_dir}/heur.pt and heur_targ.pt")


if __name__ == "__main__":
    main()
