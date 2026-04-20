"""Quantitative evaluation of a trained lesion-evo heuristic.

Runs greedy-Q over N random instances bucketed by reverse-walk length, and
reports solve rate + path optimality (path_len / |s△g| lower bound).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

import deepxube_hw4.train_evolution  # noqa: F401
from deepxube.factories.domain_factory import domain_factory
from deepxube.factories.nnet_input_factory import get_nnet_input_t
from deepxube.nnet.nnet_utils import load_nnet
from deepxube_hw4.evolution import COST_MIN, EXPAND, SHRINK
from deepxube_hw4.train_evolution import LesionEvoMLP
from deepxube_hw4.viz_with_heur import greedy_q_search


def tight_cost_lb(domain, state, goal) -> float:
    """Sum the true bio-cost of the exact parcel flips required by s△g.

    Each parcel in s△g must be flipped exactly once by the unique
    correct action (EXPAND if in goal but not state; SHRINK otherwise).
    This is the lowest cost any path that reaches the goal can achieve
    without wasted moves, so a model path that only touches s△g parcels
    hits this floor exactly.
    """
    total = 0.0
    for p in state.active ^ goal.target:
        kind = EXPAND if p in goal.target else SHRINK
        total += domain.action_cost(kind, p)
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--heur_dir", required=True)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--n_trials", type=int, default=100)
    p.add_argument("--steps_min", type=int, default=2)
    p.add_argument("--steps_max", type=int, default=15)
    p.add_argument("--budget", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dual_head", action="store_true",
                   help="Checkpoint is a dual-head net (out_dim=2: [len, cost]).")
    p.add_argument("--lambda_len", type=float, default=0.5,
                   help="Inference-time mix for dual-head: λ·L + (1-λ)·C.")
    args = p.parse_args()

    name, args_str = args.domain.split(".", 1)
    domain = domain_factory.build_class(name, domain_factory.get_kwargs(name, args_str))
    nin = get_nnet_input_t(("lesion_evo", "lesion_evo_sga"))(domain=domain)
    out_dim = 2 if args.dual_head else 1
    heur = LesionEvoMLP(nin, out_dim=out_dim, q_fix=False,
                        hidden=args.hidden, n_layers=args.n_layers)
    load_nnet(str(Path(args.heur_dir) / "heur.pt"), heur); heur.eval()

    rng = np.random.default_rng(args.seed)
    by_bucket: dict[int, list] = defaultdict(list)
    solved_total = 0
    len_opts = []
    cost_opts = []
    cost_opts_tight = []
    for _ in range(args.n_trials):
        n = int(rng.integers(args.steps_min, args.steps_max + 1))
        np.random.seed(int(rng.integers(1 << 30)))
        states, goals = domain.sample_problem_instances([n])
        s0, g = states[0], goals[0]
        lb = len(s0.active ^ g.target)
        if lb == 0:
            continue
        tight_lb = tight_cost_lb(domain, s0, g)
        _, path_a, solved = greedy_q_search(
            domain, heur, nin, s0, g, max_steps=args.budget,
            lambda_len=args.lambda_len,
        )
        path_cost = sum(domain.action_cost(a.kind, a.parcel) for a in path_a)
        bucket = min(lb, 20)
        entry = {
            "solved": solved, "lb": lb, "tight_lb": tight_lb,
            "len": len(path_a), "cost": path_cost,
        }
        by_bucket[bucket].append(entry)
        solved_total += int(solved)
        if solved:
            len_opts.append(len(path_a) / lb)
            # Loose LB: every step pays at least COST_MIN.
            cost_opts.append(path_cost / (COST_MIN * lb))
            # Tight LB: the exact bio-cost of flipping the parcels in s△g
            # once each — no-wasted-moves floor.
            if tight_lb > 0:
                cost_opts_tight.append(path_cost / tight_lb)

    N = sum(len(v) for v in by_bucket.values())
    print(f"Trials: {N}  solved: {solved_total}/{N} = {100*solved_total/max(N,1):.1f}%")
    if len_opts:
        print(f"Mean length-optimality (path_len/|s△g|, solved): {np.mean(len_opts):.3f}")
        print(f"Mean cost-optimality loose  (path_cost/(c_min·|s△g|)): "
              f"{np.mean(cost_opts):.3f}")
        print(f"Mean cost-optimality tight  (path_cost/Σ_{{s△g}} cost(a_p)): "
              f"{np.mean(cost_opts_tight):.3f}  (1.0 = no wasted moves)")
    print("\nBy |s△g| bucket:")
    print(f"  {'lb':>4} {'n':>4} {'solved':>7}  mean_len  mean_cost  "
          f"len_opt  co_loose  co_tight")
    for b in sorted(by_bucket):
        rs = by_bucket[b]
        n = len(rs); ns = sum(r["solved"] for r in rs)
        ml = float(np.mean([r["len"] for r in rs])) if rs else 0.0
        mc = float(np.mean([r["cost"] for r in rs])) if rs else 0.0
        lo = float(np.mean([r["len"] / r["lb"] for r in rs if r["solved"]])) \
            if ns else float("nan")
        col = float(np.mean([r["cost"] / (COST_MIN * r["lb"]) for r in rs if r["solved"]])) \
            if ns else float("nan")
        cot = float(np.mean([r["cost"] / r["tight_lb"]
                             for r in rs if r["solved"] and r["tight_lb"] > 0])) \
            if ns else float("nan")
        print(f"  {b:>4} {n:>4} {ns:>3}/{n:<3}  {ml:>8.1f}  {mc:>9.2f}  "
              f"{lo:>7.3f}  {col:>8.3f}  {cot:>8.3f}")


if __name__ == "__main__":
    main()
