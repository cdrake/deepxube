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
from deepxube_hw4.train_evolution import LesionEvoMLP
from deepxube_hw4.viz_with_heur import greedy_q_search


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
    args = p.parse_args()

    name, args_str = args.domain.split(".", 1)
    domain = domain_factory.build_class(name, domain_factory.get_kwargs(name, args_str))
    nin = get_nnet_input_t(("lesion_evo", "lesion_evo_sga"))(domain=domain)
    heur = LesionEvoMLP(nin, out_dim=1, q_fix=False,
                        hidden=args.hidden, n_layers=args.n_layers)
    load_nnet(str(Path(args.heur_dir) / "heur.pt"), heur); heur.eval()

    rng = np.random.default_rng(args.seed)
    by_bucket: dict[int, list] = defaultdict(list)
    solved_total = 0
    opts = []
    for _ in range(args.n_trials):
        n = int(rng.integers(args.steps_min, args.steps_max + 1))
        np.random.seed(int(rng.integers(1 << 30)))
        states, goals = domain.sample_problem_instances([n])
        s0, g = states[0], goals[0]
        lb = len(s0.active ^ g.target)
        if lb == 0:
            continue
        _, path_a, solved = greedy_q_search(domain, heur, nin, s0, g, max_steps=args.budget)
        bucket = min(lb, 20)
        entry = {"solved": solved, "lb": lb, "len": len(path_a)}
        by_bucket[bucket].append(entry)
        solved_total += int(solved)
        if solved:
            opts.append(len(path_a) / lb)

    N = sum(len(v) for v in by_bucket.values())
    print(f"Trials: {N}  solved: {solved_total}/{N} = {100*solved_total/max(N,1):.1f}%")
    if opts:
        print(f"Mean optimality (path/lb, solved only): {np.mean(opts):.3f}  "
              f"(1.0 = optimal)")
    print("\nBy |s△g| bucket:")
    print(f"  {'lb':>4} {'n':>4} {'solved':>7}  mean_len  mean_opt")
    for b in sorted(by_bucket):
        rs = by_bucket[b]
        n = len(rs); ns = sum(r["solved"] for r in rs)
        ml = float(np.mean([r["len"] for r in rs])) if rs else 0.0
        mo = float(np.mean([r["len"] / r["lb"] for r in rs if r["solved"]])) \
            if ns else float("nan")
        print(f"  {b:>4} {n:>4} {ns:>3}/{n:<3}  {ml:>8.1f}  {mo:>8.3f}")


if __name__ == "__main__":
    main()
