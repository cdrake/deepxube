"""Generate test problem instances and save as .pkl files.

Two files:
- easy.pkl: short reverse-random-walk goals (|s△g| ∈ [1, 4]).  These are solvable
  by uniform-cost search within a modest budget — they exist to show the
  heuristic produces equally optimal paths but with far fewer node expansions.
- hard.pkl: biologically-calibrated simulator goals.  The simulator rolls out
  expand/shrink trajectories of length ~10–20 weighted by DWI intensity and
  parcel coverage, so the start→goal pair resembles a real acute→chronic
  lesion evolution rather than a uniform random walk.  |s△g| is typically
  ≥ 8, and the action branching factor (~80 per state) is large enough that
  UCS exceeds practical node budgets on most instances.

Both files use the standard DeepXube format:
    {"states": [EvolutionState, ...], "goals": [EvolutionGoal, ...]}

Usage:
    python -m deepxube_hw4.make_instances --domain lesion_evo.sub-1.300 \
        --out deepxube_hw4/instances --n_easy 30 --n_hard 30
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

import deepxube_hw4.train_evolution  # noqa: F401
from deepxube.factories.domain_factory import domain_factory
from deepxube_hw4.evolution import EvolutionGoal, EvolutionState
from deepxube_hw4.simulator import simulate
from deepxube_hw4.train_evolution import TrainableLesionEvo


def gen_easy(
    domain: TrainableLesionEvo, n: int, rng: np.random.Generator,
) -> Tuple[List[EvolutionState], List[EvolutionGoal]]:
    states: List[EvolutionState] = []
    goals: List[EvolutionGoal] = []
    tries = 0
    while len(states) < n and tries < n * 20:
        tries += 1
        steps = int(rng.integers(2, 6))
        np.random.seed(int(rng.integers(1 << 30)))
        s_l, g_l = domain.sample_problem_instances([steps])
        s, g = s_l[0], g_l[0]
        lb = len(s.active ^ g.target)
        if 1 <= lb <= 4:
            states.append(s); goals.append(g)
    return states, goals


def gen_hard(
    domain: TrainableLesionEvo, n: int, rng: np.random.Generator,
) -> Tuple[List[EvolutionState], List[EvolutionGoal]]:
    states: List[EvolutionState] = []
    goals: List[EvolutionGoal] = []
    tries = 0
    while len(states) < n and tries < n * 20:
        tries += 1
        traj = simulate(domain, rng, max_steps=25)
        start = domain.start_state()
        goal = EvolutionGoal(traj.states[-1].active)
        lb = len(start.active ^ goal.target)
        if lb >= 8:
            states.append(start); goals.append(goal)
    return states, goals


def summarize(label: str, states, goals):
    lbs = [len(s.active ^ g.target) for s, g in zip(states, goals)]
    print(f"{label}: n={len(states)}  |s△g| min/mean/max = "
          f"{min(lbs)}/{np.mean(lbs):.1f}/{max(lbs)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_easy", type=int, default=30)
    p.add_argument("--n_hard", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    name, args_str = args.domain.split(".", 1)
    domain = domain_factory.build_class(name, domain_factory.get_kwargs(name, args_str))
    print(f"Domain: {domain}  K={domain.K}")

    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    easy_s, easy_g = gen_easy(domain, args.n_easy, rng)
    summarize("easy", easy_s, easy_g)
    pickle.dump({"states": easy_s, "goals": easy_g}, open(out / "easy.pkl", "wb"),
                protocol=-1)

    hard_s, hard_g = gen_hard(domain, args.n_hard, rng)
    summarize("hard", hard_s, hard_g)
    pickle.dump({"states": hard_s, "goals": hard_g}, open(out / "hard.pkl", "wb"),
                protocol=-1)

    print(f"Wrote {out}/easy.pkl and {out}/hard.pkl")


if __name__ == "__main__":
    main()
