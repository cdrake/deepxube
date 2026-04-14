"""Interactive viz with a trained heuristic in the loop.

Two modes:
  interactive — user types actions (expand/shrink/stop); each step shows
                h(state, goal) and the top-k Q-values for legal actions.
  solve       — runs greedy-Q search from start to goal (or step cap) and
                steps through the produced trajectory with n/p/idx nav.

Usage:
  python -m deepxube_hw4.viz_with_heur --domain lesion_evo.sub-1.300 \
      --heur_dir deepxube_hw4/output --steps 12 [--mode solve]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

import deepxube_hw4.train_evolution  # noqa: F401  (registers domain + heur)
from deepxube.factories.domain_factory import domain_factory
from deepxube.factories.nnet_input_factory import get_nnet_input_t
from deepxube.nnet.nnet_utils import load_nnet
from deepxube_hw4.evolution import EvolutionAction, EvolutionGoal, EvolutionState
from deepxube_hw4.train_evolution import LesionEvoMLP, TrainableLesionEvo


def build_heur(domain: TrainableLesionEvo, heur_dir: Path, hidden: int, n_layers: int):
    nnet_input_t = get_nnet_input_t(("lesion_evo", "lesion_evo_sga"))
    nnet_input = nnet_input_t(domain=domain)
    heur = LesionEvoMLP(nnet_input, out_dim=1, q_fix=False, hidden=hidden, n_layers=n_layers)
    load_nnet(str(heur_dir / "heur.pt"), heur)
    heur.eval()
    return heur, nnet_input


@torch.no_grad()
def q_values(
    heur: LesionEvoMLP,
    nnet_input,
    state: EvolutionState,
    goal: EvolutionGoal,
    actions: List[EvolutionAction],
) -> np.ndarray:
    feats = nnet_input.to_np([state] * len(actions), [goal] * len(actions), actions)
    inputs = [torch.from_numpy(x) for x in feats]
    out = heur(inputs)[0].cpu().numpy().ravel()
    return out


def print_q_top(acts: List[EvolutionAction], q: np.ndarray, k: int = 8) -> None:
    order = np.argsort(q)
    print(f"  Top-{k} (lowest Q = best):")
    for i in order[:k]:
        print(f"    Q={q[i]:+.3f}  {acts[i]}")


def greedy_q_search(
    domain: TrainableLesionEvo, heur, nnet_input,
    start: EvolutionState, goal: EvolutionGoal, max_steps: int = 40,
) -> Tuple[List[EvolutionState], List[EvolutionAction], bool]:
    states = [start]
    actions: List[EvolutionAction] = []
    s = start
    for _ in range(max_steps):
        if domain.is_solved([s], [goal])[0]:
            return states, actions, True
        acts = [a for a in domain.legal_actions(s) if a.kind != 0]
        if not acts:
            break
        q = q_values(heur, nnet_input, s, goal, acts)
        a = acts[int(np.argmin(q))]
        s = domain.next_state([s], [a])[0][0]
        states.append(s); actions.append(a)
    return states, actions, domain.is_solved([s], [goal])[0]


def _render(domain, state, goal, fig, title_suffix: str = "") -> None:
    fig.clear()
    domain.visualize_state_goal(state, goal, fig)
    if title_suffix:
        fig.suptitle(fig._suptitle.get_text() + f"  |  {title_suffix}")
    fig.canvas.draw()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--heur_dir", required=True)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--mode", choices=["interactive", "solve"], default="interactive")
    p.add_argument("--max_steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    name, args_str = args.domain.split(".", 1)
    kwargs = domain_factory.get_kwargs(name, args_str)
    domain: TrainableLesionEvo = domain_factory.build_class(name, kwargs)
    heur, nnet_input = build_heur(domain, Path(args.heur_dir), args.hidden, args.n_layers)

    np.random.seed(args.seed)
    states, goals = domain.sample_problem_instances([args.steps])
    state, goal = states[0], goals[0]
    h0 = float(q_values(heur, nnet_input, state, goal, [EvolutionAction(0)])[0])
    print(f"start |active|={len(state.active)}  goal |target|={len(goal.target)}  h(start)={h0:+.3f}")

    fig = plt.figure(figsize=(12, 4))
    _render(domain, state, goal, fig, f"h={h0:+.2f}")

    if args.mode == "solve":
        path_s, path_a, solved = greedy_q_search(
            domain, heur, nnet_input, state, goal, max_steps=args.max_steps
        )
        print(f"greedy-Q: {'SOLVED' if solved else 'not solved'} in {len(path_a)} steps")
        idx, idx_max = 0, len(path_s) - 1
        plt.show(block=False)
        while True:
            cmd = input(f"State {idx}/{idx_max}. n=next p=prev <int>=idx q=quit: ").strip()
            if cmd in {"", "q"}:
                break
            if cmd.lower() == "n" and idx < idx_max:
                idx += 1
            elif cmd.lower() == "p" and idx > 0:
                idx -= 1
            else:
                try:
                    idx = max(0, min(idx_max, int(cmd)))
                except ValueError:
                    continue
            s_now = path_s[idx]
            h_now = float(q_values(heur, nnet_input, s_now, goal, [EvolutionAction(0)])[0])
            action_str = f"last={path_a[idx - 1]}" if idx > 0 else "start"
            _render(domain, s_now, goal, fig, f"h={h_now:+.2f}  {action_str}")
    else:
        print(domain.string_to_action_help())
        plt.show(block=False)
        while True:
            acts = domain.legal_actions(state)
            q = q_values(heur, nnet_input, state, goal, acts)
            print_q_top(acts, q, k=6)
            cmd = input("action (or blank to quit): ").strip()
            if not cmd:
                break
            a = domain.string_to_action(cmd)
            if a is None:
                print(f"  bad action: {cmd!r}"); continue
            state = domain.next_state([state], [a])[0][0]
            h_now = float(q_values(heur, nnet_input, state, goal, [EvolutionAction(0)])[0])
            print(f"  |active|={len(state.active)}  h={h_now:+.3f}  solved={domain.is_solved([state],[goal])[0]}")
            _render(domain, state, goal, fig, f"h={h_now:+.2f}")


if __name__ == "__main__":
    main()
