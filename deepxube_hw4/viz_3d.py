"""3D volumetric view of a lesion-evolution solution.

Uses matplotlib's `ax.voxels` (same pattern as examples/volume_viz.py) to show
faint brain + cyan acute + green goal + red current-state, stepping through
a greedy-Q trajectory produced by the trained heuristic.

Usage:
    python -m deepxube_hw4.viz_3d --domain lesion_evo.sub-1.300 \
        --heur_dir deepxube_hw4/output_warm --steps 10 --stride 4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

import deepxube_hw4.train_evolution  # noqa: F401
from deepxube.factories.domain_factory import domain_factory
from deepxube.factories.nnet_input_factory import get_nnet_input_t
from deepxube.nnet.nnet_utils import load_nnet
from deepxube_hw4.evolution import EvolutionGoal, EvolutionState
from deepxube_hw4.train_evolution import LesionEvoMLP
from deepxube_hw4.viz_with_heur import greedy_q_search, q_values


def _downsample(mask: np.ndarray, s: int) -> np.ndarray:
    if s <= 1:
        return mask.astype(bool)
    # Any voxel in the s×s×s block counts.
    di, dj, dk = mask.shape
    ni, nj, nk = di // s, dj // s, dk // s
    m = mask[: ni * s, : nj * s, : nk * s]
    return m.reshape(ni, s, nj, s, nk, s).any(axis=(1, 3, 5))


def _render_3d(
    ax,
    dwi: np.ndarray,
    brain: np.ndarray,
    acute: np.ndarray,
    goal: np.ndarray,
    current: np.ndarray,
    stride: int,
    title: str,
) -> None:
    ax.clear()
    br = _downsample(brain, stride)
    ac = _downsample(acute, stride)
    gl = _downsample(goal, stride)
    cu = _downsample(current, stride)

    # Layers: only-brain (faint) | acute-only | goal-only | current | overlap cases
    # Priority for color assignment (higher wins): current > goal∩current shown red,
    # goal-only green, acute-only cyan, brain-only gray.
    shape = br.shape
    colors = np.zeros(shape + (4,), dtype=np.float32)
    visible = np.zeros(shape, dtype=bool)

    # Brain background (skip if too many — keeps voxel count manageable).
    brain_only = br & ~(ac | gl | cu)
    colors[brain_only] = [0.7, 0.7, 0.7, 0.03]
    visible |= brain_only

    # Acute (cyan, low alpha).
    acute_layer = ac & ~cu & ~gl
    colors[acute_layer] = [0.0, 0.9, 0.9, 0.25]
    visible |= acute_layer

    # Goal-only (green).
    goal_only = gl & ~cu
    colors[goal_only] = [0.2, 1.0, 0.2, 0.5]
    visible |= goal_only

    # Current (red, fully opaque).
    colors[cu] = [1.0, 0.15, 0.15, 0.85]
    visible |= cu

    ax.voxels(visible, facecolors=colors, edgecolor=None)
    ax.set_box_aspect(shape)
    ax.set_xlabel("i"); ax.set_ylabel("j"); ax.set_zlabel("k")
    ax.set_title(title)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--heur_dir", required=True)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_gif", type=str, default="")
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

    np.random.seed(args.seed)
    states, goals = domain.sample_problem_instances([args.steps])
    s0, g = states[0], goals[0]
    path_s, path_a, solved = greedy_q_search(
        domain, heur, nin, s0, g, max_steps=args.max_steps,
        lambda_len=args.lambda_len,
    )
    print(f"|start|={len(s0.active)}  |goal|={len(g.target)}  |s△g|={len(s0.active ^ g.target)}")
    print(f"greedy-Q: {'SOLVED' if solved else 'FAILED'} in {len(path_a)} steps")

    labels = domain.parc.labels
    brain = domain.subject.mask.astype(bool) if domain.subject.mask.any() else labels > 0
    dwi = domain.subject.dwi
    acute_mask = np.isin(labels, np.fromiter(domain.acute, dtype=np.int32))
    goal_mask = np.isin(labels, np.fromiter(g.target, dtype=np.int32)) if g.target else np.zeros_like(labels, bool)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    if args.save_gif:
        import imageio
        frames = []
        for t, s in enumerate(path_s):
            cur = np.isin(labels, np.fromiter(s.active, dtype=np.int32)) if s.active else np.zeros_like(labels, bool)
            _render_3d(ax, dwi, brain, acute_mask, goal_mask, cur, args.stride,
                       f"step {t}/{len(path_a)}  |active|={len(s.active)}")
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames.append(frame)
        imageio.mimsave(args.save_gif, frames, fps=2)
        print(f"Saved {args.save_gif} ({len(frames)} frames)")
        return

    idx = [0]
    def redraw():
        t = idx[0]
        s = path_s[t]
        cur = np.isin(labels, np.fromiter(s.active, dtype=np.int32)) if s.active else np.zeros_like(labels, bool)
        _render_3d(ax, dwi, brain, acute_mask, goal_mask, cur, args.stride,
                   f"step {t}/{len(path_a)}  |active|={len(s.active)}  "
                   f"{'SOLVED' if (solved and t==len(path_a)) else ''}")
        fig.canvas.draw_idle()

    def on_key(ev):
        if ev.key in ("n", "right") and idx[0] < len(path_s) - 1:
            idx[0] += 1; redraw()
        elif ev.key in ("p", "left") and idx[0] > 0:
            idx[0] -= 1; redraw()
        elif ev.key == "home":
            idx[0] = 0; redraw()
        elif ev.key == "end":
            idx[0] = len(path_s) - 1; redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    print("Keys: n/right = next, p/left = prev, home/end = first/last")
    plt.show()


if __name__ == "__main__":
    main()
