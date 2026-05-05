"""Build the quantitative figures cited in WRITEUP.md §5.

Reads results/{easy,hard}_{ucs,heur}/results.pkl and emits:

  fig_solve_rate.png   — grouped bars: % solved and mean nodes generated
                          (solved-only), UCS vs. heuristic, easy vs. hard.
  fig_nodes_scatter.png — per-instance UCS vs. heuristic nodes generated,
                          log-log, one panel per split; markers distinguish
                          solved-both / solved-one / solved-neither.

Usage:
    python -m deepxube_hw4.make_figures
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path(__file__).parent / "results"
OUT = Path(__file__).parent


def load(name: str) -> dict:
    with open(RESULTS / name / "results.pkl", "rb") as f:
        return pickle.load(f)


def solve_rate_fig() -> None:
    splits = ["easy", "hard"]
    data = {s: {alg: load(f"{s}_{alg}") for alg in ("ucs", "heur")} for s in splits}

    fig, (ax_rate, ax_nodes) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(splits))
    w = 0.35

    rate_ucs = [100 * np.mean(data[s]["ucs"]["solved"]) for s in splits]
    rate_heur = [100 * np.mean(data[s]["heur"]["solved"]) for s in splits]
    ax_rate.bar(x - w / 2, rate_ucs, w, label="UCS", color="#888888")
    ax_rate.bar(x + w / 2, rate_heur, w, label="Heuristic A*", color="#1f77b4")
    ax_rate.set_xticks(x)
    ax_rate.set_xticklabels([s.capitalize() for s in splits])
    ax_rate.set_ylabel("% solved (10 s / instance)")
    ax_rate.set_ylim(0, 105)
    ax_rate.set_title("Solve rate")
    for xi, rate in zip(x - w / 2, rate_ucs):
        ax_rate.text(xi, rate + 2, f"{rate:.0f}%", ha="center", fontsize=9)
    for xi, rate in zip(x + w / 2, rate_heur):
        ax_rate.text(xi, rate + 2, f"{rate:.0f}%", ha="center", fontsize=9)
    ax_rate.legend(loc="upper right")
    ax_rate.grid(axis="y", alpha=0.3)

    def mean_nodes_solved(d: dict) -> float:
        n = np.asarray(d["num_nodes_generated"], dtype=float)
        s = np.asarray(d["solved"], dtype=bool)
        return float(np.mean(n[s])) if s.any() else float("nan")

    nodes_ucs = [mean_nodes_solved(data[s]["ucs"]) for s in splits]
    nodes_heur = [mean_nodes_solved(data[s]["heur"]) for s in splits]

    finite = [v for v in nodes_ucs + nodes_heur if np.isfinite(v)]
    y_lo, y_hi = 10, max(finite) * 3

    plot_ucs = [v if np.isfinite(v) else 0 for v in nodes_ucs]
    plot_heur = [v if np.isfinite(v) else 0 for v in nodes_heur]
    ax_nodes.bar(x - w / 2, plot_ucs, w, label="UCS", color="#888888")
    ax_nodes.bar(x + w / 2, plot_heur, w, label="Heuristic A*", color="#1f77b4")
    ax_nodes.set_xticks(x)
    ax_nodes.set_xticklabels([s.capitalize() for s in splits])
    ax_nodes.set_ylabel("Mean nodes generated (solved only)")
    ax_nodes.set_yscale("log")
    ax_nodes.set_ylim(y_lo, y_hi)
    ax_nodes.set_title("Search effort")
    for xi, n in zip(x - w / 2, nodes_ucs):
        label = f"{n:,.0f}" if np.isfinite(n) else "none solved"
        ypos = n * 1.1 if np.isfinite(n) else y_lo * 1.3
        ax_nodes.text(xi, ypos, label, ha="center", va="bottom", fontsize=9)
    for xi, n in zip(x + w / 2, nodes_heur):
        label = f"{n:,.0f}" if np.isfinite(n) else "none solved"
        ypos = n * 1.1 if np.isfinite(n) else y_lo * 1.3
        ax_nodes.text(xi, ypos, label, ha="center", va="bottom", fontsize=9)
    ax_nodes.legend(loc="upper right")
    ax_nodes.grid(axis="y", which="both", alpha=0.3)

    fig.suptitle("UCS vs. learned heuristic: 10 s/instance budget, 30 instances per split",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_solve_rate.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def nodes_scatter_fig() -> None:
    splits = ["easy", "hard"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)

    for ax, split in zip(axes, splits):
        ucs = load(f"{split}_ucs")
        heur = load(f"{split}_heur")
        n_ucs = np.asarray(ucs["num_nodes_generated"], dtype=float)
        n_heur = np.asarray(heur["num_nodes_generated"], dtype=float)
        s_ucs = np.asarray(ucs["solved"], dtype=bool)
        s_heur = np.asarray(heur["solved"], dtype=bool)

        n_ucs = np.clip(n_ucs, 1, None)
        n_heur = np.clip(n_heur, 1, None)

        both = s_ucs & s_heur
        heur_only = ~s_ucs & s_heur
        ucs_only = s_ucs & ~s_heur
        neither = ~s_ucs & ~s_heur

        ax.scatter(n_ucs[both], n_heur[both], s=40, c="#1f77b4",
                   label=f"both solved ({both.sum()})", zorder=3)
        ax.scatter(n_ucs[heur_only], n_heur[heur_only], s=40, c="#2ca02c",
                   marker="^", label=f"heur only ({heur_only.sum()})", zorder=3)
        ax.scatter(n_ucs[ucs_only], n_heur[ucs_only], s=40, c="#ff7f0e",
                   marker="s", label=f"UCS only ({ucs_only.sum()})", zorder=3)
        ax.scatter(n_ucs[neither], n_heur[neither], s=40, c="#d62728",
                   marker="x", label=f"neither ({neither.sum()})", zorder=3)

        lo, hi = 1, max(n_ucs.max(), n_heur.max()) * 1.2
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, zorder=1, label="y = x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("UCS nodes generated")
        ax.set_ylabel("Heuristic nodes generated")
        ax.set_title(f"{split.capitalize()} (n={len(n_ucs)})")
        ax.grid(which="both", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Per-instance search effort: points below y=x favor the heuristic")
    fig.tight_layout()
    fig.savefig(OUT / "fig_nodes_scatter.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    solve_rate_fig()
    nodes_scatter_fig()
    print("Wrote:")
    print(f"  {OUT / 'fig_solve_rate.png'}")
    print(f"  {OUT / 'fig_nodes_scatter.png'}")


if __name__ == "__main__":
    main()
