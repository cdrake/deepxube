"""Run UCS baseline vs. trained heuristic with DeepXube's solve CLI.

Calls `deepxube solve` (via solve_launch.py) on each .pkl with two configs:
  - UCS baseline: graph_q with weight=0 (heuristic ignored -> Dijkstra).
  - Heuristic:    graph_q with weight=1 (A*-style) using our trained QIn net.

Summarizes solve rate, mean solution cost, and mean nodes generated.

Usage:
    python -m deepxube_hw4.run_experiments \
        --domain lesion_evo.sub-1.300 \
        --heur_dir deepxube_hw4/output_warm \
        --instances_dir deepxube_hw4/instances \
        --results_dir deepxube_hw4/results \
        --time_limit 15
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np


PYTHON = sys.executable


def run_solve(
    domain: str,
    pkl: Path,
    results_dir: Path,
    pathfind: str,
    heur: str | None,
    heur_file: Path | None,
    time_limit: float,
    extra_env: dict | None = None,
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-m", "deepxube_hw4.solve_launch",
        "--domain", domain,
        "--pathfind", pathfind,
        "--file", str(pkl),
        "--results", str(results_dir),
        "--time_limit", str(time_limit),
        "--heur_type", "QIn",
        "--redo",
    ]
    if heur is not None and heur_file is not None:
        cmd += ["--heur", heur, "--heur_file", str(heur_file)]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    res = pickle.load(open(results_dir / "results.pkl", "rb"))
    return res


def summarize(label: str, res: dict) -> dict:
    solved = np.array(res["solved"], dtype=bool)
    ng = np.array(res["num_nodes_generated"], dtype=float)
    pc = np.array([c if np.isfinite(c) else np.nan for c in res["path_costs"]], dtype=float)
    row = {
        "label": label,
        "n": len(solved),
        "solved": int(solved.sum()),
        "rate": float(solved.mean()),
        "mean_nodes": float(ng[solved].mean()) if solved.any() else float("nan"),
        "mean_cost": float(np.nanmean(pc[solved])) if solved.any() else float("nan"),
        "mean_nodes_all": float(ng.mean()),
    }
    return row


def print_table(rows: list[dict]) -> None:
    hdr = f"{'config':<30} {'n':>3} {'solved':>7} {'rate':>6} {'mean_cost':>10} {'mean_nodes_solved':>18} {'mean_nodes_all':>15}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:<30} {r['n']:>3} {r['solved']:>3}/{r['n']:<3} "
              f"{r['rate']:>6.2%} {r['mean_cost']:>10.2f} "
              f"{r['mean_nodes']:>18.1f} {r['mean_nodes_all']:>15.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--heur_dir", required=True)
    p.add_argument("--heur", default="lesion_evo_mlp.512H_3L")
    p.add_argument("--instances_dir", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--time_limit", type=float, default=15.0)
    args = p.parse_args()

    heur_file = Path(args.heur_dir) / "heur.pt"
    pkls = {
        "easy": Path(args.instances_dir) / "easy.pkl",
        "hard": Path(args.instances_dir) / "hard.pkl",
    }
    rows: list[dict] = []
    for diff, pkl in pkls.items():
        if not pkl.exists():
            print(f"skip {diff}: {pkl} missing"); continue
        # UCS baseline: graph_q with weight=0 ignores the heuristic.
        ucs = run_solve(
            args.domain, pkl,
            Path(args.results_dir) / f"{diff}_ucs",
            pathfind="graph_q.1B_0.0W", heur=None, heur_file=None,
            time_limit=args.time_limit,
        )
        rows.append(summarize(f"{diff}/UCS (graph_q w=0)", ucs))
        # Trained heuristic: graph_q with weight=1 uses our Q predictions.
        trn = run_solve(
            args.domain, pkl,
            Path(args.results_dir) / f"{diff}_heur",
            pathfind="graph_q.1B_1.0W", heur=args.heur, heur_file=heur_file,
            time_limit=args.time_limit,
        )
        rows.append(summarize(f"{diff}/Heur (graph_q w=1)", trn))

    print("\n=== SUMMARY ===")
    print_table(rows)


if __name__ == "__main__":
    main()
