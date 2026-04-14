"""Launch `deepxube viz` on the lesion-evolution domain.

Usage (interactive step-through from random start):
    python -m deepxube_hw4.viz_launch --domain lesion_evo.sub-1.300 --steps 10

Usage (replay a solved trajectory from a viz.pkl file produced by `deepxube test`):
    python -m deepxube_hw4.viz_launch --domain lesion_evo.sub-1.300 --file <path> --idx 0 --soln
"""
import argparse
import sys

import deepxube_hw4.train_evolution  # noqa: F401  (domain registration)
from deepxube._cli import _parse_viz_info, viz

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    _parse_viz_info(p)
    args = p.parse_args(sys.argv[1:])
    viz(args)
