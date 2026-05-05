"""Wrapper around `deepxube solve` that registers the lesion-evolution domain."""
import argparse
import sys

import deepxube_hw4.train_evolution  # noqa: F401  (domain registration)
from deepxube._solve import parse_solve, solve_cli

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    parse_solve(p)
    args = p.parse_args(sys.argv[1:])
    solve_cli(args)
