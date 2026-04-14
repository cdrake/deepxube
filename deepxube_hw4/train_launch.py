"""Wrapper around `deepxube train` that registers our lesion-evolution domain first."""
import argparse
import sys

import deepxube_hw4.train_evolution  # noqa: F401  (triggers factory registration)
from deepxube._train_cli import parser_train, train_cli

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    parser_train(p)
    args = p.parse_args(sys.argv[1:])
    train_cli(args)
