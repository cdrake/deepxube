"""Interactive step-through of a lesion-as-goal path (placeholder solver).

Mirrors the `deepxube viz ... --soln` REPL: n = next, p = previous,
<int> = jump to index, <Enter> = quit.

Usage:
    python examples/lesion_step_viz.py
    python examples/lesion_step_viz.py --downsample 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from deepxube_hw4.domain import DEFAULT_T1, LesionGoal, LesionPathDomain


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--t1", type=Path, default=DEFAULT_T1)
    p.add_argument("--downsample", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    domain = LesionPathDomain(t1_path=args.t1, downsample=args.downsample)
    start = domain.sample_start_state()
    goal = LesionGoal()
    states_on_path, actions = domain.straight_line_path(start, domain.lesion_centroid)

    print(f"shape={domain.shape}  path_len={len(states_on_path)}  "
          f"start={start}  centroid={tuple(domain.lesion_centroid)}")
    print("Controls: n = next, p = prev, <int> = jump, <Enter> = quit.")

    fig = plt.figure(figsize=(12, 4.5))
    idx = 0
    idx_max = len(states_on_path) - 1

    def redraw() -> None:
        fig.clear()
        domain.visualize_state_goal(states_on_path[idx], goal, fig, path=states_on_path)
        fig.canvas.draw_idle()

    redraw()
    plt.show(block=False)

    while True:
        cmd = input(f"step {idx}/{idx_max} > ").strip()
        if cmd == "":
            break
        if cmd.lower() == "n":
            if idx < idx_max:
                a = actions[idx]
                (nxt,), (tc,) = domain.next_state([states_on_path[idx]], [a])
                assert nxt == states_on_path[idx + 1]
                print(f"  action={a} tc={tc}")
                idx += 1
        elif cmd.lower() == "p":
            if idx > 0:
                idx -= 1
        else:
            try:
                j = int(cmd)
            except ValueError:
                print(f"  unknown: {cmd!r}")
                continue
            if 0 <= j <= idx_max:
                idx = j
            else:
                print(f"  out of range [0, {idx_max}]")
                continue
        print(f"  solved={domain.is_solved([states_on_path[idx]], [goal])[0]}")
        redraw()


if __name__ == "__main__":
    main()
