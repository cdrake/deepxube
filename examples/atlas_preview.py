"""Preview a single subject: T1 mid-slices with lesion overlay.

Supports the real ATLAS v2.0 loader and a synthetic MNI152-based fallback.

Usage:
    python examples/atlas_preview.py --synthetic --random
    python examples/atlas_preview.py --synthetic --id syn0007
    python examples/atlas_preview.py --atlas --random
    python examples/atlas_preview.py --atlas --id r001s001
"""
from __future__ import annotations

import argparse
import random
import sys

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--atlas", action="store_true", help="Use ATLAS v2.0 loader ($ATLAS_ROOT).")
    src.add_argument("--synthetic", action="store_true", help="Use MNI152 + synthetic lesions.")
    pick = p.add_mutually_exclusive_group(required=True)
    pick.add_argument("--id", type=str)
    pick.add_argument("--random", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def _norm(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def show(sid: str, t1: np.ndarray, mask: np.ndarray | None) -> None:
    i, j, k = (s // 2 for s in t1.shape)
    slices = [
        ("sagittal", t1[i, :, :].T, None if mask is None else mask[i, :, :].T),
        ("coronal",  t1[:, j, :].T, None if mask is None else mask[:, j, :].T),
        ("axial",    t1[:, :, k].T, None if mask is None else mask[:, :, k].T),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, (name, img, m) in zip(axes, slices):
        ax.imshow(_norm(img), cmap="gray", origin="lower")
        if m is not None and m.any():
            overlay = np.zeros(m.shape + (4,), dtype=np.float32)
            overlay[..., 0] = 1.0
            overlay[..., 3] = (m > 0).astype(np.float32) * 0.5
            ax.imshow(overlay, origin="lower")
        ax.set_title(f"{name} (mid)")
        ax.set_axis_off()
    fig.suptitle(sid + ("" if (mask is not None and mask.any()) else "  [no/empty mask]"))
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    if args.synthetic:
        from deepxube_hw4 import synthetic as src
        if args.random:
            rng = np.random.default_rng(args.seed)
            s, t1, mask, _ = src.load_random(rng=rng)
            sid = s.id
        else:
            t1, mask, _ = src.load_by_id(args.id)
            sid = args.id
    else:
        from deepxube_hw4 import atlas as src
        if args.random:
            rng = random.Random(args.seed)
            s, t1, mask, _ = src.load_random(rng=rng)
            sid = s.id
        else:
            t1, mask, _ = src.load_by_id(args.id)
            sid = args.id

    nvox = int((mask > 0).sum()) if mask is not None else 0
    print(f"{sid}: T1 shape={t1.shape}, lesion voxels={nvox}")
    if nvox == 0 and args.synthetic:
        print("warning: empty synthetic lesion — try a different seed/id.", file=sys.stderr)
    show(sid, t1, mask)


if __name__ == "__main__":
    main()
