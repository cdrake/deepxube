"""Lesion-as-goal pathfinding viz pipeline (placeholder solver).

Loads MNI152, drops a fixed spherical lesion at a known location, downsamples,
picks a start voxel on the brain surface, and renders a 3D view of:
  - brain (translucent gray)
  - lesion (solid red, the goal)
  - start marker (green)
  - placeholder path (yellow): straight-line voxel walk start -> lesion centroid

Swap the `straight_line_path` call for a real deepxube solver later.

Usage:
    python examples/lesion_path_viz.py
    python examples/lesion_path_viz.py --downsample 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

DEFAULT_T1 = Path(
    "/Users/chrisdrake/Dev/niivue/niivue/packages/niivue/demos/images/mni152.nii.gz"
)

LESION_CENTER_FRAC = (0.65, 0.55, 0.55)  # right-frontal-ish, easy to spot
LESION_RADIUS_MM = 18.0
BRAIN_PERCENTILE = 40.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--t1", type=Path, default=DEFAULT_T1)
    p.add_argument("--downsample", type=int, default=4)
    return p.parse_args()


def make_lesion(shape: tuple[int, int, int], voxel_mm: np.ndarray) -> np.ndarray:
    center = np.array([int(f * s) for f, s in zip(LESION_CENTER_FRAC, shape)])
    radii_vox = LESION_RADIUS_MM / voxel_mm
    ii, jj, kk = np.indices(shape)
    d2 = (
        ((ii - center[0]) / radii_vox[0]) ** 2
        + ((jj - center[1]) / radii_vox[1]) ** 2
        + ((kk - center[2]) / radii_vox[2]) ** 2
    )
    return (d2 <= 1.0).astype(np.uint8)


def pick_start(brain: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Pick a brain-surface voxel roughly opposite the goal (for a visible path)."""
    brain_idx = np.argwhere(brain)
    centroid = brain_idx.mean(axis=0)
    direction = centroid - goal
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    # score = projection along direction from goal (farther = higher)
    scores = (brain_idx - goal) @ direction
    return brain_idx[int(np.argmax(scores))]


def straight_line_path(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Placeholder: dense voxel samples along the straight line start->goal."""
    n = int(np.linalg.norm(goal - start) * 2) + 2
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = start[None, :] * (1 - t) + goal[None, :] * t
    return np.round(pts).astype(int)


def render(
    brain: np.ndarray,
    lesion: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    path: np.ndarray,
    shape: tuple[int, int, int],
) -> None:
    path_mask = np.zeros(shape, dtype=bool)
    path_mask[path[:, 0], path[:, 1], path[:, 2]] = True
    path_mask &= ~lesion.astype(bool)  # let lesion color dominate at the goal

    # Show brain sparsely so it doesn't hide the interior (every-Nth voxel).
    brain_sparse = brain.copy()
    stride = 2
    keep = np.zeros_like(brain_sparse)
    keep[::stride, ::stride, ::stride] = 1
    brain_sparse = brain_sparse & keep.astype(bool)

    occ = brain_sparse | lesion.astype(bool) | path_mask
    colors = np.zeros(shape + (4,), dtype=np.float32)
    colors[brain_sparse] = [0.6, 0.6, 0.6, 0.08]
    colors[path_mask] = [1.0, 0.95, 0.1, 1.0]
    colors[lesion.astype(bool)] = [0.9, 0.1, 0.1, 0.9]

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(occ, facecolors=colors, edgecolor=None)
    ax.scatter(*start, c="lime", s=120, label="start", depthshade=False)
    ax.scatter(*goal, c="cyan", s=120, marker="^", label="goal (lesion centroid)",
               depthshade=False)
    ax.set_box_aspect(shape)
    ax.set_xlabel("i"); ax.set_ylabel("j"); ax.set_zlabel("k")
    ax.set_title("lesion-as-goal pathfinding (placeholder straight line)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    img = nib.load(str(args.t1))
    t1 = np.asarray(img.get_fdata(), dtype=np.float32)
    voxel_mm = np.abs(np.diag(img.affine)[:3]).astype(np.float32)
    print(f"Loaded {args.t1.name}: shape={t1.shape} voxel_mm={voxel_mm}")

    f = max(1, args.downsample)
    t1_ds = t1[::f, ::f, ::f]
    voxel_mm_ds = voxel_mm * f
    shape = t1_ds.shape

    brain = t1_ds > np.percentile(t1_ds, BRAIN_PERCENTILE)
    lesion = make_lesion(shape, voxel_mm_ds) & brain.astype(np.uint8)
    if lesion.sum() == 0:
        raise RuntimeError("Lesion ended up empty after brain masking — adjust center/radius.")

    goal = np.argwhere(lesion).mean(axis=0).round().astype(int)
    start = pick_start(brain, goal)
    path = straight_line_path(start, goal)

    print(f"downsampled shape={shape}  brain vox={int(brain.sum())}  "
          f"lesion vox={int(lesion.sum())}")
    print(f"start={tuple(start)}  goal={tuple(goal)}  path len={len(path)}")

    render(brain, lesion, start, goal, path, shape)


if __name__ == "__main__":
    main()
