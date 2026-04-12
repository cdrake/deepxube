"""Load a NIfTI volume with nibabel and render it as an interactive 3D cuboid.

Drag in the matplotlib window to rotate; scroll to zoom.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

DEFAULT_FILE = (
    "/Users/chrisdrake/Dev/niivue/niivue/packages/niivue/demos/images/mni152.nii.gz"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", type=Path, default=Path(DEFAULT_FILE),
                   help="Path to a .nii or .nii.gz volume.")
    p.add_argument("--downsample", type=int, default=4,
                   help="Integer stride applied to each axis before rendering.")
    p.add_argument("--threshold", type=float, default=60.0,
                   help="Intensity percentile (0-100); voxels below are hidden.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    img = nib.load(str(args.file))
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    print(f"Loaded {args.file.name}: shape={data.shape} dtype={data.dtype}")

    f = max(1, args.downsample)
    data = data[::f, ::f, ::f]

    vmax = float(data.max())
    if vmax > 0:
        data = data / vmax

    cutoff = np.percentile(data, args.threshold)
    mask = data > cutoff
    print(f"Rendering {int(mask.sum())} / {mask.size} voxels "
          f"(downsample={f}, threshold p{args.threshold:g}).")

    # Grayscale RGBA, alpha scaled by intensity.
    colors = np.zeros(mask.shape + (4,), dtype=np.float32)
    colors[..., 0] = data
    colors[..., 1] = data
    colors[..., 2] = data
    colors[..., 3] = np.clip(data, 0.0, 1.0)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 8))
    ax.voxels(mask, facecolors=colors, edgecolor=None)
    # TODO: honor img.affine for anisotropic voxel sizes; for now use voxel-count aspect.
    ax.set_box_aspect(data.shape)
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.set_zlabel("k")
    ax.set_title(args.file.name)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
