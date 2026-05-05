"""Render a clean triplanar view of a SOOP subject with both lesion masks.

Supersedes the stale `soop_sub1_domain.png` left over from the removed
surgical-corridor framing. The old figure had unit-aspect sagittal/coronal
panels on a DWI volume whose through-plane voxel size is ~4–6× the in-plane
spacing, so those views looked horizontally squashed.

This version uses physical-unit extents + equal aspect + gridspec
width-ratios matched to voxel dimensions so every plane is anatomically
proportioned.

Usage:
    python -m deepxube_hw4.make_soop_triplanar --sid sub-1 \
        --out deepxube_hw4/soop_sub1_domain.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deepxube_hw4.soop import load_subject


def centroid(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return tuple(s // 2 for s in mask.shape)  # type: ignore[return-value]
    c = coords.mean(axis=0).round().astype(int)
    return int(c[0]), int(c[1]), int(c[2])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sid", default="sub-1")
    p.add_argument("--out", default="deepxube_hw4/soop_sub1_domain.png")
    args = p.parse_args()

    acute = load_subject(args.sid, kind="lesionAcute")
    chronic = load_subject(args.sid, kind="lesionChronic")

    dwi = acute.dwi
    mx, my, mz = acute.voxel_mm  # mm/voxel
    X, Y, Z = dwi.shape

    i, j, k = centroid(acute.mask | chronic.mask)

    vmax = float(np.percentile(dwi, 99.5))
    vmin = float(dwi.min())

    sag_w, sag_h = Y * my, Z * mz
    cor_w, cor_h = X * mx, Z * mz
    ax_w, ax_h = X * mx, Y * my

    total_w = sag_w + cor_w + ax_w
    total_h = max(sag_h, cor_h, ax_h)

    scale = 10.0 / total_w
    fig = plt.figure(figsize=(total_w * scale, total_h * scale * 1.15))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[sag_w, cor_w, ax_w], wspace=0.05,
    )
    ax_sag = fig.add_subplot(gs[0, 0])
    ax_cor = fig.add_subplot(gs[0, 1])
    ax_axi = fig.add_subplot(gs[0, 2])

    def show(ax, img, mask_a, mask_c, extent, title):
        ax.imshow(img.T, cmap="gray", origin="lower", extent=extent,
                  vmin=vmin, vmax=vmax, aspect="equal")
        for m, color in ((mask_a, (1.0, 0.45, 0.75)),   # acute = pink
                         (mask_c, (0.95, 0.15, 0.15))): # chronic = red
            if m.any():
                rgba = np.zeros(m.T.shape + (4,), dtype=np.float32)
                rgba[..., 0] = color[0]
                rgba[..., 1] = color[1]
                rgba[..., 2] = color[2]
                rgba[..., 3] = 0.55 * (m.T > 0)
                ax.imshow(rgba, origin="lower", extent=extent,
                          interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    show(ax_sag,
         dwi[i, :, :], acute.mask[i, :, :], chronic.mask[i, :, :],
         extent=(0, sag_w, 0, sag_h), title=f"sagittal  i={i}")
    show(ax_cor,
         dwi[:, j, :], acute.mask[:, j, :], chronic.mask[:, j, :],
         extent=(0, cor_w, 0, cor_h), title=f"coronal  j={j}")
    show(ax_axi,
         dwi[:, :, k], acute.mask[:, :, k], chronic.mask[:, :, k],
         extent=(0, ax_w, 0, ax_h), title=f"axial  k={k}")

    fig.suptitle(
        f"{args.sid}: DWI TRACE with lesion masks  "
        f"(pink = acute, red = chronic)",
        fontsize=12, y=0.98,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
