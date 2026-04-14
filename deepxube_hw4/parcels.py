"""SLIC supervoxel parcellation on native-DWI volumes (Option B, atlas-free)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from skimage.segmentation import slic


@dataclass
class Parcellation:
    labels: NDArray        # (X, Y, Z) int32, 0 = non-brain, 1..K = parcel IDs
    n_parcels: int
    centroids: NDArray     # (K, 3) float, voxel-space centroids (0-indexed over parcels 1..K)
    parcel_of_lesion_voxel: NDArray  # (K,) float, lesion-coverage fraction per parcel

    def lesion_parcel_set(self, mask: NDArray, coverage: float = 0.3) -> NDArray:
        """Parcel IDs (1..K) whose lesion coverage >= `coverage`."""
        frac = self.parcel_coverage(mask)
        return np.where(frac >= coverage)[0] + 1

    def parcel_coverage(self, mask: NDArray) -> NDArray:
        """(K,) fraction of each parcel's voxels that fall inside `mask`."""
        flat_lab = self.labels.ravel()
        flat_msk = mask.ravel().astype(bool)
        sizes = np.bincount(flat_lab, minlength=self.n_parcels + 1)[1:]
        hits = np.bincount(flat_lab, weights=flat_msk.astype(np.float32),
                           minlength=self.n_parcels + 1)[1:]
        return np.where(sizes > 0, hits / np.maximum(sizes, 1), 0.0)


def parcellate_dwi(
    dwi: NDArray, brain: NDArray, n_parcels: int = 300, compactness: float = 0.1,
) -> Parcellation:
    """SLIC supervoxels over brain voxels only. Background gets label 0."""
    vol = dwi.astype(np.float32)
    lo, hi = np.percentile(vol[brain], [1, 99])
    norm = np.clip((vol - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    labels = slic(
        norm, n_segments=n_parcels, compactness=compactness,
        mask=brain.astype(bool), channel_axis=None, start_label=1,
    ).astype(np.int32)
    labels[~brain.astype(bool)] = 0
    used = np.unique(labels)
    used = used[used > 0]
    # Remap to contiguous 1..K
    remap = np.zeros(labels.max() + 1, dtype=np.int32)
    remap[used] = np.arange(1, len(used) + 1, dtype=np.int32)
    labels = remap[labels]
    K = int(len(used))
    idx = np.argwhere(labels > 0)
    ids = labels[labels > 0]
    centroids = np.zeros((K, 3), dtype=np.float32)
    for pid in range(1, K + 1):
        centroids[pid - 1] = idx[ids == pid].mean(axis=0)
    return Parcellation(labels, K, centroids, np.zeros(K, dtype=np.float32))


def parcel_adjacency(labels: NDArray) -> NDArray:
    """(K+1, K+1) bool adjacency matrix (indexed by parcel ID, row/col 0 unused)."""
    K = int(labels.max())
    adj = np.zeros((K + 1, K + 1), dtype=bool)
    for axis in range(3):
        a = np.moveaxis(labels, axis, 0)
        left, right = a[:-1], a[1:]
        pairs = np.stack([left.ravel(), right.ravel()], axis=1)
        pairs = pairs[(pairs[:, 0] != pairs[:, 1]) & (pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
        adj[pairs[:, 0], pairs[:, 1]] = True
        adj[pairs[:, 1], pairs[:, 0]] = True
    return adj
