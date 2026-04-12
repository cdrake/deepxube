"""Synthetic lesion dataset built on top of a single MNI152 T1.

Drop-in replacement for `deepxube_hw4.atlas` while real ATLAS v2.0 access is
pending. Lesions are random ellipsoidal blobs placed inside brain tissue
(intensity-thresholded region of the T1). Same return signature as
`atlas.load_subject` so downstream code doesn't care which source it came from.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import nibabel as nib
import numpy as np

MNI152_PATH = Path(os.environ.get(
    "MNI152_PATH",
    "/Users/chrisdrake/Dev/niivue/niivue/packages/niivue/demos/images/mni152.nii.gz",
))

DEFAULT_N = 50            # synthetic cohort size
BRAIN_PERCENTILE = 40.0   # voxels above this percentile are candidate lesion sites


@dataclass(frozen=True)
class Subject:
    id: str               # e.g. "sub-syn0007"
    ses: str              # always "ses-1" for synthetic
    t1_path: Path         # all synthetic subjects share the MNI152 T1
    mask_path: None = None
    seed: int = 0         # drives lesion generation


_T1_CACHE: tuple[np.ndarray, np.ndarray] | None = None  # (t1, affine)
_BRAIN_CACHE: np.ndarray | None = None                  # bool, same shape as t1


def _load_t1() -> tuple[np.ndarray, np.ndarray]:
    global _T1_CACHE
    if _T1_CACHE is None:
        img = nib.load(str(MNI152_PATH))
        t1 = np.asarray(img.get_fdata(), dtype=np.float32)
        _T1_CACHE = (t1, np.asarray(img.affine))
    return _T1_CACHE


def _brain_mask() -> np.ndarray:
    global _BRAIN_CACHE
    if _BRAIN_CACHE is None:
        t1, _ = _load_t1()
        thr = np.percentile(t1, BRAIN_PERCENTILE)
        _BRAIN_CACHE = t1 > thr
    return _BRAIN_CACHE


def _make_lesion(shape: tuple[int, int, int], seed: int) -> np.ndarray:
    """Random ellipsoidal lesion inside the brain mask."""
    rng = np.random.default_rng(seed)
    brain = _brain_mask()
    brain_idx = np.argwhere(brain)
    center = brain_idx[rng.integers(len(brain_idx))]

    radii = rng.integers(4, 18, size=3)  # voxels; ~8-36 mm for 1mm MNI
    # Random rotation via orthonormal basis from QR of a random matrix.
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))

    # Bounding box around the lesion to avoid materializing full-volume grid.
    pad = int(radii.max()) + 1
    lo = np.maximum(center - pad, 0)
    hi = np.minimum(center + pad + 1, shape)
    ii, jj, kk = np.meshgrid(
        np.arange(lo[0], hi[0]),
        np.arange(lo[1], hi[1]),
        np.arange(lo[2], hi[2]),
        indexing="ij",
    )
    pts = np.stack([ii - center[0], jj - center[1], kk - center[2]], axis=-1)
    local = pts @ q  # rotate
    ellipsoid = ((local / radii) ** 2).sum(axis=-1) <= 1.0

    mask = np.zeros(shape, dtype=np.uint8)
    sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sub[ellipsoid & brain[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]] = 1
    return mask


def iter_subjects(n: int = DEFAULT_N, start: int = 0) -> Iterator[Subject]:
    for i in range(start, start + n):
        yield Subject(id=f"sub-syn{i:04d}", ses="ses-1",
                      t1_path=MNI152_PATH, seed=i + 1)


def load_subject(
    subject: Subject,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t1, affine = _load_t1()
    mask = _make_lesion(t1.shape, subject.seed)
    return t1, mask, affine


def load_by_id(subject_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sid = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    # Parse trailing digits (sub-syn0007 -> seed 8).
    digits = "".join(ch for ch in sid if ch.isdigit())
    if not digits:
        raise KeyError(f"Synthetic subject id must end with digits: {sid!r}")
    idx = int(digits)
    return load_subject(Subject(id=sid, ses="ses-1",
                                t1_path=MNI152_PATH, seed=idx + 1))


def load_random(
    n: int = DEFAULT_N, rng: np.random.Generator | None = None,
) -> tuple[Subject, np.ndarray, np.ndarray, np.ndarray]:
    r = rng or np.random.default_rng()
    i = int(r.integers(n))
    s = Subject(id=f"sub-syn{i:04d}", ses="ses-1",
                t1_path=MNI152_PATH, seed=i + 1)
    t1, mask, affine = load_subject(s)
    return s, t1, mask, affine
