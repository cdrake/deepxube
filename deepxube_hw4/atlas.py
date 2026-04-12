"""ATLAS v2.0 stroke-lesion dataset loader.

Data is not bundled. Request access at
https://fcon_1000.projects.nitrc.org/indi/retro/atlas_download.html,
decrypt + unpack, then point ATLAS_ROOT at the unpacked directory.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import nibabel as nib
import numpy as np

ATLAS_ROOT = Path(os.environ.get("ATLAS_ROOT", Path.home() / "Data" / "atlas"))

T1_GLOB = "Training/sub-*/ses-*/anat/sub-*_T1w.nii.gz"
MASK_GLOB_TMPL = (
    "derivatives/ATLAS/{sub}/{ses}/anat/"
    "{sub}_{ses}_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
)


@dataclass(frozen=True)
class Subject:
    id: str               # e.g. "sub-r001s001"
    ses: str              # e.g. "ses-1"
    t1_path: Path
    mask_path: Path | None


def _resolve_root(root: Path | None) -> Path:
    r = Path(root) if root is not None else ATLAS_ROOT
    if not r.exists():
        raise FileNotFoundError(
            f"ATLAS root {r} does not exist. Set $ATLAS_ROOT or pass root=."
        )
    return r


def iter_subjects(root: Path | None = None) -> Iterator[Subject]:
    r = _resolve_root(root)
    for t1 in sorted(r.glob(T1_GLOB)):
        # .../Training/sub-XXX/ses-Y/anat/sub-XXX_ses-Y_..._T1w.nii.gz
        ses = t1.parents[1].name
        sub = t1.parents[2].name
        mask = r / MASK_GLOB_TMPL.format(sub=sub, ses=ses)
        yield Subject(id=sub, ses=ses, t1_path=t1,
                      mask_path=mask if mask.exists() else None)


def load_subject(
    subject: Subject,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    t1_img = nib.load(str(subject.t1_path))
    t1 = np.asarray(t1_img.get_fdata(), dtype=np.float32)
    affine = np.asarray(t1_img.affine)

    mask: np.ndarray | None = None
    if subject.mask_path is not None:
        m_img = nib.load(str(subject.mask_path))
        assert m_img.shape == t1_img.shape, (
            f"{subject.id}: mask shape {m_img.shape} != T1 {t1_img.shape}"
        )
        assert np.allclose(m_img.affine, t1_img.affine, atol=1e-4), (
            f"{subject.id}: mask/T1 affine mismatch"
        )
        mask = np.asarray(m_img.get_fdata(), dtype=np.uint8)
    return t1, mask, affine


def load_by_id(
    subject_id: str, root: Path | None = None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    sid = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    for s in iter_subjects(root):
        if s.id == sid:
            return load_subject(s)
    raise KeyError(f"Subject {subject_id} not found under {_resolve_root(root)}")


def load_random(
    root: Path | None = None,
    with_mask: bool = True,
    rng: random.Random | None = None,
) -> tuple[Subject, np.ndarray, np.ndarray | None, np.ndarray]:
    subjects = [s for s in iter_subjects(root)
                if (s.mask_path is not None) or not with_mask]
    if not subjects:
        raise RuntimeError("No subjects found (with_mask filter applied).")
    s = (rng or random).choice(subjects)
    t1, mask, affine = load_subject(s)
    return s, t1, mask, affine
