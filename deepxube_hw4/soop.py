"""SOOP (OpenNeuro ds004889) loader: TRACE DWI + native-space lesion mask."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import nibabel as nib
import numpy as np
from numpy.typing import NDArray

DEFAULT_ROOT = Path(__file__).parent / "data" / "soop"
MaskKind = Literal["lesion", "lesionAcute", "lesionChronic"]


@dataclass
class SOOPSubject:
    sid: str
    dwi: NDArray         # (X, Y, Z) float32, TRACE DWI in native space
    mask: NDArray        # (X, Y, Z) uint8, lesion mask in same space
    voxel_mm: NDArray    # (3,) float32
    affine: NDArray      # (4, 4) float64


def _dwi_path(root: Path, sid: str) -> Path:
    return root / sid / "dwi" / f"{sid}_rec-TRACE_dwi.nii.gz"


def _mask_path(root: Path, sid: str, kind: MaskKind) -> Path:
    return (root / "derivatives" / "lesion_masks" / sid / "dwi"
            / f"{sid}_space-TRACE_desc-{kind}_mask.nii.gz")


def list_subjects(root: Path = DEFAULT_ROOT) -> List[str]:
    """Subjects with both TRACE DWI and a combined lesion mask."""
    mask_root = root / "derivatives" / "lesion_masks"
    if not mask_root.exists():
        return []
    sids = []
    for d in sorted(mask_root.iterdir()):
        if not d.name.startswith("sub-"):
            continue
        if _dwi_path(root, d.name).exists() and _mask_path(root, d.name, "lesion").exists():
            sids.append(d.name)
    return sids


def load_subject(
    sid: str, root: Path = DEFAULT_ROOT, kind: MaskKind = "lesion",
) -> SOOPSubject:
    dwi_img = nib.load(str(_dwi_path(root, sid)))
    mask_img = nib.load(str(_mask_path(root, sid, kind)))
    dwi = np.squeeze(np.asarray(dwi_img.get_fdata(), dtype=np.float32))
    mask = np.squeeze(np.asarray(mask_img.get_fdata())).astype(np.uint8)
    if dwi.shape != mask.shape:
        raise ValueError(f"{sid}: DWI {dwi.shape} vs mask {mask.shape}")
    voxel_mm = np.abs(np.diag(dwi_img.affine)[:3]).astype(np.float32)
    return SOOPSubject(sid, dwi, mask, voxel_mm, dwi_img.affine)
