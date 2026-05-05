"""Preview SOOP sub-1: TRACE DWI with lesion mask overlay."""
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent / "data" / "soop"
dwi = np.squeeze(nib.load(ROOT / "sub-1/dwi/sub-1_rec-TRACE_dwi.nii.gz").get_fdata())
mask = nib.load(
    ROOT / "derivatives/lesion_masks/sub-1/dwi/sub-1_space-TRACE_desc-lesion_mask.nii.gz"
).get_fdata()

print(f"DWI: {dwi.shape} {dwi.dtype}  mask: {mask.shape}  lesion voxels: {int(mask.sum())}")

zs = np.where(mask.any(axis=(0, 1)))[0]
if len(zs) == 0:
    zs = [dwi.shape[2] // 2]
slices = np.linspace(zs.min(), zs.max(), min(6, len(zs))).astype(int)

fig, axes = plt.subplots(1, len(slices), figsize=(3 * len(slices), 3))
if len(slices) == 1:
    axes = [axes]
for ax, z in zip(axes, slices):
    ax.imshow(dwi[:, :, z].T, cmap="gray", origin="lower")
    m = np.ma.masked_where(mask[:, :, z] == 0, mask[:, :, z])
    ax.imshow(m.T, cmap="autumn", alpha=0.5, origin="lower")
    ax.set_title(f"z={z}")
    ax.axis("off")
plt.tight_layout()
out = Path(__file__).parent / "soop_sub1_preview.png"
plt.savefig(out, dpi=120)
print(f"wrote {out}")
