# HW4 — lesion-aware pathfinding (ATLAS v2.0)

Homework submission package for CSCE 775 HW4 extra credit. Kept as a sibling of
`deepxube/` so it can be zipped on its own without pulling the whole repo.

## 0. Synthetic fallback (no DUA required)

If you don't yet have ATLAS access, a synthetic cohort built on the single
MNI152 T1 (shipped with niivue demos) is available. Same loader API as the real
ATLAS module.

```sh
conda run -n rlclass python examples/atlas_preview.py --synthetic --random
conda run -n rlclass python examples/atlas_preview.py --synthetic --id syn0007
```

Override the base T1 path with `MNI152_PATH=...` if the niivue demo file moves.
Lesions are random ellipsoidal blobs seeded deterministically by subject id.

## 1. Real ATLAS v2.0 data access (one-time)

ATLAS v2.0 is public but requires a DUA.

1. Request a password at
   <https://fcon_1000.projects.nitrc.org/indi/retro/atlas_download.html>.
   The password is personal and non-shareable.
2. Download either the **OpenSSL** (14.4 GB) or **ccrypt** (10.6 GB) archive.
3. Decrypt + unpack. Expected post-unpack layout:

   ```
   $ATLAS_ROOT/
     Training/sub-rXXXsYYY/ses-1/anat/
         sub-rXXXsYYY_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz
     derivatives/ATLAS/sub-rXXXsYYY/ses-1/anat/
         sub-rXXXsYYY_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz
     Testing/       # no public masks
   ```

4. Export the root so the loader can find it:

   ```sh
   export ATLAS_ROOT=~/Data/atlas
   ```

   Default if unset: `~/Data/atlas`.

## 2. Environment

Uses the `rlclass` conda env (Python 3.10, torch, numpy, matplotlib, nibabel).
No extra packages required for this slice.

## 3. Sanity check

```sh
conda run -n rlclass python -c \
  "from deepxube_hw4.atlas import iter_subjects; print(sum(1 for _ in iter_subjects()))"
# expect ~655

conda run -n rlclass python examples/atlas_preview.py --random
conda run -n rlclass python examples/atlas_preview.py --id r001s001
```

The preview shows sagittal/coronal/axial mid-slices of the T1 with the lesion
mask overlaid in semi-transparent red.

## 4. What's next (not in this slice)

- Voxel pathfinding domain subclassing `StateGoalVizable` / `StringToAct`.
- Train/test problem instance generation (`.pkl`).
- Heuristic NN + training script.
- Solver + comparison against uniform cost search.
