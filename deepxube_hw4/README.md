# HW4 — Lesion Evolution as Heuristic Search

See `WRITEUP.md` for the full methodology, results, and progress log.

## Environment

Uses the `rlclass` conda env (Python 3.10, torch ≥ 2.0, nibabel, scikit-image,
matplotlib). Either activate it:

```sh
conda activate rlclass
```

…or prefix every command with the pinned interpreter:

```sh
/opt/homebrew/Caskroom/miniforge/base/envs/rlclass/bin/python -m ...
```

## Data — SOOP (OpenNeuro ds004889)

SOOP is a public acute-stroke DWI dataset (1715 subjects, 1461 with acute
ischemic stroke). We use the TRACE DWI volumes and the `derivatives/lesion_masks/`
native-space masks (acute, chronic, combined).

### One-time download

The loader expects everything under `deepxube_hw4/data/soop/`. Only the TRACE
DWI files and the lesion-mask derivatives are needed (not T1/FLAIR), which
keeps the download to roughly 1–2 GB.

```sh
pip install openneuro-py
cd deepxube_hw4/data  # directory is gitignored

# DWI TRACE volumes (skip T1/FLAIR/ADC/bold to save space):
openneuro-py download --dataset ds004889 --target-dir soop \
    --include 'sub-*/dwi/*rec-TRACE*'

# Lesion-mask derivatives in native TRACE space:
openneuro-py download --dataset ds004889 --target-dir soop \
    --include 'derivatives/lesion_masks/**'
```

Expected layout after download:

```
deepxube_hw4/data/soop/
├── sub-1/dwi/sub-1_rec-TRACE_dwi.nii.gz
├── sub-2/dwi/sub-2_rec-TRACE_dwi.nii.gz
├── ...
└── derivatives/lesion_masks/
    ├── sub-1/dwi/sub-1_space-TRACE_desc-lesion_mask.nii.gz
    ├── sub-1/dwi/sub-1_space-TRACE_desc-lesionAcute_mask.nii.gz
    └── sub-1/dwi/sub-1_space-TRACE_desc-lesionChronic_mask.nii.gz
```

Verify:

```sh
python -c "from deepxube_hw4.soop import list_subjects; \
           print(len(list_subjects()), 'subjects')"
```

Note: the `lesionAcute` and `lesionChronic` masks on a given subject are
independent lesions on the single scan, **not** paired timepoints (see the
SOOP paper, PMC11297183). We use `lesionAcute` as the training start state
and rely on the biological simulator (§3 of `WRITEUP.md`) for supervision.

## Quick start (after data download)

```sh
# 1. Supervised warm-start heuristic (~1 minute):
python -m deepxube_hw4.pretrain_heur --domain lesion_evo.sub-1.300 \
    --out deepxube_hw4/output_warm --n_traj 400 --epochs 10

# 2. 100-trial quantitative evaluation:
python -m deepxube_hw4.eval_heur --domain lesion_evo.sub-1.300 \
    --heur_dir deepxube_hw4/output_warm --n_trials 100

# 3a. Triplanar interactive viz with live Q-values:
python -m deepxube_hw4.viz_with_heur --domain lesion_evo.sub-1.300 \
    --heur_dir deepxube_hw4/output_warm --steps 10 --mode solve

# 3b. 3D volumetric viz (drag to rotate; n/p to step):
python -m deepxube_hw4.viz_3d --domain lesion_evo.sub-1.300 \
    --heur_dir deepxube_hw4/output_warm --steps 10 --stride 4

# 3c. Save a trajectory as a GIF:
python -m deepxube_hw4.viz_3d --domain lesion_evo.sub-1.300 \
    --heur_dir deepxube_hw4/output_warm --steps 10 --save_gif solve.gif
```

## Tests

```sh
conda run -n rlclass pytest tests/test_lesion_domain.py -q
```

## What's here

| File | Purpose |
|---|---|
| `soop.py` | OpenNeuro SOOP loader. |
| `parcels.py` | SLIC supervoxel parcellation + parcel adjacency. |
| `evolution.py` | `LesionEvolutionDomain` — state, actions, triplanar viz. |
| `simulator.py` | Biological forward simulator (+ calibration grid search). |
| `train_evolution.py` | Factory registration, NN input, `LesionEvoMLP`. |
| `pretrain_heur.py` | Supervised warm-start on simulator trajectories. |
| `train_launch.py` | Wrapper around `deepxube train`. |
| `viz_launch.py` | Wrapper around `deepxube viz`. |
| `viz_with_heur.py` | Triplanar viz with live Q-values, interactive + solve modes. |
| `viz_3d.py` | 3D volumetric viz of greedy-Q trajectories. |
| `eval_heur.py` | 100-trial solve rate + path-optimality eval. |
| `WRITEUP.md` | Full methodology, results, and progress log. |
