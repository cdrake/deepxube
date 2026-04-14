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

The `--domain` flag uses the format `lesion_evo.<subject_id>[.<n_parcels>]`,
e.g. `lesion_evo.sub-1.300` for SOOP subject 1 with 300 SLIC supervoxels. The
same subject id must be used for pretrain / eval / viz (the heuristic is
parcellation-specific).

### 1. Train the warm-start heuristic (~1 min on CPU)

```sh
python -m deepxube_hw4.pretrain_heur --domain lesion_evo.sub-1.300 \
    --out deepxube_hw4/output_warm --n_traj 400 --epochs 10
```

Writes `output_warm/heur.pt` and `heur_targ.pt`. Watch for final `mae` near
0.5 on cost-to-go.

### 2. Quantitative evaluation

```sh
python -m deepxube_hw4.eval_heur --domain lesion_evo.sub-1.300 \
    --heur_dir deepxube_hw4/output_warm --n_trials 100 --seed 1
```

Reports overall solve rate, mean path-optimality, and a per-`|s△g|` bucket
breakdown. Expected: ≈87% solved, 100% path-optimality on the solved subset
(`lb = |s△g|` is the unit-cost lower bound; `mean_opt = path_len / lb = 1.0`
means every solved instance found a shortest path).

### 3. Interactive visualisation

There are two independent viz frontends. Both take `--heur_dir`, a random-walk
goal length `--steps`, and an optional `--seed`.

#### 3a. Triplanar (axial/coronal/sagittal) with live Q-values

Best for debugging — you see the DWI slices and a ranked list of the
heuristic's top actions at each step.

```sh
# Interactive mode: type actions yourself, see top-6 Q-values before each step.
python -m deepxube_hw4.viz_with_heur --domain lesion_evo.sub-1.300 \
    --heur_dir deepxube_hw4/output_warm --steps 10 --mode interactive

# Solve mode: run greedy-Q to completion, step through the trajectory.
python -m deepxube_hw4.viz_with_heur --domain lesion_evo.sub-1.300 \
    --heur_dir deepxube_hw4/output_warm --steps 10 --mode solve
```

Keys in solve mode: `n` = next state, `p` = previous, `<int>` = jump to
index, `<Enter>` or `q` = quit. Overlay colors: cyan = acute mask,
lime = target/goal, red = current state, gray = DWI background.

#### 3b. 3D volumetric (matplotlib `ax.voxels`)

```sh
# Interactive: drag to rotate, scroll to zoom, n/p to step through trajectory.
python -m deepxube_hw4.viz_3d --domain lesion_evo.sub-1.300 \
    --heur_dir deepxube_hw4/output_warm --steps 10 --stride 4

# Save a GIF of the whole solve trajectory:
python -m deepxube_hw4.viz_3d --domain lesion_evo.sub-1.300 \
    --heur_dir deepxube_hw4/output_warm --steps 10 --save_gif solve.gif
```

Keys in interactive mode: `n`/`right` = next step, `p`/`left` = previous,
`home`/`end` = jump to first/last. Overlay colors are the same as the
triplanar viz. `--stride` downsamples the voxel grid before `ax.voxels`
(stride 4 ≈ 50k voxels rendered, which is the matplotlib sweet spot;
stride 2 is sharper but slow; stride 1 is unusable).

If solves are uninteresting (e.g. `|s△g|=1`), bump `--steps` (reverse random
walk length used to construct the goal) or try a different `--seed` until
you get a more visually informative trajectory.

### 4. Re-running DAVI (optional, currently a regression)

DAVI on top of the warm-start degraded performance in our setup — see
WRITEUP §4.5. To reproduce:

```sh
mkdir -p deepxube_hw4/output_davi
cp deepxube_hw4/output_warm/heur.pt deepxube_hw4/output_davi/
cp deepxube_hw4/output_warm/heur_targ.pt deepxube_hw4/output_davi/
python -m deepxube_hw4.train_launch --domain lesion_evo.sub-1.300 \
    --heur lesion_evo_mlp.512H_3L --heur_type QIn \
    --pathfind beam_q.1B_5.0T --step_max 15 --bal \
    --search_itrs 50 --up_itrs 50 --up_gen_itrs 50 \
    --batch_size 128 --up_batch_size 64 \
    --max_itrs 1000 --procs 1 --dir deepxube_hw4/output_davi
```

Then `eval_heur --heur_dir deepxube_hw4/output_davi` to compare.

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
