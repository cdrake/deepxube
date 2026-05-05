# HW4 — Lesion Evolution as Heuristic Search

Acute stroke imaging captures the injury at a single timepoint, but clinicians
need to know what the lesion **will become** once tissue at risk either
resolves (edema) or infarcts (penumbral completion). This project casts that
forward-evolution problem as **goal-conditioned heuristic search on a
parcellated brain** and solves it with a learned warm-start heuristic on top
of the DeepXube framework.

![SOOP sub-1 DWI with acute and chronic lesion masks](soop_sub1_domain.png)

*Subject 1 DWI TRACE with acute (pink) and chronic (red) lesion masks. The
acute mask is the search start state; the goal is either a simulated chronic
state (training) or an external target (inference).*

## Why it matters

**Clinical novelty.**
- Forward acute-to-chronic lesion evolution treated as search, not image-to-image
  regression — every trajectory is auditable step-by-step.
- State is a set of **SLIC supervoxels** (constrained k-means over position +
  intensity, ≈285 parcels/subject), not voxels or puzzle tiles — the search
  operates at a clinically meaningful granularity.
- SOOP's `lesionAcute` / `lesionChronic` masks are independent lesions on a
  *single* scan, not paired timepoints (see PMC11297183). A calibrated
  biological simulator supplies supervision where the dataset cannot — matched
  against published priors for mean acute/chronic ratio and trajectory length.

**DeepXube novelty.**
- First DeepXube domain on real medical imaging — prior domains were
  combinatorial puzzles (Rubik's cube, sliding tile, etc.).
- Subject-specific **variable-size action spaces**: EXPAND / SHRINK / STOP
  over a parcel-adjacency graph that differs per subject, instead of a fixed
  permutation action set.
- **Frozenset-equality goals** instead of positional equality — a new
  state/goal matching regime for the framework.

![SLIC parcellation with lesion overlay](soop_sub1_parcels.png)

*Subject-specific SLIC parcellation (K=285). Each state is a frozenset of
active parcel indices; actions are EXPAND / SHRINK over parcels adjacent to
the current boundary.*

## Prerequisites

Uses the `rlclass` conda env (Python 3.10, torch ≥ 2.0, nibabel,
scikit-image, matplotlib). Either activate it or prefix commands with the
pinned interpreter:

```sh
conda activate rlclass
# …or…
/opt/homebrew/Caskroom/miniforge/base/envs/rlclass/bin/python -m ...
```

`deepxube_hw4/data/`, `output_*/`, `results/`, and `instances/` are all
gitignored — regenerate them with the commands below.

### One-time SOOP data download (≈1–2 GB)

SOOP is a public acute-stroke DWI dataset (ds004889, 1715 subjects). We use
the TRACE DWI volumes and the native-space lesion-mask derivatives only.

```sh
pip install openneuro-py
cd deepxube_hw4/data

openneuro-py download --dataset ds004889 --target-dir soop \
    --include 'sub-*/dwi/*rec-TRACE*'

openneuro-py download --dataset ds004889 --target-dir soop \
    --include 'derivatives/lesion_masks/**'
```

Expected layout:

```
deepxube_hw4/data/soop/
├── sub-1/dwi/sub-1_rec-TRACE_dwi.nii.gz
├── …
└── derivatives/lesion_masks/
    └── sub-1/dwi/sub-1_space-TRACE_desc-lesion{,Acute,Chronic}_mask.nii.gz
```

Verify:

```sh
python -c "from deepxube_hw4.soop import list_subjects; print(len(list_subjects()), 'subjects')"
```

## 5-minute walkthrough

The `--domain` flag is `lesion_evo.<subject>[.<n_parcels>]`, e.g.
`lesion_evo.sub-1.300`. The same subject id must be used for
pretrain / eval / viz (the heuristic is parcellation-specific).

### 1. Train the warm-start heuristic (~1 min, CPU)

```sh
python -m deepxube_hw4.pretrain_heur --domain lesion_evo.sub-1.300 \
    --out deepxube_hw4/output_warm --n_traj 400 --epochs 10
```

**Expect:** `output_warm/heur.pt` + `heur_targ.pt`, final `mae ≈ 0.48` on
cost-to-go ∈ [0, 26].

### 2. Quantitative eval (100 trials)

```sh
python -m deepxube_hw4.eval_heur --domain lesion_evo.sub-1.300 \
    --heur_dir deepxube_hw4/output_warm --n_trials 100 --seed 1
```

**Expect:** `Trials: 100 solved: 87/100 = 87.0%`, `Mean optimality (solved) =
1.000` — i.e. every solved instance found a shortest path (`path_len = |s△g|`,
the unit-cost lower bound).

### 3. UCS vs. learned heuristic benchmark

This is the headline comparison — classical uniform-cost search versus the
warm-start heuristic on matched easy/hard splits.

```sh
# 3a. Generate 30 easy + 30 hard frozen instances:
python -m deepxube_hw4.make_instances --domain lesion_evo.sub-1.300 \
    --out deepxube_hw4/instances --n_easy 30 --n_hard 30

# 3b. Run both solvers (UCS = graph_q weight=0, Heur = graph_q weight=1),
#     15 s/instance budget:
python -m deepxube_hw4.run_experiments --domain lesion_evo.sub-1.300 \
    --heur_dir deepxube_hw4/output_warm \
    --instances_dir deepxube_hw4/instances \
    --results_dir deepxube_hw4/results

# 3c. Render the two benchmark figures from results/:
python -m deepxube_hw4.make_figures
```

**Expect:** `results/{easy,hard}_{ucs,heur}/results.pkl` populated, then
`fig_solve_rate.png` and `fig_nodes_scatter.png` overwritten in the HW4
directory. Summary table at step 3b should show UCS solving ~37% easy / 0%
hard, heuristic 100% easy / 70% hard.

### 4. Triplanar viz with live Q-values

```sh
# Solve mode: greedy-Q runs to completion, step through the trajectory.
python -m deepxube_hw4.viz_with_heur --domain lesion_evo.sub-1.300 \
    --heur_dir deepxube_hw4/output_warm --steps 10 --mode solve

# Interactive mode: type actions yourself, see top-6 Q-values each step.
python -m deepxube_hw4.viz_with_heur --domain lesion_evo.sub-1.300 \
    --heur_dir deepxube_hw4/output_warm --steps 10 --mode interactive
```

Solve-mode keys: `n` next, `p` previous, `<int>` jump, `q` / Enter quit.
Overlay colors: cyan = acute mask, lime = target goal, red = current active
parcels, green outline = frontier being scored.

### 5. 3D volumetric viz (optional)

```sh
# Interactive (drag rotate, scroll zoom, n/p step):
python -m deepxube_hw4.viz_3d --domain lesion_evo.sub-1.300 \
    --heur_dir deepxube_hw4/output_warm --steps 10 --stride 4

# Save a shareable GIF of the full solve:
python -m deepxube_hw4.viz_3d --domain lesion_evo.sub-1.300 \
    --heur_dir deepxube_hw4/output_warm --steps 10 --save_gif solve.gif
```

`--stride 4` gives ≈50k rendered voxels (matplotlib's `ax.voxels` sweet
spot); stride 2 is sharper but slow; stride 1 is unusable.

## Results at a glance

> The benchmark figures below were produced with the **unit-cost baseline**
> (`output_warm/`). The **bio-cost variant** (`output_warm_bio/`, §4.6 of
> `WRITEUP.md`) is a newer extension with clinically-weighted step costs —
> see the "Bio-cost extension" section below and `WRITEUP.md` §5.3 for
> 100-trial numbers with that variant.

### Warm-start heuristic, 100-trial eval (seed=1, reverse-walk ∈ [2, 15])

```
Trials: 100  solved: 87/100 = 87.0%
Mean optimality (path/lb, solved only): 1.000   (lb = |s△g|)
```

| `|s△g|` bucket | solved / n | mean path length |
|---|---|---|
| 1–4   | 17 / 17 | = lb exactly |
| 5–8   | 37 / 37 | = lb exactly |
| 9–12  | 23 / 30 | exact when solved |
| 13–15 | 11 / 16 | exact when solved |

**Every solved instance finds an optimal path.** Failures concentrate at
`|s△g| ≥ 9`, where greedy-Q enters local minima.

### UCS vs. learned heuristic (30 easy + 30 hard, 15 s/instance)

![Solve rate, UCS vs. heuristic](fig_solve_rate.png)

*UCS solves 37% of easy and 0% of hard within budget. The learned heuristic
solves 100% of easy and 70% of hard.*

![Per-instance nodes expanded](fig_nodes_scatter.png)

*On instances both solvers handle (blue, easy), every point sits below y = x
— the heuristic is uniformly, not just on-average, more efficient. Green
triangles on the right edge are instances the heuristic solved while UCS
timed out.*

![Interactive Q-value viz](viz_stepthrough.png)

*Mid-search frame from `viz_with_heur.py`. Cyan = acute start, lime = target
goal, red = current parcels, green outline = frontier being scored.*

### Bio-cost extension

Each `EXPAND` / `SHRINK` now costs between `0.5` (biologically plausible —
dim low-coverage SHRINK = edema resolution, bright EXPAND = penumbra
completion) and `2.0` (implausible — would require un-infarcting core or
seeding new infarct in dim cortex). STOP stays at `1.0`.

`pretrain_heur.py` takes a `--lambda_len` mix factor so the training target
is a weighted sum of length-to-go and bio-cost-to-go:

```
target = λ · len_remaining + (1 − λ) · cost_remaining
```

- `--lambda_len 0.0` → pure bio-cost (minimum-biological-cost path)
- `--lambda_len 1.0` → pure length (minimum-step path, unit-cost baseline)
- `--lambda_len 0.5` → balanced (recommended, see table below)

Recommended run:

```sh
python -m deepxube_hw4.pretrain_heur --domain lesion_evo.sub-1.300 \
    --out deepxube_hw4/output_warm_mo_05 --n_traj 400 --epochs 10 \
    --lambda_len 0.5
python -m deepxube_hw4.eval_heur --domain lesion_evo.sub-1.300 \
    --heur_dir deepxube_hw4/output_warm_mo_05 --n_trials 100 --seed 1
```

**λ sweep (100 trials, seed=1, current K=285 parcellation):**

| λ_len | solved | len_opt | cost_opt (floor=1.0) |
|-------|-------:|--------:|---------------------:|
| 0.0 (pure cost)   | 70 %   | 1.000 | 2.97 |
| 0.3               | 62.6 % | 1.000 | 2.93 |
| **0.5 (balanced)** | **72 %** | **1.000** | **2.93** |
| 0.7               | 69 %   | 1.000 | 3.05 |
| 1.0 (pure length) | 62 %   | 1.000 | 2.94 |

Two findings:
- **Solve rate peaks at λ=0.5** (72 %) — both endpoints underperform the
  balanced objective. The length signal provides coarse progress gradient,
  the cost signal provides fine discrimination among similar continuations.
- **Cost-optimality is ≈ constant (~2.9×) regardless of λ.** Scalar
  weighted-sum targets don't induce cost-seeking behavior. Hypothesis
  at the time: the length term dominates greedy-Q argmin because its
  range is ~10× wider than the cost term. The dual-head experiment
  below tests and refutes that hypothesis.

### Dual-head Q net (methodological follow-up)

To test whether the flat cost-optimality was caused by the scalar
target conflating the two signals, we trained one dual-head
checkpoint (`LesionEvoMLP(out_dim=2)` predicting
`[length_to_go, cost_to_go]`) and combined the heads at inference:

```sh
python -m deepxube_hw4.pretrain_heur --domain lesion_evo.sub-1.300 \
    --out deepxube_hw4/output_warm_dual --n_traj 400 --epochs 8 \
    --dual_head
for L in 0.0 0.3 0.5 0.7 1.0; do
  python -m deepxube_hw4.eval_heur --domain lesion_evo.sub-1.300 \
      --heur_dir deepxube_hw4/output_warm_dual --n_trials 100 \
      --dual_head --lambda_len "$L"
done
```

| λ_len (inference) | solved | len_opt | co_loose | co_tight |
|-------------------|-------:|--------:|---------:|---------:|
| 0.0 (pure cost)   | 64 %   | 1.000   | 2.94     | **1.000** |
| 0.5               | 75 %   | 1.004   | 2.99     | **1.003** |
| 1.0 (pure length) | 60 %   | 1.000   | 3.00     | **1.000** |

Separating the heads gives comparable solve rates but leaves `co_loose`
flat, *including at λ=0.0 where only the cost head drives argmin*.
This falsifies the "length drowns cost" hypothesis: the admissible LB
(`COST_MIN · |s△g|`) is the loose term — it assumes every required
parcel flip costs the minimum. Replacing it with
`co_tight = path_cost / Σ_{p∈s△g} cost(a_p)` (sum of the exact bio-costs
of the parcels that must flip) gives **1.000 at every λ**: the model
already takes no wasted moves on solved instances; the remaining
gap vs. ideal is solve rate on hard instances, not path cost. See
WRITEUP.md §5.5.

> Note: the 87 % solve rate reported for `output_warm/` in §5.1 of the
> WRITEUP was measured on the **pre-fix K=288 parcellation** (before §2.1's
> brain-mask connectivity fix) and is no longer reproducible; the current
> K=285 runs above are the live reference.

## Code map

| File | Purpose |
|---|---|
| `soop.py` | OpenNeuro SOOP loader (DWI TRACE + native masks). |
| `parcels.py` | SLIC supervoxel parcellation + parcel adjacency. |
| `evolution.py` | `LesionEvolutionDomain` — state, actions, triplanar viz. |
| `simulator.py` | Biological forward simulator + population-prior calibration. |
| `train_evolution.py` | Factory registration, NN input, `LesionEvoMLP`. |
| `pretrain_heur.py` | Supervised warm-start on simulator trajectories. |
| `eval_heur.py` | 100-trial solve rate + path-optimality eval. |
| `make_instances.py` | Generate frozen easy/hard benchmark instances. |
| `run_experiments.py` | UCS vs. heuristic head-to-head via `deepxube solve`. |
| `make_figures.py` | Render `fig_solve_rate.png` and `fig_nodes_scatter.png`. |
| `viz_with_heur.py` | Triplanar viz with live Q-values (interactive / solve). |
| `viz_3d.py` | 3D volumetric viz of greedy-Q trajectories. |
| `train_launch.py`, `viz_launch.py`, `solve_launch.py` | Thin CLI wrappers around `deepxube train` / `viz` / `solve`. |
| `tests/test_lesion_domain.py` | Smoke tests (skipped if SOOP sub-1 absent). |

## Where to go deeper

- **Simulator calibration** — `WRITEUP.md` §3 derives the shrink/expand
  Bernoulli weights and grid-searches against published priors.
- **DAVI failure post-mortem** — `WRITEUP.md` §4 documents why cold DAVI and
  refinement-on-warm-start both regressed (`87% → 31%` solve rate), and what
  would fix it (lower search temperature, wider beam, frozen-heuristic policy
  head).
- **Pivot history** — `WRITEUP.md` §6 tracks the original surgical-corridor
  direction, the SOOP paired-timepoint audit that forced the forward-evolution
  reframing, and the brain-mask connectivity fix (§2.1).

## Tests

```sh
conda run -n rlclass pytest tests/test_lesion_domain.py -q
```
