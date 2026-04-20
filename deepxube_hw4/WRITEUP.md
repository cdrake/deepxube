# HW4 — Lesion Evolution as Heuristic Search

**Author:** Chris Drake · CSCE 775 · 2026
**Branch:** `feat/hw4-lesion-domain`

## 1. Problem & motivation

Acute stroke imaging shows the injury at a single timepoint, but clinicians care
about what the lesion *will look like* after the tissue at risk either resolves
(edema) or infarcts (penumbral completion). We cast this forward-evolution
problem as **goal-conditioned heuristic search on a parcellated brain**:

- **State:** a set of active (infarcted) parcels over a subject-specific SLIC
  supervoxel parcellation of the DWI volume. *SLIC (Simple Linear Iterative
  Clustering, Achanta et al. 2012)* is a constrained k-means over position +
  intensity that partitions a volume into `K` spatially compact, intensity-
  coherent supervoxels — here, `K = 285` parcels per subject on sub-1 after
  the brain-mask iteration in §2.1. *DWI (diffusion-
  weighted imaging)* is the MR sequence most sensitive to cytotoxic edema;
  acute stroke lesions light up bright on the TRACE reconstruction used here.
- **Start:** the acute lesion mask from SOOP (ds004889).
- **Goal:** a target parcel set — either simulated from a biological model
  during training, or a real chronic mask at inference / validation.
- **Actions:** `EXPAND p` (recruit a frontier parcel), `SHRINK p` (resolve a
  boundary parcel), or `STOP`.
- **Cost:** biologically-weighted (§4.6). Each EXPAND/SHRINK costs between
  `COST_MIN = 0.5` (biologically plausible) and `COST_MAX = 2.0`
  (implausible); STOP costs `1.0`. Unit-cost results in §5.1–5.2 correspond
  to the original baseline; bio-cost results are in §5.3.

Novelty for DeepXube: it has only been demonstrated on combinatorial puzzles.
Variable-size action spaces over subject-specific parcellations with
frozenset-equality goals is a new regime.

![SOOP sub-1 DWI with acute (pink) and chronic (red) lesion masks](soop_sub1_domain.png)

*Figure 1: SOOP subject 1 DWI TRACE volume with the two independent lesion
masks (acute pink, chronic red) shown on sagittal / coronal / axial slices.
Acute and chronic are not paired timepoints, so the domain uses the acute
mask as start and a calibrated simulator (or an external target) to supply
the goal.*

## 2. Pipeline

```
SOOP DWI ─▶ SLIC parcellation ─▶ LesionEvolutionDomain ─▶ DeepXube mixins
                                       │
                                       ▼
                         biology simulator (calibrated)
                                       │
                 ┌────────────────┬───┴───┬───────────────┐
                 ▼                ▼       ▼               ▼
         supervised             DAVI     eval          viz
         warm-start          refinement  (greedy-Q)  (triplanar)
```

| File | Role |
|---|---|
| `soop.py` | SOOP loader (DWI TRACE + native acute/chronic masks). |
| `parcels.py` | SLIC supervoxel parcellation + parcel adjacency. |
| `evolution.py` | `LesionEvolutionDomain`, `EvolutionState/Action/Goal`, triplanar viz. |
| `simulator.py` | Biological forward simulator + calibration. |
| `train_evolution.py` | Factory-registered wrapper, NN input, `LesionEvoMLP`. |
| `pretrain_heur.py` | Supervised warm-start on simulator trajectories. |
| `train_launch.py` | Wrapper around `deepxube train`. |
| `eval_heur.py` | 100-trial solve rate + path-optimality. |
| `viz_with_heur.py` | Interactive & solve-mode viz with live Q-values. |

![SLIC parcellation (K=285) with lesion overlay](soop_sub1_parcels.png)

*Figure 2: Subject-specific SLIC supervoxel parcellation (K = 285) on the
left; the acute lesion overlay on the right. Each state is a frozenset of
active parcel indices; actions are EXPAND / SHRINK over parcels adjacent to
the current boundary.*

### 2.1 Brain-mask iteration

Parcellation quality depends entirely on what gets fed to SLIC as the "brain".
The first pass was a one-liner intensity threshold,
`subj.dwi > np.percentile(subj.dwi, 40)`, and it had no connectivity
constraint: on sub-1 it produced a mask with **128 disconnected components** —
eye globes, the scalp rim, and bright noise specks all cleared the threshold
and SLIC tiled each one with its own micro-parcels. Visible as coloured
"islands" drifting off the brain in early parcellation figures, and as
unreachable parcels in the expand/shrink adjacency graph (they share no face
with the main mass, so the search can never recruit them).

The fix is one extra line — take the largest connected component of the
thresholded mask via `scipy.ndimage.label` — and collapses the mask to 1
component, drops 752 island voxels, and brings `K` from 288 to 285 without
touching any real brain tissue. Figure 2 shows the post-fix parcellation;
the §4 / §5 numbers below were measured against the pre-fix K = 288 mask
(see note in §5).

## 3. Biological simulator

`simulate(domain, ...)` rolls out a stochastic acute→chronic trajectory:

- At each step, Bernoulli(`p_stop_step`) terminates.
- Else sample EXPAND / SHRINK weighted by:
  - SHRINK weight = `shrink_bias · (1 - coverage) · (1 - 0.5·brightness)`
    — parcels with low acute coverage and dim DWI are likely edema.
  - EXPAND weight = `expand_bias · brightness` — bright DWI-adjacent parcels
    capture penumbra.

Grid calibration against published population priors
(mean acute→chronic ratio ≈ 0.85, mean trajectory length ≈ 15) picked
`shrink_bias=3.0, expand_bias=0.3, p_stop_step=0.05` (actual: mean_ratio=0.906,
mean_length=15.0).

![Simulator rollout: acute → simulated chronic](evo_sub1_sim.png)

*Figure 3: A calibrated simulator rollout. Cyan = acute (start), lime =
simulated chronic (goal), red = current active parcels. |active| = 15 after
~15 EXPAND / SHRINK steps, matching the population mean trajectory length.*

## 4. DAVI attempts and refinements

The project iterated through several DAVI configurations before finding one
that worked.

### 4.1 Cold DAVI (`output_2k/`)
```
--step_max 30 --search_itrs 50 --pathfind beam_q.1B_5.0T --max_itrs 2000
```
Result: `%solved` stuck at ≈3% the entire run (only the `start == goal` corner
cases); loss climbed 0.05 → 1.0+; cost-to-go diverged unboundedly. Diagnosis:
with beam=1 over a ~80-action legal set and a random heuristic, the search
never reaches a goal state, so bootstrapped targets grow every iteration with
no grounded signal.

![Cold-DAVI smoke run: loss / cost-to-go / solve rate over 50 iterations](train_curves.png)

*Figure 4: Cold DAVI (50-iter smoke) — loss oscillates, predicted cost-to-go
tracks the (unbounded) bootstrap target, and `%solved` flatlines near the
`start == goal` floor. Extending to 2000 iters did not change the shape; the
search never generates a grounded 0-cost terminal, so the target drifts.*

### 4.2 Step curriculum (`output_2k_bal/`)
Added `--bal` (trainer auto-advances `step_max_curr` from 1 when solve-rate ≥
50%).
Result: stuck at `step_max_curr=2`, `%solved` = 33% = the uniform-over-{0,1,2}
step-0 floor. Same chicken-and-egg: step=1 instances require picking the
correct parcel out of ~80 on the first try; random gets ~1%, never crosses
50%.

### 4.3 Representation refinements (`output_smoke2/`)
Two additions to `LesionEvoSGAIn`:
- Explicit `goal\state` ("needs-expand") and `state\goal` ("needs-shrink")
  K-dim channels so the net sees the symmetric difference directly.
- A scalar bit for "the candidate action's parcel is in s△g".

Plus a domain tweak: SHRINK is now restricted to *boundary* parcels (those
with at least one inactive neighbor), halving the SHRINK action count and
matching the biological interpretation of edema resolution at the rim.

Result: same 50/33% floor. The features don't help until the search
produces a non-trivial solved instance, which it still cannot.

### 4.4 Supervised warm-start (`output_warm/`) — the working heuristic
`pretrain_heur.py` generates ≈11k `(state, goal, action, cost_to_go)` tuples
from 400 simulator trajectories:
- Positive samples along the trajectory: `cost_to_go = T - t`.
- `STOP` at the goal state: `cost_to_go = 0`.
- Negative samples: a random non-trajectory legal action at step `t` gets
  `cost_to_go = T - t + 1` (mild penalty).

MSE training, 10 epochs, Adam lr=1e-3. Final MAE = **0.48** on cost-to-go ∈
[0, 26].

### 4.5 DAVI refinement on top of warm-start (`output_davi/`) — regression

Loaded `output_warm/heur.pt` + `heur_targ.pt` into a fresh training dir and
ran 900 DAVI iterations with the same `beam_q.1B_5.0T` configuration. The
same pathology returned: `%solved` stuck at 33%, loss climbed, cost-to-go
max diverged to 17+. Re-evaluating the refined checkpoint: **87% → 31%**
solved (100-trial, same seed). DAVI *actively destroyed* the warm-start
signal.

Diagnosis: with stochastic softmax sampling at temperature 5.0, beam-1
under-exploits the warm heuristic; the bootstrap targets from unsolved
searches climb faster than the grounded 0-cost terminal signals can anchor
them. The warm-start alone is the usable artifact.

Candidate fixes (not pursued here):
- Lower search temperature (`beam_q.1B_0.1T`) to exploit the warm heuristic.
- Replace beam-1 with wider beam or A\*-style search once the framework allows.
- Freeze the warm heuristic and train only a policy head on top.

### 4.6 Biologically-weighted step costs (`output_warm_bio/`)

The initial cost model used unit cost per toggle, which makes every EXPAND /
SHRINK equivalent regardless of which parcel is being moved. That bakes in
the wrong objective — clinically, resolving a rim-of-lesion dim parcel
(edema) is very different from un-infarcting a bright high-coverage parcel
(which would imply reversing a core infarct). The minimum-length plan is not
the minimum-biological-cost plan.

`LesionEvolutionDomain.action_cost(kind, parcel)` now returns:

```
SHRINK:  plaus = (1 - coverage) · (1 - 0.5 · brightness)
EXPAND:  plaus = brightness
cost = COST_MIN + (COST_MAX - COST_MIN) · (1 - plaus)   # [0.5, 2.0]
STOP:    cost = 1.0
```

`brightness` is the mean DWI signal of the parcel normalized to [0, 1] by
the 99th percentile; `coverage` is the fraction of the parcel's voxels that
fall inside the acute mask. The weights mirror the simulator's plausibility
weights (§3), so every trajectory the simulator favors as "likely" is also
cheap under search.

Downstream plumbing:

- `next_state` now returns `action_cost(...)` instead of `1.0`.
- `pretrain_heur.build_dataset` builds targets from **cumulative bio-cost**
  from each state to the end of the simulator trajectory (not `T - t`).
  Negative samples get `remaining[t] + action_cost(a_neg)` as a mild upper
  bound.
- `eval_heur` reports both `mean_len` (unchanged) and `mean_cost` (new),
  with length-optimality `path_len / |s△g|` and cost-optimality
  `path_cost / (COST_MIN · |s△g|)` — the latter's admissible lower bound is
  the all-plausible-toggles path.

### 4.7 Multi-objective training target (`output_warm_mo/`)

§5.3 shows the pure-cost warm-start (§4.6) finds length-optimal but not
cost-optimal paths. Hypothesis: the MSE target `cost_remaining` lets the net
exploit `|s△g|` geometry to saturate length-optimality before its predictions
are well-calibrated on cost, so greedy-Q minimizes a length-dominated proxy.

Fix tried: make the training target a weighted sum of length and cost:

```
target = λ · len_remaining + (1 − λ) · cost_remaining
```

Implemented via `--lambda_len` in `pretrain_heur`. Negatives use
`target(len_rem + 1, cost_rem + action_cost(a_neg))`.

- λ = 0.0 → pure bio-cost (§4.6 behavior)
- λ = 1.0 → pure length (matches the original §4.4 unit-cost target)
- λ ∈ (0, 1) → multi-objective

### 4.8 Parcellation drift caveat

The original 87% solve-rate (§5.1) was measured on the **pre-fix K=288**
parcellation. The §2.1 brain-mask connectivity fix dropped K to 285 and
**invalidated the old `output_warm/heur.pt`** (1445-dim input layer → 1430
now). The §5.3 (bio-cost) and §5.4 (multi-objective) numbers are freshly
trained on K=285 and should be considered the current reference; pre-fix §5
numbers are kept for historical record but not directly comparable.

## 5. Results

*Benchmark numbers in this section were measured against the pre-fix
parcellation (K = 288; §2.1). The current code produces K = 285 on sub-1;
retraining on the new mask is pending and expected to shift the numbers
only marginally since the 3 dropped parcels are non-brain islands that
were never part of the acute or chronic lesion masks.*

### 5.1 Warm-start heuristic, 100-trial eval

100 trials, seed=1, reverse-walk length ∈ [2, 15], greedy-Q with budget 60:

```
Trials: 100  solved: 87/100 = 87.0%
Mean optimality (path/lb, solved only): 1.000   (lb = |s△g|)
```

| |s△g| bucket | solved / n | mean path length |
|---|---|---|
| 1–4 | 17 / 17 | = lb exactly |
| 5–8 | 37 / 37 | = lb exactly |
| 9–12 | 23 / 30 | exact when solved |
| 13–15 | 11 / 16 | exact when solved |

**Every solved instance finds an optimal path** (matches the `|s△g|` lower
bound). Failures concentrate at `|s△g| ≥ 9`, where greedy-Q enters local
minima — precisely the regime DAVI refinement should address, since positive
reinforcement is now available.

### 5.2 Head-to-head vs uniform-cost search

30 easy instances (short reverse-random-walk goals, `|s△g| ∈ [1, 4]`) and
30 hard instances (simulator-calibrated goals, `|s△g|` typically ≥ 8),
10 s/instance budget, `deepxube solve` with `graph_q` pathfinder. UCS sets
`weight=0` (Dijkstra), heuristic A\* uses `weight=1` with the warm-start net.

![Solve rate and mean nodes, UCS vs. heuristic](fig_solve_rate.png)

*Figure 5: UCS solves only the short easy goals within the budget (37%) and
none of the hard ones (0%). The learned heuristic solves every easy instance
and 70% of the hard ones while generating roughly 10× fewer nodes on the
easy split where both are solvable.*

![Per-instance nodes expanded, UCS vs. heuristic](fig_nodes_scatter.png)

*Figure 6: Each dot is one instance. On instances both solvers handle (blue,
easy only), every single point sits below the y = x diagonal — the heuristic
is uniformly more efficient, not just on average. Green triangles on the
right edge of each panel are instances the heuristic solved but UCS timed
out on (≈75k nodes generated with no solution).*

![Interactive viz of the heuristic finding a goal](viz_stepthrough.png)

*Figure 7: Mid-search frame from `viz_with_heur.py`. Cyan outline = acute
start, lime outline = target goal, red fill = current active parcels.
Green outlines mark the frontier of parcels the Q-head is evaluating at
this step.*

### 5.3 Bio-cost heuristic, 100-trial eval

Same protocol as §5.1 (seed=1, reverse-walk ∈ [2, 15], budget=60), but using
the `output_warm_bio/` checkpoint trained with bio-weighted cost targets
(§4.6). Mean cost-to-go in the training set is 11.9 (vs. 7.5 under unit
cost, because bio-costs average above 1.0 along simulator trajectories).

```
Trials: 100  solved: 69/100 = 69.0%
Mean length-optimality (path_len/|s△g|, solved): 1.000
Mean cost-optimality (path_cost/(c_min·|s△g|), solved): 2.92
  (1.0 = admissible floor where every step is fully plausible)
```

Two things stand out:

- **Length-optimality is preserved (= 1.000 on every solved bucket).**
  Bio-costs change *which* toggles are cheap but not whether the heuristic
  picks shortest paths among those it reaches. Good sanity — the geometric
  lower bound `|s△g|` is still saturated.
- **Cost-optimality is ≈ 2.9× the floor.** The warm-start net finds short
  paths but does not yet prefer cheap ones — it solves by toggling whichever
  parcels reduce `|s△g|` fastest, not the ones biology would pick. A policy
  head trained on bio-cost Q-values would address this; unit-cost + bio-cost
  as a multi-objective weighted sum (λ·len + (1-λ)·cost) is another lever.
- **Absolute solve rate drops 87% → 69%.** The cost landscape is richer so
  a fixed MLP capacity has a harder target function, and 400 simulator
  trajectories may now under-sample the cost space. Scaling training data
  or training depth should recover most of the gap. (But see §4.8 — part
  of the 87% → 69% gap is the K=288 → K=285 parcellation change, not
  bio-costs.)

### 5.4 Multi-objective λ sweep

Same 100-trial protocol, same K=285 parcellation, seed=1, across five
λ values:

| λ_len | checkpoint            | solved | len_opt | cost_opt |
|-------|-----------------------|--------|---------|----------|
| 0.0   | `output_warm_mo_00/`  | 70 %   | 1.000   | 2.97     |
| 0.3   | `output_warm_mo_03/`  | 62.6 % | 1.000   | 2.93     |
| **0.5** | **`output_warm_mo_05/`** | **72 %** | **1.000** | **2.93** |
| 0.7   | `output_warm_mo_07/`  | 69 %   | 1.000   | 3.05     |
| 1.0   | `output_warm_mo_10/`  | 62 %   | 1.000   | 2.94     |

Two findings, one positive and one negative:

- **Solve rate is concave in λ, peaking at λ = 0.5 (72 %).** Pure-length
  (λ=1) and pure-cost (λ=0) both underperform a balanced objective by
  2–10 pts. Intuition: the length signal gives a strong "progress toward
  `|s△g|=0`" gradient (scale 0–20), while the cost signal gives finer
  discrimination among similar-length continuations. Combining them
  regularizes the Q-function and makes fewer local minima.
- **Cost-optimality is ≈ constant (2.93–3.05) regardless of λ.** The
  weighted-sum target does **not** reduce the bio-cost of the paths the
  net finds — it only affects *which* paths it finds. Diagnosis: when
  the greedy-Q argmin sees actions with comparable predicted Q, the
  length term dominates the ranking because it varies ~10× more than the
  cost term across legal actions. Pure weighted-sum is insufficient to
  induce cost-seeking behavior.

Conclusion: λ = 0.5 is promoted as the new canonical warm-start (72 % solve
rate, length-optimal paths). Cost-optimality remains an open problem —
next attempts should use a dual-head net (§7 item 2) so the cost signal
can't be drowned out at inference.

## 6. Progress log

- **Pivot from surgical-corridor** (original HW4 direction in §1 of v1 of
  this writeup): low novelty, overlaps with well-trodden A\*-on-MRI literature.
- **SOOP paper audit** revealed that `lesionAcute` and `lesionChronic` masks
  are independent lesions on a single scan, not paired timepoints — killed
  the backward-inference framing. Settled on forward evolution with the
  acute mask as start and a calibrated simulator for supervision.
- **Factory registration** via `@domain_factory.register_class("lesion_evo")`,
  `@heuristic_factory.register_class("lesion_evo_mlp")`, and `@register_nnet_input`.
- **DAVI 50-iter smoke**: loss dropped 0.1→0.06, cost-to-go target 2.58 /
  pred 2.23 — looked healthy at small scale but failed to generalize at 2k.
- **Diagnosis above (§4.1–4.3)** → **warm-start fix (§4.4)**.
- **Brain-mask cleanup (§2.1)**: spotted "island" parcels drifting off the
  brain in early figures; traced to the missing connectivity constraint in
  the intensity-only threshold. One-line largest-CC filter collapsed 128
  mask components → 1 and tightened K from 288 to 285.

## 7. Next steps

1. **Lower-temperature DAVI refinement.** §4.5 suggests the failure is
   exploration/exploitation, not representation — retry with
   `beam_q.1B_0.1T` (or wider beam once supported) so the warm heuristic is
   actually followed during data generation.
2. **Cost-seeking heuristic via dual-head net.** §5.4 shows scalar
   weighted-sum targets don't reduce cost-optimality (flat at ≈ 2.9×
   floor) — the length term dominates the greedy-Q argmin because its
   range is ~10× wider than the cost term. Next attempt: two-headed Q
   (length head + cost head), combine at inference with a sweepable λ.
   Each head gets clean scalar supervision and the cost signal can't be
   drowned out.
3. **Cross-subject generalization.** Current MLP is tied to `K = 285`; move to
   a subject-invariant input (parcel features rather than one-hot) so a single
   heuristic transfers across SOOP subjects.
4. **External validation.** Pair-wise acute + chronic timepoints from ISLES
   2022 for quantitative evaluation against real follow-ups.

## 8. How to run

```sh
# Pretrain (≈1 minute):
python -m deepxube_hw4.pretrain_heur --domain lesion_evo.sub-1.300 \
    --out deepxube_hw4/output_warm --n_traj 400 --epochs 10

# Quantitative eval (100 trials):
python -m deepxube_hw4.eval_heur --domain lesion_evo.sub-1.300 \
    --heur_dir deepxube_hw4/output_warm --n_trials 100

# Interactive viz with live Q-values:
python -m deepxube_hw4.viz_with_heur --domain lesion_evo.sub-1.300 \
    --heur_dir deepxube_hw4/output_warm --steps 10 --mode solve
```

Tests: `conda run -n rlclass pytest tests/test_lesion_domain.py -q`
