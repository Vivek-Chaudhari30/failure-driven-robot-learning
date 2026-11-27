# Experiment Design

> Status: **draft protocol, not yet pre-registered.** This document becomes the frozen
> pre-registration (tagged commit `prereg-v1`) at the end of Milestone M7, *before* headline
> runs and before any final-test evaluation. After that, any change gets logged in §12 (Deviations) with a reason.
>
> No results exist yet. Every number below is either a **design parameter** or a
> **placeholder to be calibrated** (marked ⚙️). None of them is an expected outcome.

Related documents: [RESEARCH.md](RESEARCH.md) (motivation, literature, hypotheses context),
[ARCHITECTURE.md](ARCHITECTURE.md) (how this is implemented), [ROADMAP.md](ROADMAP.md) (when).

---

## 1. Research questions and falsifiable hypotheses

| ID | Hypothesis | Falsified if… |
|----|------------|---------------|
| **H1** (primary) | At an **equal acquisition cost**, the proposed failure-driven strategy **S5** gets higher cell-balanced test success than uniform domain-randomized acquisition **S2**. | In regime R2 (uniform initial data), the paired difference S5−S2 has a 95% CI that includes 0 or favours S2 at ≥2 of 3 budgets. (R1 alone is not sufficient evidence, see §4.) |
| **H1b** | Diagnosis adds value beyond failure frequency: S5 > S3. | The S5−S3 CI includes 0 at all budgets. Then the "intelligence" adds nothing over a frequency heuristic. |
| **H2** (efficiency) | Datasets built by S5 reach success threshold τ with fewer demonstrations than uniform collection: ρ(τ) = N\*\_uniform(τ) / N\*\_S5(τ) > 1. | The CI of ρ(τ) includes 1 for τ ∈ {0.8, 0.9}. *The "300 vs 1000" figure is illustrative. We measure ρ, we do not target it.* |
| **H3** (observability) | Targeted data improves **coverage-limited** cells much more than **latent-factor** cells. Giving the policy the latent factor (privileged observation) closes the latent-factor gap. | Targeted data improves latent-factor cells as much as coverage cells, *or* privileged observation does not help them. Either way the taxonomy in §6 is wrong and must be revised. |
| **H4** (diagnosis validity) | The diagnosis module's per-cell root-cause labels agree with intervention outcomes. Its predicted gain correlates with the realized gain. | Label accuracy against intervention-defined ground truth ≤ majority-class baseline, or Spearman ρ(predicted, realized gain) CI includes 0. |
| **H5** (no harm) | S5 does not degrade success in non-targeted cells by more than δ = 3 pp ⚙️ relative to S2. | Mean non-targeted-cell success under S5 is lower than S2 by > δ (one-sided 95% CI). |
| **H6** (generalization) | S5's improvements transfer to held-out compositional cells **H** that no strategy could acquire in. | S5−S2 difference on H cells has a CI that includes 0 or is negative. (Reported regardless of outcome.) |

**Negative outcomes are publishable and get documented.** If H1 fails in R2 but holds in R1,
the honest conclusion is: "targeted acquisition helps when the existing data is skewed, but not
beyond what uniform coverage provides." That is still a useful result.

---

## 2. Task specification

**Robot / task:** Franka Panda (7-DoF arm + parallel gripper), single cube on a table, lift.
Built as a derived config of Isaac Lab's lift task. In Isaac Lab 3.0.0 that task is
`IsaacContrib-Lift-Cube-Franka-IK-Rel` (PhysX). In ≤ 2.3.x it was `Isaac-Lift-Cube-Franka-IK-Rel-v0`.
Isaac Lab 3.0 also changed quaternion order from WXYZ to XYZW, which all our pose code must respect.
All Isaac Lab defaults quoted here must be re-checked against the pinned version (see
[docs/ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md)). The stock task's observations (relative
joint state, object position, target-pose command, last action) are replaced by ours, below.

| Item | Specification |
|------|---------------|
| Control rate | Inherit the Isaac Lab lift default (sim dt 0.01 s, decimation 2 → **50 Hz** policy rate). Verify against the pinned version. |
| Episode horizon | T = 250 steps (5 s) ⚙️. Calibrated so the expert finishes in ≲ 60% of T. |
| **Action space** (7-D) | Relative end-effector pose command through differential IK: Δposition (3), Δorientation (3, axis-angle), gripper ∈ {open, close} (1, binary). Scaled and clipped to the env's limits. |
| **Observation space** (state-based, v1) | joint pos (9) and vel (9), EE position (3) and orientation quaternion (4), gripper width (1), cube position (3), cube yaw as (sin 4ψ, cos 4ψ) to respect the cube's 90° symmetry (2), cube position relative to EE (3), previous action (7). **Friction and mass are *not* observed** (deliberately, see §3). |
| **Success** | Cube centre height ≥ z₀ + 0.08 m ⚙️, held for ≥ 25 consecutive steps (0.5 s), with the cube between the fingers. |
| **Termination** | Success (after hold), timeout at T, or a terminal failure predicate (cube off table, NaN/physics explosion). |
| Failure labelling | Ground-truth predicates over privileged sim state, §6. |

**Why state-based first.** The research question is about *which data*, not *perception*.
Vision adds an orthogonal source of failure (perception) and roughly 100× storage and compute
(see ARCHITECTURE §8), and it would confound the attribution of failures. Vision is a
pre-planned extension (M9).

---

## 3. Condition space Θ and cells

Three variation factors (the user brief asked for 2–3). They are chosen so that **each one plants
a different, known root cause**:

| Factor | Range ⚙️ | Observable by policy? | Planted cause it enables |
|--------|---------|------------------------|--------------------------|
| **Cube planar position** (x, y) offset from nominal | x ∈ [−0.10, 0.10] m, y ∈ [−0.25, 0.25] m (Isaac Lab lift default reset range; verify) | Yes | **Coverage gap**: workspace edges are rare in skewed data and need different arm configurations. |
| **Cube yaw** ψ | [−45°, +45°] (covers all orientations, because of cube symmetry) | Yes | **Coverage gap**: needs yaw-dependent wrist rotation that the policy must learn per region. |
| **Friction coefficient** μ (cube–finger contact; static = dynamic) | [0.1, 1.0], log-uniform | **No** (latent) | **Observability gap** (expert adapts to μ using privileged knowledge, the policy cannot see μ), and at the extreme low end, **expert infeasibility**. |

Mass is held fixed in v1 at a value chosen during calibration so that friction actually matters
(with a stiff gripper and a light cube, μ may have no effect). Mass becomes a fourth factor in v1.5.

**Cells (analysis and allocation units):** position 2×2 quadrants × yaw 3 bins × friction 3 bins
(log-spaced) = **36 cells**. Strategies output a *sampling distribution over continuous Θ*
(a mixture over cells, uniform within each cell). Cells are used only for allocation and for reporting.

**Compositional hold-out set H:** 4 cells ⚙️, chosen so that every individual factor bin still
appears in training cells (only *combinations* are held out). No strategy may acquire in H.
Validation never includes H. H is evaluated only in the final test (Exp 5).

### Calibration (M2–M3, before any hypothesis test)
1. Sweep each factor alone with the scripted expert. Ranges are set so that expert success is
   ≥ 95% everywhere **except** an intentionally infeasible low-μ band. We record that band's
   expert success; we do not hide it.
2. Train one baseline BC policy on a uniform dataset. Confirm that the success landscape is
   **non-flat**. If baseline success is ≥ 98% in every cell, the task is too easy for the question
   to be asked: tighten ranges or move to a lower-data regime. If it is ≤ 20% everywhere, the
   task is too hard.
3. **Calibration uses only calibration seeds** (a separate seed stream, §5) and is frozen before Exp 1.

---

## 4. Data regimes and the initial dataset D₀

The literature on active learning shows that uniform random is a hard baseline to beat (see
RESEARCH.md §4.3). The result is also regime-dependent, so we run two regimes:

- **R1 "skewed collection"**: D₀ is drawn from p_nom, a narrow distribution that mimics human
  collection habits. Position ~ truncated 𝒩(0, (0.03 m)²), yaw ~ truncated 𝒩(0, (10°)²),
  μ ≥ 0.6. This is realistic and creates genuine coverage gaps. It is also *easy to win*, so
  winning in R1 alone does not establish H1.
- **R2 "uniform collection"**: D₀ ~ Uniform(Θ_train). Here, gaps come from *difficulty*, not from
  missing coverage. **This is the regime that decides H1.**

|D₀| = N₀ = 100 demos ⚙️ (chosen from Exp 1 so that baseline success is in the 40–80% band,
where acquisition can matter).

---

## 5. Splits, seeds, and leakage prevention

| Stream | Seed range | Used by | Never used by |
|--------|-----------|---------|---------------|
| calibration | 0 – 9,999 | range calibration, hyperparameter pilot | anything after calibration freeze |
| acquisition (train) | 1e6 – 2e6 | demo generation, DAgger rollouts | evaluation |
| validation-eval | 2e6 – 3e6 | the data-intelligence engine, checkpoint sanity | final claims |
| **test-eval** | 3e6 – 4e6 | final reporting only (Exp 1–7 tables) | **any acquisition or tuning decision** |

Enforcement in code (ARCHITECTURE §5): the acquisition engine's API accepts only `EvalReport`
objects with `split == "val"`. Test evaluations write to `runs/<exp>/test/`, and that path is
not readable by `fdrl.intel`. A unit test asserts this. Test evaluation is run in one batch per
experiment, after the analysis script is frozen.

**Pairing:** within a pipeline seed k, all strategies share the same D₀, the same training seed,
and the same evaluation episodes (identical initial conditions). Comparisons are **paired**, which
removes much of the between-seed variance.

---

## 6. Failure taxonomy (ground truth from privileged sim signals)

Predicates are evaluated per step. The episode's **primary label** is the earliest-triggered
predicate, using the priority order below. All triggered flags are stored (multi-label), together
with the failure timestep.

| Code | Name | Predicate (thresholds ⚙️) |
|------|------|---------------------------|
| F1 | Approach failure | EE never within 2 cm (xy) and 3 cm (z) of the pre-grasp pose by t = 0.6T |
| F2 | Misaligned grasp | gripper closes while |yaw error| > 15° or xy offset > 1.5 cm or z offset > 2 cm |
| F3 | Missed grasp | gripper closes to width < ε with no finger–cube contact force |
| F4 | Slip | fingers closed and in contact, cube–EE relative displacement > 1.5 cm before reaching success height |
| F5 | Drop | cube exceeded z₀ + 0.03 m, then lost contact and fell |
| F6 | Disturbed object | before grasp, cube displaced > 3 cm or tilted > 30° (pushed or knocked) |
| F7 | Unstable motion | action jerk or joint velocity above limit, or joint-limit contact, for > 10 steps |
| F8 | Timeout | none of the above, and success not reached by T |

**Validation of the taxonomy (M5 exit criterion):** render 30 random failures per category. Two
passes of manual review (by the author, blind to predicted label where feasible) must give
precision ≥ 85% per category, and unclassified-or-"other" must be ≤ 10% of failures.

**Detection ≠ diagnosis.** The taxonomy labels *what happened* (detection). *Why it happened*
(root cause) is a separate, hypothesis-level output of the diagnosis module (§7.2), validated by
interventions. A correlation between a cell and a failure type is never reported as a cause.

---

## 7. Acquisition strategies

All strategies receive the same **acquisition budget B** in **cost units** (§8), and the same
validation evaluation budget where they use one. Each outputs a sampling distribution q(θ) over
Θ_train \ H. Demos are then generated by the scripted expert. Failed expert attempts are discarded
but **charged**.

| ID | Strategy | q(θ) |
|----|----------|------|
| S0 | No acquisition | reference only |
| S1 | Nominal-random ("more of the same") | the D₀ distribution (p_nom in R1, uniform in R2) |
| S2 | Uniform domain randomization | Uniform(Θ_train \ H) |
| S3 | Failure-frequency | ∝ smoothed validation failure rate per cell, (fails + 1)/(n + 2), with 10% uniform floor |
| S3b | **Failed-instance replay (AdaDemo-style)**, the closest prior method | sample θ ~ Uniform, roll out the current policy, request expert demos only at θ where the policy failed. **Policy rollouts are charged** to the budget (AdaDemo did not count them) |
| S4 | Coverage / diversity | greedy k-center in normalized Θ against existing demo conditions (fills sparse regions regardless of failures) |
| S4b | Factored coverage (after Gao et al., RSS 2024) | vary one factor at a time around a base condition, exploiting compositional generalization; non-adaptive |
| **S5** | **Failure-driven with diagnosis (proposed)** | §7.1 |
| S6 | DAgger-style relabelling (reference; different data *type*) | roll out the policy in acquisition-seed conditions, relabel visited states with the expert, charge the same cost units |
| S7 | Oracle allocation (non-deployable upper bound) | like S5 but uses *test-split* per-cell success. Clearly labelled as leakage-by-design; never a competitor |

**Controls**
- **Compute-matched S2⁺**: S2 plus extra demos whose cost equals S5's extra evaluation and
  diagnosis sim cost. This answers "would that compute have been better spent on random data?"
- **Shuffled-S5 (negative control)**: S5 with cell→score assignments randomly permuted. If it
  matches S5, then gains come from re-weighting *per se*, not from targeting.

### 7.1 S5 v1: interpretable failure-driven allocation

For each allowed cell c:
- Success posterior from validation: s_c ~ Beta(1 + successes, 1 + failures). Use the posterior
  mean ŝ_c and an upper quantile for exploration.
- Expert cost: κ_c = E[sim steps per attempt] / expert-success-rate_c, estimated from acquisition
  logs with a calibration prior.
- Fixability f_c ∈ [0, 1] from the diagnosis (§7.2). Round 1 uses fixed class priors
  (coverage 1.0, capacity/compounding 0.6, observability 0.3, infeasible 0.0) ⚙️. From round 2
  onward, f_c is replaced by a partially pooled estimate of realized gain-per-demo from previous
  rounds, by diagnosis class.
- **Priority** p_c = w_c · (1 − ŝ_c) · f_c / κ_c + β·σ_c, where w_c is the test-distribution
  weight (uniform) and σ_c is the posterior std.
- Allocation: 90% of B ∝ p_c, 10% of B uniform (floor, to guard H5).
- Per-strategy tunables (β, floor, priors) are set **only** on calibration and validation seeds,
  with the **same tuning budget** (≤ 8 configurations) given to S3 and S4.

### 7.2 Diagnosis module: per-cell root-cause hypotheses

The diagnosis computes the following evidence signals per cell. A transparent rule set maps them
to {coverage, compounding/state-coverage, observability, expert-infeasible, capacity, healthy}.
The output includes a confidence and an evidence list.

| Signal | Computed how | Points to |
|--------|-------------|-----------|
| Expert success e_c | acquisition logs | e_c < 0.8 → infeasible |
| Demo density n_c | count in dataset | low n_c and high failure → coverage |
| Train-fit error | BC action error on *training* demos in c vs global median | high → capacity or label conflict |
| Ensemble decomposition | 5-member ensemble on held-out expert-labelled states in c: member disagreement (epistemic) vs error-to-expert with agreement (aleatoric) | epistemic → data-fixable; aleatoric → observability or conflict |
| State-coverage gap | kNN distance from policy-visited states (failing rollouts) to the nearest demo states | high → compounding error (DAgger-fixable) |
| Privileged-observation probe | retrain with the latent θ appended to the observation; per-cell success Δ | large Δ → observability (sim-only diagnostic) |
| Failure type mix and timestep | §6 labels | qualitative evidence attached to the report |

**Intervention-defined ground truth (for H4):** on validation seeds, run the interventions
separately for each cell group: (a) add 100 targeted demos, (b) add privileged observation,
(c) measure expert success. The "true" label is the intervention that closes ≥ 50% of the gap
(or "infeasible" if e_c < 0.8). Planted causes (§3) give an *a priori* expectation, but
**interventions define truth, not our intent**: if friction failures turn out to be data-fixable,
that is a finding.

---

## 8. Cost accounting

Demonstrations are not equally expensive. Hard cells have lower expert success, so they need more
attempts per accepted demo. The primary budget unit is therefore:

> **1 cost unit = 1 expert sim step** (accepted *and* rejected attempts). Budgets are reported as
> "demo-equivalents" = cost / mean accepted-demo length under uniform sampling.

Also logged per run: accepted demos, transitions, rejected attempts, validation-eval sim steps,
diagnosis-probe sim steps and training GPU-seconds, acquisition wall-clock, training GPU-seconds,
and peak VRAM. **Headline comparisons are at equal acquisition cost.** A second table reports
equal *total* cost (acquisition + evaluation + diagnosis), using S2⁺.

---

## 9. Policy and training protocol (held fixed across strategies)

- **Policy:** MLP BC, 3 × 512, GELU, LayerNorm, observation normalization fit on the training
  set. Heads: Gaussian (fixed σ, i.e. MSE) for Δpose, Bernoulli (BCE) for the gripper.
- **Optimization:** AdamW, lr 3e-4 with cosine decay, weight decay 1e-4, batch 256, **fixed
  50k gradient steps** ⚙️ regardless of dataset size. This avoids an epochs-vs-size confound.
  Sensitivity to 25k/100k steps is reported (Exp 7).
- **Checkpoint:** the *final* checkpoint (no selection). This avoids an extra use of validation
  that could favour some strategies.
- **Robustness of conclusions to policy class (Exp 7):** BC with action chunking (k = 8, temporal
  ensembling) and a small Diffusion Policy on the main S2-vs-S5 comparison.
- Hyperparameters are tuned once in calibration on a uniform dataset of 200 demos, then frozen for
  every strategy.

---

## 10. Experiments

Unless noted: 36 ID cells, validation 100 episodes/cell (3,600 episodes), test 300 episodes/cell
(10,800 episodes, SE ≤ 2.9 pp per cell, ≤ 0.5 pp overall). Pilot runs use 5 pipeline seeds;
headline runs use 10.

**Exp 1: Baseline scaling.** N ∈ {25, 50, 100, 200, 400, 800, 1600}, from p_nom and from uniform,
× 10 seeds. Output: success-vs-log N curves (overall, worst-cell, per factor), and N\*(τ) (data to
reach τ). Also sets N₀ for §4.

**Exp 2: Equal-budget acquisition.** From D₀ (N₀ = 100) in R1 and R2, one round with
B ∈ {50, 100, 200} demo-equivalents, for S0–S7 (incl. S3b, S4b), S2⁺, and shuffled-S5. Primary
test of H1, H1b, H5. **S5 vs S3b is the key novelty comparison**: does a diagnosis-aware,
cost-aware model of the failure region beat instance-level failure replay?

**Exp 3: Data efficiency.**
(a) Place S5's iterative-loop datasets (Exp 4) on the Exp 1 uniform scaling curve → ρ(τ), with
bootstrap CI over seeds.
(b) Offline subset selection: from a fixed pool of 1,600 uniform demos, select k ∈ {100, 200, 400}
by random, cell-balanced, failure-weighted (using validation results of a policy trained on a
random 100-demo subset), or k-center. Influence-based selection (CUPID-style) is a stretch baseline.

**Exp 4: Iterative improvement.** N₀ = 100, then 4 rounds × B = 50, for S1–S6 (incl. S3b, S4b). Curves of test
success vs cumulative cost (acquisition-only and total).

**Exp 5: Generalization and no-harm.** Test on H cells (H6). Per-factor marginals. Change in
non-targeted cells (H5). Failure-type distributions before and after.

**Exp 6: Ablations of S5.** −diagnosis (= S3 + cost), −cost-awareness, −feasibility filter,
−exploration bonus, single model instead of ensemble, cell granularity (18 / 36 / 72 cells),
cells vs a continuous failure model (GP classifier over Θ, sample ∝ predicted failure × fixability),
validation budget (25 / 50 / 100 episodes per cell), floor (0 / 10 / 25%).

**Exp 7: Robustness.** 10 pipeline seeds for headline claims. Policy class (§9). Gradient-step
sensitivity. *Stretch:* replicate Exp 2 (R2, B = 100) in a second simulator (ManiSkill3) to show
the conclusion is not a PhysX or Isaac artifact.

**Exp D: Diagnosis validity (H3, H4).** Intervention-defined ground truth, §7.2. Confusion
matrix of diagnosis vs truth. Predicted-vs-realized gain scatter with Spearman ρ.

---

## 11. Metrics, statistics, and reporting

**Metrics:** cell-balanced overall success (primary), per-cell success, worst-cell and bottom-10%
mean success (more stable than the minimum), failure-category frequencies, H-cell success, success
gain per cost unit, N\*(τ), ρ(τ), cost breakdowns (§8), training GPU-hours.

**Statistics**
- Unit of replication = pipeline seed. Report the mean and 95% CI (stratified bootstrap,
  10,000 resamples) over seeds, never the best seed. Also report per-seed dots on plots.
- Primary comparisons are **paired differences** (same seed). Holm–Bonferroni correction across
  the primary family {H1 at 3 budgets, H1b, H5}.
- Minimum effect of interest: +5 pp overall, +10 pp worst-cell ⚙️. A statistically significant
  but smaller effect is reported as "not practically significant".
- **Sample-size decision:** the pilot (5 seeds, Exp 2 at B = 100) estimates the SD of paired
  differences. The number of seeds for headline runs is then fixed (default 10, or more if the
  pilot shows we are underpowered for 5 pp at 80% power). That number is recorded before
  headline runs.
- All runs are reported, including crashed or diverged ones (marked), with no exclusions after
  seeing results. IQM is reported alongside the mean, following rliable.

**Required plots:** success vs dataset size (Exp 1); random vs targeted at each budget (Exp 2);
failure distributions before and after (Exp 5); dataset coverage heatmaps over Θ per strategy;
improvement across iterations (Exp 4); performance vs cost (data, sim steps, GPU-hours);
diagnosis confusion matrix (Exp D).

---

## 12. Deviations log

| Date | Change | Reason | Decided before or after seeing test data? |
|------|--------|--------|-------------------------------------------|
| — | — | — | — |
