# Roadmap

Milestones are ordered by dependency. Each one has deliverables, validation checks and an
**exit criterion**: do not start the next milestone until it is met, or until the deviation
is written down. Effort is in **full-time-equivalent weeks** for one person who is new to
robotics. Treat the numbers as planning estimates, not commitments.

Mapping to the original brief's phases: A → M1–M2 · B → M3–M4 · C → M5 · D → M6–M7 ·
E → M6 · F → M8.

```
M0 Design ─▶ M1 Env feasibility ─▶ M2 Env + conditions + logging ─▶ M3 Expert + calibration
   │                                                                     │
   └─ M0.5 Literature refresh (parallel)                                 ▼
                       M5 Failure taxonomy ◀─ M4 BC baseline + eval + Exp 1 pilot
                               │
                               ▼
         M6 Strategies + loop + pilot ──[GO / NO-GO]──▶ M7 Diagnosis + Exp D + PRE-REGISTER
                                                                     │
                                                                     ▼
                                                M8 Headline experiments ─▶ M9 Extensions (gated)
```

| Milestone | FTE weeks | Cumulative |
|-----------|-----------|------------|
| M0 Research design | 1 | 1 |
| M0.5 Literature refresh | (parallel) 1 | — |
| M1 Environment feasibility | 1–2 | 3 |
| M2 Custom env, conditions, logging | 2 | 5 |
| M3 Scripted expert and calibration | 2 | 7 |
| M4 BC baseline, eval harness, Exp 1 pilot | 2 | 9 |
| M5 Failure taxonomy | 1–2 | 11 |
| M6 Strategies, loop, pilot, go/no-go | 2–3 | 14 |
| M7 Diagnosis, Exp D, pre-registration | 2–3 | 17 |
| M8 Headline experiments and write-up | 4–6 | 23 |

---

## M0: Research design ✅ (this commit)
- **Deliverables:** README, RESEARCH, EXPERIMENT_DESIGN, ARCHITECTURE, ROADMAP,
  docs/ENVIRONMENT_SETUP, `scripts/check_env.sh`, `.gitignore`.
- **Validation:** cross-document consistency (hypotheses, strategies, milestones, versions);
  every citation verified or explicitly marked.
- **Exit:** advisor has reviewed the design. In particular, they agree on the scope, the
  compute allocation, and IP/authorship (RESEARCH §7).

## M0.5: Literature refresh (parallel with M1)
- Read in full: AdaDemo, CUPID, Gao et al. (compositional), RoboMD, Demo-SCORE, AMF, and the
  2025–26 items listed under RESEARCH §4.2 "also seen".
- **Exit:** a one-page note per paper in `docs/lit/`. RESEARCH §3 is updated if any of them
  already does root-cause-aware acquisition. If one does, re-scope before M6.

## M1: Environment feasibility (Phase A)
- **Depends on:** cluster access.
- **Tasks:**
  1. Run `scripts/check_env.sh` on a GPU node. Ask the admins the §2 questions in
     ENVIRONMENT_SETUP.
  2. Pick a path using the decision table (Isaac Lab 3.0 / kit-less spike / ManiSkill3).
  3. Build or pull the container. Run the stock lift task headless with 64 → 1,024 envs.
  4. **Benchmark:** episodes/min and peak VRAM for 1,024 envs × 250 steps, with random and
     zero actions.
  5. Write `pyproject.toml` with pins matching the chosen stack, and a SLURM job template.
- **Deliverables:** ENVIRONMENT_SETUP §6 decision record; `scripts/slurm/*.sbatch`; benchmark JSON.
- **Validation:** smoke test passes twice on different days (no flaky setup).
- **Exit:** stock lift env runs headless at ≥ 1,024 parallel envs (or the fallback's equivalent),
  throughput is measured, and the ARCHITECTURE §8 estimates are updated with real numbers.
- **Learn (robotics):**
  - *Rigid-body simulation basics:* timestep, substeps, decimation (sim rate vs control rate),
    why contact is the hard part of physics engines.
  - *Franka Panda anatomy:* 7 revolute joints + 2 prismatic fingers, joint limits, end-effector
    (EE) frame.
  - *What "headless" and "parallel envs" mean on a GPU simulator.*
  - Resources: Isaac Lab tutorials (official docs); Lynch & Park, *Modern Robotics*, ch. 2–3
    (configuration space, rigid-body motions).

## M2: Custom environment, condition space, logging
- **Depends on:** M1.
- **Tasks:**
  - `fdrl.conditions`: Θ, cells, hold-out set H, seed streams (property tests on the Mac).
  - `fdrl.sim.isaaclab_lift`: derived env config that applies **per-env θ at reset** (cube x, y,
    yaw; friction via physics material; fixed mass), our observation set, our success predicate,
    finger–cube and cube–table contact sensors.
  - `fdrl.data`: EpisodeSummary/trajectory recording, sharded HDF5 + Parquet index, manifests.
- **Validation:**
  - Applied θ read back from the sim matches the logged θ (position within 1 mm, yaw within
    0.5°, friction exact).
  - Contact forces are nonzero only when touching (visual check on 5 rendered episodes).
  - Record → read round-trip is bit-exact.
  - Leakage tests pass (ARCHITECTURE §5).
- **Exit:** a random policy produces correctly logged episodes for any θ, on the cluster.
- **Learn:**
  - *Coordinate frames and transforms:* world, robot base, EE and object frames; homogeneous
    transforms.
  - *Quaternions:* Isaac Lab 3.0 uses **XYZW**. Yaw extraction. Why we encode cube yaw as
    sin/cos of 4ψ (90° symmetry).
  - *Isaac Lab managers:* scene, events (reset randomization), observations, actions,
    terminations.
  - *Friction:* Coulomb model, static vs dynamic coefficients, how PhysX combines the materials
    of two touching bodies.
  - Resources: *Modern Robotics* ch. 3; Tedrake, *Robotic Manipulation* (MIT course notes,
    online), chapters on geometric pose and on basic pick-and-place.

## M3: Scripted expert and calibration
- **Depends on:** M2.
- **Tasks:** `fdrl.expert.scripted`, a state-feedback state machine. Its phase is inferred from
  state: pre-grasp above the cube with yaw aligned → descend → close → lift (μ-aware lift
  profile) → hold. Add optional action noise for diversity. Then run calibration sweeps
  (EXPERIMENT_DESIGN §3, calibration seeds only).
- **Validation:**
  - Expert success ≥ 95% in every feasible cell (≥ 200 episodes/cell).
  - The infeasible low-μ band is characterized and its expert success recorded.
  - The expert recovers from a mid-episode perturbation (needed for DAgger relabelling).
  - Friction measurably affects slip or success for an *unadapted* lift profile (otherwise
    increase mass, per RESEARCH §6).
- **Exit:** Θ ranges, mass and success thresholds are frozen in `configs/env/lift_cube.yaml`
  and tagged `calib-v1`.
- **Learn:**
  - *Differential inverse kinematics:* Jacobian, damped least squares, why we command relative
    EE poses instead of joint angles.
  - *Grasping basics:* parallel-jaw antipodal grasps, friction cones, why grasp misalignment
    and low friction cause slip.
  - *Gripper control:* position vs effort control, grip force.
  - *State machines for manipulation.*
  - Resources: *Modern Robotics* ch. 5–6 (velocity kinematics, inverse kinematics) and ch. 12
    (grasping); Tedrake, *Robotic Manipulation*, chapters on pick-and-place and on clutter or
    contact.

## M4: BC baseline, evaluation harness, Exp 1 pilot (Phase B)
- **Depends on:** M3.
- **Tasks:**
  - `fdrl.policy.bc` + `fdrl.train` (fixed-step protocol, EXPERIMENT_DESIGN §9).
  - `fdrl.evaluate`: batched val/test rollouts → per-cell EvalReport.
  - Hyperparameter pilot on a 200-demo uniform dataset (calibration seeds).
  - **One-time sanity check vs robomimic BC** on the same data.
  - Exp 1 pilot: N ∈ {25 … 1,600}, both distributions, 5 seeds.
- **Validation:**
  - Training is deterministic on CPU.
  - Our BC is within CI of robomimic BC (or the gap is explained).
  - The success landscape over cells is non-flat (calibration criterion 2).
- **Exit:**
  - N₀ chosen so that baseline success is in the 40–80% band.
  - Seed-to-seed SD estimated.
  - Eval cost per report measured.
- **Learn:**
  - *Behavior cloning and compounding error / covariate shift* (why DAgger exists).
  - *Action and observation normalization.*
  - *Action chunking and temporal ensembling* (concept; used in Exp 7).
  - *Why validation loss is a poor proxy for rollout success.*
  - Resources: the DAgger paper (Ross et al., 2011, §1–3); the robomimic paper.

## M5: Failure taxonomy (Phase C)
- **Depends on:** M4 (needs a policy that fails).
- **Tasks:** `fdrl.failures` predicates F1–F8 (EXPERIMENT_DESIGN §6); a per-step diagnostic
  signal recorder; render 30 failures per category; audit tool (a simple HTML gallery).
- **Validation:** manual audit precision ≥ 85% per category; unclassified ≤ 10%.
- **Exit:** taxonomy thresholds frozen (`taxonomy-v1`). Failure distributions are reported for
  the Exp 1 pilot.
- **Learn:**
  - *Contact dynamics and slip detection* from force signals.
  - *What simulation contact gets wrong* (penetration, jitter) and how to spot artifacts.
  - Detection vs diagnosis (RESEARCH §1).

## M6: Strategies, closed loop, pilot, go/no-go (Phases D–E)
- **Depends on:** M5.
- **Tasks:**
  - `fdrl.intel.stats`: Beta posteriors, density, expert cost.
  - Strategies S1, S2, S3, S3b, S4, S4b, S5-lite (§7.1 with round-1 class priors and the cheap
    diagnosis signals: expert success, density, failure rate), S2⁺, shuffled-S5.
  - `fdrl.loop` with budget enforcement and the **cost ledger** (EXPERIMENT_DESIGN §8).
  - Human-readable diagnosis report (markdown + JSON) per round.
  - **Pilot:** Exp 2 in R1 and R2 at B = 100, 5 seeds (validation-split metrics only).
- **Validation:**
  - The budget ledger reconciles exactly (charged sim steps = Σ attempts).
  - Strategies are unit-tested on synthetic EvalReports (e.g., S3 allocates more to
    high-failure cells; S4 to empty regions).
  - The shuffled-S5 control behaves like S2.
- **Exit / GO–NO-GO:**
  - **GO** if S5-lite − S2 in R2 has a point estimate > 0, or its CI includes +5 pp.
  - **NO-GO** (CI entirely below +5 pp and point estimate ≤ 0): write a short negative-result
    memo, then revisit (finer cells? continuous failure model? different regime?) before
    spending headline compute.
  - Use the pilot variance for the power analysis (EXPERIMENT_DESIGN §11).
- **Learn:** *Domain randomization* (what it buys and what it doesn't); *active learning and
  why random is a strong baseline* (RESEARCH §4.3).

## M7: Full diagnosis, Exp D, pre-registration (Phase D)
- **Depends on:** M6 GO.
- **Tasks:**
  - Ensembles (5×), train-fit error per cell, state-coverage gap (kNN), privileged-observation
    probe, rule-based diagnosis with evidence strings (EXPERIMENT_DESIGN §7.2).
  - Fixability estimation from round history.
  - Exp D pilot: intervention-defined ground truth on validation seeds.
- **Validation:**
  - Diagnosis is unit-tested on synthetic signals.
  - Exp D confusion matrix computed.
  - Planted-cause expectations are compared against intervention truth, *including*
    surprises.
- **Exit:**
  - S5 v1 is frozen.
  - EXPERIMENT_DESIGN is finalized, with all ⚙️ values filled in and the seed count fixed by
    the power analysis.
  - Analysis scripts are frozen.
  - **Tag `prereg-v1`.** Optional: post the pre-registration publicly (e.g., OSF) after
    advisor approval.
- **Learn:**
  - *Observability:* why a memoryless policy cannot adapt to unobserved friction, and how
    history or force sensing changes that.
  - *Epistemic vs aleatoric uncertainty.*

## M8: Headline experiments and write-up (Phase F)
- **Depends on:** `prereg-v1`.
- **Tasks:**
  - Run Exp 1–7 and Exp D with the frozen protocol.
  - Run test evaluations once per experiment.
  - Produce all plots (EXPERIMENT_DESIGN §11).
  - Log deviations.
  - Write the paper draft. Negative results are written up with the same care.
- **Validation:**
  - `scripts/run_experiment.py` reproduces any table from configs.
  - A re-run of 2 randomly chosen configurations lands within CI.
  - Every reported number traces to a run directory.
- **Exit:** a draft paper or tech report with every pre-registered hypothesis answered
  (supported / not supported / inconclusive). All code is tagged.

## M9: Extensions (each gated on M8 being reproducible)
Ordered by scientific value per cost:
1. **Cross-simulator replication** (ManiSkill3 PickCube, Exp 2 in R2): is the effect a PhysX
   artifact?
2. **Observation-only diagnosis:** replace privileged failure labels with detectors (in the
   style of FAIL-Detect / Sentinel). This is required for any real-world or product claim.
3. **Harder task:** pick-and-place or stacking, with a generator (Isaac Lab AutoData, or
   MimicGen for research only).
4. **Vision policy** (cameras, lighting and texture factors).
5. **Attribution-based curation** (CUPID / TRAK) vs condition-level acquisition at equal budget.
6. **Real-world partner experiment** with human teleop, acquiring only controllable conditions.
7. Other embodiments (mobile manipulators, drones): only after 1–6.

---

## Week-1 checklist
- [ ] Advisor meeting: scope, compute allocation, IP and authorship, private repo.
- [ ] Run `scripts/check_env.sh` on a cluster GPU node; send the admins the access questions.
- [ ] Create the **private** GitHub repo; push these docs.
- [ ] Mac: install robosuite in a scratch venv and run Lift with Panda via the viewer. Get a feel
      for the robot, gripper and cube (learning only).
- [ ] Start M0.5 reading (AdaDemo first).
