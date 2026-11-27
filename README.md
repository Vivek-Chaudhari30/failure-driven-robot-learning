# Failure-Driven Data Selection for Robotic Manipulation

> **Don't collect more data. Diagnose, then collect the right data.**

A research project that asks: *can a robot-learning system figure out which training data it
needs, and does acquiring that data beat acquiring data blindly, at equal cost?*

**Status: M1, code complete for the full pipeline; no research results yet.** Everything runs end
to end on a laptop with a numpy stand-in simulator (`sim.backend=toy`). The Isaac Lab adapter
(`src/fdrl/sim/isaaclab_lift.py`) is written but **has not been run**: it needs the lab GPU
machines. Numbers produced by the toy backend test the plumbing; they are not results.

## The idea in 30 seconds

1. Train a behavior-cloning policy for a Franka Panda cube-lift task in simulation.
2. Evaluate it over a controlled **condition space** (cube position, cube yaw, friction) and
   label every failure with sim ground truth (missed grasp, slip, drop, …).
3. **Diagnose** each failing region. Is it a *coverage gap* (more data should help), an
   *observability gap* (the policy can't sense friction, so data won't help), an *expert
   limit*, or a *capacity* issue?
4. Spend a fixed, cost-honest acquisition budget where data is predicted to help. Retrain,
   re-evaluate, repeat.
5. Compare against uniform domain randomization, naive failure replay (AdaDemo-style), coverage
   heuristics and DAgger, under a pre-registered, multi-seed, paired statistical protocol.

"Collect data where the robot fails" alone has **already been shown** (AdaDemo, 2024; DAgger,
2011). Our bet is that **separating data-fixable failures from the rest**, validated against
intervention-defined ground truth, is what makes acquisition efficient. See
[RESEARCH.md §3](RESEARCH.md).

## Quick start

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m pytest -q                                                    # 51 tests, sim-free
python scripts/run_experiment.py --config configs/experiment/smoke.yaml   # all strategies, ~30 s
python scripts/analyze.py --config configs/experiment/smoke.yaml          # tables + plots
```

A larger laptop run: `configs/experiment/local_pilot.yaml` (3 seeds, 6 strategies, ~30 min on CPU).

On the lab GPU machines (ssh, no SLURM, no containers), follow
[docs/ENVIRONMENT_SETUP.md §6.3](docs/ENVIRONMENT_SETUP.md): install, run the tests, then
`scripts/isaac_smoke.py` to validate the Isaac Lab adapter before anything else.

## Scripts

| Script | What it does |
|--------|--------------|
| `scripts/check_env.sh` | Read-only machine inspection (GPU, driver, OS, Python, containers, disk, network) |
| `scripts/isaac_smoke.py` | First run on a GPU box: applies conditions θ, checks they read back, runs the expert |
| `scripts/calibrate.py` | Expert success and episode length per cell on calibration seeds (sets `demo_equiv_steps`) |
| `scripts/run_experiment.py` | Runs Exp 1 (scaling), Exp 2 (equal budget) or Exp 4 (iterative) from a config; resumable |
| `scripts/eval.py` | Validation or final **test** evaluation (test requires `--frozen-analysis <git-sha>`) |
| `scripts/analyze.py` | Paired comparisons vs S2, bootstrap CIs, Holm correction, plots |

## Code layout

```
src/fdrl/
  config.py        typed config (dataclasses + YAML)        cli.py       --config / overrides
  conditions.py    Θ, 36 cells, hold-out H, seed streams     episodes.py  rollout -> stored records
  sim/             ONLY place that touches a simulator
    base.py          SimBackend protocol + factory
    obs.py           simulator-agnostic observation (41-D) and privileged-state keys
    toy_lift.py      numpy stand-in (laptop / CI)
    isaaclab_lift.py Isaac Lab adapter (unrun; lines marked VERIFY)
    rollout.py       batched rollouts, success/termination, recorder
  expert/scripted.py   state-feedback, friction-aware grasp-lift expert
  data/            schema, HDF5+Parquet episode store, dataset manifests
  policy/bc.py     MLP BC policy, normalizer, ensemble      train.py     fixed-step BC training
  failures.py      taxonomy F1-F8                            evaluate.py  per-cell reports + leakage guards
  intel/           stats, rule-based diagnosis, strategies S0-S7 (+S3b, S4b, S2+, shuffled-S5)
  loop.py          acquisition rounds, cost ledger, experiment orchestration
  analysis/        bootstrap / Holm / IQM / N*(τ) / ρ(τ), plots
configs/           env, expert, policy, strategy, experiment YAMLs
tests/             unit + end-to-end tests (incl. leakage guards)
```

## Documents

| Doc | What it answers |
|-----|-----------------|
| [RESEARCH.md](RESEARCH.md) | Why this matters, which assumptions we rejected, verified literature review, novelty gap, risks, commercialization evidence gaps, IP and licensing |
| [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) | Falsifiable hypotheses H1–H6, task spec, condition space, splits and leakage rules, failure taxonomy, strategies S0–S7, cost accounting, Exp 1–7 + D, statistics |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Closed-loop system design, key decisions and trade-offs, repo layout, data contracts, leakage guards, resource estimates, reuse vs build |
| [ROADMAP.md](ROADMAP.md) | Milestones M0–M9 with dependencies, deliverables, validation checks, exit criteria, go/no-go gate, and the robotics learning roadmap |
| [docs/ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md) | Verified Isaac Lab requirements, the lab machines as inspected, chosen install path, run procedure |

## Key decisions (with the reasoning in the linked docs)

| Decision | Choice | Why |
|----------|--------|-----|
| Simulator | Isaac Lab on the lab's RTX A6000 / RTX 6000 Ada machines; numpy stand-in for laptop and CI | Isaac Sim doesn't run on macOS; the stand-in keeps 90% of the code testable anywhere ([ENVIRONMENT_SETUP](docs/ENVIRONMENT_SETUP.md)) |
| Policy | MLP behavior cloning, state observations, fixed training protocol | The question is about demonstrations, not RL or perception ([RESEARCH §2](RESEARCH.md)) |
| Expert | Scripted, state-feedback, privileged | Cheap, consistent, can relabel any state (enables DAgger) |
| Condition factors | Cube position + yaw (observable), friction (latent) | Plants known root causes so the diagnosis can be checked |
| Budget unit | Sim steps: expert attempts incl. rejected ones, plus policy rollouts used for acquisition | Hard demos cost more; failure-driven methods spend extra compute |
| Deciding regime | Uniform initial data (R2) | Beating skewed data alone is easy and not convincing |

## Repository policy

- No datasets, checkpoints, videos, run outputs or secrets in git (`.gitignore` enforces the common cases).
- Keep the GitHub repo **private** until you and your advisor agree otherwise (IP: [RESEARCH §7](RESEARCH.md)).
- Every reported number must trace to a run directory with its config, git SHA, seeds and environment metadata (`runs/<exp>/run_meta.json`).
- Negative results are reported. The pre-registration is tagged `prereg-v1` before headline runs.
