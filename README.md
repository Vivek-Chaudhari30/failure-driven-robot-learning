# Failure-Driven Data Selection for Robotic Manipulation

> **Don't collect more data. Diagnose, then collect the right data.**

A research project that asks: *can a robot-learning system figure out which training data it
needs, and does acquiring that data beat acquiring data blindly, at equal cost?*

**Status: M0, research design.** No code beyond an environment-inspection script, and **no
results yet**. Anything that reads like a result is a hypothesis or a design target.

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

## Documents

| Doc | What it answers |
|-----|-----------------|
| [RESEARCH.md](RESEARCH.md) | Why this matters, which assumptions we rejected, verified literature review, novelty gap, risks, commercialization evidence gaps, IP and licensing |
| [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) | Falsifiable hypotheses H1–H6, task spec, condition space, splits and leakage rules, failure taxonomy, strategies S0–S7, cost accounting, Exp 1–7 + D, statistics |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Closed-loop system design, key decisions and trade-offs, repo layout, data contracts, leakage guards, resource estimates, reuse vs build |
| [ROADMAP.md](ROADMAP.md) | Milestones M0–M9 with dependencies, deliverables, validation checks, exit criteria, go/no-go gate, and the robotics learning roadmap |
| [docs/ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md) | Verified Isaac Lab / Isaac Sim requirements, cluster inspection procedure, simulator decision table, fallbacks |

## Key decisions (with the reasoning in the linked docs)

| Decision | Choice | Why |
|----------|--------|-----|
| Simulator | Isaac Lab 3.0.0 + Isaac Sim 6.2 **if** the cluster has RTX GPUs; otherwise ManiSkill3 | Isaac Sim does not support A100/H100 or macOS ([ENVIRONMENT_SETUP](docs/ENVIRONMENT_SETUP.md)) |
| Policy | MLP behavior cloning, state observations, fixed training protocol | The question is about demonstrations, not RL or perception ([RESEARCH §2](RESEARCH.md)) |
| Expert | Scripted, state-feedback, privileged | Cheap, consistent, can relabel any state (enables DAgger) |
| Condition factors | Cube position + yaw (observable), friction (latent) | Plants known root causes so the diagnosis can be checked |
| Budget unit | Expert sim steps including rejected attempts; evaluation and diagnosis also charged | Hard demos cost more; failure-driven methods spend extra compute |
| Deciding regime | Uniform initial data (R2) | Beating skewed data alone is easy and not convincing |

## Quick start (today)

The only runnable artifact is a **read-only** environment inspector for the GPU cluster:

```bash
srun --gres=gpu:1 --time=00:10:00 bash scripts/check_env.sh
```

It installs nothing and writes `env_report_<host>_<date>.txt`. Review it before sharing.
Then follow [docs/ENVIRONMENT_SETUP.md §3](docs/ENVIRONMENT_SETUP.md).

## Repository policy

- No datasets, checkpoints, videos, run outputs or secrets in git (`.gitignore` enforces the common cases).
- Keep the GitHub repo **private** until you and your advisor agree otherwise (IP: [RESEARCH §7](RESEARCH.md)).
- Every reported number must trace to a run directory with its config, git SHA, seeds and environment metadata.
- Negative results are reported. The pre-registration is tagged `prereg-v1` before headline runs.
