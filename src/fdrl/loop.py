"""Acquisition rounds and experiment orchestration (ARCHITECTURE section 1).

dataset manifest -> train BC -> validation eval -> diagnose -> strategy plan -> expert demos -> ...

Pairing (EXPERIMENT_DESIGN section 5): within a pipeline seed, all strategies share the same D0,
the same training seed per round and the same validation episodes. Round 0 (the D0 policy, its
validation report and diagnosis) is computed once per (regime, pipeline seed) and cached.

Cost accounting (section 8): 1 cost unit = 1 sim step. Expert attempts are charged whether
accepted or rejected; policy rollouts used for acquisition (S3b, S6) are charged too. Steps simulated
only because of batching past the budget are logged as ``discarded_parallel_steps`` and not charged.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from fdrl.conditions import (
    CellGrid,
    Condition,
    NominalDistribution,
    sample_condition,
    seed_for,
    uniform_train,
)
from fdrl.config import Config, to_yaml
from fdrl.data.manifest import DatasetManifest, load_training_arrays
from fdrl.data.store import EpisodeStore
from fdrl.episodes import to_records
from fdrl.evaluate import EvalArtifacts, EvalReport, evaluate
from fdrl.expert.scripted import ScriptedExpert
from fdrl.intel.diagnose import Diagnosis, DiagnosisSignals, diagnose
from fdrl.intel.stats import demo_density, ensemble_decomposition, expert_stats, state_coverage_gap
from fdrl.intel.strategies import (
    NEEDS_DIAGNOSIS,
    NEEDS_POLICY,
    AcquisitionContext,
    AcquisitionPlan,
    make_plan,
)
from fdrl.policy.bc import MLPPolicy
from fdrl.sim.base import SimBackend, make_backend
from fdrl.sim.rollout import rollout
from fdrl.train import fit_error_per_cell, train_bc, train_ensemble


@dataclass
class CostLedger:
    expert_steps_accepted: int = 0
    expert_steps_rejected: int = 0
    policy_rollout_steps: int = 0
    discarded_parallel_steps: int = 0
    val_eval_steps: int = 0
    diagnosis_steps: int = 0
    train_seconds: float = 0.0
    diagnosis_train_seconds: float = 0.0
    accepted_demos: int = 0
    rejected_attempts: int = 0
    policy_rollouts: int = 0

    @property
    def acquisition_cost(self) -> int:
        return self.expert_steps_accepted + self.expert_steps_rejected + self.policy_rollout_steps

    @property
    def total_cost(self) -> int:
        return self.acquisition_cost + self.val_eval_steps + self.diagnosis_steps

    def to_dict(self) -> dict:
        d = asdict(self)
        d["acquisition_cost"] = self.acquisition_cost
        d["total_cost"] = self.total_cost
        return d


def derive_seed(*parts: int) -> int:
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


def run_metadata(cfg: Config, backend: SimBackend | None) -> dict:
    def _git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return ""

    try:
        from importlib.metadata import distributions

        pkgs = sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions())
    except Exception:
        pkgs = []
    meta = {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty_diff": _git("diff", "HEAD"),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": pkgs,
        "backend": backend.info() if backend is not None else None,
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    meta["config_yaml"] = to_yaml(cfg)
    return meta


_T0 = time.time()


def log(msg: str) -> None:
    """Progress line, flushed immediately (runs are often piped to tee inside tmux)."""
    el = time.time() - _T0
    print(f"[fdrl {time.strftime('%H:%M:%S')} +{el / 60:6.1f}m] {msg}", flush=True)


class Collector:
    """Executes an AcquisitionPlan (or a fixed-count collection) and writes episodes to the store."""

    def __init__(self, cfg: Config, backend: SimBackend, grid: CellGrid, store: EpisodeStore, expert: ScriptedExpert):
        self.cfg, self.backend, self.grid, self.store, self.expert = cfg, backend, grid, store, expert

    def _expert_ctrl(self):
        return lambda obs, priv: self.expert.act(priv)

    def _conditions(self, plan_dist, sequence, ps: int, block: int, start: int, n: int):
        items = []
        for k in range(n):
            i = start + k
            s = seed_for("acq", ps, block, i)
            if sequence is not None:
                c = sequence[i % len(sequence)]
                if self.grid.cell_of(c) in self.grid.holdout:
                    raise RuntimeError("strategy sequence contains a hold-out condition")
            else:
                c = sample_condition(self.grid, plan_dist, s, "acq")
            items.append((c, s))
        return items

    def collect(
        self,
        plan: AcquisitionPlan,
        *,
        pipeline_seed: int,
        block: int,
        tag: str,
        policy: MLPPolicy | None = None,
        target_accepted: int | None = None,
    ) -> tuple[list[str], CostLedger, dict[int, int]]:
        """Returns (accepted episode ids in acquisition order, ledger, accepted count per cell)."""
        led = CostLedger()
        accepted_ids: list[str] = []
        per_cell: dict[int, int] = {}
        if plan.mode == "none":
            return accepted_ids, led, per_cell
        budget = math.inf if target_accepted is not None else plan.budget
        n_env = self.backend.num_envs
        cursor = 0
        max_attempts = 10_000  # one seed block
        common = dict(grid=self.grid, task=self.cfg.task, fail_cfg=self.cfg.failures,
                      strategy=plan.strategy, round_idx=block, pipeline_seed=pipeline_seed, tag=tag)

        def done(pending: int = 0) -> bool:
            if target_accepted is not None:
                return len(accepted_ids) + pending >= target_accepted
            return led.acquisition_cost >= budget

        log(f"collect {tag}: mode={plan.mode} target={target_accepted if target_accepted is not None else plan.budget}")
        while not done():
            if cursor + n_env > max_attempts:
                raise RuntimeError(f"acquisition exhausted seed block {block} (budget too large?)")
            items = self._conditions(plan.distribution, plan.sequence, pipeline_seed, block, cursor, n_env)
            cursor += n_env

            if plan.mode in ("sampler", "sequence"):
                self._expert_batch(items, led, accepted_ids, per_cell, done, common)
            elif plan.mode == "failed_replay":
                if policy is None:
                    raise ValueError("S3b needs the current policy")
                # Sequential semantics, batched execution: for each instance in order, charge the
                # policy rollout; if it failed, charge the expert attempt at the same θ right after.
                res = rollout(self.backend, items, policy.controller(), self.cfg.task, record_obs=False)
                fail_idx = [k for k, r in enumerate(res) if not r.success]
                exp_res = rollout(self.backend, [items[k] for k in fail_idx], self._expert_ctrl(),
                                  self.cfg.task) if fail_idx else []
                exp_map = dict(zip(fail_idx, exp_res, strict=True))
                pol_used, exp_used, exp_acc = [], [], []
                for k, r in enumerate(res):
                    if done():
                        led.discarded_parallel_steps += r.length + (exp_map[k].length if k in exp_map else 0)
                        continue
                    led.policy_rollout_steps += r.length
                    led.policy_rollouts += 1
                    pol_used.append(r)
                    if k in exp_map:
                        e = exp_map[k]
                        if done():
                            led.discarded_parallel_steps += e.length
                            continue
                        self._charge_expert(e, led)
                        exp_used.append(e)
                        exp_acc.append(e.success)
                recs = to_records(pol_used, split="acq", source="policy", accepted=[False] * len(pol_used),
                                  policy_id=policy.policy_id(), **common)
                self.store.write([s for s, _ in recs])
                self._record_expert(exp_used, exp_acc, accepted_ids, per_cell, common)
                if led.policy_rollouts >= plan.notes.get("max_policy_rollouts", 100_000):
                    break
            elif plan.mode == "dagger":
                if policy is None:
                    raise ValueError("S6 needs the current policy")
                res = rollout(self.backend, items, policy.controller(), self.cfg.task, labeler=self.expert.act)
                keep = []
                for r in res:
                    if done():
                        led.discarded_parallel_steps += r.length
                        continue
                    led.policy_rollout_steps += r.length
                    led.policy_rollouts += 1
                    keep.append(r)
                recs = to_records(keep, split="acq", source="dagger", accepted=[True] * len(keep),
                                  policy_id=policy.policy_id(), **common)
                self.store.write([s for s, _ in recs], {s.episode_id: t for s, t in recs})
                for s, _ in recs:
                    accepted_ids.append(s.episode_id)
                    per_cell[s.cell] = per_cell.get(s.cell, 0) + 1
                led.accepted_demos += len(recs)
            else:
                raise ValueError(plan.mode)
            log(f"  collect {tag}: attempts={cursor} accepted={len(accepted_ids)} cost={led.acquisition_cost}")
        return accepted_ids, led, per_cell

    def _expert_batch(self, items, led, accepted_ids, per_cell, done, common):
        n_env = self.backend.num_envs
        for start in range(0, len(items), n_env):
            chunk = items[start : start + n_env]
            res = rollout(self.backend, chunk, self._expert_ctrl(), self.cfg.task)
            used, acc, n_new = [], [], 0
            for r in res:
                if done(n_new):
                    led.discarded_parallel_steps += r.length
                    continue
                used.append(r)
                acc.append(r.success)
                self._charge_expert(r, led)
                n_new += int(r.success)
            self._record_expert(used, acc, accepted_ids, per_cell, common)

    @staticmethod
    def _charge_expert(r, led: CostLedger) -> None:
        if r.success:
            led.expert_steps_accepted += r.length
            led.accepted_demos += 1
        else:
            led.expert_steps_rejected += r.length
            led.rejected_attempts += 1

    def _record_expert(self, used, acc, accepted_ids, per_cell, common) -> None:
        """Store all expert attempts (rejected ones as summaries only) and register accepted demos."""
        if not used:
            return
        recs = to_records(used, split="acq", source="expert", accepted=acc, **common)
        trajs = {s.episode_id: t for s, t in recs if s.accepted}
        self.store.write([s for s, _ in recs], trajs)
        for s, _ in recs:
            if s.accepted:
                accepted_ids.append(s.episode_id)
                per_cell[s.cell] = per_cell.get(s.cell, 0) + 1


@dataclass
class RoundResult:
    strategy: str
    regime: str
    pipeline_seed: int
    round: int
    budget_demo_eq: float
    n_episodes: int
    val_success: float
    ledger: dict
    out_dir: str
    cell_weights: dict | None = None
    extra: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, cfg: Config, exp_root: str | Path | None = None, backend: SimBackend | None = None):
        self.cfg = cfg
        self.root = Path(exp_root or Path(cfg.experiment.run_root) / cfg.experiment.name)
        self.root.mkdir(parents=True, exist_ok=True)
        self.grid = CellGrid(cfg.conditions)
        self.backend = backend or make_backend(cfg)
        self.store = EpisodeStore(self.root / "episodes")
        self.expert = ScriptedExpert(cfg.expert, cfg.task, cfg.sim.toy.gripper_open_width)
        self.collector = Collector(cfg, self.backend, self.grid, self.store, self.expert)
        meta_path = self.root / "run_meta.json"
        if not meta_path.exists():
            meta_path.write_text(json.dumps(run_metadata(cfg, self.backend), indent=1))

    # ------------------------------------------------------------------ helpers
    def _dist(self, regime: str):
        if regime == "R1":
            return NominalDistribution(self.cfg.conditions)
        if regime == "R2":
            return uniform_train(self.grid)
        raise ValueError(regime)

    def shared_dir(self, regime: str, ps: int) -> Path:
        return self.root / "shared" / regime / f"seed{ps}"

    def train(self, manifest: DatasetManifest, ps: int, round_idx: int, include_latent: bool = False):
        arr = load_training_arrays(self.store, manifest, self.grid, include_latent=include_latent)
        log(f"train {manifest.name}: {len(manifest)} demos, {len(arr.obs)} samples")
        res = train_bc(arr.obs, arr.actions, self.cfg.train, self.cfg.policy, seed=derive_seed(ps, round_idx, 7),
                       include_latent=include_latent, mu_range=tuple(self.cfg.conditions.mu_range))
        log(f"  train done in {res.seconds:.0f}s")
        return res, arr

    def val_eval(self, policy: MLPPolicy, ps: int, out_dir: Path, artifacts: bool, tag: str,
                 strategy: str | None, round_idx: int) -> tuple[EvalReport, EvalArtifacts | None]:
        log(f"val eval {tag} round {round_idx} (artifacts={artifacts})")
        out = evaluate(policy.controller(), self.backend, self.grid, self.cfg, split="val", pipeline_seed=ps,
                        policy_id=policy.policy_id(), store=self.store, out_dir=out_dir,
                        expert=self.expert.act if artifacts else None, collect_artifacts=artifacts,
                        tag=tag, strategy=strategy, round_idx=round_idx)
        log(f"  val success {out[0].overall():.3f}")
        return out

    def collect_fixed(self, regime: str, ps: int, n: int, name: str) -> DatasetManifest:
        plan = AcquisitionPlan(name, "sampler", math.inf, distribution=self._dist(regime))
        ids, led, _ = self.collector.collect(plan, pipeline_seed=ps, block=0, tag=name, target_accepted=n)
        return DatasetManifest.create(name, ids[:n], meta={"regime": regime, "ledger": led.to_dict()})

    # ------------------------------------------------------------------ round 0 (shared)
    def d0(self, regime: str, ps: int) -> DatasetManifest:
        p = self.shared_dir(regime, ps) / "d0_manifest.json"
        if p.exists():
            return DatasetManifest.load(p)
        m = self.collect_fixed(regime, ps, self.cfg.experiment.n0, f"D0_{regime}_s{ps}")
        m.save(p)
        return m

    def round0(self, regime: str, ps: int, need_diagnosis: bool) -> tuple[Path, EvalReport, Diagnosis | None]:
        d = self.shared_dir(regime, ps) / "round0"
        pol_p, rep_p, diag_p = d / "policy.pt", d / "val_report.json", d / "diagnosis.json"
        manifest = self.d0(regime, ps)
        if pol_p.exists() and rep_p.exists():
            report = EvalReport.load(rep_p)
            if diag_p.exists() or not need_diagnosis:
                return pol_p, report, (Diagnosis.load(diag_p) if diag_p.exists() else None)
            # Diagnosis requested later: re-collect artifacts with the SAME cached policy (no retrain).
            policy = MLPPolicy.load(pol_p)
            arr = load_training_arrays(self.store, manifest, self.grid)
            log("val eval round0 again, collecting diagnosis artifacts (cached policy)")
            _, art = evaluate(policy.controller(), self.backend, self.grid, self.cfg, split="val",
                              pipeline_seed=ps, policy_id=policy.policy_id(), expert=self.expert.act,
                              collect_artifacts=True)
            diag = self.compute_diagnosis(report, art, policy, arr, manifest, ps, 0)
            diag.save(diag_p)
            return pol_p, report, diag
        res, arr = self.train(manifest, ps, 0)
        res.policy.save(pol_p, {"manifest": manifest.sha256})
        report, art = self.val_eval(res.policy, ps, d, artifacts=need_diagnosis, tag="round0", strategy=None, round_idx=0)
        diag = None
        if need_diagnosis:
            diag = self.compute_diagnosis(report, art, res.policy, arr, manifest, ps, 0)
            diag.save(diag_p)
        (d / "train.json").write_text(json.dumps({"losses": res.losses, "seconds": res.seconds}))
        return pol_p, report, diag

    # ------------------------------------------------------------------ diagnosis
    def compute_diagnosis(self, report, art, policy, arr, manifest, ps: int, round_idx: int) -> Diagnosis:
        cfg = self.cfg
        log(f"diagnosis round {round_idx}")
        cells = self.grid.train_cells
        idx = self.store.index()
        sig = DiagnosisSignals(
            expert=expert_stats(idx, self.grid, pipeline_seed=ps, default_steps=cfg.experiment.demo_equiv_steps),
            density=demo_density(idx, manifest.episode_ids, self.grid),
            fit_error=fit_error_per_cell(policy, arr.obs, arr.actions, arr.cell),
        )
        cost = {"sim_steps": 0.0, "train_seconds": 0.0}
        if cfg.policy.ensemble_size > 1:
            ens, secs = train_ensemble(arr.obs, arr.actions, cfg.train, cfg.policy,
                                       seed=derive_seed(ps, round_idx, 11), k=cfg.policy.ensemble_size)
            sig.ensemble = ensemble_decomposition(ens, art)
            cost["train_seconds"] += secs
        sig.state_gap = state_coverage_gap(art, arr.obs, policy.obs_mean.cpu().numpy(), policy.obs_std.cpu().numpy(),
                                           seed=ps)
        if cfg.diagnosis.run_privileged_probe:
            res_p, _ = self.train(manifest, ps, round_idx, include_latent=True)
            cost["train_seconds"] += res_p.seconds
            rep_p, _ = evaluate(res_p.policy.controller(), self.backend, self.grid, cfg, split="val",
                                pipeline_seed=ps, policy_id=res_p.policy.policy_id())
            cost["sim_steps"] += rep_p.sim_steps
            sig.probe_gain = {c: rep_p.success(c) - report.success(c) for c in cells if c in rep_p.per_cell}
        diag = diagnose(report, sig, cells, cfg.diagnosis, cfg.strategy.fixability_priors)
        diag.cost = cost
        log("  diagnosis done")
        return diag

    # ------------------------------------------------------------------ strategies
    def run_strategy(self, regime: str, ps: int, strategy: str, budget_demo_eq: float, rounds: int,
                     out_root: Path | None = None) -> list[RoundResult]:
        cfg = self.cfg
        log(f"=== {strategy} {regime} seed{ps} B={budget_demo_eq:g} ===")
        needs_diag = strategy in NEEDS_DIAGNOSIS
        pol_p, report, diag = self.round0(regime, ps, need_diagnosis=needs_diag)
        manifest = self.d0(regime, ps)
        policy = MLPPolicy.load(pol_p)
        budget = budget_demo_eq * cfg.experiment.demo_equiv_steps
        base = out_root or (self.root / regime / f"B{budget_demo_eq:g}" / strategy / f"seed{ps}")
        results: list[RoundResult] = []
        history: list[dict] = []
        scfg = cfg.strategy
        if strategy == "S2+":
            # Compute-matched control: S2 plus the sim steps S5 spends on validation and diagnosis.
            # (Ensemble training GPU-seconds are logged separately, not converted to sim steps.)
            _, rep_d, diag_d = self.round0(regime, ps, need_diagnosis=True)
            extra = rep_d.sim_steps + (diag_d.cost.get("sim_steps", 0.0) if diag_d else 0.0)
            scfg = type(scfg)(**{**asdict(scfg), "extra_budget": float(extra)})

        for r in range(1, rounds + 1):
            out = base / f"round{r}"
            if (out / "metrics.json").exists():  # resumable
                d = json.loads((out / "metrics.json").read_text())
                results.append(RoundResult(**d))
                manifest = DatasetManifest.load(out / "manifest.json")
                policy = MLPPolicy.load(out / "policy.pt")
                report = EvalReport.load(out / "val_report.json")
                diag = Diagnosis.load(out / "diagnosis.json") if (out / "diagnosis.json").exists() else diag
                continue
            ctx = AcquisitionContext(
                grid=self.grid, cond_cfg=cfg.conditions, cfg=scfg, regime=regime, budget=budget,
                pipeline_seed=ps, round_idx=r, index=self.store.index(), dataset_ids=manifest.episode_ids,
                val_report=report, diagnosis=diag,
                expert=expert_stats(self.store.index(), self.grid, pipeline_seed=ps,
                                    default_steps=cfg.experiment.demo_equiv_steps),
                gain_history=history or None,
            )
            plan = make_plan(strategy, ctx)
            if plan.mode == "none":
                new_ids, led, per_cell = [], CostLedger(), {}
            else:
                new_ids, led, per_cell = self.collector.collect(
                    plan, pipeline_seed=ps, block=r, tag=f"{strategy}_{regime}_B{budget_demo_eq:g}",
                    policy=policy if strategy in NEEDS_POLICY else None)
            new_manifest = manifest.extend(f"{strategy}_{regime}_s{ps}_r{r}", new_ids)
            if new_ids:
                res, arr = self.train(new_manifest, ps, r)
                led.train_seconds = res.seconds
                new_policy = res.policy
            else:
                new_policy, arr = policy, None
            need_art = needs_diag and r < rounds
            new_report, art = self.val_eval(new_policy, ps, out, artifacts=need_art, tag=strategy,
                                            strategy=strategy, round_idx=r)
            led.val_eval_steps = new_report.sim_steps
            if needs_diag and diag is not None:
                for c, d in diag.cells.items():
                    n_add = per_cell.get(c, 0)
                    if n_add:
                        history.append({"label": d.label, "cell": c,
                                        "gain_per_demo": (new_report.success(c) - report.success(c)) / n_add})
            new_diag = None
            if need_art:
                if arr is None:
                    arr = load_training_arrays(self.store, new_manifest, self.grid)
                new_diag = self.compute_diagnosis(new_report, art, new_policy, arr, new_manifest, ps, r)
                led.diagnosis_steps = int(new_diag.cost.get("sim_steps", 0))
                led.diagnosis_train_seconds = float(new_diag.cost.get("train_seconds", 0.0))
                new_diag.save(out / "diagnosis.json")
            new_manifest.save(out / "manifest.json")
            new_policy.save(out / "policy.pt", {"manifest": new_manifest.sha256, "strategy": strategy})
            (out / "plan.json").write_text(json.dumps({
                "strategy": plan.strategy, "mode": plan.mode, "budget": plan.budget,
                "cell_weights": {str(k): v for k, v in (plan.cell_weights or {}).items()},
                "accepted_per_cell": {str(k): v for k, v in per_cell.items()},
                "notes": plan.notes,
            }, indent=1, default=str))
            rr = RoundResult(strategy, regime, ps, r, budget_demo_eq, len(new_manifest),
                             new_report.overall(), led.to_dict(), str(out),
                             {str(k): v for k, v in (plan.cell_weights or {}).items()} or None)
            (out / "metrics.json").write_text(json.dumps(asdict(rr), indent=1))
            results.append(rr)
            manifest, policy, report = new_manifest, new_policy, new_report
            diag = new_diag or diag
        return results

    # ------------------------------------------------------------------ experiments
    def run_scaling(self, regime: str, ps: int, sizes: list[int]) -> list[dict]:
        pool_p = self.shared_dir(regime, ps) / "scaling_pool.json"
        if pool_p.exists():
            pool = DatasetManifest.load(pool_p)
        else:
            pool = self.collect_fixed(regime, ps, max(sizes), f"pool_{regime}_s{ps}")
            pool.save(pool_p)
        out = []
        for n in sorted(sizes):
            d = self.root / "scaling" / regime / f"N{n}" / f"seed{ps}"
            if (d / "metrics.json").exists():
                out.append(json.loads((d / "metrics.json").read_text()))
                continue
            m = pool.prefix(f"scaling_{regime}_N{n}_s{ps}", n)
            res, _ = self.train(m, ps, 0)
            rep, _ = self.val_eval(res.policy, ps, d, artifacts=False, tag=f"scaling_N{n}", strategy=None, round_idx=0)
            m.save(d / "manifest.json")
            res.policy.save(d / "policy.pt", {"manifest": m.sha256})
            rec = {"regime": regime, "n": n, "pipeline_seed": ps, "val_success": rep.overall(),
                   "train_seconds": res.seconds, "out_dir": str(d)}
            (d / "metrics.json").write_text(json.dumps(rec, indent=1))
            out.append(rec)
        return out

    def run_experiment(self) -> list:
        e = self.cfg.experiment
        out: list = []
        for regime in e.regimes:
            for ps in e.pipeline_seeds:
                if e.kind == "scaling":
                    out.extend(self.run_scaling(regime, ps, e.scaling_sizes))
                elif e.kind in ("equal_budget", "iterative"):
                    rounds = 1 if e.kind == "equal_budget" else e.rounds
                    for b in e.budgets:
                        for s in e.strategies:
                            out.extend(self.run_strategy(regime, ps, s, b, rounds))
                else:
                    raise ValueError(e.kind)
        return out


def load_conditions(rows) -> list[Condition]:
    return [Condition(r["x"], r["y"], r["yaw"], r["mu"]) for r in rows]
