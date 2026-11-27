"""Rule-based per-cell root-cause hypotheses (EXPERIMENT_DESIGN 7.2).

Output labels: coverage, compounding, observability, expert_infeasible, capacity, healthy.
Each label comes with a confidence and the evidence that produced it. These are HYPOTHESES:
Exp D validates them against intervention-defined ground truth (H4).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from fdrl.config import DiagnosisCfg
from fdrl.evaluate import EvalReport
from fdrl.intel.stats import EnsembleSignal, ExpertCellStats, require_val, success_posteriors

LABELS = ("coverage", "compounding", "observability", "expert_infeasible", "capacity", "healthy")


@dataclass
class DiagnosisSignals:
    expert: dict[int, ExpertCellStats]
    density: dict[int, int]
    fit_error: dict[int, float] = field(default_factory=dict)
    ensemble: dict[int, EnsembleSignal] = field(default_factory=dict)
    state_gap: dict[int, float] = field(default_factory=dict)
    probe_gain: dict[int, float] = field(default_factory=dict)  # privileged-observation probe Δsuccess


@dataclass
class CellDiagnosis:
    label: str
    confidence: float
    fixability: float
    evidence: dict[str, float]


@dataclass
class Diagnosis:
    cells: dict[int, CellDiagnosis]
    cost: dict[str, float] = field(default_factory=dict)  # sim steps / GPU-seconds spent diagnosing

    def labels(self) -> dict[int, str]:
        return {c: d.label for c, d in self.cells.items()}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        d = {"cells": {str(c): asdict(v) for c, v in self.cells.items()}, "cost": self.cost}
        Path(path).write_text(json.dumps(d, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> Diagnosis:
        d = json.loads(Path(path).read_text())
        return cls({int(c): CellDiagnosis(**v) for c, v in d["cells"].items()}, d.get("cost", {}))


def diagnose(report: EvalReport, sig: DiagnosisSignals, cells, cfg: DiagnosisCfg,
             fixability_priors: dict[str, float]) -> Diagnosis:
    require_val(report)
    post = success_posteriors(report, cells)
    dens_vals = [sig.density.get(c, 0) for c in cells]
    med_density = float(np.median(dens_vals)) if dens_vals else 0.0
    fit_vals = [v for c, v in sig.fit_error.items() if c in cells]
    med_fit = float(np.median(fit_vals)) if fit_vals else float("nan")
    out: dict[int, CellDiagnosis] = {}
    for c in cells:
        ev: dict[str, float] = {
            "success_mean": post[c].mean,
            "expert_success": sig.expert[c].success if c in sig.expert else float("nan"),
            "density": float(sig.density.get(c, 0)),
            "median_density": med_density,
        }
        if c in sig.fit_error:
            ev["fit_error"] = sig.fit_error[c]
            ev["median_fit_error"] = med_fit
        if c in sig.ensemble:
            ev["disagreement"] = sig.ensemble[c].disagreement
            ev["ensemble_error"] = sig.ensemble[c].error
            ev["aleatoric_share"] = sig.ensemble[c].aleatoric_share
        if c in sig.state_gap:
            ev["state_gap_ratio"] = sig.state_gap[c]
        if c in sig.probe_gain:
            ev["probe_gain"] = sig.probe_gain[c]

        label, conf = _rules(ev, cfg)
        out[c] = CellDiagnosis(label, conf, float(fixability_priors.get(label, 0.5)), ev)
    return Diagnosis(out)


def _rules(ev: dict[str, float], cfg: DiagnosisCfg) -> tuple[str, float]:
    e = ev.get("expert_success", float("nan"))
    if np.isfinite(e) and e < cfg.infeasible_expert_success:
        return "expert_infeasible", float(np.clip((cfg.infeasible_expert_success - e) / cfg.infeasible_expert_success + 0.5, 0, 1))
    if ev["success_mean"] >= cfg.healthy_success:
        return "healthy", float(ev["success_mean"])
    if ev.get("probe_gain", 0.0) >= cfg.probe_gain:
        return "observability", float(np.clip(0.5 + ev["probe_gain"], 0, 1))
    if ev["median_density"] > 0 and ev["density"] < cfg.low_density_ratio * ev["median_density"]:
        return "coverage", 0.8
    if ev.get("aleatoric_share", 0.0) >= cfg.aleatoric_share:
        return "observability", float(ev["aleatoric_share"]) * 0.8
    if ev.get("state_gap_ratio", 0.0) >= cfg.state_gap_ratio:
        return "compounding", 0.6
    mf = ev.get("median_fit_error", float("nan"))
    if np.isfinite(mf) and ev.get("fit_error", 0.0) >= cfg.fit_error_ratio * mf:
        return "capacity", 0.5
    return "coverage", 0.4  # default: assume data-fixable, low confidence
