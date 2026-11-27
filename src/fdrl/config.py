"""Typed configuration (ARCHITECTURE AD7): dataclasses + YAML via OmegaConf.

Every value marked "calibrate" in EXPERIMENT_DESIGN.md is a placeholder until calibration (M2-M3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass
class ConditionSpaceCfg:
    # Offsets from the nominal cube position (metres), yaw (degrees), friction (log-uniform).
    x_range: tuple[float, float] = (-0.10, 0.10)
    y_range: tuple[float, float] = (-0.25, 0.25)
    yaw_range_deg: tuple[float, float] = (-45.0, 45.0)
    mu_range: tuple[float, float] = (0.1, 1.0)
    n_yaw_bins: int = 3
    n_mu_bins: int = 3
    # Compositional hold-out cells H as (quadrant, yaw_bin, mu_bin). Quadrants: 0=(x-,y-),
    # 1=(x-,y+), 2=(x+,y-), 3=(x+,y+). Every single-factor bin must still appear in training.
    holdout: list[list[int]] = field(
        default_factory=lambda: [[0, 0, 1], [1, 2, 2], [2, 1, 0], [3, 2, 1]]
    )
    # R1 "skewed" nominal distribution p_nom (EXPERIMENT_DESIGN section 4).
    nom_pos_std: float = 0.03
    nom_yaw_std_deg: float = 10.0
    nom_mu_min: float = 0.6


@dataclass
class TaskCfg:
    horizon: int = 250
    policy_dt: float = 0.02  # 50 Hz
    cube_half: float = 0.02  # half edge length (m); calibrate against the Isaac asset
    nominal_cube_xy: tuple[float, float] = (0.5, 0.0)
    success_height: float = 0.08  # above z0
    success_hold_steps: int = 25
    pos_scale: float = 0.01  # metres per step for a normalized action of 1.0
    rot_scale: float = 0.05  # radians per step for a normalized action of 1.0
    include_latent_obs: bool = False  # privileged-observation probe (EXPERIMENT_DESIGN 7.2)


@dataclass
class ToySimCfg:
    """Parameters of the numpy stand-in simulator (fdrl.sim.toy_lift). Not a physics claim."""

    cube_mass: float = 0.6
    grip_force: float = 20.0  # per finger, N
    gravity: float = 9.81
    gripper_open_width: float = 0.08
    gripper_speed: float = 0.01  # width change per step
    grasp_xy_tol: float = 0.015
    grasp_z_tol: float = 0.02
    grasp_yaw_tol_deg: float = 20.0
    max_slip: float = 0.02  # relative cube-finger slip before the cube falls out
    jam_yaw_deg: float = 30.0  # beyond this yaw error, closing fingers jam on the cube corners
    action_noise: float = 0.02  # std of actuation noise, in normalized action units
    ee_start: tuple[float, float, float] = (0.4, 0.0, 0.25)
    workspace_lo: tuple[float, float, float] = (0.25, -0.45, 0.0)
    workspace_hi: tuple[float, float, float] = (0.80, 0.45, 0.45)


@dataclass
class IsaacCfg:
    task: str = "IsaacContrib-Lift-Cube-Franka-IK-Rel"
    device: str = "cuda:0"
    headless: bool = True
    quat_order: str = "xyzw"  # Isaac Lab 3.0 uses XYZW; <=2.3 used WXYZ
    friction_body: str = "object"  # scene entity whose material friction is set to mu
    # Isaac's relative IK re-anchors each command at the CURRENT (lagging) hand pose, so only a
    # fraction of each Δ is realised. The adapter therefore integrates actions into a target pose
    # and commands (target - current). The target may lead the hand by at most these amounts.
    track_target: bool = True
    max_lead_m: float = 0.05
    max_lead_rad: float = 0.3


@dataclass
class SimCfg:
    backend: str = "toy"  # "toy" | "isaaclab"
    num_envs: int = 64
    toy: ToySimCfg = field(default_factory=ToySimCfg)
    isaac: IsaacCfg = field(default_factory=IsaacCfg)


@dataclass
class ExpertCfg:
    pregrasp_height: float = 0.08
    gain: float = 0.5  # fraction of the remaining error commanded per step
    align_xy_tol: float = 0.004
    align_yaw_tol_deg: float = 3.0
    descend_xy_tol: float = 0.006
    descend_yaw_tol_deg: float = 5.0
    # Hysteresis: once below the pre-grasp height, keep descending while within these looser
    # bounds (a real arm drifts sideways while moving down). Beyond descend_xy_tol the expert
    # pauses its descent and corrects xy instead of climbing back to the pre-grasp pose.
    descend_keep_xy_tol: float = 0.02
    descend_keep_yaw_tol_deg: float = 10.0
    close_z_tol: float = 0.004
    lift_target_height: float = 0.12  # above z0
    lift_vmax: float = 0.3  # m/s
    lift_vmin: float = 0.06
    accel_safety: float = 0.4  # fraction of the friction-limited acceleration the expert uses
    # Friction-limited upward acceleration model a_cap(mu) = grip_accel_per_mu * mu - g.
    # Toy default = 2 * grip_force / cube_mass. Calibrate for Isaac (M3).
    grip_accel_per_mu: float = 66.7
    gravity: float = 9.81
    mu_full_speed: float = 0.5  # at or above this mu the expert lifts at lift_vmax
    mu_aware: bool = True  # the expert uses privileged mu; the policy never sees it


@dataclass
class FailureCfg:
    approach_xy: float = 0.02
    approach_z: float = 0.03
    approach_frac_T: float = 0.6
    misalign_yaw_deg: float = 15.0
    misalign_xy: float = 0.015
    misalign_z: float = 0.02
    closed_width_eps: float = 0.005
    contact_force_eps: float = 0.1
    slip_disp: float = 0.015
    drop_height: float = 0.03
    disturb_disp: float = 0.03
    disturb_tilt_deg: float = 30.0
    jerk_limit: float = 1.5  # normalized action units per step^2
    joint_vel_limit: float = 2.0
    unstable_steps: int = 10


@dataclass
class PolicyCfg:
    hidden: int = 512
    layers: int = 3
    ensemble_size: int = 5  # used by the diagnosis only


@dataclass
class TrainCfg:
    steps: int = 50_000
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    gripper_loss_weight: float = 1.0
    device: str = "cpu"
    log_every: int = 1000


@dataclass
class EvalCfg:
    n_per_cell_val: int = 100
    n_per_cell_test: int = 300
    store_success_frac: float = 0.1  # fraction of successful rollouts whose trajectory is stored
    max_traj_per_cell: int = 10  # cap on stored eval trajectories per cell (failures first)


@dataclass
class StrategyCfg:
    name: str = "S2"
    floor: float = 0.10
    beta: float = 0.5
    fixability_priors: dict[str, float] = field(
        default_factory=lambda: {
            "coverage": 1.0,
            "compounding": 0.6,
            "capacity": 0.6,
            "observability": 0.3,
            "expert_infeasible": 0.0,
            "healthy": 0.2,
        }
    )
    kcenter_pool: int = 4000
    s3b_max_policy_rollouts: int = 100_000
    use_diagnosis: bool = True
    use_cost: bool = True
    use_feasibility: bool = True
    shuffle: bool = False
    extra_budget: float = 0.0  # S2+ (compute-matched): added cost units


@dataclass
class DiagnosisCfg:
    infeasible_expert_success: float = 0.8
    healthy_success: float = 0.9
    low_density_ratio: float = 0.5
    probe_gain: float = 0.2
    aleatoric_share: float = 0.6
    state_gap_ratio: float = 1.5
    fit_error_ratio: float = 2.0
    run_privileged_probe: bool = False
    max_states_per_cell: int = 2000


@dataclass
class ExperimentCfg:
    name: str = "smoke"
    kind: str = "equal_budget"  # "scaling" | "equal_budget" | "iterative"
    run_root: str = "runs"
    regimes: list[str] = field(default_factory=lambda: ["R2"])
    pipeline_seeds: list[int] = field(default_factory=lambda: [0])
    strategies: list[str] = field(default_factory=lambda: ["S0", "S2", "S3", "S5"])
    n0: int = 100
    budgets: list[float] = field(default_factory=lambda: [100.0])  # demo-equivalents
    rounds: int = 1
    scaling_sizes: list[int] = field(default_factory=lambda: [25, 50, 100, 200, 400, 800, 1600])
    demo_equiv_steps: float = 100.0  # mean accepted-demo length under uniform sampling (calibrate)


@dataclass
class Config:
    task: TaskCfg = field(default_factory=TaskCfg)
    conditions: ConditionSpaceCfg = field(default_factory=ConditionSpaceCfg)
    sim: SimCfg = field(default_factory=SimCfg)
    expert: ExpertCfg = field(default_factory=ExpertCfg)
    failures: FailureCfg = field(default_factory=FailureCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    strategy: StrategyCfg = field(default_factory=StrategyCfg)
    diagnosis: DiagnosisCfg = field(default_factory=DiagnosisCfg)
    experiment: ExperimentCfg = field(default_factory=ExperimentCfg)


def load_config(*paths: str | Path, overrides: list[str] | None = None) -> Config:
    """Merge defaults <- YAML files (in order) <- dotlist overrides, return a typed Config."""
    base = OmegaConf.structured(Config)
    merged = [base]
    for p in paths:
        merged.append(OmegaConf.load(p))
    if overrides:
        merged.append(OmegaConf.from_dotlist(list(overrides)))
    cfg = OmegaConf.merge(*merged)
    return OmegaConf.to_object(cfg)  # type: ignore[return-value]


def to_yaml(cfg: Config) -> str:
    return OmegaConf.to_yaml(OmegaConf.structured(cfg))


def to_dict(cfg: Any) -> dict:
    return OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)  # type: ignore[return-value]
