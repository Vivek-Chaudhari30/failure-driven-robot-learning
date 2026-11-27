import numpy as np

from fdrl.failures import label_episode
from fdrl.sim.obs import quat_xyzw_from_yaw


def _traj(T=100, **over):
    priv = {
        "ee_pos": np.tile([0.4, 0.0, 0.25], (T, 1)),
        "ee_quat": quat_xyzw_from_yaw(np.zeros(T)),
        "cube_pos": np.tile([0.5, 0.0, 0.02], (T, 1)),
        "cube_quat": quat_xyzw_from_yaw(np.zeros(T)),
        "contact_lr": np.zeros((T, 2)),
        "gripper_width": np.full(T, 0.08),
        "joint_vel": np.zeros((T, 9)),
        "at_joint_limit": np.zeros(T),
        "cube_z0": np.full(T, 0.02),
    }
    priv.update(over)
    acts = np.zeros((T, 7))
    acts[:, 6] = 1.0
    return priv, acts


def test_approach_failure(cfg):
    priv, acts = _traj(T=250)
    fl = label_episode(priv, acts, False, cfg.task, cfg.failures)
    assert fl.label == "F1"


def test_timeout_when_nothing_else(cfg):
    T = 250
    priv, acts = _traj(T=T, ee_pos=np.tile([0.5, 0.0, 0.03], (T, 1)))
    fl = label_episode(priv, acts, False, cfg.task, cfg.failures)
    assert fl.label == "F8"


def test_misaligned_grasp_and_missed(cfg):
    T = 100
    ee = np.tile([0.5, 0.0, 0.02], (T, 1))
    ee[:, 0] += 0.025  # 2.5 cm off in x
    width = np.full(T, 0.08)
    width[30:] = 0.0  # closes fully on nothing
    priv, acts = _traj(T=T, ee_pos=ee, gripper_width=width)
    acts[30:, 6] = -1.0
    fl = label_episode(priv, acts, False, cfg.task, cfg.failures)
    assert "F2" in fl.flags and "F3" in fl.flags
    assert fl.label == "F2"  # earliest (same step, F2 before F3 by priority)


def test_slip_detected(cfg):
    T = 100
    ee = np.tile([0.5, 0.0, 0.02], (T, 1))
    ee[40:, 2] = np.linspace(0.02, 0.10, 60)
    cube = ee.copy()
    cube[60:, 2] -= np.linspace(0, 0.04, 40)  # cube lags the fingers
    contact = np.zeros((T, 2))
    contact[30:90] = 10.0
    priv, acts = _traj(T=T, ee_pos=ee, cube_pos=cube, contact_lr=contact)
    acts[30:, 6] = -1
    fl = label_episode(priv, acts, False, cfg.task, cfg.failures)
    assert fl.label == "F4"


def test_success_has_no_label(cfg):
    priv, acts = _traj()
    fl = label_episode(priv, acts, True, cfg.task, cfg.failures)
    assert fl.label is None and fl.t is None
