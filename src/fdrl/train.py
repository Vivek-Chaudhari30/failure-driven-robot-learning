"""Fixed-step BC training, identical across strategies (EXPERIMENT_DESIGN section 9).

Fixed number of gradient steps regardless of dataset size (no epochs-vs-size confound), final
checkpoint (no selection on validation), AdamW + cosine decay. Deterministic on CPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from fdrl.config import PolicyCfg, TrainCfg
from fdrl.policy.bc import Ensemble, MLPPolicy


@dataclass
class TrainResult:
    policy: MLPPolicy
    losses: list[tuple[int, float]]
    seconds: float
    n_samples: int


def train_bc(
    obs: np.ndarray,
    actions: np.ndarray,
    train_cfg: TrainCfg,
    policy_cfg: PolicyCfg,
    seed: int,
    include_latent: bool = False,
    mu_range: tuple[float, float] = (0.1, 1.0),
) -> TrainResult:
    if len(obs) != len(actions) or len(obs) == 0:
        raise ValueError("obs/actions empty or misaligned")
    device = torch.device(train_cfg.device)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    if device.type == "cpu":
        torch.use_deterministic_algorithms(True, warn_only=True)
    policy = MLPPolicy(obs.shape[1], policy_cfg, include_latent=include_latent, mu_range=mu_range)
    policy.set_normalizer(obs)
    policy.to(device)

    O = torch.as_tensor(obs, dtype=torch.float32, device=device)
    A = torch.as_tensor(actions[:, :6], dtype=torch.float32, device=device)
    G = torch.as_tensor((actions[:, 6] > 0).astype(np.float32), device=device)
    opt = torch.optim.AdamW(policy.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(train_cfg.steps, 1))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = len(obs)
    losses: list[tuple[int, float]] = []
    t0 = time.time()
    policy.train()
    for step in range(train_cfg.steps):
        idx = torch.randint(0, n, (train_cfg.batch_size,), generator=gen).to(device)
        pose, logit = policy(O[idx])
        loss = F.mse_loss(pose, A[idx]) + train_cfg.gripper_loss_weight * F.binary_cross_entropy_with_logits(logit, G[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % train_cfg.log_every == 0 or step == train_cfg.steps - 1:
            losses.append((step, float(loss.item())))
    policy.eval()
    return TrainResult(policy, losses, time.time() - t0, n)


def train_ensemble(
    obs: np.ndarray,
    actions: np.ndarray,
    train_cfg: TrainCfg,
    policy_cfg: PolicyCfg,
    seed: int,
    k: int,
) -> tuple[Ensemble, float]:
    members, secs = [], 0.0
    for j in range(k):
        r = train_bc(obs, actions, train_cfg, policy_cfg, seed=seed * 1000 + 17 + j)
        members.append(r.policy)
        secs += r.seconds
    return Ensemble(members), secs


@torch.no_grad()
def fit_error_per_cell(policy: MLPPolicy, obs: np.ndarray, actions: np.ndarray, cells: np.ndarray) -> dict[int, float]:
    """Mean Δpose squared error on TRAINING data, per cell (diagnosis 'train-fit error')."""
    pred = policy.predict(obs)[:, :6]
    err = ((pred - actions[:, :6]) ** 2).mean(axis=1)
    return {int(c): float(err[cells == c].mean()) for c in np.unique(cells)}
