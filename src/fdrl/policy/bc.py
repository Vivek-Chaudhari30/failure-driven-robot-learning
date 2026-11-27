"""MLP behavior-cloning policy, observation normalizer and ensemble (EXPERIMENT_DESIGN section 9).

Heads: 6-D Δpose (MSE, i.e. fixed-σ Gaussian) and gripper open-probability (BCE).
Gripper label convention: action[6] > 0 means open.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fdrl.config import PolicyCfg
from fdrl.sim.obs import append_latent


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, cfg: PolicyCfg, include_latent: bool = False,
                 mu_range: tuple[float, float] = (0.1, 1.0)):
        super().__init__()
        self.obs_dim = obs_dim
        self.include_latent = include_latent
        self.mu_range = tuple(mu_range)
        self.hidden = cfg.hidden
        self.n_layers = cfg.layers
        layers: list[nn.Module] = []
        d = obs_dim
        for _ in range(cfg.layers):
            layers += [nn.Linear(d, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.GELU()]
            d = cfg.hidden
        self.trunk = nn.Sequential(*layers)
        self.pose_head = nn.Linear(d, 6)
        self.grip_head = nn.Linear(d, 1)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def set_normalizer(self, obs: np.ndarray) -> None:
        mean = obs.mean(axis=0)
        std = obs.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        self.obs_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk((obs - self.obs_mean) / self.obs_std)
        return self.pose_head(h), self.grip_head(h).squeeze(-1)

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        dev = self.obs_mean.device
        pose, logit = self(torch.as_tensor(obs, dtype=torch.float32, device=dev))
        pose = pose.clamp(-1.0, 1.0).cpu().numpy()
        grip = np.where(torch.sigmoid(logit).cpu().numpy() >= 0.5, 1.0, -1.0)
        return np.concatenate([pose, grip[:, None]], axis=1).astype(np.float32)

    def controller(self):
        """Callable for fdrl.sim.rollout: (obs, priv) -> actions."""
        self.eval()

        def _ctrl(obs: np.ndarray, priv: dict[str, np.ndarray]) -> np.ndarray:
            if self.include_latent:
                obs = append_latent(obs, priv["mu"], self.mu_range)
            return self.predict(obs)

        return _ctrl

    # ------------------------------------------------------------------ persistence
    def policy_id(self) -> str:
        """Content hash of the weights: identifies a checkpoint in episode records."""
        h = hashlib.sha256()
        for k, v in sorted(self.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    def save(self, path: str | Path, meta: dict | None = None) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pid = self.policy_id()
        torch.save(
            {
                "state_dict": self.state_dict(),
                "obs_dim": self.obs_dim,
                "hidden": self.hidden,
                "layers": self.n_layers,
                "include_latent": self.include_latent,
                "mu_range": list(self.mu_range),
                "policy_id": pid,
                "meta": meta or {},
            },
            path,
        )
        return pid

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> MLPPolicy:
        ck = torch.load(path, map_location=device, weights_only=False)
        pol = cls(ck["obs_dim"], PolicyCfg(hidden=ck["hidden"], layers=ck["layers"]),
                  include_latent=ck["include_latent"], mu_range=tuple(ck["mu_range"]))
        pol.load_state_dict(ck["state_dict"])
        pol.to(device)
        if pol.policy_id() != ck["policy_id"]:
            raise ValueError(f"checkpoint {path} content hash mismatch")
        return pol

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        torch.save(self.state_dict(), buf)
        return buf.getvalue()


class Ensemble:
    """K independently-seeded policies; used for the epistemic/aleatoric diagnosis signal."""

    def __init__(self, members: list[MLPPolicy]):
        if not members:
            raise ValueError("empty ensemble")
        self.members = members

    @torch.no_grad()
    def predict_pose(self, obs: np.ndarray) -> np.ndarray:
        """[K, N, 6] Δpose predictions."""
        out = []
        for m in self.members:
            m.eval()
            pose, _ = m(torch.as_tensor(obs, dtype=torch.float32, device=m.obs_mean.device))
            out.append(pose.clamp(-1, 1).cpu().numpy())
        return np.stack(out)
