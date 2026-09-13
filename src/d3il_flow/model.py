"""Conditional flow matching over action chunks.

x_1 = normalized action chunk (T_a * act_dim), x_0 ~ N(0, I), straight-line (OT) path
x_tau = (1 - tau) x_0 + tau x_1,  target velocity u = x_1 - x_0,  loss = ||v_theta(x_tau, tau | obs) - u||^2.

Hooks for RL later:
* ``sample(..., x0=...)``      -> noise-space control (DSRL-style: RL picks x_0, head frozen)
* ``sample(..., return_path)`` -> the full denoising chain (ReinFlow/DPPO-style policy gradient on steps)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .env import Normalizer


@dataclass
class ModelConfig:
    obs_dim: int = 4
    act_dim: int = 2
    act_steps: int = 4
    hidden: int = 256
    n_layers: int = 3
    time_dim: int = 32
    sigma_min: float = 0.0  # >0 -> Lipman-style tube around the path; 0 -> pure OT-CFM


class SinusoidalEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, tau: torch.Tensor) -> torch.Tensor:  # (B, 1) in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, device=tau.device) / half)
        ang = tau * 1000.0 * freqs  # scale to the usual diffusion-timestep range
        return torch.cat([ang.sin(), ang.cos()], dim=-1)


class FlowMLP(nn.Module):
    """v_theta(x_tau, tau | obs). Residual MLP; small enough to overfit Avoiding in minutes on CPU."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.x_dim = cfg.act_steps * cfg.act_dim
        self.time_emb = nn.Sequential(
            SinusoidalEmb(cfg.time_dim), nn.Linear(cfg.time_dim, cfg.hidden), nn.Mish()
        )
        self.inp = nn.Linear(self.x_dim + cfg.obs_dim + cfg.hidden, cfg.hidden)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, cfg.hidden), nn.Mish()
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.out = nn.Linear(cfg.hidden, self.x_dim)
        nn.init.zeros_(self.out.weight), nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, tau: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        h = F.mish(self.inp(torch.cat([x, obs, self.time_emb(tau)], dim=-1)))
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


def cfm_loss(model: FlowMLP, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """obs: (B, obs_dim); act: (B, T_a, act_dim) normalized to [-1, 1]."""
    x1 = act.flatten(1)
    x0 = torch.randn_like(x1)
    tau = torch.rand(x1.shape[0], 1, device=x1.device)
    s = model.cfg.sigma_min
    x_tau = (1 - (1 - s) * tau) * x0 + tau * x1
    target = x1 - (1 - s) * x0
    return F.mse_loss(model(x_tau, tau, obs), target)


@torch.no_grad()
def sample(
    model: FlowMLP,
    obs: torch.Tensor,
    n_steps: int = 8,
    x0: torch.Tensor | None = None,
    method: str = "euler",
    return_path: bool = False,
):
    """Integrate dx/dtau = v(x, tau | obs) from tau=0 to 1. Returns (B, T_a, act_dim) in [-1, 1]."""
    B = obs.shape[0]
    x = torch.randn(B, model.x_dim, device=obs.device) if x0 is None else x0
    path = [x]
    dt = 1.0 / n_steps
    for k in range(n_steps):
        tau = torch.full((B, 1), k * dt, device=obs.device)
        if method == "midpoint":
            x_mid = x + 0.5 * dt * model(x, tau, obs)
            x = x + dt * model(x_mid, tau + 0.5 * dt, obs)
        else:
            x = x + dt * model(x, tau, obs)
        path.append(x)
    out = x.clamp(-1, 1).view(B, model.cfg.act_steps, model.cfg.act_dim)
    return (out, torch.stack(path, 1)) if return_path else out


class FlowPolicy:
    """Inference wrapper: numpy obs -> numpy action chunk. Bundles model + EMA weights + normalizer."""

    def __init__(
        self,
        model: FlowMLP,
        norm: Normalizer,
        n_steps: int = 8,
        method: str = "euler",
        device: str = "cpu",
    ):
        self.model = model.to(device).eval()
        self.norm, self.n_steps, self.method, self.device = norm, n_steps, method, device

    @torch.no_grad()
    def act(self, obs: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        o = torch.as_tensor(np.asarray(obs, np.float32), device=self.device)
        squeeze = o.ndim == 1
        o = o.unsqueeze(0) if squeeze else o
        x0_t = None if x0 is None else torch.as_tensor(x0, device=self.device).flatten(1)
        a = sample(self.model, o, self.n_steps, x0_t, self.method).cpu().numpy()
        return a[0] if squeeze else a

    # -- persistence -----------------------------------------------------------------------------
    @staticmethod
    def save(
        path, model: FlowMLP, ema_state: dict | None, norm: Normalizer, extra: dict | None = None
    ):
        torch.save(
            {
                "model_cfg": asdict(model.cfg),
                "state_dict": model.state_dict(),
                "ema_state_dict": ema_state,
                "norm": norm.as_dict(),
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(cls, path, device: str = "cpu", use_ema: bool = True, **kw) -> FlowPolicy:
        ck = torch.load(path, map_location=device, weights_only=False)
        model = FlowMLP(ModelConfig(**ck["model_cfg"]))
        sd = ck["ema_state_dict"] if (use_ema and ck.get("ema_state_dict")) else ck["state_dict"]
        model.load_state_dict(sd)
        return cls(
            model,
            Normalizer(**{k: np.asarray(v, np.float32) for k, v in ck["norm"].items()}),
            device=device,
            **kw,
        )


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}
