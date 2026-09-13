"""CFM behaviour cloning on D3IL Avoiding.

d3il-flow-train --data data/d3il/avoid_m1/train.npz --out outputs/avoid_m1_flow
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import ChunkDataset
from .env import load_normalizer
from .model import EMA, FlowMLP, FlowPolicy, ModelConfig, cfm_loss


def parse():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", default="data/d3il/avoid_m1/train.npz")
    p.add_argument(
        "--norm-dir", default=None, help="dir with normalization.npz (default: dirname of --data)"
    )
    p.add_argument("--out", default="outputs/avoid_m1_flow")
    p.add_argument("--act-steps", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--sigma-min", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def main():
    a = parse()
    torch.manual_seed(a.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    norm = load_normalizer(a.norm_dir or Path(a.data).parent)
    ds = ChunkDataset(a.data, act_steps=a.act_steps, norm=norm)
    print(f"[data] {ds.summary()}")
    dl = DataLoader(
        ds,
        batch_size=a.bs,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=device == "cuda",
    )

    mcfg = ModelConfig(
        obs_dim=ds.obs_dim,
        act_dim=ds.act_dim,
        act_steps=a.act_steps,
        hidden=a.hidden,
        n_layers=a.layers,
        sigma_min=a.sigma_min,
    )
    model = FlowMLP(mcfg).to(device)
    ema = EMA(model, a.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))
        ),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params / 1e3:.1f}k params on {device}; cfg={mcfg}")

    run = None
    if a.wandb:
        import wandb

        run = wandb.init(project="d3il-flow", config={**vars(a), **mcfg.__dict__})

    (out / "config.json").write_text(json.dumps({**vars(a), "model": mcfg.__dict__}, indent=2))

    step, t0, running = 0, time.time(), 0.0
    model.train()
    while step < a.steps:
        for obs, act in dl:
            obs, act = obs.to(device, non_blocking=True), act.to(device, non_blocking=True)
            loss = cfm_loss(model, obs, act)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(model)
            step += 1
            running += loss.item()
            if step % a.log_every == 0:
                msg = {
                    "step": step,
                    "loss": running / a.log_every,
                    "lr": sched.get_last_lr()[0],
                    "sps": step / (time.time() - t0),
                }
                print(f"[train] {msg}")
                if run:
                    run.log(msg, step=step)
                running = 0.0
            if step % a.save_every == 0 or step == a.steps:
                FlowPolicy.save(
                    out / "ckpt.pt",
                    model,
                    ema.state_dict(),
                    norm,
                    extra={"step": step, "args": vars(a)},
                )
            if step >= a.steps:
                break
    print(f"[done] saved {out / 'ckpt.pt'} ({time.time() - t0:.0f}s)")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
