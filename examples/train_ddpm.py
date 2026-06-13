"""DDPM on a 2-D toy distribution (two moons / spirals).

A real denoising diffusion probabilistic model: cosine/linear noise schedule,
an MLP that predicts the added noise, the standard simplified DDPM loss, and
an ancestral sampler used for evaluation. Data is generated on the fly (no
sklearn dependency — the moons are constructed directly).

This is a deliberately different architecture and loss from train_gpt.py, so a
gpusched run mixing the two exercises heterogeneous VRAM footprints. VRAM
scales with --hidden, --depth, and --batch-size. Single-GPU.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (common_args, make_logger, get_device, install_sigterm_handler,
                    reset_vram_peak, save_checkpoint, vram_mb)


# --------------------------------------------------------------------------- #
# Target distribution: two interleaving moons, generated directly.
# --------------------------------------------------------------------------- #
def sample_two_moons(n: int, device: torch.device, noise: float = 0.08) -> torch.Tensor:
    n1 = n // 2
    n2 = n - n1
    t1 = math.pi * torch.rand(n1, device=device)
    moon1 = torch.stack([torch.cos(t1), torch.sin(t1)], dim=1)
    t2 = math.pi * torch.rand(n2, device=device)
    moon2 = torch.stack([1 - torch.cos(t2), 1 - torch.sin(t2) - 0.5], dim=1)
    x = torch.cat([moon1, moon2], dim=0)
    x = x + noise * torch.randn_like(x)
    # normalize roughly to zero-mean unit-ish scale
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


# --------------------------------------------------------------------------- #
# Noise schedule (linear beta) and closed-form q(x_t | x_0).
# --------------------------------------------------------------------------- #
class Diffusion:
    def __init__(self, timesteps: int, device: torch.device):
        self.T = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = torch.cumprod(alphas, dim=0)

    def q_sample(self, x0, t, noise):
        ab = self.alpha_bar[t].unsqueeze(1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise

    @torch.no_grad()
    def sample(self, model, n, device):
        x = torch.randn(n, 2, device=device)
        for t in reversed(range(self.T)):
            tt = torch.full((n,), t, device=device, dtype=torch.long)
            eps = model(x, tt)
            alpha = self.alphas[t]
            ab = self.alpha_bar[t]
            coef = (1 - alpha) / torch.sqrt(1 - ab)
            mean = (x - coef * eps) / torch.sqrt(alpha)
            if t > 0:
                x = mean + torch.sqrt(self.betas[t]) * torch.randn_like(x)
            else:
                x = mean
        return x


# --------------------------------------------------------------------------- #
# Noise-prediction MLP with sinusoidal timestep embedding.
# --------------------------------------------------------------------------- #
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


class NoiseMLP(nn.Module):
    def __init__(self, hidden: int, depth: int, t_dim: int = 64):
        super().__init__()
        self.t_dim = t_dim
        self.t_proj = nn.Sequential(nn.Linear(t_dim, hidden), nn.SiLU())
        self.inp = nn.Linear(2, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)])
        self.out = nn.Linear(hidden, 2)

    def forward(self, x, t):
        h = self.inp(x) + self.t_proj(timestep_embedding(t, self.t_dim))
        for layer in self.layers:
            h = h + F.silu(layer(h))
        return self.out(h)


@torch.no_grad()
def energy_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cheap distribution-match metric: lower = samples closer to target.
    Uses mean pairwise distances (energy distance, subsampled for speed)."""
    a, b = a[:512], b[:512]
    def pdist_mean(x, y):
        return torch.cdist(x, y).mean()
    return (2 * pdist_mean(a, b) - pdist_mean(a, a) - pdist_mean(b, b)).item()


def main():
    p = argparse.ArgumentParser(description="DDPM on a 2-D two-moons distribution")
    common_args(p)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--timesteps", type=int, default=100)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    reset_vram_peak(device)

    config = vars(args) | {"model": "ddpm"}
    run_name = f"ddpm_h{args.hidden}_d{args.depth}_lr{args.lr:g}"
    logger = make_logger(args, config, run_name)
    stop = install_sigterm_handler()

    model = NoiseMLP(args.hidden, args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    diff = Diffusion(args.timesteps, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"[init] device={device} params={n_params/1e6:.3f}M "
          f"hidden={args.hidden} depth={args.depth} batch={args.batch_size} "
          f"T={args.timesteps}", flush=True)

    best_ed = float("inf")
    final_epoch = 0
    for epoch in range(1, args.epochs + 1):
        final_epoch = epoch
        running = 0.0
        for _ in range(args.steps_per_epoch):
            x0 = sample_two_moons(args.batch_size, device)
            t = torch.randint(0, diff.T, (args.batch_size,), device=device)
            noise = torch.randn_like(x0)
            xt = diff.q_sample(x0, t, noise)
            pred = model(xt, t)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
        # evaluate: sample and compare to a fresh target batch
        gen = diff.sample(model, 512, device)
        ref = sample_two_moons(512, device)
        ed = energy_distance(gen, ref)
        best_ed = min(best_ed, ed)
        logger.log_epoch(epoch, loss=running / args.steps_per_epoch,
                         energy_dist=ed, vram_mib=vram_mb(device))
        if stop.stop:
            save_checkpoint(f"{args.out}/ckpt.pt", model, opt, epoch, {"best_ed": best_ed})
            logger.finalize("cancelled", best_energy_dist=round(best_ed, 4),
                            epochs_done=epoch, peak_vram_mib=round(vram_mb(device)))
            return

    save_checkpoint(f"{args.out}/ckpt.pt", model, opt, final_epoch, {"best_ed": best_ed})
    logger.finalize("completed", best_energy_dist=round(best_ed, 4),
                    epochs_done=final_epoch, peak_vram_mib=round(vram_mb(device)),
                    params_m=round(n_params / 1e6, 3))


if __name__ == "__main__":
    main()
