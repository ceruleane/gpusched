"""Small GPT on a synthetic sorting task.

The model learns to sort a short sequence of integers: given a scrambled
sequence, predict the sorted sequence (a classic toy task that genuinely
exercises a causal transformer and whose loss visibly decreases). No data is
downloaded — sequences are generated on the fly.

VRAM scales with --d-model, --n-layer, and --batch-size, so different sweep
configs land at different memory footprints. Run with --help for knobs.

Single-GPU; gpusched sets CUDA_VISIBLE_DEVICES so this always sees one device.
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
# Synthetic task: sort sequences of integers in [0, VOCAB).
# Input  : a scrambled sequence of length L
# Target : the same multiset, sorted ascending
# We frame it as next-token prediction over [input | sorted] and only score
# the sorted half (standard "copy/sort" transformer toy task).
# --------------------------------------------------------------------------- #
VOCAB = 50          # token values 0..49
SEQ_LEN = 16        # length of the sequence to sort


def make_batch(batch_size: int, device: torch.device):
    x = torch.randint(0, VOCAB, (batch_size, SEQ_LEN), device=device)
    y = torch.sort(x, dim=1).values
    # full sequence: [scrambled, sorted]; predict position t+1 from <= t
    full = torch.cat([x, y], dim=1)                  # (B, 2L)
    inp = full[:, :-1]                               # (B, 2L-1)
    tgt = full[:, 1:].clone()                        # (B, 2L-1)
    # only score predictions of the sorted half
    tgt[:, : SEQ_LEN - 1] = -100                     # ignore_index in the input half
    return inp, tgt


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, block: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.register_buffer("mask", torch.tril(torch.ones(block, block)).view(1, 1, block, block))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        h = self.n_head
        q = q.view(B, T, h, C // h).transpose(1, 2)
        k = k.view(B, T, h, C // h).transpose(1, 2)
        v = v.view(B, T, h, C // h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // h)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, block: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, d_model: int, n_layer: int, n_head: int, block: int):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d_model)
        self.pos = nn.Embedding(block, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, block) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB, bias=False)
        self.block = block

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


@torch.no_grad()
def sort_accuracy(model, device, n_batches: int, batch_size: int) -> float:
    """Exact-token accuracy on the sorted half."""
    model.eval()
    correct = total = 0
    for _ in range(n_batches):
        inp, tgt = make_batch(batch_size, device)
        logits = model(inp)
        pred = logits.argmax(-1)
        scored = tgt != -100
        correct += (pred[scored] == tgt[scored]).sum().item()
        total += scored.sum().item()
    model.train()
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser(description="Small GPT on a synthetic sort task")
    common_args(p)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--spike-every", type=int, default=0,
                   help="every N epochs, transiently allocate a big scratch tensor "
                        "to mimic a periodic VRAM spike (0 = off). Exercises gpusched's "
                        "--spike-buffer protection for fluctuating jobs.")
    p.add_argument("--spike-mib", type=int, default=2048,
                   help="size of the transient spike allocation in MiB")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    reset_vram_peak(device)
    block = 2 * SEQ_LEN - 1

    config = vars(args) | {"model": "gpt", "vocab": VOCAB, "seq_len": SEQ_LEN}
    run_name = f"gpt_d{args.d_model}_l{args.n_layer}_lr{args.lr:g}"
    logger = make_logger(args, config, run_name)
    stop = install_sigterm_handler()

    model = GPT(args.d_model, args.n_layer, args.n_head, block).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"[init] device={device} params={n_params/1e6:.2f}M "
          f"d_model={args.d_model} n_layer={args.n_layer} batch={args.batch_size}", flush=True)

    best_acc = 0.0
    final_epoch = 0
    for epoch in range(1, args.epochs + 1):
        final_epoch = epoch
        # Optional periodic VRAM spike: transiently grab a big scratch tensor,
        # then free it. On CUDA this raises the polled memory for a moment, which
        # is exactly the fluctuating-VRAM pattern gpusched's --spike-buffer is
        # meant to absorb without double-booking the card.
        if args.spike_every and epoch % args.spike_every == 0 and device.type == "cuda":
            scratch = torch.empty(args.spike_mib * 1024 * 1024 // 4,
                                  dtype=torch.float32, device=device)
            scratch += 1.0
            del scratch
            torch.cuda.synchronize(device)
        running = 0.0
        for _ in range(args.steps_per_epoch):
            inp, tgt = make_batch(args.batch_size, device)
            logits = model(inp)
            loss = F.cross_entropy(logits.view(-1, VOCAB), tgt.view(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
        acc = sort_accuracy(model, device, n_batches=5, batch_size=args.batch_size)
        best_acc = max(best_acc, acc)
        logger.log_epoch(epoch, loss=running / args.steps_per_epoch,
                         sort_acc=acc, vram_mib=vram_mb(device))
        if stop.stop:
            save_checkpoint(f"{args.out}/ckpt.pt", model, opt, epoch,
                            {"best_acc": best_acc})
            logger.finalize("cancelled", best_acc=round(best_acc, 4),
                            epochs_done=epoch, peak_vram_mib=round(vram_mb(device)))
            return

    save_checkpoint(f"{args.out}/ckpt.pt", model, opt, final_epoch, {"best_acc": best_acc})
    logger.finalize("completed", best_acc=round(best_acc, 4),
                    epochs_done=final_epoch, peak_vram_mib=round(vram_mb(device)),
                    params_m=round(n_params / 1e6, 2))


if __name__ == "__main__":
    main()
