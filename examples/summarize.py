"""Summarize a finished sweep: rank runs by their metric, per architecture.

    python summarize.py --out-root runs

Reads every runs/*/summary.json, groups by model, and prints a leaderboard
(GPT by sort accuracy desc; DDPM by energy distance asc). Runs still in
progress or cancelled show up with whatever they reached. No dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summaries(out_root: str) -> list[dict]:
    rows = []
    for summ in Path(out_root).glob("*/summary.json"):
        try:
            data = json.loads(summ.read_text())
            data["_run"] = summ.parent.name
            rows.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return rows


def fmt_config(cfg: dict, keys: list[str]) -> str:
    return "  ".join(f"{k}={cfg.get(k)}" for k in keys if k in cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default="runs")
    args = p.parse_args()

    rows = load_summaries(args.out_root)
    if not rows:
        print(f"No summary.json files under {args.out_root}/ yet.")
        return

    gpt = [r for r in rows if r.get("config", {}).get("model") == "gpt"]
    ddpm = [r for r in rows if r.get("config", {}).get("model") == "ddpm"]

    if gpt:
        gpt.sort(key=lambda r: r.get("best_acc", -1), reverse=True)
        print("\n=== GPT sweep — ranked by sort accuracy (higher is better) ===")
        print(f"{'run':<28} {'acc':>7} {'status':>10} {'epochs':>7} {'peakMiB':>9} {'sec':>6}")
        for r in gpt:
            cfg = r.get("config", {})
            print(f"{r['_run']:<28} {r.get('best_acc', 0):>7.3f} "
                  f"{r.get('status', '?'):>10} {r.get('epochs_done', '?'):>7} "
                  f"{r.get('peak_vram_mib', '?'):>9} {r.get('elapsed_s', 0):>6.0f}")

    if ddpm:
        ddpm.sort(key=lambda r: r.get("best_energy_dist", 1e9))
        print("\n=== DDPM sweep — ranked by energy distance (lower is better) ===")
        print(f"{'run':<28} {'enDist':>8} {'status':>10} {'epochs':>7} {'peakMiB':>9} {'sec':>6}")
        for r in ddpm:
            print(f"{r['_run']:<28} {r.get('best_energy_dist', 0):>8.4f} "
                  f"{r.get('status', '?'):>10} {r.get('epochs_done', '?'):>7} "
                  f"{r.get('peak_vram_mib', '?'):>9} {r.get('elapsed_s', 0):>6.0f}")
    print()


if __name__ == "__main__":
    main()
