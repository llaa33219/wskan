"""Aggregate the V8 matrix into the performance table (best eval CE)."""

import csv
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["wskan7bc", "mamba2", "tf", "conv", "lstm"]
SIZES = ["1k", "10k", "100k", "1m", "10m"]
DATASETS = ["tinystories", "ultrachat", "wikitext"]
SEEDS = [42, 123, 2024]
STEPS = {"1k": 20000, "10k": 20000, "100k": 20000, "1m": 10000, "10m": 5000}


def best(tag):
    p = ROOT / "checkpoints" / tag / "train_log.csv"
    if not p.exists():
        return None, 0
    rows = [r for r in csv.DictReader(open(p)) if r["eval_loss"] not in ("nan", "")]
    if not rows:
        return None, 0
    b = min(rows, key=lambda r: float(r["eval_loss"]))
    return float(b["eval_loss"]), int(rows[-1]["step"])


def main():
    done = total = 0
    for ds in DATASETS:
        print(f"\n===== {ds} (best eval CE, mean +- std over seeds) =====")
        hdr = f"{'model':>10} | " + " | ".join(f"{s:>18}" for s in SIZES)
        print(hdr)
        for m in MODELS:
            cells = []
            for size in SIZES:
                vals = []
                for seed in SEEDS:
                    v, last = best(f"{m}_{ds}_{size}_s{seed}")
                    if v is not None:
                        vals.append(v)
                    total += 1
                    if last >= STEPS[size]:
                        done += 1
                if len(vals) == 3:
                    cells.append(f"{st.mean(vals):.4f}±{st.stdev(vals):.4f}")
                elif vals:
                    cells.append(f"{st.mean(vals):.4f} ({len(vals)}s)")
                else:
                    cells.append("—")
            print(f"{m:>10} | " + " | ".join(cells))
    print(f"\ncompleted runs: {done}/{total}")


if __name__ == "__main__":
    main()
