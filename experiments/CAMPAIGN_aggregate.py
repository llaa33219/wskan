"""Aggregate the 3-epoch campaign: 5-seed tables + paired wavelet ablation."""

import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["wskan11", "wskan11real", "mamba2", "tf"]
SIZES = ["1k", "10k", "100k", "1m", "10m"]
DATASETS = ["tinystories", "ultrachat", "wikitext"]
SEEDS = [42, 123, 2024, 7, 31337]


def best(tag):
    p = ROOT / "checkpoints" / tag / "train_log.csv"
    if not p.exists():
        return None
    rows = [r for r in csv.DictReader(open(p)) if r["eval_loss"] not in ("nan", "")]
    return float(min(float(r["eval_loss"]) for r in rows)) if rows else None


def main():
    for ds in DATASETS:
        print(f"\n===== {ds} (best eval CE; mean±std over 5 seeds) =====")
        print(f"{'model':>12} | " + " | ".join(f"{s:>16}" for s in SIZES))
        for m in MODELS:
            cells = []
            for size in SIZES:
                vals = [best(f"{m}_{ds}_{size}_3ep_s{s}") for s in SEEDS]
                vals = [v for v in vals if v is not None]
                if len(vals) == 5:
                    cells.append(f"{st.mean(vals):.4f}±{st.stdev(vals):.4f}")
                elif vals:
                    cells.append(f"{st.mean(vals):.4f}({len(vals)}s)")
                else:
                    cells.append("—")
            print(f"{m:>12} | " + " | ".join(cells))

    print("\n===== wavelet ablation: wskan11 − wskan11real (paired per seed) =====")
    for ds in DATASETS:
        print(f"--- {ds} ---")
        print(f"{'size':>6} | " + " | ".join(f"s{s}" for s in SEEDS) + " | mean±std")
        for size in SIZES:
            diffs = []
            for s in SEEDS:
                a = best(f"wskan11_{ds}_{size}_3ep_s{s}")
                b = best(f"wskan11real_{ds}_{size}_3ep_s{s}")
                diffs.append(None if (a is None or b is None) else a - b)
            ok = [d for d in diffs if d is not None]
            cells = ["—" if d is None else f"{d:+.4f}" for d in diffs]
            mean = f"{st.mean(ok):+.4f}±{st.stdev(ok):.4f}" if len(ok) >= 2 else "—"
            print(f"{size:>6} | " + " | ".join(f"{c:>8}" for c in cells) + f" | {mean}")


if __name__ == "__main__":
    main()
