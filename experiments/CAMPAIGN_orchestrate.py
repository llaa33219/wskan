"""3-epoch campaign orchestrator (writes per-GPU slices; launch via tmux).

4 models (wskan7bc, wskan7bcreal [omega=0 ablation], mamba2, tf) x 5 sizes
x 3 datasets x 5 seeds. Tiered step counts (see report recommendation):
100k tier runs the full protocol (TS 2.6ep, UC/WT 3ep); others are
capacity-appropriate caps. batch=64 for all runs (documented protocol change).

Data (measured): tinystories ~1.9GB full, ultrachat ~1.18GB full,
wikitext-103 ~542MB.

Usage: .venv/bin/python experiments/CAMPAIGN_orchestrate.py
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path("/tmp/opencode")
MODELS = ["wskan11", "wskan11real", "mamba2", "tf"]
SIZES = ["10m", "1m", "100k", "10k", "1k"]
DATASETS = ["ultrachat", "wikitext", "tinystories"]
SEEDS = [42, 123, 2024, 7, 31337]
BATCH = 64
BLOCK = {"tinystories": 256, "ultrachat": 512, "wikitext": 256}
STEPS = {
    ("tinystories", "1k"): 30_000, ("tinystories", "10k"): 60_000, ("tinystories", "100k"): 300_000,
    ("tinystories", "1m"): 100_000, ("tinystories", "10m"): 33_000,
    ("ultrachat", "1k"): 36_000, ("ultrachat", "10k"): 54_000, ("ultrachat", "100k"): 108_000,
    ("ultrachat", "1m"): 54_000, ("ultrachat", "10m"): 36_000,
    ("wikitext", "1k"): 33_000, ("wikitext", "10k"): 50_000, ("wikitext", "100k"): 99_000,
    ("wikitext", "1m"): 66_000, ("wikitext", "10m"): 33_000,
}
TRAIN_STORIES = {"tinystories": 3_000_000, "ultrachat": 1_400_000, "wikitext": 2_000_000}
CKPT_EVERY = {"1k": 25_000, "10k": 25_000, "100k": 25_000, "1m": 10_000, "10m": 5_000}


def job_cmd(model, size, dataset, seed):
    compile_ = " --compile" if (
        (model in ("mamba2", "tf") and size in ("100k", "1m", "10m"))
        or (model in ("wskan11", "wskan11real"))
    ) else ""
    lr = "3e-3" if size in ("1k", "10k", "100k") else "1e-3"
    steps = STEPS[(dataset, size)]
    bf16 = " --bf16" if model.startswith("wskan11") else ""
    return (f".venv/bin/python -u experiments/V1_train_tinystories_lm.py "
            f"--model {model} --scale {size} --dataset {dataset} --seed {seed} "
            f"--batch {BATCH} --block {BLOCK[dataset]} "
            f"--train-stories {TRAIN_STORIES[dataset]} --eval-stories 500 "
            f"--steps {steps} --lr {lr} --lr-schedule cosine "
            f"--ckpt-every {CKPT_EVERY[size]} --eval-every 1000 --out-tag 3ep{compile_}{bf16}")


def done(tag, dataset, size):
    csv = ROOT / "checkpoints" / tag / "train_log.csv"
    if not csv.exists():
        return False
    rows = [l for l in csv.read_text().splitlines()[1:] if l.strip()]
    return bool(rows) and rows[-1].split(",")[0] == str(STEPS[(dataset, size)])


def free_gpus():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader"]).decode()
    return [l.split(",")[0] for l in out.strip().splitlines()
            if int(l.split(",")[1].split()[0]) < 500]


def main():
    gpus = free_gpus()
    n = max(1, len(gpus))
    slices = [[] for _ in range(n)]
    pending = 0
    for size in SIZES:
        for ds in DATASETS:
            for model in MODELS:
                for seed in SEEDS:
                    tag = f"{model}_{ds}_{size}_3ep_s{seed}"
                    if done(tag, ds, size):
                        continue
                    slices[pending % n].append(f"{tag};{job_cmd(model, size, ds, seed)}")
                    pending += 1
    for k, sl in enumerate(slices):
        (TMP / f"camp_slice_{k}.cmd").write_text("\n".join(sl) + "\n")
    (ROOT / ".v8_logs").mkdir(exist_ok=True)
    print(f"{n} GPUs, {pending} pending jobs, batch {BATCH}")
    for k, g in enumerate(gpus):
        print(f"camp{k} GPU={g}")


if __name__ == "__main__":
    main()
