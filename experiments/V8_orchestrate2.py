"""File-based orchestrator for the V8 matrix (robust replacement).

Prepare: writes one job-slice file per free GPU to /tmp/opencode/v8_slice_<n>.cmd
Each line: <tag>;<command>. Launch each slice with:
  setsid nohup bash experiments/V8_worker.sh <slice-file> <gpu-uuid> &

Resumable: completed runs (train_log.csv with final step) are skipped.

Usage: .venv/bin/python experiments/V8_orchestrate2.py [--gpus U1,U2,...]
"""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["wskan7bc", "mamba2", "tf", "conv", "lstm"]
SIZES = ["10m", "1m", "100k", "10k", "1k"]
DATASETS = ["ultrachat", "wikitext", "tinystories"]
SEEDS = [42, 123, 2024]
STEPS = {"1k": 20000, "10k": 20000, "100k": 20000, "1m": 10000, "10m": 5000}
TMP = Path("/tmp/opencode")


def job_cmd(model, size, dataset, seed):
    compile_ = " --compile" if (
        (model in ("mamba2", "tf", "conv") and size in ("100k", "1m", "10m"))
        or (model == "wskan7bc" and size == "100k")
    ) else ""
    return (f".venv/bin/python -u experiments/V1_train_tinystories_lm.py "
            f"--model {model} --scale {size} --dataset {dataset} --seed {seed} "
            f"--block 256 --train-stories 200000 --eval-stories 500 "
            f"--ckpt-every 100000 --eval-every 1000{compile_}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=str, default=None)
    args = ap.parse_args()
    if args.gpus:
        gpus = args.gpus.split(",")
    else:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader"]).decode()
        gpus = [l.split(",")[0] for l in out.strip().splitlines()
                if int(l.split(",")[1].split()[0]) < 500]
    n = len(gpus)
    slices = [[] for _ in range(n)]
    i = 0
    pending = 0
    for size in SIZES:
        for dataset in DATASETS:
            for model in MODELS:
                for seed in SEEDS:
                    tag = f"{model}_{dataset}_{size}_s{seed}"
                    csv = ROOT / "checkpoints" / tag / "train_log.csv"
                    if csv.exists():
                        rows = [l for l in csv.read_text().splitlines()[1:] if l.strip()]
                        if rows and rows[-1].split(",")[0] == str(STEPS[size]):
                            continue
                    slices[i % n].append(f"{tag};{job_cmd(model, size, dataset, seed)}")
                    i += 1
                    pending += 1
    for k, sl in enumerate(slices):
        p = TMP / f"v8_slice_{k}.cmd"
        p.write_text("\n".join(sl) + "\n")
    (ROOT / ".v8_logs").mkdir(exist_ok=True)
    print(f"{n} GPUs, {pending} pending jobs")
    for k, g in enumerate(gpus):
        print(f"setsid nohup bash experiments/V8_worker.sh {TMP}/v8_slice_{k}.cmd {g} "
              f"> {TMP}/v8_worker_{k}.log 2>&1 < /dev/null &")


if __name__ == "__main__":
    main()
