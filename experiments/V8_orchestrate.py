"""Orchestrates the V8 cross-family benchmark matrix.

5 models (wskan7bc, mamba2, tf, conv, lstm) x 5 sizes (1k..10m) x 3 datasets
(tinystories, ultrachat, wikitext) x 3 seeds = 225 runs, scheduled onto free
GPUs (one sequential worker per GPU). Resumable: a run whose train_log.csv
already contains the final step is skipped.

Usage: .venv/bin/python experiments/V8_orchestrate.py [--gpus UUID,UUID,...] [--dry]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["wskan7bc", "mamba2", "tf", "conv", "lstm"]
SIZES = ["10m", "1m", "100k", "10k", "1k"]  # big first
DATASETS = ["ultrachat", "wikitext", "tinystories"]
SEEDS = [42, 123, 2024]
STEPS = {"1k": 20000, "10k": 20000, "100k": 20000, "1m": 10000, "10m": 5000}


def final_step(scale: str) -> int:
    return STEPS[scale]


def job_cmd(model, size, dataset, seed):
    compile_ = " --compile" if (
        (model in ("mamba2", "tf", "conv") and size in ("100k", "1m", "10m"))
        or (model == "wskan7bc" and size in ("100k",))  # 1m/10m: internal chunk compile only
    ) else ""
    return (f".venv/bin/python -u experiments/V1_train_tinystories_lm.py "
            f"--model {model} --scale {size} --dataset {dataset} --seed {seed} "
            f"--block 256 --train-stories 200000 --eval-stories 500 "
            f"--ckpt-every 100000 --eval-every 1000{compile_}")


def jobs():
    for size in SIZES:
        for dataset in DATASETS:
            for model in MODELS:
                for seed in SEEDS:
                    tag = f"{model}_{dataset}_{size}_s{seed}"
                    out = ROOT / "checkpoints" / tag / "train_log.csv"
                    if out.exists():
                        rows = [l for l in out.read_text().splitlines()[1:] if l.strip()]
                        if rows and rows[-1].split(",")[0] == str(final_step(size)):
                            continue  # done
                    yield tag, job_cmd(model, size, dataset, seed)


def free_gpus():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"]).decode()
    gpus = []
    for line in out.strip().splitlines():
        idx, uuid, used = [x.strip() for x in line.split(",")]
        if int(used.split()[0]) < 500:
            gpus.append(uuid)
    return gpus


def worker(uuid, queue, log):
    env_note = f"CUDA_VISIBLE_DEVICES={uuid} TORCHINDUCTOR_CACHE_DIR=/tmp/opencode/inductor"
    while True:
        if not queue:
            break
        tag, cmd = queue.pop(0)
        t0 = time.time()
        print(f"[{uuid[:16]}] START {tag}", flush=True)
        r = subprocess.run(f"env {env_note} {cmd}", shell=True, cwd=ROOT,
                           capture_output=True, text=True)
        dt = (time.time() - t0) / 60
        ok = r.returncode == 0
        print(f"[{uuid[:16]}] {'DONE ' if ok else 'FAIL '}{tag} ({dt:.1f} min)", flush=True)
        (ROOT / ".v8_logs").mkdir(exist_ok=True)
        (ROOT / ".v8_logs" / f"{tag}.log").write_text(r.stdout[-4000:] + "\n" + r.stderr[-4000:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    gpus = args.gpus.split(",") if args.gpus else free_gpus()
    queue = list(jobs())
    print(f"free GPUs: {len(gpus)} | jobs remaining: {len(queue)}")
    if args.dry:
        for tag, cmd in queue[:10]:
            print(tag, "::", cmd)
        return

    import threading
    threads = []
    for g in gpus:
        t = threading.Thread(target=worker, args=(g, queue, None), daemon=True)
        t.start()
        time.sleep(20)  # stagger data loading
    for t in threads:
        t.join()
    print("ALL QUEUES EMPTY")


if __name__ == "__main__":
    main()
