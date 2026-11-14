"""
Run every experiment listed in experiments.csv by calling train.py.

Example:
    uv run run_experiments.py                      # all experiments, 3 at a time
    uv run run_experiments.py --parallel 1 --total-steps 500000

Experiments that already have a saved dqn_model.pth are skipped, so the
script can be stopped and started again without redoing finished runs.
Each run's console output is saved to models/<member>/exp_<N>/train_output.txt.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def run_one(row, total_steps):
    out_dir = os.path.join("models", row["member"], f"exp_{int(row['exp']):02d}")
    name = f"{row['member']} exp {row['exp']}"
    if os.path.exists(os.path.join(out_dir, "dqn_model.pth")):
        print(f"[skip] {name} (already finished)", flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)

    command = [
        sys.executable, "train.py",
        "--member", row["member"], "--exp", row["exp"],
        "--lr", row["lr"], "--gamma", row["gamma"],
        "--batch-size", row["batch_size"],
        "--eps-start", row["eps_start"], "--eps-end", row["eps_end"],
        "--eps-decay", row["eps_decay"],
        "--total-steps", str(total_steps),
    ]
    print(f"[start] {name}", flush=True)
    start = time.time()
    with open(os.path.join(out_dir, "train_output.txt"), "w") as log_file:
        result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
    minutes = (time.time() - start) / 60
    status = "done" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"[{status}] {name} in {minutes:.0f} min", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run all experiments from experiments.csv")
    parser.add_argument("--file", default="experiments.csv")
    parser.add_argument("--parallel", type=int, default=3, help="how many runs at the same time")
    parser.add_argument("--total-steps", type=int, default=500_000)
    args = parser.parse_args()

    with open(args.file) as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} experiments, {args.parallel} at a time, {args.total_steps} steps each", flush=True)

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for row in rows:
            pool.submit(run_one, row, args.total_steps)


if __name__ == "__main__":
    main()
