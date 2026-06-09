#!/usr/bin/env python3
import subprocess
import os
import sys

datasets = ['DD', 'PROTEINS']
modes = ['reweight_random_FN']
seeds = [0, 1, 2, 3, 4]

log_file = "result/experiment_summary_missing.txt"
os.makedirs("result", exist_ok=True)

with open(log_file, "w") as f:
    f.write("=== Missing Group B Runs (DD & PROTEINS) ===\n")

for ds in datasets:
    for mode in modes:
        for seed in seeds:
            print(f"Running {ds} {mode} seed={seed}...", flush=True)
            cmd = f"python gsimclr.py --DS {ds} --lr 1e-4 --local --num-gc-layers 5 --aug dnodes --mode {mode} --log_interval 1 --epochs 100 --batch-size 128 --aug_ratio 1 --seed {seed}"
            
            try:
                proc = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=10800)
                acc = None
                for line in proc.stdout.splitlines():
                    if line.startswith('BEST_VAL_ACC='):
                        try:
                            acc = float(line.split('=')[1])
                        except:
                            pass
                status = f"{acc:.4f}" if acc is not None else "parse_failed"
            except subprocess.TimeoutExpired:
                status = "TIMEOUT"
            except subprocess.CalledProcessError as e:
                status = f"FAILED (code {e.returncode})"
            except Exception as e:
                status = f"FAILED ({str(e)})"
            
            res = f"[{ds}] {mode} seed={seed} : {status}\n"
            print(res, flush=True)
            with open(log_file, "a") as f:
                f.write(res)

print(f"All done! Summary saved to {log_file}")
