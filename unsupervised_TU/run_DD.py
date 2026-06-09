#!/usr/bin/env python3
"""
GraphCL Multi-Seed Experiment Runner (For DD only)
=====================================
Group A: normal
Group B: reweight_random_FN
Batch size reduced to 8 to prevent OOM for large graphs in DD.
"""

import subprocess
import json
import os
import re
import numpy as np
import sys
from datetime import datetime

DATASETS   = ['DD']
MODES      = ['normal', 'reweight_random_FN']
SEEDS      = [0, 1, 2, 3, 4]
EPOCHS     = 100
LOG_INT    = 1
LR         = '1e-4'
AUG        = 'dnodes'
NUM_GC     = 5
BATCH_SIZE = 8
AUG_RATIO  = 1

CUDA_DEV   = sys.argv[1] if len(sys.argv) > 1 else '0'
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SUMMARY_FILE = os.path.join(BASE_DIR, 'result', 'experiment_summary_DD.txt')
os.makedirs(os.path.join(BASE_DIR, 'result'), exist_ok=True)

summary_lines = []
header = (
    '=' * 70 + '\n'
    f'GraphCL DD Experiment Summary  —  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
    f'Epochs={EPOCHS}  BatchSize={BATCH_SIZE}  LR={LR}  Aug={AUG}\n'
    '=' * 70
)
summary_lines.append(header)
print(header)

for ds in DATASETS:
    ds_block = [f'\n{"─"*60}', f'Dataset: {ds}', f'{"─"*60}']
    print('\n'.join(ds_block))
    summary_lines.extend(ds_block)

    for mode in MODES:
        group_label = 'A (normal)' if mode == 'normal' else 'B (reweight_random_FN)'
        print(f'\n  ▶ Group {group_label}')
        seed_accs = []
        seed_precisions = []
        seed_recalls = []
        seed_f1s = []

        for seed in SEEDS:
            cmd = [
                'python', os.path.join(BASE_DIR, 'gsimclr.py'),
                '--DS', ds,
                '--lr', LR,
                '--local',
                '--num-gc-layers', str(NUM_GC),
                '--aug', AUG,
                '--mode', mode,
                '--log_interval', str(LOG_INT),
                '--epochs', str(EPOCHS),
                '--batch-size', str(BATCH_SIZE),
                '--aug_ratio', str(AUG_RATIO),
                '--seed', str(seed),
            ]
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = CUDA_DEV

            print(f'    [{ds}] {mode} seed={seed}  ... ', end='', flush=True)
            acc = None
            fn_metrics = {}
            try:
                proc = subprocess.run(
                    cmd, cwd=BASE_DIR, env=env, capture_output=True, text=True, timeout=7200
                )
                if proc.returncode != 0:
                    print(f'FAILED (code {proc.returncode})')
                    print(f'      stderr tail: {proc.stderr[-800:]}')
                    seed_accs.append(None)
                    if mode == 'reweight_random_FN': seed_precisions.append(None)
                    continue
                
                for line in proc.stdout.splitlines():
                    if line.startswith('BEST_VAL_ACC='):
                        acc = float(line.split('=')[1])
                    if line.startswith('BEST_FN_METRICS='):
                        try: fn_metrics = json.loads(line.split('=', 1)[1])
                        except: pass
            except subprocess.TimeoutExpired:
                print('TIMEOUT')
                seed_accs.append(None)
                if mode == 'reweight_random_FN': seed_precisions.append(None)
                continue
            except Exception as e:
                print(f'ERROR: {e}')
                seed_accs.append(None)
                if mode == 'reweight_random_FN': seed_precisions.append(None)
                continue

            seed_accs.append(acc)
            output_str = f'val_acc={acc:.4f}' if acc is not None else 'parse_failed'
            if mode == 'reweight_random_FN' and fn_metrics:
                prec = fn_metrics.get('precision', None)
                rec = fn_metrics.get('recall', None)
                f1 = fn_metrics.get('f1', None)
                seed_precisions.append(prec)
                seed_recalls.append(rec)
                seed_f1s.append(f1)
                if prec is not None: output_str += f', pFN prec={prec:.4f}'
            print(output_str)

        valid = [a for a in seed_accs if a is not None]
        per_seed_str = ', '.join([f'{a:.4f}' if a is not None else 'N/A' for a in seed_accs])
        
        if valid:
            mean_acc = np.mean(valid)
            std_acc  = np.std(valid)
            result_str = (
                f'\n  [Group {group_label}]\n'
                f'    Per-seed ACC : [{per_seed_str}]\n'
                f'    Mean±Std ACC : {mean_acc:.4f} ± {std_acc:.4f}\n'
            )
            print(f'\n  → {mode}  ACC Mean: {mean_acc:.4f} ± {std_acc:.4f}')
            
            if mode == 'reweight_random_FN':
                valid_prec = [p for p in seed_precisions if p is not None]
                if valid_prec:
                    mean_prec = np.mean(valid_prec)
                    std_prec = np.std(valid_prec)
                    per_seed_prec_str = ', '.join([f'{p:.4f}' if p is not None else 'N/A' for p in seed_precisions])
                    result_str += (
                        f'    Per-seed Precision : [{per_seed_prec_str}]\n'
                        f'    Mean±Std Precision : {mean_prec:.4f} ± {std_prec:.4f}\n'
                    )
                    print(f'  → {mode}  Precision Mean: {mean_prec:.4f} ± {std_prec:.4f}')
        else:
            result_str = f'\n  [Group {group_label}]\n    ALL RUNS FAILED\n'
            print(f'\n  → {mode}  ALL RUNS FAILED')
        summary_lines.append(result_str)

        with open(SUMMARY_FILE, 'w') as f:
            f.write('\n'.join(summary_lines) + '\n')

summary_lines.append('\n' + '=' * 70 + '\n')
full_summary = '\n'.join(summary_lines)
with open(SUMMARY_FILE, 'w') as f:
    f.write(full_summary)
print(f'\nSummary saved → {SUMMARY_FILE}')
