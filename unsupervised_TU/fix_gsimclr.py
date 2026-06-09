#!/usr/bin/env python3
import os

with open('/home/qiannnhui/GraphCL/unsupervised_TU/gsimclr.py', 'r') as f:
    current_lines = f.readlines()

with open('/home/qiannnhui/GraphCL/unsupervised_TU/gsimclr.py.bak', 'r') as f:
    bak_lines = f.readlines()

# Find where current ends
end_idx_current = 0
for i, line in enumerate(current_lines):
    if "optimizer = torch.optim.Adam" in line:
        end_idx_current = i + 1

# Find where the rest of the code begins in bak
start_idx_bak = 0
for i, line in enumerate(bak_lines):
    if "tmp_loader = DataLoader(dataset_eval" in line:
        start_idx_bak = i
        break

# The lines from start_idx_bak to the end of bak are what we need to append.
# BUT we need to skip the tmp_loader logic that caused OOM!
rest_lines = bak_lines[start_idx_bak:]
filtered_rest_lines = []
skip = False
for line in rest_lines:
    if "tmp_loader = DataLoader(dataset_eval" in line:
        skip = True
    if skip and "del batch_all, tmp_loader" in line:
        skip = False
        continue
    if not skip:
        filtered_rest_lines.append(line)

# Now we need to insert args.mode == 'reweight_random_FN' logic inside the train loop
# and print BEST_VAL_ACC at the very end.

final_lines = current_lines[:end_idx_current] + filtered_rest_lines

# Inject reweight_random_FN
for i, line in enumerate(final_lines):
    if "elif args.mode == 'MSE_infonce':" in line:
        # Before this line, insert reweight_random_FN logic
        inject_code = """            elif args.mode == 'reweight_random_FN':
                loss, pos_sim, neg_sim, fn_metrics = unified_loss(
                    x, x_aug, labels, 
                    pos_strategy='normal', 
                    neg_strategy='random_drop_fn', drop_p=0.5,
                    sim_measure=args.similarity_measure
                )
                if 'precision' not in epoch_metrics:
                    epoch_metrics.update({'precision': 0.0, 'recall': 0.0, 'f1': 0.0})
                epoch_metrics['precision'] += fn_metrics.get('precision', 0)
                epoch_metrics['recall'] += fn_metrics.get('recall', 0)
                epoch_metrics['f1'] += fn_metrics.get('f1', 0)
"""
        final_lines.insert(i, inject_code)
        break

# Inject epoch_metrics initialization if reweight_random_FN
for i, line in enumerate(final_lines):
    if "if args.mode == 'focal_infonce' or args.mode == 'PPR_infonce':" in line:
        inject_code = """        if args.mode == 'reweight_random_FN':
            epoch_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
"""
        final_lines.insert(i, inject_code)
        break

# Inject BEST_VAL_ACC and BEST_FN_METRICS at the end of the file
best_acc_code = """    print(f'BEST_VAL_ACC={best_val_acc}')
    if args.mode == 'reweight_random_FN':
        if 'precision' in epoch_metrics:
            # We approximate the best by the last epoch's metrics or keep an average, 
            # here we just print the best_fn_metrics. Wait, best_fn_metrics was saved during best_val_acc
            print(f'BEST_FN_METRICS={json.dumps(best_fn_metrics)}')
"""
final_lines.append(best_acc_code)

# Inject saving best_fn_metrics when best_val_acc is found
for i, line in enumerate(final_lines):
    if "best_val_acc = acc_val" in line:
        inject_code = """                if args.mode == 'reweight_random_FN':
                    num_batches = len(dataloader)
                    best_fn_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
"""
        final_lines.insert(i+1, inject_code)
        break

with open('/home/qiannnhui/GraphCL/unsupervised_TU/gsimclr.py', 'w') as f:
    f.writelines(final_lines)

print("gsimclr.py has been fully restored and patched!")
