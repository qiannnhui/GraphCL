#!/usr/bin/env python3
import os

# 1. Patch unified_loss.py
with open('/home/qiannnhui/GraphCL/unsupervised_TU/unified_loss.py', 'r') as f:
    ul_content = f.read()

if "elif neg_strategy == 'random_drop_fn':" not in ul_content:
    # We need to insert the random_drop_fn strategy in unified_loss.py
    # Find the neg_strategy block
    insert_str = """
    elif neg_strategy == 'random_drop_fn':
        # Drop FNs from the denominator with probability drop_p
        drop_p = kwargs.get('drop_p', 0.5)
        # Random mask for TP pairs (False Negatives)
        drop_mask = (torch.rand(TP_mask.shape, device=device) >= drop_p)
        N_sum = (sim_matrix * TN_mask).sum(dim=1) + (sim_matrix * (TP_mask & drop_mask)).sum(dim=1)
"""
    # Wait, unified_loss signature doesn't have **kwargs. I will just add drop_p to the signature!
    # Let's replace the signature of unified_loss
    old_sig = """                     neg_include_self: bool = False,
                     tn_weight: float = 1.0):"""
    new_sig = """                     neg_include_self: bool = False,
                     tn_weight: float = 1.0,
                     drop_p: float = 0.5):"""
    ul_content = ul_content.replace(old_sig, new_sig)
    
    old_neg_strategy = """    elif neg_strategy == 'tn_add_weight':
        tn_term = (sim_matrix * TN_mask).sum(dim=1)
        fn_term = (sim_matrix * TP_mask).sum(dim=1)
        N_sum = (tn_weight * tn_term) + fn_term"""
        
    new_neg_strategy = old_neg_strategy + """

    elif neg_strategy == 'random_drop_fn':
        drop_mask = (torch.rand(TP_mask.shape, device=device) >= drop_p)
        N_sum = (sim_matrix * TN_mask).sum(dim=1) + (sim_matrix * (TP_mask & drop_mask)).sum(dim=1)"""
    
    ul_content = ul_content.replace(old_neg_strategy, new_neg_strategy)
    
    # Also unified_loss returns 3 values but gsimclr expects 4 values for random_drop_fn
    # In gsimclr.py: loss, pos_sim, neg_sim, fn_metrics = unified_loss(...)
    old_return = """    pos_sim = P_term.mean()
    neg_sim = N_sum.mean()

    return loss, pos_sim, neg_sim"""
    new_return = """    pos_sim = P_term.mean()
    neg_sim = N_sum.mean()

    # Return empty dict for fn_metrics if it's expected
    if neg_strategy == 'random_drop_fn':
        return loss, pos_sim, neg_sim, {}
    return loss, pos_sim, neg_sim"""
    
    ul_content = ul_content.replace(old_return, new_return)

    with open('/home/qiannnhui/GraphCL/unsupervised_TU/unified_loss.py', 'w') as f:
        f.write(ul_content)

# 2. Patch gsimclr.py to fix the O(V*E) CPU bottleneck
with open('/home/qiannnhui/GraphCL/unsupervised_TU/gsimclr.py', 'r') as f:
    gsim_content = f.read()

old_bottleneck = """                # node_num_aug, _ = data_aug.x.size()
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]

                data_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)"""

new_bottleneck = """                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                
                # Optimized bottleneck using numpy
                idx_not_missing = np.unique(edge_idx).tolist()
                
                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]
                data_aug.batch = data.batch[idx_not_missing]
                
                idx_dict = np.zeros(node_num, dtype=np.int64)
                idx_dict[idx_not_missing] = np.arange(node_num_aug)
                
                new_edge_idx = idx_dict[edge_idx]
                mask = new_edge_idx[0] != new_edge_idx[1]
                data_aug.edge_index = torch.from_numpy(new_edge_idx[:, mask])"""

if "idx_not_missing = [n for n in range(node_num)" in gsim_content:
    gsim_content = gsim_content.replace(old_bottleneck, new_bottleneck)
    with open('/home/qiannnhui/GraphCL/unsupervised_TU/gsimclr.py', 'w') as f:
        f.write(gsim_content)
    print("gsimclr.py bottleneck optimized!")
else:
    print("Bottleneck already optimized or not found.")

print("Patch complete.")
