import torch
import torch.nn.functional as F

def create_pos_and_neg_mask(labels):

    labels = labels.view(-1, 1)
    pos_mask = labels.eq(labels.T)
    neg_mask = ~pos_mask
    # remove self-positive pairs
    pos_mask = pos_mask & ~torch.diag(torch.ones(labels.size(0), device=labels.device, dtype=torch.bool))

    return pos_mask, neg_mask

def calculate_f1_scores(all_x, all_x_aug, all_labels, T=0.2, eps=1e-6, deg_boundary=30.0):
    """
    Calculate (F1, Precision, Recall) per epoch based on False Negative (FN) analysis.
    """
    z_a = torch.cat(all_x, dim=0).to(all_x[0].device)
    z_b = torch.cat(all_x_aug, dim=0).to(all_x[0].device)
    labels = torch.cat(all_labels, dim=0).to(all_x[0].device)
    
    z_a_norm = F.normalize(z_a, dim=1)
    z_b_norm = F.normalize(z_b, dim=1)
    
    sim_ab = (z_a_norm @ z_b_norm.T).clamp(-1.0, 1.0)
    angle_ab_deg = torch.rad2deg(torch.acos(sim_ab.clamp(-1.0 + eps, 1.0 - eps))) 
    
    # FN 條件: 負樣本角度 (z_a[i] vs z_b[j]) 小於 角度閾值 deg_boundary
    FN_angle_matrix = angle_ab_deg <= deg_boundary # (N, N)
    FN_angle_matrix.fill_diagonal_(False)           # 排除正樣本 i==i
    
    # --- Ground Truth Positive Mask (GT Pos) ---
    gt_pos_mask, gt_neg_mask = create_pos_and_neg_mask(labels)
    
    # --- Calculate Metrics ---
    tp = (FN_angle_matrix & gt_pos_mask).sum().float()
    fp = (FN_angle_matrix & gt_neg_mask).sum().float()
    fn_missed = (~FN_angle_matrix & gt_pos_mask).sum().float()
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn_missed + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return f1, precision, recall

  # 保持原有的 reweighted_by_angle_corrected 方法不變，它用於損失計算