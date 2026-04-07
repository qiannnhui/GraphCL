def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
# from core.encoders import *
import networkx as nx
from torch_geometric.utils import to_networkx, degree, k_hop_subgraph, subgraph
from torch_geometric.utils import to_dense_adj

# from torch_geometric.datasets import TUDataset
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *

from arguments import arg_parse
from check_dim_collapse import check_dimensional_collapse
from ortho_loss import l2_reg_ortho
from torch.utils.tensorboard import SummaryWriter
import time
from sklearn.metrics import confusion_matrix
from plot_similarity import plot_similarity_matrix
from plot_sim_distribution import plot_similarity_distribution
from plot_theta_l2_scatter import plot_theta_l2, plot_theta_l2_epoch
from plot_single_anchor_FP_FN_distribution import plot_theta_l2_distribution
from plot_tsne import visualize_embeddings
from plot_KDE import plot_kde_unitcircle_kde
from plot_theta_epoch import plot_theta_per_epoch
from plot_sim_epoch import plot_sim_per_epoch
from plot_RBO_heatmap import plot_sorted_rbo_heatmap
from plot_RBO_cosine_diag import plot_rbo_cosine_diagnosis
from make_save_dir import make_save_dir # 引入創建儲存目錄的函數
from save_load_ckpts import load_checkpoint, save_checkpoint # 引入檢查點函數
from unified_loss import unified_loss, get_pair_angles, flexible_hard_mining_loss
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def get_soft_metrics(pred_mtx, gt_mtx):
    batch_size = pred_mtx.size(0)
    device = pred_mtx.device
    diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    tp = torch.sum(pred_mtx * gt_mtx)
    fp = torch.sum(pred_mtx * (1 - gt_mtx).masked_fill(diag_mask, 0.0))
    fn = torch.sum((1 - pred_mtx).masked_fill(diag_mask, 0.0) * gt_mtx)
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p.item(), r.item(), f1.item()

class simclr(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, dataset_size, shuffle_DBN=False, dataset_num_features=1, device='cuda'):
    super(simclr, self).__init__()
    self.embedding_dim = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
    self.device = device

    bn_layer = [DecorrelatedShuffledBatchNorm1d(self.embedding_dim)] if shuffle_DBN else []
    self.proj_head = nn.Sequential(
        nn.Linear(self.embedding_dim, self.embedding_dim), 
        *bn_layer,
        nn.ReLU(inplace=True), 
        nn.Linear(self.embedding_dim, self.embedding_dim)
    )
    self.init_emb()
    self.register_buffer('rbo_consistency_count', torch.zeros(dataset_size, dataset_size, dtype=torch.int32))

  def init_emb(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

  def forward(self, x, edge_index, batch):
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)
    
    y = self.proj_head(y)
    
    return y, M


  def validate_rbo_ranking(self, coverage, gt_fn_mask, gt_tn_mask, k_list=[1, 5, 10]):
    """
    同時驗證 FN (高分) 與 TN (低分) 的可靠性
    """
    batch_size = coverage.size(0)
    device = coverage.device
    ranking_stats = {}

    _, fn_sorted_indices = torch.sort(coverage, dim=1, descending=True)
    tmp_coverage = coverage.clone()
    diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    tmp_coverage.masked_fill_(diag_mask, 999.0) 
    _, tn_sorted_indices = torch.sort(tmp_coverage, dim=1, descending=False)

    for k in k_list:
        # FN Precision@K
        topk_fn = fn_sorted_indices[:, :k]
        fn_hits = torch.gather(gt_fn_mask, 1, topk_fn).sum()
        ranking_stats[f'RBO_FN_Precision@{k}'] = (fn_hits / (batch_size * k)).item()
        ranking_stats[f'RBO_FN_Recall@{k}'] = (fn_hits / gt_fn_mask.sum()).item() if gt_fn_mask.sum() > 0 else 0

        # TN Precision@K
        topk_tn = tn_sorted_indices[:, :k]
        tn_hits = torch.gather(gt_tn_mask, 1, topk_tn).sum()
        ranking_stats[f'RBO_TN_Precision@{k}'] = (tn_hits / (batch_size * k)).item()
        ranking_stats[f'RBO_TN_Recall@{k}'] = (tn_hits / gt_tn_mask.sum()).item() if gt_tn_mask.sum() > 0 else 0
        
    return ranking_stats
  
  def identify_fn_by_rbo_coverage(self, sim_matrix, labels, p=0.98, depth=50):
        batch_size = sim_matrix.size(0)
        device = sim_matrix.device
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        
        # === 關鍵修改：確保 depth 不超過當前 Batch Size ===
        curr_depth = min(depth, batch_size) 
        
        # 1. 獲取排序索引
        _, sorted_indices = torch.sort(sim_matrix, dim=1, descending=True)
        top_indices = sorted_indices[:, :curr_depth] # 使用動態的 curr_depth

        # 2. 矩陣化 RBO 計算
        # 這裡的維度第一維也要改成 curr_depth
        masks = torch.zeros((curr_depth, batch_size, batch_size), device=device)
        d_idx = torch.arange(curr_depth, device=device)
        b_idx = torch.arange(batch_size, device=device)
        
        # 進行索引賦值
        masks[d_idx.view(-1, 1), b_idx.view(1, -1), top_indices.t()] = 1.0
        
        current_top_masks_all = torch.cumsum(masks, dim=0) 
        shared_counts_all = torch.bmm(current_top_masks_all, current_top_masks_all.transpose(1, 2))
        
        # d_vec 也要同步調整
        d_vec = torch.arange(1, curr_depth + 1, device=device).float().view(-1, 1, 1)
        agreements_all = shared_counts_all / d_vec
        
        # 權重也需要根據新的 curr_depth 重新歸一化或計算
        weights = (1 - p) * (p ** (d_vec - 1))
        coverage = torch.sum(weights * agreements_all, dim=0).masked_fill(diag_mask, 0.0)

        # === [新增] Min-Max Stretching 確保 Parallel Plot 不會下墜 ===
        c_min, c_max = coverage.min(), coverage.max()
        stretched_coverage = (coverage - c_min) / (c_max - c_min + 1e-8) if c_max > c_min else coverage

        # === [保留] 你原本的 F1 / Precision / Recall 統計邏輯 ===
        labels_col = labels.view(-1, 1)
        gt_fn_mask = (labels_col.eq(labels_col.T) & ~diag_mask).float() 
        gt_tn_mask = (~labels_col.eq(labels_col.T) & ~diag_mask).float() 

        # 使用 stretched 版本來計算 soft metrics 會更穩定
        soft_tp = torch.sum(stretched_coverage * gt_fn_mask)
        soft_fp = torch.sum(stretched_coverage * gt_tn_mask)
        soft_fn = torch.sum((1 - stretched_coverage) * gt_fn_mask)

        stats = {
            'soft_pFN_recall': (soft_tp / (soft_tp + soft_fn + 1e-8)).item(),
            'soft_pFN_precision': (soft_tp / (soft_tp + soft_fp + 1e-8)).item(),
            'avg_coverage': coverage[gt_fn_mask.bool()].mean().item() if gt_fn_mask.any() else 0,
        }
        stats['soft_pFN_f1'] = 2 * stats['soft_pFN_precision'] * stats['soft_pFN_recall'] / (stats['soft_pFN_precision'] + stats['soft_pFN_recall'] + 1e-8)
        
        # 額外補上你原本可能需要的 Ranking Stats
        ranking_stats = self.validate_rbo_ranking(stretched_coverage, gt_fn_mask, gt_tn_mask)
        stats.update(ranking_stats)

        return stretched_coverage, stats
  
#   def loss_cal_structural_PPR_infonce(self, x, x_aug, labels, T=0.2, alpha_ppr=0.8, gamma=3.0):
#         batch_size = x.size(0)
#         device = x.device
#         diag_mask = torch.eye(batch_size, device=device).bool()

#         # --- A. 核心邏輯 (PPR -> Target -> Suppression) ---
#         cos_sim = torch.mm(F.normalize(x, dim=1), F.normalize(x_aug, dim=1).T)
#         jaccard_mtx = self.compute_geometry_jaccard_sim(data)
#         # jaccard_mtx = self.compute_adamic_adar_sim(jaccard_mtx)
        
#         # PPR 擴散 (作為 Target)
#         W = jaccard_mtx.clone() * cos_sim.clone()

#         # W[W < 0.2] = 0.0 
#         D_inv = torch.diag(1.0 / (W.sum(dim=1) + 1e-8))
#         P = torch.mm(D_inv, W)
#         Target = torch.eye(batch_size, device=device)
#         for _ in range(3):
#             Target = alpha_ppr * torch.eye(batch_size, device=device) + (1 - alpha_ppr) * torch.mm(P, Target)
#         # target f1, precision, recall before norm
#         Target = F.softmax(Target / T, dim=1) # 這裡改成 softmax 會更合理，因為我們後面是用 KL Divergence Loss

#         # t_min, t_max = Target.min(), Target.max()
#         # Target = (Target - t_min) / (t_max - t_min + 1e-8)
#         Target = Target.masked_fill(diag_mask, 0.0)

#         # Focal Gap 計算 (作為 Suppression)
#         # 強烈建議在計算 Focal 之前加入這行
#         # modified
#         # cos_sim_norm = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-8)
#         # gap = torch.relu(Target - cos_sim_norm)
#         # gap = torch.relu(Target - cos_sim)
#         # suppression = torch.pow(1.0 - gap, gamma)
#         # suppression = torch.clamp(suppression, min=0.01)

#         # --- B. 計算六條線的數據 (Soft Metrics) ---
#         labels_col = labels.view(-1, 1)
#         gt_mask = labels_col.eq(labels_col.T).float().masked_fill(diag_mask, 0.0)

#         # 1-3. Target 的 P, R, F1
#         t_p, t_r, t_f1 = get_soft_metrics(Target, gt_mask)
#         # s_p, s_r, s_f1 = get_soft_metrics(1-suppression, gt_mask)

#         # --- C. InfoNCE Loss ---
#         exp_sim = torch.exp(cos_sim / T)
#         weighted_neg_sim = exp_sim * Target
#         neg_denom = weighted_neg_sim.masked_fill(diag_mask, 0.0).sum(dim=1)
#         loss = -torch.log(exp_sim.diag() / (neg_denom + 1e-8) + 1e-8).mean()
#         # KL Divergence Loss
#         # log_prob = F.log_softmax(cos_sim / T, dim=1)
#         # loss = F.kl_div(log_prob, Target, reduction='batchmean')
#         # exp_sim = torch.exp(cos_sim / T)
#         # pos_sim = exp_sim.diag()
#         # neg_denom = exp_sim.sum(dim=1)

#         # # 模型預測的機率分佈 (Softmax over current batch)
#         # p_pred = exp_sim / (neg_denom.view(-1, 1) + 1e-8)

#         # # 結構專家提供的「理想」分佈 (對 Target 做 Softmax)
#         # T_teacher = T
#         # p_target = F.softmax(Target / T_teacher, dim=1) 

#         # # 使用 KL Divergence 或 Cross Entropy 讓模型去對齊結構地圖
#         # loss = F.kl_div(p_pred.log(), p_target, reduction='batchmean')

#         stats = {
#             't_p': t_p, 't_r': t_r, 't_f1': t_f1,
#             's_p': s_p, 's_r': s_r, 's_f1': s_f1
#         }
        
#         return loss, Target, stats
  
  def loss_cal_structural_PPR_infonce(self, x, x_aug, labels, T=0.2, alpha_ppr=0.8, gamma=3.0):
        batch_size = x.size(0)
        device = x.device
        diag_mask = torch.eye(batch_size, device=device).bool()

        # --- A. 核心邏輯 (PPR -> Target -> Suppression) ---
        cos_sim = torch.mm(F.normalize(x, dim=1), F.normalize(x_aug, dim=1).T)
        jaccard_mtx = self.compute_geometry_jaccard_sim(data)
        jaccard_mtx = self.compute_adamic_adar_sim(jaccard_mtx)
        
        # PPR 擴散 (作為 Target)
        W = jaccard_mtx.clone()

        W[W < 0.2] = 0.0 
        D_inv = torch.diag(1.0 / (W.sum(dim=1) + 1e-8))
        P = torch.mm(D_inv, W)
        Target = torch.eye(batch_size, device=device)
        for _ in range(3):
            Target = alpha_ppr * torch.eye(batch_size, device=device) + (1 - alpha_ppr) * torch.mm(P, Target)
        # target f1, precision, recall before norm

        t_min, t_max = Target.min(), Target.max()
        Target = (Target - t_min) / (t_max - t_min + 1e-8)
        Target = Target.masked_fill(diag_mask, 0.0)

        # Focal Gap 計算 (作為 Suppression)
        # 強烈建議在計算 Focal 之前加入這行
        # modified
        cos_sim_norm = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-8)
        # gap = torch.relu(Target - cos_sim_norm)
        fn_confidence = Target * cos_sim_norm
        suppression = torch.pow(1.0 - fn_confidence, gamma)
        suppression = torch.clamp(suppression, min=0.01)
        # --- B. 計算六條線的數據 (Soft Metrics) ---
        labels_col = labels.view(-1, 1)
        gt_mask = labels_col.eq(labels_col.T).float().masked_fill(diag_mask, 0.0)

        # 1-3. Target 的 P, R, F1
        t_p, t_r, t_f1 = get_soft_metrics(Target, gt_mask)
        s_p, s_r, s_f1 = get_soft_metrics(1-suppression, gt_mask)
        
        # --- C. InfoNCE Loss ---
        exp_sim = torch.exp(cos_sim / T)
        weighted_neg_sim = exp_sim * suppression
        neg_denom = weighted_neg_sim.masked_fill(diag_mask, 0.0).sum(dim=1)
        loss = -torch.log(exp_sim.diag() / (neg_denom + 1e-8) + 1e-8).mean()

        stats = {
            't_p': t_p, 't_r': t_r, 't_f1': t_f1,
            's_p': s_p, 's_r': s_r, 's_f1': s_f1
        }
        
        return loss, Target, stats

  def loss_cal_structural_focal_infonce(self, x, x_aug, labels, T=0.2, alpha_ppr=0.8, gamma=3.0):
        batch_size = x.size(0)
        device = x.device
        diag_mask = torch.eye(batch_size, device=device).bool()

        # --- A. 核心邏輯 (PPR -> Target -> Suppression) ---
        cos_sim = torch.mm(F.normalize(x, dim=1), F.normalize(x_aug, dim=1).T)
        jaccard_mtx = self.compute_geometry_jaccard_sim(data)
        jaccard_mtx = self.compute_adamic_adar_sim(jaccard_mtx)
        
        # PPR 擴散 (作為 Target)
        W = jaccard_mtx.clone()

        W[W < 0.2] = 0.0 
        D_inv = torch.diag(1.0 / (W.sum(dim=1) + 1e-8))
        P = torch.mm(D_inv, W)
        Target = torch.eye(batch_size, device=device)
        for _ in range(3):
            Target = alpha_ppr * torch.eye(batch_size, device=device) + (1 - alpha_ppr) * torch.mm(P, Target)
        # target f1, precision, recall before norm

        t_min, t_max = Target.min(), Target.max()
        Target = (Target - t_min) / (t_max - t_min + 1e-8)
        Target = Target.masked_fill(diag_mask, 0.0)

        # Focal Gap 計算 (作為 Suppression)
        # 強烈建議在計算 Focal 之前加入這行
        # modified
        # cos_sim_norm = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-8)
        # gap = torch.relu(Target - cos_sim_norm)
        gap = torch.relu(Target - cos_sim)
        suppression = torch.pow(1.0 - gap, gamma)
        suppression = torch.clamp(suppression, min=0.01)

        # --- B. 計算六條線的數據 (Soft Metrics) ---
        labels_col = labels.view(-1, 1)
        gt_mask = labels_col.eq(labels_col.T).float().masked_fill(diag_mask, 0.0)

        # 1-3. Target 的 P, R, F1
        t_p, t_r, t_f1 = get_soft_metrics(Target, gt_mask)
        s_p, s_r, s_f1 = get_soft_metrics(1-suppression, gt_mask)
        
        # --- C. InfoNCE Loss ---
        exp_sim = torch.exp(cos_sim / T)
        weighted_neg_sim = exp_sim * suppression
        neg_denom = weighted_neg_sim.masked_fill(diag_mask, 0.0).sum(dim=1)
        loss = -torch.log(exp_sim.diag() / (neg_denom + 1e-8) + 1e-8).mean()

        stats = {
            't_p': t_p, 't_r': t_r, 't_f1': t_f1,
            's_p': s_p, 's_r': s_r, 's_f1': s_f1
        }
        
        return loss, Target, stats

  def compute_geometry_jaccard_sim(self, data, max_deg=20):
        edge_index = data.edge_index
        batch = data.batch
        num_graphs = data.num_graphs
        deg = degree(edge_index[0], dtype=torch.long)
        deg_clamped = deg.clamp(max=max_deg)
        deg_onehot = F.one_hot(deg_clamped, num_classes=max_deg + 1).float()
        
        graph_hist = torch.zeros((num_graphs, max_deg + 1), device=self.device)
        graph_hist.scatter_add_(0, batch.view(-1, 1).expand(-1, max_deg + 1), deg_onehot)

        A = graph_hist.unsqueeze(1) 
        B = graph_hist.unsqueeze(0) 
        intersection = torch.min(A, B).sum(dim=-1)
        union = torch.max(A, B).sum(dim=-1)
        return intersection / (union + 1e-8)


  def loss_cal_reweighted(self, z_a, z_b, T=0.2, eps=1e-6, dist=False, neg_aug=False):

        # --- normalize for cosine ---
        z_a_n = F.normalize(z_a, dim=1)
        z_b_n = F.normalize(z_b, dim=1)
        # --- cosines ---
        sim_ab = (z_a_n @ z_b_n.T).clamp(-1 + 1e-7, 1 - 1e-7)   # pos pairs (i,i)
        sim_aa = (z_a_n @ z_a_n.T).clamp(-1 + 1e-7, 1 - 1e-7)   # others (neg pairs)

        # --- angles ---
        angle_pos_ab = torch.acos(sim_ab.diagonal()).unsqueeze(1)  # (B,1)  z_a[i] vs z_b[i]
        angle_aa     = torch.acos(sim_aa)                          # (B,B)  z_a[i] vs z_a[j]
        if neg_aug:
            # the negatives are the augmented pairs
            angle_ab     = torch.acos(sim_ab)                          # (B,B)  z_a[i] vs z_a[j]
            close_theta = angle_ab <= angle_pos_ab            # (B,B)
        else:
            # the negatives are the original pairs
            close_theta = angle_aa <= angle_pos_ab            # (B,B)
        # no matter neg_aug is True or False, FN_matrix is the same (both are from non-augmented anchors but use augmented positives to determine closeness)
        FN_theta_matrix = angle_aa <= angle_pos_ab            # (B,B)

        # --- distances ---
        if dist:
            dist_pos_ab = torch.cdist(z_a, z_b, p=2).diagonal().unsqueeze(1)  # (B,1)
            dist_aa     = torch.cdist(z_a, z_a, p=2)                          # (B,B)
            if neg_aug:
                dist_ab     = torch.cdist(z_a, z_b, p=2)                          # (B,B)
                close_dist = dist_ab <= dist_pos_ab           # (B,B)
            else:
                close_dist = dist_aa <= dist_pos_ab           # (B,B)
            false_mask = close_theta & close_dist

            # get FN_matrix using positive pair boundary on non-augmented anchors
            FN_dist_matrix = dist_aa <= dist_pos_ab            # (B,B)
            FN_matrix = FN_theta_matrix & FN_dist_matrix
        else:
            false_mask = close_theta
            FN_matrix = FN_theta_matrix

        false_mask.fill_diagonal_(False)

        # reweighting
        W = torch.full_like(sim_aa, 1.0)   # Assume all easy
        W[false_mask] = 0.0                        # reweight false negatives
        W.fill_diagonal_(0.0)                      # avoid i==i

        # === pos pairs ===
        log_pos = sim_ab.diagonal() / T            # (B,)

        # === neg pairs ===
        logit_neg = torch.log(W + eps) + sim_aa / T        # (B,B)
        log_denom = torch.logsumexp(logit_neg, dim=1)      # (B,)
        pos_sim_ = sim_ab.diagonal().mean()
        neg_sim_ = (sim_aa * W).sum(dim=1).mean()

        return -(log_pos - log_denom).mean(), FN_matrix, pos_sim_, neg_sim_

  def reweighted_by_angle(self, z_a, z_b, T=0.2, eps=1e-6, deg_boundary=30.0):
    
    z_a_norm = F.normalize(z_a, dim=1)
    z_b_norm = F.normalize(z_b, dim=1)
    sim_ab = (z_a_norm @ z_b_norm.T).clamp(-1.0, 1.0)
    
    angle_ab_deg = torch.rad2deg(torch.acos(sim_ab.clamp(-1.0 + eps, 1.0 - eps))) # 避免 acos(+-1) 導致 NaN
    FN_candidate_mask = angle_ab_deg <= deg_boundary 
    diag_mask = torch.eye(sim_ab.size(0), dtype=torch.bool, device=sim_ab.device)
    FN_angle_matrix = FN_candidate_mask & (~diag_mask)
    
    # Pos_Set_Mask: 分子中的所有樣本 (i==i 的正樣本 + FN 樣本)
    pos_set_mask = FN_candidate_mask # 因為 diag_mask 在 FN_candidate_mask 內
    
    # === Pos Pairs Term (Numerator) ===
    pos_logits = sim_ab / T 
    pos_logits.masked_fill_(~pos_set_mask, -1e9) 
    log_numerator = torch.logsumexp(pos_logits, dim=1) 
    
    # === Neg Pairs Term (Denominator) ===
    neg_set_mask = angle_ab_deg > deg_boundary
    neg_logits = sim_ab / T
    neg_logits.masked_fill_(~neg_set_mask, -1e9)
    log_denominator = torch.logsumexp(neg_logits, dim=1) 
    
    # === Final Loss (InfoNCE: - log(Numerator / Denominator)) ===    
    loss = -(log_numerator - log_denominator).mean()
    pos_sim_avg = sim_ab[pos_set_mask].mean()
    
    fn_sim_avg = sim_ab[FN_angle_matrix].mean()

    return loss, pos_sim_avg, fn_sim_avg
  
  def loss_cal_reweighted_FNs_by_RBO(self, x, x_aug, labels, geo_sim_matrix, causal_sim_matrix, cur_epoch, total_epochs):
        T = 0.2
        batch_size = x.size(0)
        cos_sim_raw = torch.mm(F.normalize(x, dim=1), F.normalize(x_aug, dim=1).T)
        
        # Curriculum: 隨 Epoch 增加，從 Jaccard(geo) 轉向 Causal
        alpha = max(0.5, 0.9 - (cur_epoch / (total_epochs + 1e-8)) * 0.4)
        consensus = alpha * geo_sim_matrix + (1 - alpha) * causal_sim_matrix
        
        # 取得 stretched RBO 作為信心值
        rbo_confidence, stats = self.identify_fn_by_rbo_coverage(consensus, labels, p=0.98)
        
        # Gap-based Reweighting
        gap = torch.relu(consensus - cos_sim_raw)
        reweight_factors = 1.0 + rbo_confidence * (gap ** 2) * 20.0

        sim_matrix = torch.exp(cos_sim_raw / T)
        weighted_sim = sim_matrix * reweight_factors
        diag_mask = torch.eye(batch_size, device=x.device).bool()
        weighted_sim.masked_fill_(diag_mask, 0.0)
        
        loss = -torch.log(sim_matrix.diag() / (weighted_sim.sum(dim=1) + 1e-8) + 1e-8).mean()
        
        stats.update({'alpha': alpha, 'gap': gap.mean().item()})
        return loss, rbo_confidence, stats
  
  def compute_adamic_adar_sim(self, adj_matrix):
        """
        計算 Batch 內部的 Adamic-Adar 相似度
        adj_matrix: [B, B] 的初始相似度矩陣 (例如 Jaccard)
        """
        device = adj_matrix.device
        # 1. 計算每個節點的度數 (Degree)
        # 這裡我們將 adj_matrix 視為權重圖，計算加權度數
        degree = adj_matrix.sum(dim=1)
        
        # 2. 計算 AA 權重: 1 / log(degree)
        # 加上 1.1 確保 log 內部大於 1，避免分母為 0 或負數
        weights = 1.0 / torch.log(degree + 1.1) 
        weights[torch.isinf(weights)] = 0
        weights[torch.isnan(weights)] = 0
        
        # 3. 透過矩陣乘法計算共同鄰居的加權總和
        # 公式: AA = A * D_weight * A.T
        # 其中 D_weight 是以 weights 為對角線的矩陣
        weighted_adj = adj_matrix * weights.unsqueeze(0) # 廣播相乘
        aa_matrix = torch.mm(weighted_adj, adj_matrix.t())
        
        # 4. 歸一化到 0~1 之間以利 Parallel Plot 顯示
        aa_min, aa_max = aa_matrix.min(), aa_matrix.max()
        aa_norm = (aa_matrix - aa_min) / (aa_max - aa_min + 1e-8)
        
        return aa_norm

  def compute_structural_consensus_metrics(self, data, alpha=0.15, ppr_steps=10, heat_t=1.0, walk_len=5, num_walk_samples=30):
        """
        高效稀疏運算版本：整合 PPR, Heat Kernel, High-Pass 與 Anonymous Walk
        解決維度衝突 (Node vs Edge) 與 OOM 問題
        """
        from torch_geometric.nn import global_mean_pool
        from torch_geometric.utils import degree
        from collections import Counter
        
        # 處理 random_walk 函式庫版本相容性
        try:
            from torch_cluster import random_walk
        except ImportError:
            from torch_geometric.utils import random_walk

        edge_index = data.edge_index
        num_nodes = data.num_nodes
        batch = data.batch # 標記每個節點屬於哪張圖 [N]
        num_graphs = data.num_graphs
        device = self.device

        # --- 1. 譜域預處理：計算歸一化係數 ---
        row, col = edge_index
        deg = degree(col, num_nodes)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        deg_inv_sqrt = deg_inv_sqrt.view(-1, 1) # 強制轉為 [N, 1] 以利廣播計算

        # 定義 SpMV (Sparse Matrix-Vector Multiplication) 運算：D^-1/2 * A * D^-1/2 * x
        def sparse_op(input_x):
            # input_x shape: [N, 1]
            # (1) 先做對源節點的歸一化
            x_scaled = input_x * deg_inv_sqrt # [N, 1]
            # (2) 透過邊傳遞訊息 (A * x)
            norm_src = x_scaled[row] # [Edges, 1]
            out = torch.zeros((num_nodes, input_x.size(1)), device=device)
            out.scatter_add_(0, col.unsqueeze(-1), norm_src) # [N, 1]
            # (3) 最後做對目標節點的歸一化
            return out * deg_inv_sqrt

        # 初始信號：全 1 向量 [N, 1]
        x_ones = torch.ones((num_nodes, 1), device=device)

        # --- 2. 譜域指標計算 ---
        # (A) PPR 疊代 (Power Iteration)
        x_ppr = x_ones.clone()
        for _ in range(ppr_steps):
            x_ppr = (1 - alpha) * sparse_op(x_ppr) + alpha * x_ones
        # 聚合至圖級特徵 [B, 1]
        ppr_repr = global_mean_pool(x_ppr, batch)

        # (B) High-Pass (L*x = x - S*x)
        Sx = sparse_op(x_ones)
        Lx = x_ones - Sx
        # Lx 代表結構變化的能量，聚合至圖級 [B, 1]
        high_pass_repr = global_mean_pool(Lx, batch)

        # (C) Heat Kernel (近似 exp(-tL)x ≈ x - tLx)
        x_heat = x_ones - heat_t * Lx
        heat_repr = global_mean_pool(x_heat, batch)

        # 輔助函式：計算 [B, B] 相似度矩陣並歸一化
        def get_sim_matrix(repr_vec):
            # repr_vec shape: [B, 1]
            sim = torch.mm(repr_vec, repr_vec.t())
            c_min, c_max = sim.min(), sim.max()
            return (sim - c_min) / (c_max - c_min + 1e-8)

        sub_ppr = get_sim_matrix(ppr_repr)
        sub_high = get_sim_matrix(high_pass_repr)
        sub_heat = get_sim_matrix(heat_repr)
        sub_aa = self.compute_adamic_adar_sim(sub_ppr)

        # --- 3. Anonymous Walk 計算 ---
        # 為每張圖隨機選擇起點
        subset_indices = []
        for i in range(num_graphs):
            nodes_in_g = (batch == i).nonzero(as_tuple=True)[0]
            if len(nodes_in_g) > 0:
                # 隨機抽樣，若節點數不足則允許重複
                idx = nodes_in_g[torch.randint(0, len(nodes_in_g), (num_walk_samples,), device=device)]
                subset_indices.append(idx)
        
        if len(subset_indices) > 0:
            start_nodes = torch.cat(subset_indices)
            # 使用 GPU 進行平行隨機走訪
            walks = random_walk(row, col, start_nodes, walk_length=walk_len-1)
            
            # 模式統計 (CPU 處理)
            walks_cpu = walks.cpu().numpy()
            graph_patterns = []
            for i in range(num_graphs):
                g_walks = walks_cpu[i*num_walk_samples : (i+1)*num_walk_samples]
                counts = Counter()
                for w in g_walks:
                    # 匿名化編碼 (例如: [102, 45, 102] -> [0, 1, 0])
                    d = {}
                    p = tuple([d.setdefault(node, len(d)) for node in w])
                    counts[p] += 1
                graph_patterns.append(counts)

            # 計算圖與圖之間的 Anonymous Walk 相似度 [B, B]
            sub_anon = torch.zeros((num_graphs, num_graphs), device=device)
            for i in range(num_graphs):
                for j in range(i, num_graphs):
                    c1, c2 = graph_patterns[i], graph_patterns[j]
                    intersection = sum((c1 & c2).values())
                    union = sum((c1 | c2).values())
                    val = intersection / (union + 1e-8)
                    sub_anon[i, j] = sub_anon[j, i] = val
        else:
            sub_anon = torch.zeros((num_graphs, num_graphs), device=device)

        return {
            'PPR': sub_ppr,
            'Heat_Kernel': sub_heat,
            'High_Pass': sub_high,
            'Anonymous_Walk': sub_anon,
            'Adamic_Adar': sub_aa
        }

  def plot_advanced_diagnosis(self, labels, metrics_dict, epoch, save_path, n_samples=500):
        # todos: upper-triangle only

        # 1. 建立 Mask 移除對角線
        B = labels.size(0)
        mask = ~torch.eye(B, dtype=torch.bool, device=labels.device)
        labels_col = labels.view(-1, 1)
        gt_mask = labels_col.eq(labels_col.T) & mask
        
        # 2. 拉平所有指標並放入 DataFrame
        data_for_df = {}
        for name, mtx in metrics_dict.items():
            data_for_df[name] = mtx[mask].cpu().numpy()
        
        flat_gt = gt_mask[mask].cpu().numpy()
        data_for_df['Label'] = flat_gt.astype(int)
        for name in data_for_df:
            if name != 'Label':
                # 確保數值在 0~1 之間，並處理掉可能的 nan
                data_for_df[name] = np.nan_to_num(np.clip(data_for_df[name], 0, 1))
        
        full_df = pd.DataFrame(data_for_df)

        # 3. 平衡抽樣
        pos_indices = full_df[full_df['Label'] == 1].index
        neg_indices = full_df[full_df['Label'] == 0].index
        n_pos = min(n_samples // 2, len(pos_indices))
        n_neg = n_samples - n_pos
        
        sel_idx = np.concatenate([
            np.random.choice(pos_indices, n_pos, replace=False),
            np.random.choice(neg_indices, n_neg, replace=False)
        ])
        df = full_df.loc[sel_idx].reset_index(drop=True)

        # --- Plot A: Parallel Plot (現在會顯示所有指標) ---
        fig_parallel = px.parallel_coordinates(
            df, color="Label",
            dimensions=list(metrics_dict.keys()), # 自動包含所有傳入的指標
            color_continuous_scale=[(0, 'red'), (1, 'green')],
            title=f"Global Parallel Flow - Epoch {epoch}"
        )
        fig_parallel.write_html(f"{save_path}/parallel_E{epoch}.html")

        # --- Plot B: Radar Plot (看各指標平均) ---
        categories = list(metrics_dict.keys())
        avg_df = df.groupby('Label')[categories].mean().reset_index()
        fig_radar = go.Figure()
        for _, row in avg_df.iterrows():
            name = "Same Label" if row['Label'] == 1 else "Diff Label"
            fig_radar.add_trace(go.Scatterpolar(r=row[categories].values, theta=categories, fill='toself', name=name))
        fig_radar.write_html(f"{save_path}/radar_E{epoch}.html")

  def plot_3d_interactive_diagnosis(self, labels, metrics_dict, epoch, save_path, n_samples=1000):
        """
        實作完全自由選擇 X, Y, Z 軸的 3D 互動圖
        """
        import pandas as pd
        import plotly.graph_objects as go

        # 1. 準備數據
        B = labels.size(0)
        mask = ~torch.eye(B, dtype=torch.bool, device=labels.device)
        labels_col = labels.view(-1, 1)
        gt_label = (labels_col.eq(labels_col.T) & mask)[mask].cpu().numpy()
        
        df_data = {}
        for name, mtx in metrics_dict.items():
            # 確保資料是 numpy array
            val = mtx[mask].cpu().numpy()
            df_data[name] = val
        
        df = pd.DataFrame(df_data)
        df['Label'] = gt_label
        df = df.sample(n=min(len(df), n_samples)) # 採樣
        
        # 取得所有可選的指標名稱
        columns = [c for c in df.columns if c != 'Label']
        
        # 定義顏色與標籤
        colors = df['Label'].map({True: 'green', False: 'red'})
        hover_text = df['Label'].map({True: 'Same Class', False: 'Diff Class'})

        # 2. 建立基礎 Figure (預設顯示前三維)
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=df[columns[0]], 
            y=df[columns[1]], 
            z=df[columns[2]],
            mode='markers',
            marker=dict(size=3, color=colors, opacity=0.6),
            text=hover_text,
            name="Data Points"
        ))

        # 3. 建立三個獨立的下拉選單 (X, Y, Z)
        def create_menu(target_axis, button_index):
            return dict(
                buttons=[
                    dict(
                        label=col,
                        method="update",
                        args=[{target_axis: [df[col]]}, 
                              {"scene": {f"{target_axis}axis": {"title": col}}}]
                    ) for col in columns
                ],
                direction="down",
                showactive=True,
                x=0.1 + (button_index * 0.25), # 平排選單
                y=1.15,
                xanchor="left",
                yanchor="top"
            )

        # 4. 更新佈局
        fig.update_layout(
            updatemenus=[
                create_menu("x", 0),
                create_menu("y", 1),
                create_menu("z", 2)
            ],
            annotations=[
                dict(text="Select X:", x=0.05, y=1.12, xref="paper", yref="paper", showarrow=False),
                dict(text="Select Y:", x=0.30, y=1.12, xref="paper", yref="paper", showarrow=False),
                dict(text="Select Z:", x=0.55, y=1.12, xref="paper", yref="paper", showarrow=False),
            ],
            title=f"3D Flexible Diagnostic Axis - Epoch {epoch}",
            scene=dict(
                xaxis_title=columns[0],
                yaxis_title=columns[1],
                zaxis_title=columns[2]
            ),
            margin=dict(l=0, r=0, b=0, t=60)
        )

        fig.write_html(f"{save_path}/3D_Flexible_Diagnosis_E{epoch}.html")

import random
def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def compute_pairwise_differences(metrics_dict):
    """
    計算字典中所有矩陣指標兩兩之間的絕對差值
    """
    import itertools
    diff_results = {}
    metric_names = list(metrics_dict.keys())
    
    # 兩兩組合 (例如: (Cosine, PPR), (PPR, Jaccard)...)
    for name_a, name_b in itertools.combinations(metric_names, 2):
        # 計算絕對差值 |A - B|
        # 這代表了兩個視角之間的「不一致程度」
        diff_mat = torch.abs(metrics_dict[name_a] - metrics_dict[name_b])
        
        # 命名規則: Diff_(A_vs_B)
        diff_results[f'Diff_({name_a}_vs_{name_b})'] = diff_mat
        
    return diff_results

if __name__ == '__main__':
    
    args = arg_parse()
    setup_seed(args.seed)
    save_dir = make_save_dir(
        base="./result/GCL",
        args=args,
    )
    os.makedirs(f'{save_dir}', exist_ok=True)
    ckpt_filename = f'{save_dir}/ckpts.pth.tar'
    best_val_acc = 0.0
    # tensorboard
    writer = SummaryWriter(log_dir=f'{save_dir}/tensorboard_{time.ctime(time.time())}_{args.or_loss}')

    accuracies = {'val':[], 'test':[]}
    epochs = args.epochs
    log_interval = args.log_interval
    batch_size = args.batch_size
    aug_ratio = round(args.aug_ratio * 0.1, 1)
    loss_list = []
    loss_min = float('inf')
    stage_finish_epochs = []
    self_theta_list = []
    neg_theta_list = []
    TP_theta_list = []
    TN_theta_list = []
    odecay = args.odecay
    lr = args.lr
    DS = args.DS
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    path = osp.join(args.path, DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    dataset = TUDataset(path, name=DS, aug=args.aug, aug_ratio=aug_ratio).shuffle()
    dataset_eval = TUDataset(path, name=DS, aug='none').shuffle()
    dataset_size = len(dataset)
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = simclr(args.hidden_dim, args.num_gc_layers, shuffle_DBN=args.shuffle_DBN, dataset_num_features=dataset_num_features, dataset_size=dataset_size).to(device)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tmp_loader = DataLoader(dataset_eval, batch_size=len(dataset_eval), shuffle=False)
    batch_all = next(iter(tmp_loader))

    if isinstance(batch_all, list) or isinstance(batch_all, tuple):
        raw_data = batch_all[0]
    else:
        raw_data = batch_all

    # 3. 搬移到 GPU 並計算
    raw_data = raw_data.to(device)
    global_jaccard_matrix = model.compute_geometry_jaccard_sim(raw_data)  

    # start_epoch, best_test_acc = load_checkpoint(
    #     ckpt_filename, 
    #     model, 
    #     optimizer, 
    #     device
    # )
    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    model.eval()
    emb, y = model.encoder.get_embeddings(dataloader_eval)
    # print(emb.shape, y.shape)

    """
    acc_val, acc = evaluate_embedding(emb, y)
    accuracies['val'].append(acc_val)
    accuracies['test'].append(acc)
    """

    # === Fixed Set Monitoring Initialization ===
    if args.neg_sim_analysis:
        print("Initializing Fixed Set for FN/HN/EN Analysis...")
        monitor_loader = dataloader  # Use eval set as monitor set
        with torch.no_grad():
            emb_init, y_init = model.encoder.get_embeddings(monitor_loader)
            emb_init = torch.from_numpy(emb_init).to(device)
            y_init = torch.from_numpy(y_init).to(device)
            
            # Initial Similarity Matrix
            emb_init_norm = F.normalize(emb_init, dim=1)
            sim_init = torch.matmul(emb_init_norm, emb_init_norm.T) # (N, N)
            
            # Masks
            labels_col = y_init.view(-1, 1)
            mask_same_label = labels_col.eq(labels_col.T)
            mask_diff_label = ~mask_same_label
            mask_self = torch.eye(y_init.shape[0], dtype=torch.bool, device=device)
            
            # FN: Same Label but NOT self
            mask_FN_fixed = mask_same_label & ~mask_self
            
            # HN: Diff Label but High Similarity (> 0.5)
            # You can adjust threshold here. User asked for "Low Sim" for EN and "High Sim" for HN.
            hn_threshold = 0.8
            mask_HN_fixed = mask_diff_label & (sim_init > hn_threshold)
            
            # EN: Diff Label but Low Similarity (< 0.0)
            mask_EN_fixed = mask_diff_label & (sim_init < hn_threshold)
            
            print(f"Fixed Set Stats: N={y_init.shape[0]}")
            print(f"Num FN Pairs: {mask_FN_fixed.sum().item()}")
            print(f"Num HN Pairs (> {hn_threshold}): {mask_HN_fixed.sum().item()}")
            print(f"Num EN Pairs (< {hn_threshold}): {mask_EN_fixed.sum().item()}")
            
            # === Dynamic Threshold based on FN Mean ===
            fn_mean_init = sim_init[mask_FN_fixed].mean().item() if mask_FN_fixed.any() else 0.5
            print(f"Initializing Dynamic Threshold based on FN Mean: {fn_mean_init:.4f}")
            
            mask_HN_dynamic = mask_diff_label & (sim_init > fn_mean_init)
            mask_EN_dynamic = mask_diff_label & (sim_init < fn_mean_init)
            
            print(f"Num HN Pairs (Dynamic > {fn_mean_init:.4f}): {mask_HN_dynamic.sum().item()}")
            print(f"Num EN Pairs (Dynamic < {fn_mean_init:.4f}): {mask_EN_dynamic.sum().item()}")


            # Storage for plotting
            fixed_FN_sims = []
            fixed_HN_sims = []
            fixed_EN_sims = []
            dynamic_HN_sims = []
            dynamic_EN_sims = []
            
    for epoch in range(0, epochs + 1):
        if args.mode == 'focal_infonce' or args.mode == 'PPR_infonce':
            epoch_metrics = {k: 0.0 for k in ['t_p', 't_r', 't_f1', 's_p', 's_r', 's_f1']}

    # for epoch in range(start_epoch, epochs + 1):
        # lr = scheduler.step()
        if epoch % 5 == 0:
            model.eval()
            diag_dir = f'{save_dir}/Global_Diagnosis/E{epoch}'
            os.makedirs(diag_dir, exist_ok=True)
            os.makedirs(f"{diag_dir}/base_metrics", exist_ok=True)
            os.makedirs(f"{diag_dir}/diff_metrics", exist_ok=True)
            os.makedirs(f"{diag_dir}/all_metrics", exist_ok=True)

            with torch.no_grad():
                # 注意：這裡應使用不 Shuffle 的 tmp_loader
                emb, y = model.encoder.get_embeddings(tmp_loader) 
                emb_t = F.normalize(torch.from_numpy(emb).to(device), dim=1)
                
                # 隨機抽 500 個點
                s_idx = np.random.choice(emb_t.size(0), min(emb_t.size(0), 200), replace=False)
                sub_labels = torch.from_numpy(y[s_idx]).to(device).long()
                
                # 計算診斷指標
                sub_cos = torch.mm(emb_t[s_idx], emb_t[s_idx].t())
                sub_jaccard = global_jaccard_matrix[s_idx][:, s_idx]
                
                # Causal 批次計算
                sub_data_list = []
                for i in s_idx:
                    d = dataset_eval[i]
                    if isinstance(d, (list, tuple)): d = d[0] # 處理 [data, aug]
                    sub_data_list.append(d)
                from torch_geometric.data import Batch as PyGBatch
                sub_batch = PyGBatch.from_data_list(sub_data_list).to(device)
                # 計算 Consensus 與 stretched RBO
                # alpha_curr = max(0.5, 0.9 - (epoch/args.epochs)*0.4)
                # sub_cons = alpha_curr * sub_jaccard + (1 - alpha_curr) * sub_cos
                
                struct_metrics = model.compute_structural_consensus_metrics(sub_batch)
                base_metrics = {
                    'Cosine': sub_cos,
                    'Jaccard': sub_jaccard,
                    # 'Consensus': sub_cons
                }
                base_metrics.update(struct_metrics)
                diff_metrics = compute_pairwise_differences(base_metrics)
                                
                # 繪圖字典
                plot_metrics = {}
                plot_metrics.update(base_metrics)  # 原始數值軸
                plot_metrics.update(diff_metrics)

                model.plot_advanced_diagnosis(sub_labels, base_metrics, epoch, f"{diag_dir}/base_metrics")
                model.plot_advanced_diagnosis(sub_labels, diff_metrics, epoch, f"{diag_dir}/diff_metrics")
                model.plot_advanced_diagnosis(sub_labels, plot_metrics, epoch, f"{diag_dir}/all_metrics")
                model.plot_3d_interactive_diagnosis(sub_labels, plot_metrics, epoch, diag_dir)
            model.train()

        loss_all = 0
        pos_sim_all = 0
        neg_sim_all = 0
        
        # === Fixed Set Analysis Logging ===
        if args.neg_sim_analysis:
            model.eval()
            with torch.no_grad():
                emb_curr, _ = model.encoder.get_embeddings(monitor_loader)
                emb_curr = torch.from_numpy(emb_curr).to(device)
                emb_curr_norm = F.normalize(emb_curr, dim=1)
                sim_curr = torch.matmul(emb_curr_norm, emb_curr_norm.T)
                
                # Calculate Mean Similarities using Fixed Masks
                sim_FN_val = float('nan')
                if mask_FN_fixed.any():
                    sim_FN_val = sim_curr[mask_FN_fixed].mean().item()
                    writer.add_scalar('Neg_sim_Analysis/FN_Mean_Sim', sim_FN_val, epoch)
                
                sim_HN_val = float('nan')
                if mask_HN_fixed.any():
                    sim_HN_val = sim_curr[mask_HN_fixed].mean().item()
                    writer.add_scalar('Neg_sim_Analysis/HN_Mean_Sim', sim_HN_val, epoch)
                    
                sim_EN_val = float('nan')
                if mask_EN_fixed.any():
                    sim_EN_val = sim_curr[mask_EN_fixed].mean().item()
                    writer.add_scalar('Neg_sim_Analysis/EN_Mean_Sim', sim_EN_val, epoch)

                # === Dynamic Set Analysis Logging ===
                sim_HN_dynamic_val = float('nan')
                if mask_HN_dynamic.any():
                    sim_HN_dynamic_val = sim_curr[mask_HN_dynamic].mean().item()
                    writer.add_scalar('Neg_sim_Analysis/Dynamic_HN_Mean_Sim', sim_HN_dynamic_val, epoch)

                sim_EN_dynamic_val = float('nan')
                if mask_EN_dynamic.any():
                    sim_EN_dynamic_val = sim_curr[mask_EN_dynamic].mean().item()
                    writer.add_scalar('Neg_sim_Analysis/Dynamic_EN_Mean_Sim', sim_EN_dynamic_val, epoch)
                    
                print(f"Epoch {epoch} Fixed Set Sim: FN={sim_FN_val:.4f}, HN={sim_HN_val:.4f}, EN={sim_EN_val:.4f}, Dyn_HN={sim_HN_dynamic_val:.4f}, Dyn_EN={sim_EN_dynamic_val:.4f}")

                # === Collect Data for Plotting (Similarities) ===
                def get_sims_from_mask(sim_matrix, mask):
                    if not mask.any():
                        return np.array([])
                    sim_vals = sim_matrix[mask]
                    # Direct similarity values
                    return sim_vals.cpu().numpy()

                fixed_FN_sims.append(get_sims_from_mask(sim_curr, mask_FN_fixed))
                fixed_HN_sims.append(get_sims_from_mask(sim_curr, mask_HN_fixed))
                fixed_EN_sims.append(get_sims_from_mask(sim_curr, mask_EN_fixed))
                dynamic_HN_sims.append(get_sims_from_mask(sim_curr, mask_HN_dynamic))
                dynamic_EN_sims.append(get_sims_from_mask(sim_curr, mask_EN_dynamic))
            
            model.train()

        model.train()
        self_epoch_theta_list = []
        neg_epoch_theta_list = []
        TP_epoch_theta_list = []
        TN_epoch_theta_list = []
        total_tp = 0
        total_fp = 0
        total_fn_missed = 0

        if args.plot_theta_l2 and epoch % 50 == 0:
            all_pos_l2, all_pos_theta = [], []
            all_neg_l2, all_neg_theta = [], []
            all_real_pos_l2, all_real_pos_theta = [], []
            all_real_neg_l2, all_real_neg_theta = [], []

        if args.plot_kde and epoch % log_interval == 0:
            all_anchor_embeddings = []
            all_anchor_labels = []
            all_pos_embeddings = []
            all_pos_labels = []

        # labels = torch.empty(0, dtype=torch.long, device=device)
        # for data in dataloader:
        for batch_idx, data in enumerate(dataloader):
            first_batch = (batch_idx == 0)
            last_batch = (batch_idx == len(dataloader) - 1)

            data, data_aug = data
            # labels = torch.cat([labels, data.y.to(device)], dim=0)

            labels = data.y.to(device)
            optimizer.zero_grad()
            
            node_num, _ = data.x.size()
            data = data.to(device)
            x, _ = model(data.x, data.edge_index, data.batch)

            if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4':
                # node_num_aug, _ = data_aug.x.size()
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]

                data_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

            data_aug = data_aug.to(device)

            x_aug, _ = model(data_aug.x, data_aug.edge_index, data_aug.batch)

            # Assuming unified_loss is defined and handles all pos/neg strategies.
            if args.mode == 'normal':
                # Standard InfoNCE (Self-Pos / All Negs)
                if args.denominator_anchor:
                    loss, pos_sim, neg_sim = unified_loss(x, x_aug, labels, pos_strategy='normal', neg_strategy='denominator_anchor', sim_measure=args.similarity_measure)
                else:
                    loss, pos_sim, neg_sim = unified_loss(x, x_aug, labels, sim_measure=args.similarity_measure)

            elif args.mode == 'TP1_normal':
                # P = S(x_i, x_i+ sampled), N = All Negs
                loss, pos_sim, neg_sim = unified_loss(x, x_aug, labels, pos_strategy='sum_sample_tp', pos_num_samples=1, neg_strategy='normal', sim_measure=args.similarity_measure)

            elif args.mode == 'TP1_Nnormal':
                # P = S(x_i, x_i+ sampled), N = Normalized All Negs
                loss, pos_sim, neg_sim = unified_loss(x, x_aug, labels, pos_strategy='sum_sample_tp', pos_num_samples=1, neg_strategy='normalized_normal', sim_measure=args.similarity_measure)

            elif args.mode == 'TPs_TNs':
                # P = Sum(TPs), N = Sum(TNs) (Removes FN and FP)
                loss, pos_sim, neg_sim = unified_loss(x, x_aug, labels, pos_strategy='sum_tp', neg_strategy='sum_tn', sim_measure=args.similarity_measure)

            elif args.mode == 'normal_TNs' or args.mode == 'rm_FN':
                # P = S(x_i, x_i+), N = Sum(TNs) (Removes FN)
                loss, pos_sim, neg_sim = unified_loss(x, x_aug, labels, pos_strategy='normal', neg_strategy='sum_tn', sim_measure=args.similarity_measure)

            elif args.mode == 'TPs_normal':
                # P = Sum(TPs), N = All Negs (Removes FP)
                loss, pos_sim, neg_sim = unified_loss(x, x_aug, labels, pos_strategy='sum_tp', neg_strategy='normal', sim_measure=args.similarity_measure)

            elif args.mode == 'FP1_FNs':
                # P = Sum(1 random FP), N = Sum(FNs/TPs)
                loss, pos_sim, neg_sim = unified_loss(x, x_aug, labels, pos_strategy='sum_sample_fp', pos_num_samples=1, neg_strategy='sum_fn', sim_measure=args.similarity_measure)

            elif args.mode == 'normal_FNs':
                # P = S(x_i, x_i+), N = Sum(FNs/TPs) (Removes TN/FPs)
                loss, pos_sim, neg_sim = unified_loss(x, x_aug, labels, pos_strategy='normal', neg_strategy='sum_fn', sim_measure=args.similarity_measure)

            elif args.mode == 'FP1_normal':
                # P = Sum(1 random FP), N = All Negs
                loss, pos_sim, neg_sim = unified_loss(x, x_aug, labels, pos_strategy='sum_sample_fp', pos_num_samples=1, neg_strategy='normal', sim_measure=args.similarity_measure)

            elif args.mode == 'FPs_FNs':
                # P = Sum(FPs/TNs), N = Sum(FNs/TPs)
                loss, pos_sim, neg_sim = unified_loss(x, x_aug, labels, pos_strategy='sum_fp', neg_strategy='sum_fn', sim_measure=args.similarity_measure)

            elif args.mode == 'FPs_normal':
                # P = Sum(FPs/TNs), N = All Negs
                loss, pos_sim, neg_sim = unified_loss(x, x_aug, labels, pos_strategy='sum_fp', neg_strategy='normal', sim_measure=args.similarity_measure)

            elif args.mode == 'TP1_TN2':
                # P = Sum(1 random TP), N = Sum(2 random TNs)
                loss, pos_sim, neg_sim = unified_loss(x, x_aug, labels, pos_strategy='sum_sample_tp', pos_num_samples=1, neg_strategy='sum_sample_tn', neg_num_samples=2, sim_measure=args.similarity_measure)
            elif args.mode == "normal_TN_add_weight":
                # P = S(x_i, x_i+), N = Sum(all neg) with additional weight on all TNs
                loss, pos_sim, neg_sim = unified_loss(x, x_aug, labels, pos_strategy='normal', neg_strategy='tn_add_weight', sim_measure=args.similarity_measure, tn_weight=args.tn_weight)
            elif args.mode == 'TPs_HNs': # cheated rm Easy Negatives
                # P = Sum(TPs), N = Sum(Hard Negatives)
                loss, pos_sim, neg_sim = flexible_hard_mining_loss(x, x_aug, labels, args.hard_sim_threshold, num_sets='ALL_POS', den_sets='HN', sim_measure=args.similarity_measure)
            elif args.mode == 'TPs_ENs': # cheated rm Hard Negatives
                # P = Sum(TPs), N = Sum(Easy Negatives)
                loss, pos_sim, neg_sim = flexible_hard_mining_loss(x, x_aug, labels, args.hard_sim_threshold, num_sets='ALL_POS', den_sets='EN', sim_measure=args.similarity_measure)
            elif args.mode == 'normal_rm_HNs':
                # P = S(x_i, x_i+), N = Sum(Easy Negatives) (Removes Hard Negatives)
                loss, pos_sim, neg_sim = flexible_hard_mining_loss(x, x_aug, labels, args.hard_sim_threshold, num_sets='SP', den_sets='EN,HP,EP', sim_measure=args.similarity_measure)
            elif args.mode == 'normal_rm_ENs':
                # P = S(x_i, x_i+), N = Sum(Hard Negatives) (Removes Easy Negatives)
                loss, pos_sim, neg_sim = flexible_hard_mining_loss(x, x_aug, labels, args.hard_sim_threshold, num_sets='SP', den_sets='HN,HP,EP', sim_measure=args.similarity_measure)
            elif args.mode == 'normal_HNs':
                # P = S(x_i, x_i+), N = Sum(Hard Negatives)
                loss, pos_sim, neg_sim = flexible_hard_mining_loss(x, x_aug, labels, args.hard_sim_threshold, num_sets='SP', den_sets='HN', sim_measure=args.similarity_measure)
            elif args.mode == 'normal_ENs':
                # P = S(x_i, x_i+), N = Sum(Easy Negatives)
                loss, pos_sim, neg_sim = flexible_hard_mining_loss(x, x_aug, labels, args.hard_sim_threshold, num_sets='SP', den_sets='EN', sim_measure=args.similarity_measure)

            elif args.mode == 'reweight_FNs_by_RBO':
                rbo_save_dir = f'{save_dir}/plot_RBO_weight/RBO_epoch_{epoch}'
                os.makedirs(rbo_save_dir, exist_ok=True)
                
                jaccard_sim = model.compute_geometry_jaccard_sim(data.to(device))
                # causal_sub_sim = model.compute_causal_subgraph_sim(data, model, device).to(device)
                # ppr = model.compute_ppr_matrix(data.to(device))
                loss, coverage, en_stats = model.loss_cal_reweighted_FNs_by_RBO(
                                                                                    x, x_aug, labels, 
                                                                                    geo_sim_matrix=jaccard_sim, 
                                                                                    causal_sim_matrix=jaccard_sim, 
                                                                                    cur_epoch=epoch, 
                                                                                    total_epochs=args.epochs
                                                                                )
            elif args.mode == 'focal_infonce':
                loss, _, stats = model.loss_cal_structural_focal_infonce(x, x_aug, labels)
                for k in epoch_metrics:
                    epoch_metrics[k] += stats[k]
            elif args.mode == 'PPR_infonce':
                loss, _, stats = model.loss_cal_structural_PPR_infonce(x, x_aug, labels)
                for k in epoch_metrics:
                    epoch_metrics[k] += stats[k]

            else:
                # Handles all other unmatched modes
                raise RuntimeError(f"no mode matching {args.mode}, input should be: normal, TPs_TNs, etc.")
            # scatter plot for theta and l2 norm
            self_theta_degree = get_pair_angles(x, x_aug, labels=labels, pair_type='Self')
            neg_theta_degree = get_pair_angles(x, x_aug, labels=labels, pair_type='ALL_NoneSelf')
            TP_theta_degree = get_pair_angles(x, x_aug, labels=labels, pair_type='TP')
            TN_theta_degree = get_pair_angles(x, x_aug, labels=labels, pair_type='TN')
            self_epoch_theta_list.append(self_theta_degree.cpu())
            neg_epoch_theta_list.append(neg_theta_degree.cpu())
            TP_epoch_theta_list.append(TP_theta_degree.cpu())
            TN_epoch_theta_list.append(TN_theta_degree.cpu())
            if args.plot_theta_l2 and epoch % log_interval == 0:
                if first_batch and hasattr(model, 'all_pos_l2'):
                    del model.all_pos_l2, model.all_pos_theta, model.all_neg_l2, model.all_neg_theta, model.all_pos_cos, model.all_neg_cos
                model.plot_theta_l2(x, x_aug, labels)
                if last_batch:
                    # tensorboard
                    result = model.plot_theta_l2_epoch(args=args, epoch=epoch, similarity_measure=args.similarity_measure)
                    writer.add_scalar('Theta/pos_avg_cos', result['pos_avg_cos'], epoch)
                    writer.add_scalar('Theta/pos_avg_theta', result['pos_avg_theta'], epoch)
                    writer.add_scalar('Theta/pos_avg_l2', result['pos_avg_l2'], epoch)
                    writer.add_scalar('Theta/neg_avg_cos', result['neg_avg_cos'], epoch)
                    writer.add_scalar('Theta/neg_avg_theta', result['neg_avg_theta'], epoch)
                    writer.add_scalar('Theta/neg_avg_l2', result['neg_avg_l2'], epoch)
            if args.plot_kde and epoch % log_interval == 0:
                all_anchor_embeddings.append(x.detach().cpu())
                all_anchor_labels.append(data.y.detach().cpu())
                all_pos_embeddings.append(x_aug.detach().cpu())
                all_pos_labels.append(data.y.detach().cpu())

            oloss = odecay * l2_reg_ortho(model)
            loss_all += loss.item() * data.num_graphs
            if not (args.mode == 'reweight_FNs_by_RBO' or args.mode == 'focal_infonce' or args.mode == 'PPR_infonce'):
                pos_sim_all += pos_sim.item()
                neg_sim_all += neg_sim.item()
            if args.or_loss:
                loss += oloss
            loss.backward()
            optimizer.step()
            if args.plot_theta_l2 and epoch % 50 == 0:
                # pos_l2, pos_theta, neg_l2, neg_theta = plot_theta_l2(x_anchor, x_graph_pos)
                pos_l2, pos_theta, neg_l2, neg_theta, real_pos_l2, real_pos_theta, real_neg_l2, real_neg_theta = plot_theta_l2(x, x_aug, data.y)
                all_pos_l2.append(pos_l2)
                all_pos_theta.append(pos_theta)
                all_neg_l2.append(neg_l2)
                all_neg_theta.append(neg_theta)
                all_real_pos_l2.append(real_pos_l2)
                all_real_pos_theta.append(real_pos_theta)
                all_real_neg_l2.append(real_neg_l2)
                all_real_neg_theta.append(real_neg_theta)
            if args.plot_theta_l2_distribution and epoch % 100 == 0:
                plot_theta_l2_distribution(x, x_aug, labels=data.y, args=args, epoch=epoch)
        # tensorboard
        writer.add_scalar('Loss/train', loss_all / len(dataloader.dataset), epoch)
        if not (args.mode == 'reweight_FNs_by_RBO' or args.mode == 'focal_infonce' or args.mode == 'PPR_infonce'):
            writer.add_scalar('Similarity/pos_sim', pos_sim_all / len(dataloader), epoch)
            writer.add_scalar('Similarity/neg_sim', neg_sim_all / len(dataloader), epoch)
        elif args.mode == 'focal_infonce' or args.mode == 'PPR_infonce':
            num_batches = len(dataloader)
            for k in epoch_metrics:
                writer.add_scalar(f'Metric_Group/{k}', epoch_metrics[k] / num_batches, epoch)

        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader.dataset)))
        loss_list.append(loss_all / len(dataloader.dataset))

        if epoch % log_interval == 0:
            if args.plot_kde:
                os.makedirs(f'{save_dir}/KDE/anchor', exist_ok=True)
                os.makedirs(f'{save_dir}/KDE/graph_pos', exist_ok=True)
                X_anchor = torch.cat(all_anchor_embeddings, dim=0).numpy()
                y_anchor = torch.cat(all_anchor_labels, dim=0).numpy()
                X_pos = torch.cat(all_pos_embeddings, dim=0).numpy()
                y_pos = torch.cat(all_pos_labels, dim=0).numpy()
                
                if len(X_anchor) > 20000:
                    idx = np.random.choice(len(X_anchor), 20000, replace=False)
                    X_anchor = X_anchor[idx]
                    y_anchor = y_anchor[idx]

                plot_kde_unitcircle_kde(
                    X_anchor, y_anchor,
                    classes=None,
                    save_path=f'{save_dir}/KDE/{args.kde_reduction_method}/anchor/epoch_{epoch}_kde_unit_circle',
                    plot_scatter=True,
                    reduction_method=args.kde_reduction_method,
                )
                plot_kde_unitcircle_kde(
                    X_pos, y_pos,
                    classes=None,
                    save_path=f'{save_dir}/KDE/{args.kde_reduction_method}/graph_pos/epoch_{epoch}_kde_unit_circle',
                    plot_scatter=True,
                    reduction_method=args.kde_reduction_method,
                )

                del all_anchor_embeddings, all_anchor_labels, all_pos_embeddings, all_pos_labels
                torch.cuda.empty_cache()

            if args.plot_theta_l2 and epoch % 50 == 0:
                result = plot_theta_l2_epoch(all_pos_l2, all_pos_theta, all_neg_l2, all_neg_theta, all_real_pos_l2,
                                             all_real_pos_theta, all_real_neg_l2, all_real_neg_theta, args=args, epoch=epoch, save_dir=save_dir)
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            # visualize_embeddings(emb, y, args, epoch, method="t-SNE")
            acc_val, acc = evaluate_embedding(emb, y)
            self_theta_list.append(torch.cat(self_epoch_theta_list, dim=0).numpy())
            neg_theta_list.append(torch.cat(neg_epoch_theta_list, dim=0).numpy())
            TP_theta_list.append(torch.cat(TP_epoch_theta_list, dim=0).numpy())
            TN_theta_list.append(torch.cat(TN_epoch_theta_list, dim=0).numpy())
            # singular_values = check_dimensional_collapse(emb)
            # for i, value in enumerate(singular_values):
            #     writer.add_scalar(f'Singular_Values/{epoch}_{args.DS}', np.log10(value), i)
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_epoch = epoch
                save_checkpoint(
                    epoch, 
                    model, 
                    optimizer, 
                    acc, # 保存測試準確度
                    ckpt_filename
                )

            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)
            # tensorboard
            writer.add_scalar('Accuracy/val', acc_val, epoch)
            writer.add_scalar('Accuracy/test', acc, epoch)

    if args.plot_anchor_aug_pair_theta_per_epoch:
        plot_theta_per_epoch(args, self_theta_list, save_dir=f'{save_dir}/self_theta')
        plot_theta_per_epoch(args, neg_theta_list, save_dir=f'{save_dir}/neg_theta')
        plot_theta_per_epoch(args, TP_theta_list, save_dir=f'{save_dir}/TP_theta')
        plot_theta_per_epoch(args, TN_theta_list, save_dir=f'{save_dir}/TN_theta')

    if args.neg_sim_analysis:
        plot_sim_per_epoch(args, fixed_FN_sims, save_dir=f'{save_dir}/Fixed_FN_sim', title_suffix="(Fixed FN)")
        plot_sim_per_epoch(args, fixed_HN_sims, save_dir=f'{save_dir}/Fixed_HN_sim', title_suffix="(Fixed HN)")
        plot_sim_per_epoch(args, fixed_EN_sims, save_dir=f'{save_dir}/Fixed_EN_sim', title_suffix="(Fixed EN)")
        plot_sim_per_epoch(args, dynamic_HN_sims, save_dir=f'{save_dir}/Dynamic_HN_sim_mean_{fn_mean_init:.2f}', title_suffix="(Dynamic HN)")
        plot_sim_per_epoch(args, dynamic_EN_sims, save_dir=f'{save_dir}/Dynamic_EN_sim_mean_{fn_mean_init:.2f}', title_suffix="(Dynamic EN)")

    with open((f'{save_dir}/{aug_ratio}_'+str(args.seed)), 'a+') as f:
        s1 = json.dumps(stage_finish_epochs)
        s2 = json.dumps(loss_list)
        s3 = json.dumps(accuracies)
        f.write('{},{},{},{},{},{},{},{}\n'.format(args.DS, args.num_gc_layers, epochs, log_interval, lr, s1, s2, s3))
    
