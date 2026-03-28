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
# from grakel.kernels import WeisfeilerLehman, VertexHistogram
from torch_geometric.utils import to_networkx, degree, k_hop_subgraph, subgraph
from torch_scatter import scatter_max

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
from utils import create_pos_and_neg_mask, calculate_f1_scores_by_deg_boundary
from analyze_high_similarity_negatives import analyze_high_similarity_negatives
from unified_loss import unified_loss, get_pair_angles, flexible_hard_mining_loss
from rotate_by_angle import rotate_embedding_high_dim_by_angle, rotate_embedding_high_dim, rotate_embedding_targeted_angle

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

  def plot_theta_l2(self, x, x_aug, labels):
        """
        accumulate theta and l2 norm data for plotting
        """
        pos_mask, neg_mask = create_pos_and_neg_mask(labels=labels)

        # ==== 計算 cosine similarity matrix ====
        x_norm = x / x.norm(dim=1, keepdim=True)
        x_aug_norm = x_aug / x_aug.norm(dim=1, keepdim=True)
        cos_sim_matrix = torch.einsum('ik,jk->ij', x_norm, x_aug_norm)  # (B, B)

        # ==== theta (夾角) ====
        theta_matrix = torch.acos(torch.clamp(cos_sim_matrix, -1.0 + 1e-7, 1.0 - 1e-7))
        theta_matrix_deg = theta_matrix * 180.0 / torch.pi  # 轉換成度

        # ==== l2 norm matrix ====
        x_sq = (x ** 2).sum(dim=1, keepdim=True)  # (B, 1)
        x_aug_sq = (x_aug ** 2).sum(dim=1, keepdim=True).T  # (1, B)
        l2_matrix = torch.sqrt(x_sq + x_aug_sq - 2 * torch.einsum('ik,jk->ij', x, x_aug) + 1e-8)

        # ==== Use only the upper triangular part ====
        upper_tri_mask = torch.triu(torch.ones_like(cos_sim_matrix), diagonal=1).bool()  # Exclude diagonal
        pos_mask = pos_mask & upper_tri_mask
        neg_mask = neg_mask & upper_tri_mask

        # ==== 擷取資料 ====
        pos_theta = theta_matrix_deg[pos_mask]
        pos_l2 = l2_matrix[pos_mask]

        neg_theta = theta_matrix_deg[neg_mask]
        neg_l2 = l2_matrix[neg_mask]

        # ==== Accumulate data ====
        if not hasattr(self, 'all_pos_l2'):
            self.all_pos_l2, self.all_pos_theta = [], []
            self.all_neg_l2, self.all_neg_theta = [], []
            self.all_pos_cos, self.all_neg_cos = [], []

        self.all_pos_l2.append(pos_l2.detach().cpu().numpy())
        self.all_pos_theta.append(pos_theta.detach().cpu().numpy())
        self.all_neg_l2.append(neg_l2.detach().cpu().numpy())
        self.all_neg_theta.append(neg_theta.detach().cpu().numpy())
        self.all_pos_cos.append(cos_sim_matrix[pos_mask].detach().cpu().numpy())
        self.all_neg_cos.append(cos_sim_matrix[neg_mask].detach().cpu().numpy())


  def plot_theta_l2_epoch(self, args=None, epoch=None, similarity_measure="cosine"):
        """
        Plot theta vs l2 norm for all accumulated data
        """
        # ==== 將所有資料合併 ====
        pos_l2 = np.concatenate(self.all_pos_l2)
        pos_theta = np.concatenate(self.all_pos_theta)
        neg_l2 = np.concatenate(self.all_neg_l2)
        neg_theta = np.concatenate(self.all_neg_theta)
        pos_cos = np.concatenate(self.all_pos_cos)
        neg_cos = np.concatenate(self.all_neg_cos)

        # ==== 計算平均值 ====
        pos_avg_cos = pos_cos.mean()
        pos_avg_theta = pos_theta.mean()
        pos_avg_l2 = pos_l2.mean()

        neg_avg_cos = neg_cos.mean()
        neg_avg_theta = neg_theta.mean()
        neg_avg_l2 = neg_l2.mean()

        result = {
            'pos_avg_cos': pos_avg_cos,
            'pos_avg_theta': pos_avg_theta,
            'pos_avg_l2': pos_avg_l2,
            'neg_avg_cos': neg_avg_cos,
            'neg_avg_theta': neg_avg_theta,
            'neg_avg_l2': neg_avg_l2
        }

        # ==== Plotting ====
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        fontsize = 36
        dot_size = 30

        fig, axes = plt.subplots(1, 3, figsize=(36, 8))
        plt.subplots_adjust(wspace=0.35)

        # --- 子圖 1 ---
        axes[0].scatter(pos_l2, pos_theta, color='#2ca02c', label='Positive Pairs', alpha=0.4, s=dot_size)
        axes[0].set_xlabel('Distance (L2 Norm)', fontsize=fontsize)
        axes[0].set_ylabel('Angle (degrees)', fontsize=fontsize)
        axes[0].tick_params(labelsize=fontsize)
        axes[0].text(0.5, -0.25, 'Angle vs Distance (Positive Pairs)',
                    transform=axes[0].transAxes,
                    ha='center', va='top',
                    fontsize=fontsize, fontweight='bold')
        axes[0].grid(True, linestyle='--', alpha=0.3)

        # --- 子圖 2 ---
        axes[1].scatter(neg_l2, neg_theta, color='#d62728', label='Negative Pairs', alpha=0.4, s=dot_size)
        axes[1].set_xlabel('Distance (L2 Norm)', fontsize=fontsize)
        axes[1].set_ylabel('Angle (degrees)', fontsize=fontsize)
        axes[1].tick_params(labelsize=fontsize)
        # axes[1].set_title('Angle vs Distance (Negative Pairs)', fontsize=fontsize)
        axes[1].text(0.5, -0.25, 'Angle vs Distance (Negative Pairs)',
                    transform=axes[1].transAxes,
                    ha='center', va='top',
                    fontsize=fontsize, fontweight='bold')
        axes[1].grid(True, linestyle='--', alpha=0.3)

        axes[2].scatter(pos_l2, pos_theta, color='#2ca02c', label='Positive Pairs', alpha=0.4, s=dot_size)
        axes[2].scatter(neg_l2, neg_theta, color='#d62728', label='Negative Pairs', alpha=0.4, s=dot_size)
        axes[2].set_xlabel('Distance (L2 Norm)', fontsize=fontsize)
        axes[2].set_ylabel('Angle (degrees)', fontsize=fontsize)
        axes[2].tick_params(labelsize=fontsize)
        axes[2].text(0.5, -0.25, 'Angle vs Distance (Overlayed View)',
                    transform=axes[2].transAxes,
                    ha='center', va='top',
                    fontsize=fontsize, fontweight='bold')
        axes[2].grid(True, linestyle='--', alpha=0.3)

        # --- legend with big markers ---
        custom_lines = [
            Line2D([0], [0], marker='o', color='w', label='Positive Pairs', markerfacecolor='#2ca02c', markersize=15),
            Line2D([0], [0], marker='o', color='w', label='Negative Pairs', markerfacecolor='#d62728', markersize=15)
        ]
        axes[2].legend(handles=custom_lines, loc='lower right', fontsize=fontsize)

        # Save the figure with all plots
        # os.makedirs(f'./logs/theta_vs_l2/{args.DS}/lr_{args.lr}/{similarity_measure}', exist_ok=True)
        os.makedirs(f'./logs/theta_vs_l2_paper', exist_ok=True)
        plt.tight_layout()  # Makes sure everything fits without overlap
        # plt.savefig(f'./logs/theta_vs_l2/{args.DS}/lr_{args.lr}/{similarity_measure}/epoch_{epoch}_theta_vs_l2_{args.aug}_{args.mode}.png')
        plt.savefig(f'./logs/theta_vs_l2_paper/Cheated_GCL_{args.DS}_{epoch}.pdf', bbox_inches='tight', dpi=300)
        plt.close()  # Close the figure to free memory

        return result


  def identify_fn_by_en_curriculum(self, sim_matrix, labels, epoch, total_epochs, 
                               base_en_threshold=0.1, max_en_threshold=0.5, coverage_threshold=0.5):
        """
        分析 EN-based FN 識別的動態指標
        Args:
            sim_matrix: (B, B) 相似度矩陣 (exp(cos/T))
            labels: (B,) 標籤
            base_en_threshold/max_en_threshold: 用於動態增加 EN 比例
        """
        batch_size = sim_matrix.size(0)
        device = sim_matrix.device
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        labels_col = labels.view(-1, 1)
        
        # Ground Truth Masks
        gt_fn_mask = labels_col.eq(labels_col.T) & ~diag_mask  # 真正的同類 (FN)
        gt_tn_mask = ~labels_col.eq(labels_col.T)             # 真正的異類 (TN)
        
        total_gt_fn = gt_fn_mask.sum().float()
        total_gt_tn = gt_tn_mask.sum().float()
        
        # --- 動態 EN 策略 ---
        # 隨著 Epoch 線性增加觀察 EN 的比例
        curr_en_threshold = base_en_threshold + (max_en_threshold - base_en_threshold) * (epoch / total_epochs)
        k = max(1, int(batch_size * curr_en_threshold))
        
        # 識別預測的 FN (Pred FN)
        _, en_indices = torch.topk(sim_matrix, k=k, dim=1, largest=False)
        en_mask = torch.zeros_like(sim_matrix).scatter_(1, en_indices, 1.0)
        shared_count = torch.matmul(en_mask, en_mask.T)
        pred_fn_mask = (shared_count / k >= coverage_threshold) & ~diag_mask
        
        # --- 指標統計 ---
        tp_fn = (pred_fn_mask & gt_fn_mask).sum().float() # 拿對的
        fp_fn = (pred_fn_mask & gt_tn_mask).sum().float() # 拿錯的 (誤殺 TN)
        

        # 1. FN 拿對率 (Recall)
        pFN_recall = tp_fn / (total_gt_fn + 1e-8)

        # 2. 移除精準度 (Precision)
        pFN_precision = tp_fn / (pred_fn_mask.sum().float() + 1e-8)

        pFN_f1 = 2 * (pFN_precision * pFN_recall) / (pFN_precision + pFN_recall + 1e-8)
        
        # 3. FN 拿錯率 (False Alarm Rate / FPR) - 你最關心的指標
        pFN_wrong_rate = fp_fn / (total_gt_tn + 1e-8)
        
        # 4. 分母純淨度分析
        denom_mask_after = ~diag_mask & ~pred_fn_mask
        remaining_fn_count = (denom_mask_after & gt_fn_mask).sum().float()
        remaining_pFN_ratio = remaining_fn_count / (denom_mask_after.sum().float() + 1e-8)
        original_pFN_ratio = total_gt_fn / (batch_size * (batch_size - 1) + 1e-8)

        stats = {
            'pFN_recall': pFN_recall.item(),
            'pFN_wrong_rate': pFN_wrong_rate.item(),
            'pFN_precision': pFN_precision.item(),
            'pFN_f1': pFN_f1.item(),
            'remaining_pFN_ratio': remaining_pFN_ratio.item(),
            'original_pFN_ratio': original_pFN_ratio.item(),
            'curr_en_threshold': curr_en_threshold
        }
        
        return pred_fn_mask, stats
  
#   def loss_cal_rm_FNs_by_ENs(self, x, x_aug, labels, top_k_en=10, coverage_threshold=0.5, neg_include_self=True):
  def loss_cal_rm_FNs_by_ENs(self, x, x_aug, labels, cur_epoch, total_epochs, base_en_threshold=0.1, max_en_threshold=0.5, coverage_threshold=0.5, neg_include_self=True):
        T = 0.2
        num_samples, _ = x.size()
        sim_matrix = torch.exp(torch.mm(F.normalize(x, dim=1), F.normalize(x_aug, dim=1).T) / T)

        pred_fn_mask, en_stats = self.identify_fn_by_en_curriculum(
                    sim_matrix, labels, cur_epoch, total_epochs,
                    base_en_threshold=base_en_threshold, max_en_threshold=max_en_threshold, coverage_threshold=coverage_threshold
                )
        
        num_deleted_pFN = pred_fn_mask.sum().item()
        # 總負樣本對數量 (不含對角線) 為 N * (N - 1)
        total_neg_pairs = batch_size * (batch_size - 1)
        pFN_ratio = num_deleted_pFN / (total_neg_pairs + 1e-8)
        
        # 將這些資訊加入 en_stats 回傳
        en_stats['num_deleted_pFN'] = num_deleted_pFN
        en_stats['pFN_ratio_in_batch'] = pFN_ratio

        # --- 計算 Loss ---
        self_pos = sim_matrix.diag() # (B,)
        
        denom_mask = torch.ones_like(sim_matrix, dtype=torch.bool)
        diag_mask = torch.eye(num_samples, dtype=torch.bool, device=device)
        denom_mask[diag_mask] = False if not neg_include_self else True    # 扣除自己 if neg_include_self=False
        denom_mask[pred_fn_mask] = False # 扣除推斷出的 FN
        
        neg_sim_sum = (sim_matrix * denom_mask).sum(dim=1)
        
        loss = -torch.log(self_pos / (neg_sim_sum + 1e-8) + 1e-8).mean()

        return loss, en_stats

  def identify_fn_by_en_coverage(self, sim_matrix, labels, epoch, total_epochs, 
                                   base_en_threshold=0.3, max_en_threshold=0.6):
        batch_size = sim_matrix.size(0)
        device = sim_matrix.device
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        
        # 1. 動態計算 EN 數量 k
        curr_en_threshold = base_en_threshold + (max_en_threshold - base_en_threshold) * (epoch / total_epochs)
        k = max(1, int(batch_size * curr_en_threshold))
        
        # 2. 獲取 EN 掩碼
        _, en_indices = torch.topk(sim_matrix, k=k, dim=1, largest=False)
        en_mask = torch.zeros_like(sim_matrix).scatter_(1, en_indices, 1.0)
        
        # 3. 計算 Coverage (Shared Ratio) -> 這就是我們的連續預測值 [0, 1]
        shared_count = torch.matmul(en_mask, en_mask.T)
        coverage = (shared_count / k).masked_fill(diag_mask, 0.0)
        
        # 4. 真實標籤矩陣 (Ground Truth Matrix)
        labels_col = labels.view(-1, 1)
        gt_fn_mask = (labels_col.eq(labels_col.T) & ~diag_mask).float() # 同類為 1
        gt_tn_mask = (~labels_col.eq(labels_col.T) & ~diag_mask).float() # 異類為 1

        # 5. 計算 Soft Metrics (非離散)
        # Soft TP: 在標籤為同類的地方，coverage 的總和
        soft_tp = torch.sum(coverage * gt_fn_mask)
        # Soft FP: 在標籤為異類的地方，coverage 的總和 (即誤判的總機率)
        soft_fp = torch.sum(coverage * gt_tn_mask)
        # Soft FN: 在標籤為同類的地方，漏掉的機率 (1 - coverage) 總和
        soft_fn = torch.sum((1 - coverage) * gt_fn_mask)

        soft_precision = soft_tp / (soft_tp + soft_fp + 1e-8)
        soft_recall = soft_tp / (soft_tp + soft_fn + 1e-8)
        soft_f1 = (2 * soft_precision * soft_recall) / (soft_precision + soft_recall + 1e-8)
        
        stats = {
            'soft_pFN_recall': soft_recall.item(),
            'soft_pFN_precision': soft_precision.item(),
            'soft_pFN_f1': soft_f1.item(),
            'avg_coverage': coverage[gt_fn_mask.bool()].mean().item() if gt_fn_mask.any() else 0,
            'curr_en_threshold': curr_en_threshold
        }
        ranking_stats = self.validate_rbo_ranking(coverage, gt_fn_mask, k_list=[1, 5, 10, 50])
        stats.update(ranking_stats)
        
        return coverage, stats
  
  def loss_cal_reweighted_FNs_by_ENs(self, x, x_aug, labels, cur_epoch, total_epochs, 
                                       base_en_threshold=0.3, max_en_threshold=0.6, 
                                       neg_include_self=True, reweight_strategy="1-coverage", 
                                       coverage_threshold=0.5, renormalization=True):
        T = 0.2
        batch_size, _ = x.size()
        sim_matrix = torch.exp(torch.mm(F.normalize(x, dim=1), F.normalize(x_aug, dim=1).T) / T)
        
        coverage, stats = self.identify_fn_by_en_coverage(
            sim_matrix, labels, cur_epoch, total_epochs, base_en_threshold, max_en_threshold
        )
        
        if reweight_strategy == "1-coverage":
            negative_weights = 1.0 - coverage
        elif reweight_strategy == "thresholded":
            negative_weights = torch.where(coverage > coverage_threshold, 1.0 - coverage, torch.ones_like(coverage))
        else:
            negative_weights = torch.ones_like(coverage)

        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=x.device)
        if not neg_include_self:
            negative_weights = negative_weights.masked_fill(diag_mask, 0.0)
            target_sum = float(batch_size - 1) # 每個 anchor 應該對應的總推力
        else:
            negative_weights = negative_weights.masked_fill(diag_mask, 1.0)
            target_sum = float(batch_size)

        # Renormalize weights to maintain the same total contribution as normal InfoNCE
        if renormalization:
            current_sum = negative_weights.sum(dim=1, keepdim=True) # (B, 1)
            scale_factor = target_sum / (current_sum + 1e-8)
            normalized_weights = negative_weights * scale_factor

        self_pos = sim_matrix.diag()
        weighted_neg_sim = (sim_matrix * normalized_weights).sum(dim=1)
        loss = -torch.log(self_pos / (weighted_neg_sim + 1e-8) + 1e-8).mean()

        stats['avg_scale_factor'] = scale_factor.mean().item()

        return loss, coverage, stats

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
  
  def compute_causal_subgraph_sim(self, data, model, device, num_trials=10, num_hops=2):
    model.eval()
    data = data.to(device)
    batch_size = data.num_graphs
    num_total_nodes = data.batch.size(0)
    
    with torch.no_grad():
        # 1. 取得基準 Embedding
        orig_z, _ = model.encoder(data.x, data.edge_index, data.batch)
        orig_z = F.normalize(orig_z, dim=1)
    
    max_sims = torch.full((batch_size,), -1.0, device=device)
    best_subgraph_embs = orig_z.clone()

    with torch.no_grad():
        for _ in range(num_trials):
            rand_vals = torch.rand(num_total_nodes, device=device)
            
            seeds_tensor = torch.zeros(batch_size, dtype=torch.long, device=device)
            for i in range(batch_size):
                mask_i = (data.batch == i)
                if mask_i.any():
                    seeds_tensor[i] = rand_vals.masked_fill(~mask_i, -1e9).argmax()

            subset, sub_edge_index, _, _ = k_hop_subgraph(
                node_idx=seeds_tensor,
                num_hops=num_hops,
                edge_index=data.edge_index,
                relabel_nodes=True, 
                num_nodes=num_total_nodes
            )

            sub_x = data.x[subset] if data.x is not None else None
            sub_batch = data.batch[subset]

            try:
                z_sub, _ = model.encoder(sub_x, sub_edge_index, sub_batch)
                z_sub = F.normalize(z_sub, dim=1)
                
                current_sim = (orig_z * z_sub).sum(dim=1) 
                
                update_mask = current_sim > max_sims
                best_subgraph_embs[update_mask] = z_sub[update_mask]
                max_sims[update_mask] = current_sim[update_mask]
            except Exception:
                continue
    causal_sim = torch.mm(best_subgraph_embs, best_subgraph_embs.t())
    
    return causal_sim.cpu()

  def identify_fn_by_rbo_coverage(self, sim_matrix, labels, p=0.9, depth=None):
    """
    使用純 RBO 權重計算樣本間的 Coverage。
    sim_matrix: [B, B] 相似度矩陣
    p: RBO 的持久度參數，越小越看重 Top 排名
    """
    B = sim_matrix.size(0)
    if depth is None:
        depth = B // 2
    device = sim_matrix.device
    diag_mask = torch.eye(B, dtype=torch.bool, device=device)
    # remove self-similarity for ranking
    # sim_matrix = sim_matrix.masked_fill(diag_mask, -1.0) # 先把自己設為 -1，確保不會被選為 EN 或 TN
    
    # 1. 獲取排序索引 (相似度由高到低，即最相似的排在前面)
    # sorted_indices: [batch_size, batch_size]
    _, sorted_indices = torch.sort(sim_matrix, dim=1, descending=True)
    top_indices = sorted_indices[:, :depth] 

    # 2. 建立 (depth, B, B) 的掩碼矩陣
    # 我們改用一個更安全的方法來填充 masks
    masks = torch.zeros((depth, B, B), device=device)
    
    # 利用進階索引一次性填充所有深度的 one-hot
    d_idx = torch.arange(depth, device=device)
    b_idx = torch.arange(B, device=device)
    
    # 這裡的邏輯：對於每個深度 d，在第 b 列的 top_indices[b, d] 位置填入 1
    # masks[d_idx, b_idx, top_indices[b_idx, d_idx]] = 1.0
    # 我們需要轉置 top_indices 來匹配 (depth, B)
    masks[d_idx.view(-1, 1), b_idx.view(1, -1), top_indices.t()] = 1.0
    
    # 3. 累積 Agreement (等同於 RBO 的逐層交集)
    # 沿著 depth 維度做前綴和，得到「前 d 名的成員集合」
    current_top_masks_all = torch.cumsum(masks, dim=0) # (depth, B, B)
    
    # 4. 矩陣化計算交集數
    # 使用 bmm (batch matrix multiplication): (depth, B, B) x (depth, B, B)
    shared_counts_all = torch.bmm(current_top_masks_all, current_top_masks_all.transpose(1, 2))
    
    # 5. 計算 RBO 加權平均
    d_vec = torch.arange(1, depth + 1, device=device).float().view(-1, 1, 1)
    agreements_all = shared_counts_all / d_vec
    weights = (1 - p) * (p ** (d_vec - 1))
    
    coverage = torch.sum(weights * agreements_all, dim=0) # (B, B)

    # --- 後續統計邏輯不變 ---
    diag_mask = torch.eye(B, dtype=torch.bool, device=device)
    coverage = coverage.masked_fill(diag_mask, 0.0)
    
    labels_col = labels.view(-1, 1)
    gt_fn_mask = (labels_col.eq(labels_col.T) & ~diag_mask).float() 
    gt_tn_mask = (~labels_col.eq(labels_col.T) & ~diag_mask).float() 

    soft_tp = torch.sum(coverage * gt_fn_mask)
    soft_fp = torch.sum(coverage * gt_tn_mask)
    soft_fn = torch.sum((1 - coverage) * gt_fn_mask)

    soft_precision = soft_tp / (soft_tp + soft_fp + 1e-8)
    soft_recall = soft_tp / (soft_tp + soft_fn + 1e-8)
    soft_f1 = (2 * soft_precision * soft_recall) / (soft_precision + soft_recall + 1e-8)
    
    stats = {
        'soft_pFN_recall': soft_recall.item(),
        'soft_pFN_precision': soft_precision.item(),
        'soft_pFN_f1': soft_f1.item(),
        'avg_coverage': coverage[gt_fn_mask.bool()].mean().item() if gt_fn_mask.any() else 0,
    }
    ranking_stats = self.validate_rbo_ranking(coverage, gt_fn_mask, gt_tn_mask, k_list=[1, 5, 10, 50])
    stats.update(ranking_stats)
    
    return coverage, stats

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

  def loss_cal_reweighted_FNs_by_RBO(self, x, x_aug, labels, batch_indices, geo_sim_matrix, cur_epoch=0, total_epochs=0, 
                                       neg_include_self=True, reweight_strategy="1-coverage", RBO_p=0.9,
                                       coverage_threshold=0.5, renormalization=True, RBO_anchor=False, 
                                       denominator_anchor=False, RBO_save_path="./", boost_factor=2.0):
        T = 0.2
        batch_size, _ = x.size()
        # raw_cos_matrix = torch.mm(F.normalize(x, dim=1), F.normalize(x_aug, dim=1).T)
        sim_matrix = torch.exp(torch.mm(F.normalize(x, dim=1), F.normalize(x_aug, dim=1).T) / T)
        sim_matrix_anchor = torch.exp(torch.mm(F.normalize(x, dim=1), F.normalize(x, dim=1).T) / T)
        coverage, stats = self.identify_fn_by_rbo_coverage(geo_sim_matrix, labels, p=RBO_p)
        
        # if RBO_anchor:
        #     coverage, stats = self.identify_fn_by_rbo_coverage(sim_matrix_anchor, labels, p=RBO_p)
        # else:
        #     coverage, stats = self.identify_fn_by_rbo_coverage(sim_matrix, labels, p=RBO_p)
        
        '''
        # add consistency count for each pair (temporal information)
        idx_i, idx_j = batch_indices.view(-1, 1), batch_indices.view(1, -1)
        current_hit = coverage > 0.4  # 當前門檻
        
        # 核心邏輯：命中則加1，沒命中則歸零
        with torch.no_grad():
            updated_count = torch.where(current_hit, 
                                        self.rbo_consistency_count[idx_i, idx_j] + 1, 
                                        torch.zeros_like(self.rbo_consistency_count[idx_i, idx_j]))
            self.rbo_consistency_count[idx_i, idx_j] = updated_count

        # 3. 篩選出「轉正」的 FN (例如連續 3 次命中)
        # 這些是我們真正要拉近的樣本
        final_fn_mask = (updated_count >= 5)
        
        # 4. [你的需求] 計算 Precision
        # A. 當前這一秒的 Precision (包含閃現者)
        gt_mask = labels.view(-1, 1).eq(labels.view(1, -1)) & ~torch.eye(batch_size, device=labels.device).bool()
        current_precision = (current_hit.float() * gt_mask.float()).sum() / (current_hit.sum() + 1e-8)
        
        # B. 「連續穩定者」的 Precision (這是你真正拉近的人)
        stable_fn_count = final_fn_mask.sum().item()
        if stable_fn_count > 0:
            stable_precision = (final_fn_mask.float() * gt_mask.float()).sum() / (stable_fn_count + 1e-8)
        else:
            stable_precision = 0.0

        print(f"Epoch {cur_epoch} | Batch Precision: {current_precision:.4f} | Stable Precision: {stable_precision:.4f} | Final FNs: {stable_fn_count}")
        '''
        if reweight_strategy == "1-coverage":
            negative_weights = 1.0 - coverage
        elif reweight_strategy == "boost_TNs":
            mask_no_diag = ~torch.eye(batch_size, dtype=torch.bool, device=x.device)
            flat_coverage = coverage[mask_no_diag]
            threshold_val = torch.quantile(flat_coverage, q=0.3) # 先抓錢30% 的 coverage 作為門檻
            negative_weights = torch.where(coverage <= threshold_val, 
                                        torch.full_like(coverage, boost_factor), 
                                        torch.ones_like(coverage))
        elif reweight_strategy == "hybrid":
            mask_no_diag = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            flat_coverage = coverage[mask_no_diag]
            
            tn_threshold = torch.quantile(flat_coverage, q=0.3) 
            fn_threshold = torch.quantile(flat_coverage, q=0.7) 

            negative_weights = torch.ones_like(coverage)

            negative_weights = torch.where(coverage <= tn_threshold, 
                                           torch.full_like(coverage, boost_factor), 
                                           negative_weights)
            
            down_weight = 1.0 / boost_factor
            negative_weights = torch.where(coverage >= fn_threshold, 
                                           torch.full_like(coverage, down_weight), 
                                           negative_weights)
        elif reweight_strategy == "thresholded":
            negative_weights = torch.where(coverage > coverage_threshold, 1.0 - coverage, torch.ones_like(coverage))
        else:
            negative_weights = torch.ones_like(coverage)

        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=x.device)
        if not neg_include_self:
            negative_weights = negative_weights.masked_fill(diag_mask, 0.0)
            target_sum = float(batch_size - 1)
        else:
            negative_weights = negative_weights.masked_fill(diag_mask, 1.0)
            target_sum = float(batch_size)

        if renormalization:
            current_sum = negative_weights.sum(dim=1, keepdim=True)
            scale_factor = target_sum / (current_sum + 1e-8)
            normalized_weights = negative_weights * scale_factor
            plot_sorted_rbo_heatmap(normalized_weights.cpu().numpy(), labels.cpu().numpy(), cur_epoch, f"{RBO_save_path}/rbo_heatmap_renorm.png")
        else:
            normalized_weights = negative_weights
            scale_factor = torch.ones(batch_size, 1, device=x.device)

        plot_sorted_rbo_heatmap(negative_weights.cpu().numpy(), labels.cpu().numpy(), cur_epoch, f"{RBO_save_path}/rbo_heatmap_ori.png")
        # plot_rbo_cosine_diagnosis(
        #     sim_matrix=raw_cos_matrix, 
        #     coverage=coverage, 
        #     labels=labels, 
        #     epoch=cur_epoch, 
        #     save_dir=f"{RBO_save_path}/diagnosis"
        # )

        self_pos = sim_matrix.diag()
        weighted_neg_sim = (sim_matrix * normalized_weights).sum(dim=1) if not denominator_anchor else (sim_matrix_anchor * normalized_weights).sum(dim=1)
        loss = -torch.log(self_pos / (weighted_neg_sim + 1e-8) + 1e-8).mean()

        stats['avg_scale_factor'] = scale_factor.mean().item()
        return loss, coverage, stats

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

import random
def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def analyze_embedding_structure(model, dataloader, device):
    model.eval()
    all_emb = []
    all_y = []
    with torch.no_grad():
        for data, _ in dataloader: # 這裡用 eval 模式不需要 aug
            data = data.to(device)
            emb, _ = model.encoder(data.x, data.edge_index, data.batch)
            all_emb.append(emb.cpu())
            all_y.append(data.y.cpu())
    
    z = F.normalize(torch.cat(all_emb, dim=0), p=2, dim=1) # (N, D)
    y = torch.cat(all_y, dim=0)
    num_samples = z.size(0)
    
    # 1. 計算所有 Pair 的相似度矩陣
    sim_matrix = torch.mm(z, z.t()) # (N, N)
    
    # 2. 建立標籤 Mask
    y_col = y.view(-1, 1)
    mask_same = y_col.eq(y_col.t())
    mask_diff = ~mask_same
    diag = torch.eye(num_samples, dtype=torch.bool)
    
    # 3. 提取指標
    # 真 FN 相似度 (同類且非自己)
    true_fn_sims = sim_matrix[mask_same & ~diag].mean().item()
    # 不同類相似度
    true_tn_sims = sim_matrix[mask_diff].mean().item()
    
    # 4. Alignment & Uniformity
    # 這裡簡化計算，Uniformity 通常計算所有對的 RBF kernel
    uniformity = torch.pdist(z).pow(2).mul(-2).exp().mean().log().item()

    return {
        'avg_true_fn_sim': true_fn_sims,
        'avg_true_tn_sim': true_tn_sims,
        'uniformity': uniformity,
        'separation_margin': true_fn_sims - true_tn_sims # 預期這個值越來越大
    }

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
    # for epoch in range(start_epoch, epochs + 1):
        # lr = scheduler.step()
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
        if args.mode == 'rm_FNs_by_ENs':
                    epoch_en_stats = {
                        'pFN_recall': 0.0,
                        'pFN_wrong_rate': 0.0,
                        'pFN_precision': 0.0,
                        'pFN_f1': 0.0,
                        'num_deleted_pFN': 0.0,
                        'pFN_ratio': 0.0
                    }
        if args.mode == 'reweight_FNs_by_ENs':
                    epoch_en_stats = {
                        'soft_pFN_recall': 0.0,
                        'soft_pFN_precision': 0.0,
                        'soft_pFN_f1': 0.0,
                        'avg_coverage': 0.0,
                        'Precision@1': 0.0,
                        'Precision@10': 0.0,
                        'Precision@5': 0.0,
                        'Precision@50': 0.0,
                        'Recall@1': 0.0,
                        'Recall@10': 0.0,
                        'Recall@5': 0.0,
                        'Recall@50': 0.0,
                        # 'Recall@1': 0.0,
                        # 'Recall@5': 0.0,
                        # 'Recall@10': 0.0,
                        # 'mAP': 0.0
                    }
        if args.mode == 'reweight_FNs_by_RBO':
            epoch_en_stats = {
                'soft_pFN_recall': 0.0,
                'soft_pFN_precision': 0.0,
                'soft_pFN_f1': 0.0,
                'avg_coverage': 0.0,
                'RBO_FN_Precision@1': 0.0,
                'RBO_FN_Precision@10': 0.0,
                'RBO_FN_Precision@5': 0.0,
                'RBO_FN_Precision@50': 0.0,
                'RBO_TN_Precision@1': 0.0,
                'RBO_TN_Precision@10': 0.0,
                'RBO_TN_Precision@5': 0.0,
                'RBO_TN_Precision@50': 0.0,
                'RBO_FN_Recall@1': 0.0,
                'RBO_FN_Recall@10': 0.0,
                'RBO_FN_Recall@5': 0.0,
                'RBO_FN_Recall@50': 0.0,
                'RBO_TN_Recall@1': 0.0,
                'RBO_TN_Recall@10': 0.0,
                'RBO_TN_Recall@5': 0.0,
                'RBO_TN_Recall@50': 0.0,
            }
        if args.get_f1_scores_by_deg_boundary:
            all_x_embeddings = []
            all_x_aug_embeddings = []
            all_labels = []

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
            if args.rotate == 'by_angle':
                # x_aug = rotate_embedding_high_dim_by_angle(x, angle_degree=30.0, random_plane=True)
                x_aug = rotate_embedding_targeted_angle(x, angle_degree=args.rotate_angle_deg)
            elif args.rotate == 'random':
                x_aug = rotate_embedding_high_dim(x, rotation_type='random')

            if args.get_f1_scores_by_deg_boundary:
                all_x_embeddings.append(x.detach().cpu())
                all_x_aug_embeddings.append(x_aug.detach().cpu())
                all_labels.append(data.y.detach().cpu())

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

            # Reweighted Loss Modes (assuming these are defined within the model class)
            elif args.mode == 'reweighted':
                loss, _, pos_sim, neg_sim = model.loss_cal_reweighted(x, x_aug)

            elif args.mode == 'reweighted_by_angle':
                loss, pos_sim, neg_sim = model.reweighted_by_angle(x, x_aug, deg_boundary=args.rotate_angle_deg)
            elif args.mode == 'rm_FNs_by_ENs':
                loss, en_stats = model.loss_cal_rm_FNs_by_ENs(x, x_aug, labels, epoch, epochs, base_en_threshold=args.base_en_threshold, max_en_threshold=args.max_en_threshold, coverage_threshold=args.coverage_threshold, neg_include_self=args.neg_include_self)
                # === 累加每個 Batch 的指標 ===
                epoch_en_stats['pFN_recall'] += en_stats['pFN_recall']
                epoch_en_stats['pFN_wrong_rate'] += en_stats['pFN_wrong_rate']
                epoch_en_stats['pFN_precision'] += en_stats['pFN_precision']
                epoch_en_stats['pFN_f1'] += en_stats['pFN_f1']
                epoch_en_stats['num_deleted_pFN'] += en_stats['num_deleted_pFN']
                epoch_en_stats['pFN_ratio'] += en_stats['pFN_ratio_in_batch']

            elif args.mode == 'reweight_FNs_by_ENs':
                loss, coverage, en_stats = model.loss_cal_reweighted_FNs_by_ENs(x, x_aug, labels, epoch, epochs, base_en_threshold=args.base_en_threshold, max_en_threshold=args.max_en_threshold, neg_include_self=args.neg_include_self, reweight_strategy=args.reweight_strategy, coverage_threshold=args.coverage_threshold, renormalization=args.renormalization)
                # === 累加每個 Batch 的指標 ===
                epoch_en_stats['soft_pFN_recall'] += en_stats['soft_pFN_recall']
                epoch_en_stats['soft_pFN_precision'] += en_stats['soft_pFN_precision']
                epoch_en_stats['soft_pFN_f1'] += en_stats['soft_pFN_f1']
                epoch_en_stats['avg_coverage'] += en_stats['avg_coverage']
                for k in [1, 5, 10, 50]: # 根據你在 validate_rbo_ranking 設的 k
                    epoch_en_stats[f'Precision@{k}'] += en_stats.get(f'RBO_FN_Precision@{k}', 0)
                    epoch_en_stats[f'Recall@{k}'] += en_stats.get(f'RBO_FN_Recall@{k}', 0)

            elif args.mode == 'reweight_FNs_by_RBO':
                rbo_save_dir = f'{save_dir}/plot_RBO_weight/RBO_epoch_{epoch}'
                os.makedirs(rbo_save_dir, exist_ok=True)
                
                jaccard_sim = model.compute_geometry_jaccard_sim(data.to(device))
                causal_sub_sim = model.compute_causal_subgraph_sim(data, model, device).to(device)
                consensus_geo_sim = torch.sqrt(jaccard_sim * causal_sub_sim)

                if epoch % 50 == 0:
                    raw_cos_matrix = torch.mm(F.normalize(x, dim=1), F.normalize(x_aug, dim=1).T)
                    sim_matrix = torch.exp(torch.mm(F.normalize(x, dim=1), F.normalize(x_aug, dim=1).T) / 0.2)
                    # jaccard_sim = model.compute_geometry_jaccard_sim(data.to(device))
                    # causal_sub_sim = model.compute_causal_subgraph_sim(data, model, device).to(device)
                    # consensus_geo_sim = torch.sqrt(jaccard_sim * causal_sub_sim)

                    all_matrices = {
                        "Cosine": raw_cos_matrix,
                        "Jaccard": jaccard_sim,
                        "Causal": causal_sub_sim,
                        "Consensus": consensus_geo_sim
                    }

                    analysis_pool = {**all_matrices}
                    for name, mtx in all_matrices.items():
                        # 對每一種幾何指標都算一次 RBO Coverage
                        cov, _ = model.identify_fn_by_rbo_coverage(mtx, labels, p=args.RBO_p)
                        analysis_pool[f"RBO_{name}"] = cov

                    matrix_plot_dir = f"{rbo_save_dir}/all_pairs_diagnosis"
                    os.makedirs(matrix_plot_dir, exist_ok=True)
                    
                    keys = list(analysis_pool.keys())
                    for i in range(len(keys)):
                        for j in range(i + 1, len(keys)):
                            name_x = keys[i]
                            name_y = keys[j]
                            mtx_x = analysis_pool[name_x]
                            mtx_y = analysis_pool[name_y]
                            
                            # A. 計算相關性指標 (Alignment)
                            mask = ~torch.eye(labels.size(0), device=device).bool()
                            correlation = torch.corrcoef(torch.stack([mtx_x[mask], mtx_y[mask]]))[0, 1]
                            writer.add_scalar(f"Alignment/{name_x}_vs_{name_y}", correlation.item(), epoch)
                            
                            # B. 繪製診斷圖 (X 軸為 name_x, Y 軸為 name_y)
                            # 建立子目錄避免檔案太亂
                            pair_save_dir = f"{matrix_plot_dir}/{name_x}_vs_{name_y}"
                            os.makedirs(pair_save_dir, exist_ok=True)
                            
                            # 調用妳原本的繪圖函數
                            plot_rbo_cosine_diagnosis(
                                sim_matrix=mtx_x,  # X 軸指標
                                coverage=mtx_y,    # Y 軸指標
                                labels=labels,
                                epoch=epoch,
                                save_dir=pair_save_dir
                            )

                loss, coverage, en_stats = model.loss_cal_reweighted_FNs_by_RBO(x, x_aug, labels, batch_indices=data.idx.to(device), neg_include_self=args.neg_include_self, 
                                                                                reweight_strategy=args.reweight_strategy, coverage_threshold=args.coverage_threshold, 
                                                                                renormalization=args.renormalization, RBO_anchor=args.RBO_anchor,
                                                                                denominator_anchor=args.denominator_anchor, RBO_p=args.RBO_p,
                                                                                RBO_save_path=rbo_save_dir, boost_factor=args.tn_weight, cur_epoch=epoch, geo_sim_matrix=consensus_geo_sim)
                epoch_en_stats['soft_pFN_recall'] += en_stats['soft_pFN_recall']
                epoch_en_stats['soft_pFN_precision'] += en_stats['soft_pFN_precision']
                epoch_en_stats['soft_pFN_f1'] += en_stats['soft_pFN_f1']
                epoch_en_stats['avg_coverage'] += en_stats['avg_coverage']
                for k in [1, 5, 10, 50]: # 根據你在 validate_rbo_ranking 設的 k
                    epoch_en_stats[f'RBO_FN_Precision@{k}'] += en_stats.get(f'RBO_FN_Precision@{k}', 0)
                    epoch_en_stats[f'RBO_FN_Recall@{k}'] += en_stats.get(f'RBO_FN_Recall@{k}', 0)
                    epoch_en_stats[f'RBO_TN_Precision@{k}'] += en_stats.get(f'RBO_TN_Precision@{k}', 0)
                    epoch_en_stats[f'RBO_TN_Recall@{k}'] += en_stats.get(f'RBO_TN_Recall@{k}', 0)
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

            # print(x)
            # print(x_aug)
            oloss = odecay * l2_reg_ortho(model)
            loss_all += loss.item() * data.num_graphs
            if not (args.mode == 'rm_FNs_by_ENs' or args.mode == 'reweight_FNs_by_ENs' or args.mode == 'reweight_FNs_by_RBO'):
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
        if not (args.mode == 'rm_FNs_by_ENs' or args.mode == 'reweight_FNs_by_ENs' or args.mode == 'reweight_FNs_by_RBO'):
            writer.add_scalar('Similarity/pos_sim', pos_sim_all / len(dataloader), epoch)
            writer.add_scalar('Similarity/neg_sim', neg_sim_all / len(dataloader), epoch)
        elif args.mode == 'rm_FNs_by_ENs':
            num_batches = len(dataloader)
            
            avg_recall = epoch_en_stats['pFN_recall'] / num_batches
            avg_wrong_rate = epoch_en_stats['pFN_wrong_rate'] / num_batches
            avg_precision = epoch_en_stats['pFN_precision'] / num_batches
            avg_f1 = epoch_en_stats['pFN_f1'] / num_batches
            avg_deleted_count = epoch_en_stats['num_deleted_pFN'] / num_batches
            avg_deleted_ratio = epoch_en_stats['pFN_ratio'] / num_batches

            writer.add_scalar('EN_FN_Dynamics/Removal_Recall', avg_recall, epoch)
            writer.add_scalar('EN_FN_Dynamics/Wrong_Rate_TN_Killed', avg_wrong_rate, epoch)
            writer.add_scalar('EN_FN_Dynamics/Removal_Precision', avg_precision, epoch)
            writer.add_scalar('EN_FN_Dynamics/Removal_F1_Score', avg_f1, epoch)
            # writer.add_scalar('EN_FN_Dynamics/Current_EN_THRESHOLD', current_en_threshold_val, epoch)
            writer.add_scalar('FN_Stats/Avg_Deleted_FN_Count_Per_Batch', avg_deleted_count, epoch)
            writer.add_scalar('FN_Stats/Avg_Deleted_FN_Ratio_Per_Batch', avg_deleted_ratio, epoch)
            writer.add_scalars('EN_FN_Dynamics/Purity_Check', {
                'Cleaned_FN_Ratio': en_stats['remaining_pFN_ratio'],
                'Original_FN_Ratio': en_stats['original_pFN_ratio']
            }, epoch)
        elif args.mode == 'reweight_FNs_by_ENs':
            num_batches = len(dataloader)
            
            avg_recall = epoch_en_stats['soft_pFN_recall'] / num_batches
            avg_precision = epoch_en_stats['soft_pFN_precision'] / num_batches
            avg_f1 = epoch_en_stats['soft_pFN_f1'] / num_batches
            avg_coverage = epoch_en_stats['avg_coverage'] / num_batches
            # avg_recall_k = {
            #     'R@1': epoch_en_stats['Recall@1'] / num_batches,
            #     'R@5': epoch_en_stats['Recall@5'] / num_batches,
            #     'R@10': epoch_en_stats['Recall@10'] / num_batches,
            # }
            # avg_map = epoch_en_stats['mAP'] / num_batches

            writer.add_scalar('EN_FN_Dynamics/Reweight_Recall', avg_recall, epoch)
            writer.add_scalar('EN_FN_Dynamics/Reweight_Precision', avg_precision, epoch)
            writer.add_scalar('EN_FN_Dynamics/Reweight_F1_Score', avg_f1, epoch)
            # writer.add_scalar('EN_FN_Dynamics/Current_EN_THRESHOLD', current_en_threshold_val, epoch)
            writer.add_scalar('FN_Stats/Avg_Coverage', avg_coverage, epoch)
            # writer.add_scalars('Ranking_Performance/Recall_at_K', avg_recall_k, epoch)
            # writer.add_scalar('Ranking_Performance/mAP', avg_map, epoch)
            writer.add_scalar('Ranking/Precision@1', epoch_en_stats['Precision@1'] / num_batches, epoch)
            writer.add_scalar('Ranking/Precision@10', epoch_en_stats['Precision@10'] / num_batches, epoch)
            writer.add_scalar('Ranking/Precision@5', epoch_en_stats['Precision@5'] / num_batches, epoch)
            writer.add_scalar('Ranking/Precision@50', epoch_en_stats['Precision@50'] / num_batches, epoch)
            writer.add_scalar('Ranking/Recall@1', epoch_en_stats['Recall@1'] / num_batches, epoch)
            writer.add_scalar('Ranking/Recall@10', epoch_en_stats['Recall@10'] / num_batches, epoch)
            writer.add_scalar('Ranking/Recall@5', epoch_en_stats['Recall@5'] / num_batches, epoch)
            writer.add_scalar('Ranking/Recall@50', epoch_en_stats['Recall@50'] / num_batches, epoch)
        elif args.mode == 'reweight_FNs_by_RBO':
            num_batches = len(dataloader)
            
            avg_recall = epoch_en_stats['soft_pFN_recall'] / num_batches
            avg_precision = epoch_en_stats['soft_pFN_precision'] / num_batches
            avg_f1 = epoch_en_stats['soft_pFN_f1'] / num_batches
            avg_coverage = epoch_en_stats['avg_coverage'] / num_batches

            writer.add_scalar('RBO_Dynamics/Reweight_Recall', avg_recall, epoch)
            writer.add_scalar('RBO_Dynamics/Reweight_Precision', avg_precision, epoch)
            writer.add_scalar('RBO_Dynamics/Reweight_F1_Score', avg_f1, epoch)
            writer.add_scalar('RBO_Dynamics/Avg_Coverage', avg_coverage, epoch)

            writer.add_scalar('RBO_Ranking/FN_Precision@1', epoch_en_stats['RBO_FN_Precision@1'] / num_batches, epoch)
            writer.add_scalar('RBO_Ranking/FN_Precision@10', epoch_en_stats['RBO_FN_Precision@10'] / num_batches, epoch)
            writer.add_scalar('RBO_Ranking/FN_Precision@5', epoch_en_stats['RBO_FN_Precision@5'] / num_batches, epoch)
            writer.add_scalar('RBO_Ranking/FN_Precision@50', epoch_en_stats['RBO_FN_Precision@50'] / num_batches, epoch)
            writer.add_scalar('RBO_Ranking/FN_Recall@1', epoch_en_stats['RBO_FN_Recall@1'] / num_batches, epoch)
            writer.add_scalar('RBO_Ranking/FN_Recall@10', epoch_en_stats['RBO_FN_Recall@10'] / num_batches, epoch)
            writer.add_scalar('RBO_Ranking/FN_Recall@5', epoch_en_stats['RBO_FN_Recall@5'] / num_batches, epoch)
            writer.add_scalar('RBO_Ranking/FN_Recall@50', epoch_en_stats['RBO_FN_Recall@50'] / num_batches, epoch)
            writer.add_scalar('RBO_Ranking/TN_Precision@1', epoch_en_stats['RBO_TN_Precision@1'] / num_batches, epoch)
            writer.add_scalar('RBO_Ranking/TN_Precision@10', epoch_en_stats['RBO_TN_Precision@10'] / num_batches, epoch)
            writer.add_scalar('RBO_Ranking/TN_Precision@5', epoch_en_stats['RBO_TN_Precision@5'] / num_batches, epoch)
            writer.add_scalar('RBO_Ranking/TN_Precision@50', epoch_en_stats['RBO_TN_Precision@50'] / num_batches, epoch)
            writer.add_scalar('RBO_Ranking/TN_Recall@1', epoch_en_stats['RBO_TN_Recall@1'] / num_batches, epoch)
            writer.add_scalar('RBO_Ranking/TN_Recall@10', epoch_en_stats['RBO_TN_Recall@10'] / num_batches, epoch)
            writer.add_scalar('RBO_Ranking/TN_Recall@5', epoch_en_stats['RBO_TN_Recall@5'] / num_batches, epoch)
            writer.add_scalar('RBO_Ranking/TN_Recall@50', epoch_en_stats['RBO_TN_Recall@50'] / num_batches, epoch)

        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader.dataset)))
        # print("pos sim = ", pos_sim_all, "; neg sim = ", neg_sim_all)
        loss_list.append(loss_all / len(dataloader.dataset))

        if args.get_f1_scores_by_deg_boundary:
            if len(all_x_embeddings) > 0:
                f1_epoch, precision_epoch, recall_epoch = calculate_f1_scores_by_deg_boundary(
                    all_x_embeddings, 
                    all_x_aug_embeddings, 
                    all_labels, 
                    deg_boundary=args.rotate_angle_deg
                )
                writer.add_scalar('FN_Analysis_Epoch/Precision', precision_epoch, epoch)
                writer.add_scalar('FN_Analysis_Epoch/Recall', recall_epoch, epoch)
                writer.add_scalar('FN_Analysis_Epoch/F1', f1_epoch, epoch)
                
                print(f"Epoch {epoch} FN Analysis: F1={f1_epoch:.4f}, P={precision_epoch:.4f}, R={recall_epoch:.4f}")
            else:
                print(f"Epoch {epoch}: No data accumulated for FN Analysis.")

        if epoch % log_interval == 0:
            if args.do_hn_analysis:
                num_high_sim_pairs, num_fp_hn, num_tp = analyze_high_similarity_negatives(
                    model=model, 
                    dataloader_eval=dataloader_eval, 
                    device=device, 
                    args=args, 
                    similarity_threshold=0.8, # 例如，固定使用 0.8
                    epoch=epoch
                )
                writer.add_scalar('HN_Analysis/Num_High_Sim_Pairs', num_high_sim_pairs, epoch)
                writer.add_scalar('HN_Analysis/Num_FP_High_Sim', num_fp_hn, epoch)
                writer.add_scalar('HN_Analysis/Num_TP', num_tp, epoch)
                writer.add_scalar('HN_Analysis/FP_Rate', num_fp_hn / (num_high_sim_pairs + 1e-8), epoch)

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

        struct_stats = analyze_embedding_structure(model, dataloader_eval, device)
        writer.add_scalar('Structure/True_FN_Similarity', struct_stats['avg_true_fn_sim'], epoch)
        writer.add_scalar('Structure/Separation_Margin', struct_stats['separation_margin'], epoch)
        writer.add_scalar('Structure/Uniformity', struct_stats['uniformity'], epoch)

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
    
