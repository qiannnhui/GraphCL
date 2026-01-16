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
from make_save_dir import make_save_dir # 引入創建儲存目錄的函數
from save_load_ckpts import load_checkpoint, save_checkpoint # 引入檢查點函數
from utils import create_pos_and_neg_mask, calculate_f1_scores_by_deg_boundary
from analyze_high_similarity_negatives import analyze_high_similarity_negatives
from unified_loss import unified_loss, get_pair_angles, flexible_hard_mining_loss
from rotate_by_angle import rotate_embedding_high_dim_by_angle, rotate_embedding_high_dim, rotate_embedding_targeted_angle

# class GcnInfomax(nn.Module):
#   def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
#     super(GcnInfomax, self).__init__()

#     self.alpha = alpha
#     self.beta = beta
#     self.gamma = gamma
#     self.prior = args.prior

#     self.embedding_dim = mi_units = hidden_dim * num_gc_layers
#     self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

#     self.local_d = FF(self.embedding_dim)
#     self.global_d = FF(self.embedding_dim)
#     # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
#     # self.global_d = MIFCNet(self.embedding_dim, mi_units)

#     if self.prior:
#         self.prior_d = PriorDiscriminator(self.embedding_dim)

#     self.init_emb()

#   def init_emb(self):
#     initrange = -1.5 / self.embedding_dim
#     for m in self.modules():
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)


#   def forward(self, x, edge_index, batch, num_graphs):

#     # batch_size = data.num_graphs
#     if x is None:
#         x = torch.ones(batch.shape[0]).to(device)

#     y, M = self.encoder(x, edge_index, batch)
    
#     g_enc = self.global_d(y)
#     l_enc = self.local_d(M)

#     mode='fd'
#     measure='JSD'
#     local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)
 
#     if self.prior:
#         prior = torch.rand_like(y)
#         term_a = torch.log(self.prior_d(prior)).mean()
#         term_b = torch.log(1.0 - self.prior_d(y)).mean()
#         PRIOR = - (term_a + term_b) * self.gamma
#     else:
#         PRIOR = 0
    
#     return local_global_loss + PRIOR


class simclr(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1, shuffle_DBN=False, dataset_num_features=1):
    super(simclr, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    bn_layer = [DecorrelatedShuffledBatchNorm1d(self.embedding_dim)] if shuffle_DBN else []
    self.proj_head = nn.Sequential(
        nn.Linear(self.embedding_dim, self.embedding_dim), 
        *bn_layer,
        nn.ReLU(inplace=True), 
        nn.Linear(self.embedding_dim, self.embedding_dim)
    )
    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index, batch, num_graphs):

    # batch_size = data.num_graphs
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)
    
    y = self.proj_head(y)
    
    return y, M

  def min_max_normalization(self, sim_matrix):
      
      min_sim = sim_matrix.min()
      max_sim = sim_matrix.max()
      normalized_sim = (sim_matrix - min_sim) / (max_sim - min_sim)
        
      return normalized_sim

  def calculate_confusion_matrix(self, sim_matrix, pos_mask):

      true_labels = pos_mask.float().view(-1).cpu().numpy()  # 轉換為 1D
      pred_labels = (sim_matrix.view(-1) > 0.8).float().cpu().numpy()  # 相似度 > 0.5 判斷為正樣本
      cm = confusion_matrix(true_labels, pred_labels)

      return cm

  def get_confusion_matrix(self, labels, sim_matrix, args, epoch=None):
        pos_mask, neg_mask = create_pos_and_neg_mask(labels=labels)
        # normalized_sim_matrix = self.min_max_normalization(sim_matrix=sim_matrix)
        # cm = self.calculate_confusion_matrix(pos_mask=pos_mask, sim_matrix=normalized_sim_matrix)
        # plot_similarity_matrix(sim_matrix=sim_matrix, pos_mask=pos_mask, file_name=f'epoch_{epoch}_not_normalized_{args.aug}_{args.mode}')
        # plot_similarity_matrix(sim_matrix=normalized_sim_matrix, pos_mask=pos_mask, file_name=f'epoch_{epoch}_normalized_{args.aug}_{args.mode}')
        # plot_similarity_distribution(sim_matrix=normalized_sim_matrix, pos_mask=pos_mask, file_name=f'sim_distribution_epoch_{epoch}_normalized_{args.aug}_{args.mode}')
        plot_similarity_distribution(sim_matrix=sim_matrix, pos_mask=pos_mask, neg_mask=neg_mask, args=args, epoch=epoch)
        # print("cm = ", cm)


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

  def _get_similarity_matrix(self, x, x_aug, similarity_measure="cosine", T=0.2):
        
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        if similarity_measure == "cosine":
            # Cosine similarity
            cos_sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
            cos_sim_matrix = torch.clamp(cos_sim_matrix, -1.0, 1.0)  # Clamp to avoid invalid values
            sim_matrix = torch.exp(cos_sim_matrix / T)

        elif similarity_measure == "l2":
            # L2 norm similarity
            l2_matrix = torch.cdist(x, x_aug, p=2)  # Pairwise L2 distance
            l2_matrix = l2_matrix / l2_matrix.max()  # Normalize L2 norm to [0, 1]
            sim_matrix = torch.exp(-l2_matrix / T)  # Convert distance to similarity

        elif similarity_measure == "cosine+l2":
            # Compute cosine similarity
            cos_sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
            cos_sim_matrix = torch.clamp(cos_sim_matrix, -1.0, 1.0)  # Clamp to avoid invalid values

            # Compute L2 norm
            l2_matrix = torch.cdist(x, x_aug, p=2)  # Pairwise L2 distance
            l2_matrix = l2_matrix / l2_matrix.max()  # Normalize L2 norm to [0, 1]

            # Combine cosine similarity and L2 norm
            combined_matrix = cos_sim_matrix - l2_matrix  # Example: subtract L2 norm from cosine similarity
            sim_matrix = torch.exp(combined_matrix / T)  # Convert to similarity

        else:
            raise ValueError(f"Unsupported similarity measure: {similarity_measure}")
        
        return sim_matrix
  
  def _sample_negative_mask(self, labels, batch_size, device, num_negatives=2):
    """
    為每個 Anchor 隨機採樣 num_negatives 個真正的負樣本 (不同標籤) 作為分母。
    """
    labels = labels.view(-1, 1)
    neg_mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        # 找出所有真正的負樣本 (不同標籤)
        true_neg_indices = (labels != labels[i]).nonzero(as_tuple=True)[0]
        
        if true_neg_indices.numel() > 0:
            # 隨機採樣 num_negatives 個索引 (或所有可用的負樣本，如果少於 num_negatives)
            num_to_sample = min(num_negatives, true_neg_indices.numel())
            
            # 確保採樣是唯一的
            if num_to_sample > 0:
                sampled_indices = true_neg_indices[torch.randperm(true_neg_indices.numel())[:num_to_sample]]
                neg_mask[i, sampled_indices] = True
        
    return neg_mask

  # 由於您沒有提供原始 simclr 類別，我們將假設這是 simclr 的方法
  def _get_exp_indices(self, labels, batch_size, device):
    """
    為每個 Anchor z_i 找到一個實驗用的 Positive (FP) 和 Negatives (FN)。
    
    Args:
        labels: 批次的真實標籤 (B, 1).
    
    Returns:
        pos_indices (B,): 每個 Anchor 的 FP 樣本索引 (不同標籤的隨機樣本).
        neg_mask (B, B): 每個 Anchor 的 FN 樣本掩碼 (相同標籤的所有其他樣本).
    """
    labels = labels.view(-1, 1)
    
    # 1. 構建 FP 樣本索引 (隨機選擇一個不同標籤的樣本)
    fp_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for i in range(batch_size):
        # 找出所有不同標籤的樣本索引
        diff_label_indices = (labels != labels[i]).nonzero(as_tuple=True)[0]
        
        if diff_label_indices.numel() > 0:
            # 隨機挑選一個不同標籤的索引
            rand_idx = torch.randint(0, diff_label_indices.numel(), (1,), device=device)
            fp_indices[i] = diff_label_indices[rand_idx]
        else:
            # 如果 batch 內所有標籤都相同 (罕見)，則使用自身 (將導致 log(0) 或 NaN，但在 large batch 下應避免)
            fp_indices[i] = i 

    # 2. 構建 FN 樣本掩碼 (相同標籤的所有其他樣本)
    pos_mask_all = labels.eq(labels.T) # (B, B)
    neg_mask = pos_mask_all & ~torch.diag(torch.ones(batch_size, device=device, dtype=torch.bool))
    
    return fp_indices, neg_mask

  def loss_cal_FN_FP(self, x, x_aug, labels=None, get_cm=False, epoch=None, FN=True, FP=True, FP_all=False):
    '''極端的case看看FN, FP是否真的對結果造成很大的影響'''

    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    # Note: 由於 x_aug 應該是 x 的複製，我們使用 x 和 x_aug_T
    cos_sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(cos_sim_matrix / T)

    fp_indices, neg_mask = self._get_exp_indices(labels, batch_size, x.device)

    # === 1. 分子 (Positive Sim - 使用 FP 樣本) ===
    # 這裡的 fp_indices 是 z_j 的索引，所以我們從 sim_matrix 中取出 z_i 對應 z_j 的相似度
    if FP:
        fp_sim = sim_matrix[range(batch_size), fp_indices] # (B,)
    elif not FP:
        fp_sim = sim_matrix[range(batch_size), range(batch_size)]

    if FP_all:
        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.T)
        pos_sim = sim_matrix * pos_mask
        fp_sim = pos_sim.sum(dim=1) - sim_matrix.diag()

    # === 2. 分母 (Negative Sim - 相同標籤的所有其他樣本) ===
    # neg_mask 已經是相同標籤但排除自身的掩碼
    if FN:
        neg_sim_sum = (sim_matrix * neg_mask).sum(dim=1) # (B,)
    elif not FN:
        neg_sim_sum = sim_matrix.sum(dim=1) - sim_matrix.diag()

    # === 3. 損失計算 (Logit = FP / Negatives) ===
    # 這是 InfoNCE 的形式，但 FP 位於分子，Negatives 位於分母
    loss = fp_sim / (neg_sim_sum + 1e-8)
    loss = -torch.log(loss + 1e-8).mean()
    
    pos_sim_ = fp_sim.mean()
    neg_sim_ = neg_sim_sum.mean()

    return loss, pos_sim_, neg_sim_

  def identify_fn_by_en_curriculum(self, sim_matrix, labels, epoch, total_epochs, 
                               base_en_ratio=0.1, max_en_ratio=0.5, overlap_threshold=0.5):
        """
        分析 EN-based FN 識別的動態指標
        Args:
            sim_matrix: (B, B) 相似度矩陣 (exp(cos/T))
            labels: (B,) 標籤
            base_en_ratio/max_en_ratio: 用於動態增加 EN 比例
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
        curr_en_ratio = base_en_ratio + (max_en_ratio - base_en_ratio) * (epoch / total_epochs)
        k = max(1, int(batch_size * curr_en_ratio))
        
        # 識別預測的 FN (Pred FN)
        _, en_indices = torch.topk(sim_matrix, k=k, dim=1, largest=False)
        en_mask = torch.zeros_like(sim_matrix).scatter_(1, en_indices, 1.0)
        shared_count = torch.matmul(en_mask, en_mask.T)
        pred_fn_mask = (shared_count / k >= overlap_threshold) & ~diag_mask
        
        # --- 指標統計 ---
        tp_fn = (pred_fn_mask & gt_fn_mask).sum().float() # 拿對的
        fp_fn = (pred_fn_mask & gt_tn_mask).sum().float() # 拿錯的 (誤殺 TN)
        

        # 1. FN 拿對率 (Recall)
        fn_recall = tp_fn / (total_gt_fn + 1e-8)

        # 2. 移除精準度 (Precision)
        fn_precision = tp_fn / (pred_fn_mask.sum().float() + 1e-8)

        fn_f1 = 2 * (fn_precision * fn_recall) / (fn_precision + fn_recall + 1e-8)
        
        # 3. FN 拿錯率 (False Alarm Rate / FPR) - 你最關心的指標
        fn_wrong_rate = fp_fn / (total_gt_tn + 1e-8)
        
        # 4. 分母純淨度分析
        denom_mask_after = ~diag_mask & ~pred_fn_mask
        remaining_fn_count = (denom_mask_after & gt_fn_mask).sum().float()
        remaining_fn_ratio = remaining_fn_count / (denom_mask_after.sum().float() + 1e-8)
        original_fn_ratio = total_gt_fn / (batch_size * (batch_size - 1) + 1e-8)

        stats = {
            'fn_recall': fn_recall.item(),
            'fn_wrong_rate': fn_wrong_rate.item(),
            'fn_precision': fn_precision.item(),
            'fn_f1': fn_f1.item(),
            'remaining_fn_ratio': remaining_fn_ratio.item(),
            'original_fn_ratio': original_fn_ratio.item(),
            'curr_en_ratio': curr_en_ratio
        }
        
        return pred_fn_mask, stats

  def identify_fn_by_en(self, sim_matrix, labels, top_k_en=10, overlap_threshold=0.5):
        """
        基於共享 Easy Negatives 識別 False Negatives 並計算指標
        Args:
            sim_matrix: (B, B) 相似度矩陣 (已經過 exp/T)
            labels: (B,) 真實標籤
            top_k_en: 取相似度最低的前 K 個作為 EN 集合
            overlap_threshold: 共享 EN 的比例門檻，超過則判定為 FN
        """
        batch_size = sim_matrix.size(0)
        device = sim_matrix.device
        
        # 1. 為每個樣本找出相似度最低的 Top-K (Easy Negatives)
        # 取得排序索引，前 top_k_en 個就是相似度最小的
        _, en_indices = torch.topk(sim_matrix, k=top_k_en, dim=1, largest=False)
        
        # 2. 構建 EN 掩碼矩陣 (B, B) -> 若 j 是 i 的 EN 則為 1
        en_mask = torch.zeros_like(sim_matrix).scatter_(1, en_indices, 1.0)
        
        # 3. 計算樣本間共享 EN 的數量 (Matrix Multiplication)
        # shared_count[i, j] 代表 i 和 j 共同擁有的 EN 數量
        shared_count = torch.matmul(en_mask, en_mask.T)
        
        # 4. 計算共享比例並判定推斷出的 FN (Predicted FN)
        # 排除自己 (對角線)
        shared_ratio = shared_count / top_k_en
        pred_fn_mask = (shared_ratio >= overlap_threshold)
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        pred_fn_mask = pred_fn_mask & ~diag_mask
        
        # 5. 真實 FN 統計 (Ground Truth)
        # 真實 FN 定義：標籤相同但不是自己
        labels_col = labels.view(-1, 1)
        gt_fn_mask = labels_col.eq(labels_col.T) & ~diag_mask
        
        # 6. 計算 Precision, Recall, F1
        tp = (pred_fn_mask & gt_fn_mask).sum().float()
        fp = (pred_fn_mask & ~gt_fn_mask).sum().float()
        fn = (~pred_fn_mask & gt_fn_mask).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return pred_fn_mask, precision.item(), recall.item(), f1.item()
  
#   def loss_cal_rm_FNs_by_ENs(self, x, x_aug, labels, top_k_en=10, overlap_threshold=0.5, neg_include_self=True):
  def loss_cal_rm_FNs_by_ENs(self, x, x_aug, labels, cur_epoch, total_epochs, base_en_ratio=0.1, max_en_ratio=0.5, overlap_threshold=0.5, neg_include_self=True):
        T = 0.2
        num_samples, _ = x.size()
        # 計算全矩陣相似度
        sim_matrix = torch.exp(torch.mm(F.normalize(x, dim=1), F.normalize(x_aug, dim=1).T) / T)
        
        # 使用我們新寫的函數
        # top_k 可以設為 batch_size 的 20%~30%
        # pred_fn_mask, prec, rec, f1_score = self.identify_fn_by_en(
        #     sim_matrix, labels, top_k_en=top_k_en, overlap_threshold=overlap_threshold
        # )
        pred_fn_mask, en_stats = self.identify_fn_by_en_curriculum(
                    sim_matrix, labels, cur_epoch, total_epochs,
                    base_en_ratio=base_en_ratio, max_en_ratio=max_en_ratio, overlap_threshold=overlap_threshold
                )
        
        # --- 計算 Loss ---
        self_pos = sim_matrix.diag() # (B,)
        
        # 定義分母：原本是 sum(sim) - self_pos
        # 現在要額外扣除 pred_fn_mask 標註出的項
        # 我們建立一個 mask，只保留真正的 Negative (非 Self 且 非推斷出的 FN)
        denom_mask = torch.ones_like(sim_matrix, dtype=torch.bool)
        diag_mask = torch.eye(num_samples, dtype=torch.bool, device=device)
        denom_mask[diag_mask] = False if not neg_include_self else True    # 扣除自己 if neg_include_self=False
        denom_mask[pred_fn_mask] = False # 扣除推斷出的 FN
        
        neg_sim_sum = (sim_matrix * denom_mask).sum(dim=1)
        
        loss = -torch.log(self_pos / (neg_sim_sum + 1e-8) + 1e-8).mean()

        return loss, en_stats
        # return loss, prec, rec, f1_score

  def loss_cal(self, x, x_aug, labels, get_cm=False, epoch=None):

    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    self_pos = sim_matrix[range(batch_size), range(batch_size)]

    labels = labels.view(-1, 1)
    pos_mask = labels.eq(labels.T)

    # cal negative cheated (take away the real positive)
    pos_sim = sim_matrix * pos_mask
    pos_sim_sum = pos_sim.sum(dim=1)
    neg_sim_sum = sim_matrix.sum(dim=1) - pos_sim_sum

    loss = self_pos / neg_sim_sum
    loss = -torch.log(loss + 1e-8).mean()  # 避免 log(0)
    pos_sim_ = self_pos.mean()
    neg_sim_ = neg_sim_sum.mean()

    if get_cm:
       self.get_confusion_matrix(labels=labels, sim_matrix=sim_matrix, args=args, epoch=epoch)

    return loss, pos_sim_, neg_sim_

  def loss_cal_normal(self, x, x_aug, labels=None, get_cm=False, epoch=None):

    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    cos_sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(cos_sim_matrix / T)
    self_pos = sim_matrix[range(batch_size), range(batch_size)]

    neg_sim = (sim_matrix.sum(dim=1) - self_pos)
    loss = self_pos / neg_sim
    loss = - torch.log(loss + 1e-8).mean()
    pos_sim_ = self_pos.mean()
    neg_sim_ = neg_sim.mean()

    if get_cm:
       self.get_confusion_matrix(labels=labels, sim_matrix=cos_sim_matrix, args=args, epoch=epoch)

    return loss, pos_sim_, neg_sim_


  def loss_cal_rm_FN_only(self, x, x_aug, labels=None, get_cm=False, epoch=None):

    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    cos_sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(cos_sim_matrix / T)
    self_pos = sim_matrix[range(batch_size), range(batch_size)]

    # modified: self -> positive; negative:take away all cheated
    labels = labels.view(-1, 1)
    pos_mask = labels.eq(labels.T)

    # cal negative cheated (take away the real positive)
    pos_sim = sim_matrix * pos_mask
    pos_sim_sum = pos_sim.sum(dim=1)
    neg_sim_sum = sim_matrix.sum(dim=1) - pos_sim_sum

    loss = self_pos / neg_sim_sum
    loss = -torch.log(loss + 1e-8).mean()  # avoid log(0)
    pos_sim_ = self_pos.mean()
    neg_sim_ = neg_sim_sum.mean()

    if get_cm:
       self.get_confusion_matrix(labels=labels, sim_matrix=cos_sim_matrix, args=args, epoch=epoch)

    return loss, pos_sim_, neg_sim_
  

  def loss_cal_cheated(self, x, x_aug, labels=None, get_cm=False, epoch=None, similarity_measure="cosine"):

    T = 0.2
    batch_size, _ = x.size()
    sim_matrix = self._get_similarity_matrix(x=x, x_aug=x_aug, similarity_measure=similarity_measure, T=T)
    self_pos = sim_matrix[range(batch_size), range(batch_size)]

    # modified:all pos and all neg cal
    labels = labels.view(-1, 1)  # 轉換為 (batch_size, 1) 方便比較
    pos_mask = labels.eq(labels.T)  # 創建相同類別的對應矩陣

    # 選擇 positive pairs（不一定是對角線）
    pos_sim = sim_matrix * pos_mask  # 只保留相同類別的相似度值
    # pos_sim_sum = pos_sim.sum(dim=1)
    pos_sim_sum = pos_sim.sum(dim=1) - torch.diag(pos_sim)  # 排除自己本身

    neg_sim_sum = sim_matrix.sum(dim=1) - pos_sim_sum - torch.diag(pos_sim)  # 所有樣本總和 - 正樣本總和

    loss = pos_sim_sum / neg_sim_sum
    loss = -torch.log(loss + 1e-8).mean()  # 避免 log(0)
    pos_sim_ = pos_sim_sum.mean() # not self pos
    # pos_sim_ = self_pos.mean() # self pos
    neg_sim_ = neg_sim_sum.mean()

    if get_cm:
       self.get_confusion_matrix(labels=labels, sim_matrix=cos_sim_matrix, args=args, epoch=epoch)

    return loss, pos_sim_, neg_sim_

  def loss_cal_rm_FP_only(self, x, x_aug, labels=None, get_cm=False, epoch=None):

    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    cos_sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(cos_sim_matrix / T)
    self_pos = sim_matrix[range(batch_size), range(batch_size)]

    # modified:all pos and all neg cal
    labels = labels.view(-1, 1)  # 轉換為 (batch_size, 1) 方便比較
    pos_mask = labels.eq(labels.T)  # 創建相同類別的對應矩陣

    # 選擇 positive pairs（不一定是對角線）
    pos_sim = sim_matrix * pos_mask  # 只保留相同類別的相似度值
    # pos_sim_sum = pos_sim.sum(dim=1)
    pos_sim_sum = pos_sim.sum(dim=1) - torch.diag(pos_sim)  # 排除自己本身

    neg_sim_sum = sim_matrix.sum(dim=1) - self_pos  # 所有樣本總和 - 正樣本總和

    loss = pos_sim_sum / neg_sim_sum
    loss = -torch.log(loss + 1e-8).mean()  # 避免 log(0)
    pos_sim_ = pos_sim_sum.mean() # not self pos
    # pos_sim_ = self_pos.mean() # self pos
    neg_sim_ = neg_sim_sum.mean()

    if get_cm:
       self.get_confusion_matrix(labels=labels, sim_matrix=cos_sim_matrix, args=args, epoch=epoch)

    return loss, pos_sim_, neg_sim_
  
#   def loss_cal_reweighted(self, x, x_aug, labels=None, get_cm=False, epoch=None):
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

  def reweighted_l2_loss(self, z_a, z_b, T=0.2, eps=1e-6, neg_aug=False): # Note: dist=True is now enforced for L2
    
    # z_a, z_b 是圖嵌入 (B, D)

    # --- 1. L2 Distance Squared ---
    # D_ab: z_a[i] vs z_b[j], D_aa: z_a[i] vs z_a[j]
    D_ab_sq = torch.cdist(z_a, z_b, p=2).pow(2)  # (B, B)
    if neg_aug:
        D_aa_sq = D_ab_sq.clone()            # (B, B)
        # D_ab_sq_full = torch.cdist(z_a, z_b, p=2).pow(2)
    else:
        D_aa_sq = torch.cdist(z_a, z_a, p=2).pow(2)  # (B, B)
    
    # pos_dist_sq: 正樣本對距離的平方 (B, 1)
    pos_dist_sq = D_ab_sq.diagonal().unsqueeze(1) 

    # --- 2. L2 Similarity (Sim_{L2} = exp(-D^2 / T)) ---
    S_ab = torch.exp(-D_ab_sq / T) 
    S_aa = torch.exp(-D_aa_sq / T) 
    pos_sim = S_ab.diagonal() # S_i,i+ term (B,)
    
    # --- 3. False Negative (FN) / Reweighting Logic ---
    
    # FN 條件: 負樣本 z_a[j] 比正樣本 z_b[i] 更接近錨點 z_a[i] (距離小於)
    # 由於我們只使用 L2 距離，且已將 dist 邏輯納入，因此簡化 FN 判斷
    FN_matrix = D_aa_sq <= pos_dist_sq             # (B,B)
    
    # 將 FN 矩陣作為 false_mask (忽略 dist 和 neg_aug 複雜邏輯)
    false_mask = FN_matrix 
    false_mask.fill_diagonal_(False)               # 排除 i==i
    
    # reweighting
    W = torch.full_like(S_aa, self.w_easy)         # Assume all easy
    W[false_mask] = 0.0                            # reweight false negatives (W=0)
    W.fill_diagonal_(0.0)                          # 確保 i==i 權重為 0

    # === 4. Pos Pairs Loss Term ===
    # log(S_i,i+)
    log_pos = torch.log(pos_sim + eps)             # (B,)

    # === 5. Neg Pairs Loss Term (logsumexp) ===
    # logit_neg = log(W * S_aa) = log(W) + log(S_aa) = log(W) - D_aa_sq / T
    log_W = torch.log(W + eps)
    log_S_aa = -D_aa_sq / T
    logit_neg = log_W + log_S_aa                   # (B,B)
    
    # log(sum_{j != i} exp(logit_neg))
    # 由於 W[i,i] = 0，logsumexp 會自動排除對角線項
    log_denom = torch.logsumexp(logit_neg, dim=1)  # (B,)

    # === 6. Final Loss ===
    # InfoNCE loss: - log(pos / denom) = - (log(pos) - log(denom))
    pos_sim_ = pos_sim.mean()
    neg_sim_ = (S_aa * W).sum(dim=1).mean()

    return -(log_pos - log_denom).mean(), FN_matrix, pos_sim_, neg_sim_
  
  def loss_cal_custom_sampling(self, x, x_aug, labels=None, get_cm=False, epoch=None, mode='sparse_neg'):
    """
    實現兩種客製化採樣模式的 InfoNCE 損失。
    
    Args:
        mode ('sparse_neg' or 'random_pos'):
            'sparse_neg': 分子為真正樣本，分母為兩個隨機真負樣本。
            'random_pos': 分子為隨機一半樣本，分母為剩餘一半樣本。
    """
    T = 0.2
    batch_size, _ = x.size()
    
    # --- 相似度計算 ---
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    cos_sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(cos_sim_matrix / T)
    
    labels = labels.view(-1, 1)
    
    if mode == 'sparse_neg':
        # ==== 選擇 A: 稀疏負樣本 (模擬極端 InfoNCE) ====
        
        # 1. 分子 (Positive Sim): 隨機選擇一個同標籤但非自身的樣本
        pos_indices = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        for i in range(batch_size):
            # 找出所有真正的正樣本 (同標籤且非自身)
            true_pos_indices = (labels.eq(labels[i]) & ~torch.diag(torch.ones(batch_size, device=x.device, dtype=torch.bool))).nonzero(as_tuple=True)[0]
            
            if true_pos_indices.numel() > 0:
                # 隨機挑選一個真正的正樣本
                rand_idx = torch.randint(0, true_pos_indices.numel(), (1,), device=x.device)
                pos_indices[i] = true_pos_indices[rand_idx]
            else:
                # 如果 batch 內沒有其他同類樣本，則使用自身 (這是回退，不理想)
                pos_indices[i] = i 

        fp_sim = sim_matrix[range(batch_size), pos_indices] # (B,)        
        
        # 2. 分母 (Negative Sim): 隨機採樣兩個真正的負樣本 (不同標籤)
        neg_mask = self._sample_negative_mask(labels, batch_size, x.device, num_negatives=2)
        
        # 計算分母總和
        neg_sim_sum = (sim_matrix * neg_mask).sum(dim=1) # (B,)
        
    elif mode == 'random_pos':
        # ==== 選擇 B: 隨機分子 (隨機選擇一半作為 Positive Pairs) ====
        
        # 1. 隨機分數矩陣
        # 創建一個與 sim_matrix 大小相同的隨機分數，用於獨立排序
        # 這張分數矩陣的每一行都會被獨立地用來決定哪些是 Positives
        random_scores = torch.rand((batch_size, batch_size), device=x.device)
        
        # 2. 計算分割點 (約 N/2)
        split_count = batch_size // 2
        
        # 3. 獲取 Positives Set (分子) 的掩碼
        # 對每一行獨立地找出分數最高的 split_count 個索引
        # argsort() 進行排序，[-split_count:] 獲取最大的 N/2 個索引
        
        # 獲取排序後的索引 (從最小到最大)
        sorted_indices = torch.argsort(random_scores, dim=1)
        
        # 找出作為 Positives 的 N/2 個索引 (分數最大的 N/2 個)
        pos_indices_for_each_row = sorted_indices[:, -split_count:] # (B, split_count)
        
        # 構建 Positives Mask (pos_mask)
        pos_mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=x.device)
        
        # 使用 scatter_ 設置掩碼：對於每一行 i，將 pos_indices_for_each_row[i] 對應的列設置為 True
        # 這一步實現了：pos_mask[i, j] = True 如果 j 被 Anchor i 隨機選為 Positives
        pos_mask.scatter_(dim=1, index=pos_indices_for_each_row, src=torch.ones_like(pos_indices_for_each_row, dtype=torch.bool))

        # 4. 構建 Negatives Set (分母) 的掩碼
        # Negatives Set 就是 Positives Set 之外的所有樣本
        # neg_mask = ~pos_mask
        
        # 5. 清理對角線 (防止 Self-Similarity 被錯誤地計入分子或分母)
        # 通常 InfoNCE 的分子是 Positives Set，分母是 Positives Set 之外的所有樣本
        # 由於您明確要求分子是 Positives Set 且分母是 Negatives Set，我們需要清理對角線，確保 i 和 i+ 是互斥的。
        
        # 清理對角線 (i vs i)：InfoNCE 的分子/分母通常都不包含 i vs i
        diag_mask = torch.diag(torch.ones(batch_size, device=x.device, dtype=torch.bool))
        
        # 確保對角線在分子和分母中都為 False
        pos_mask = pos_mask & ~diag_mask 
        # neg_mask = neg_mask & ~diag_mask 

        # 6. 計算分子和分母總和
        
        # 分子 (Positive Sim Sum): Anchor i 對其 Positives Set 的總和
        fp_sim = (sim_matrix * pos_mask).sum(dim=1) # (B,)
        # random_indices = torch.randint(0, batch_size, (batch_size,), device=x.device)
        # fp_sim = sim_matrix[range(batch_size), random_indices] # (B,)
        
        # 分母 (Negative Sim): 批次中剩餘的所有樣本 (Total - Random Pos)
        # 由於 random_indices 可能是重複的，我們使用掩碼來保證正確的總和
        
        # 構建分母掩碼 (排除 Random Pos)
        neg_mask = torch.ones((batch_size, batch_size), dtype=torch.bool, device=x.device)
        # neg_mask[range(batch_size), random_indices] = False # 排除分子項
        
        neg_sim_sum = (sim_matrix * neg_mask).sum(dim=1) # (B,)
        
    else:
        raise ValueError("Invalid mode. Must be 'sparse_neg' or 'random_pos'.")

    # === 3. 損失計算 ===
    loss = fp_sim / (neg_sim_sum + 1e-8)
    loss = -torch.log(loss + 1e-8).mean()
    
    pos_sim_ = fp_sim.mean()
    neg_sim_ = neg_sim_sum.mean()

    return loss, pos_sim_, neg_sim_

import random
def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


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
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = simclr(args.hidden_dim, args.num_gc_layers, shuffle_DBN=args.shuffle_DBN, dataset_num_features=dataset_num_features).to(device)
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
        total_fp = 0
        total_fn_missed = 0

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
            x, _ = model(data.x, data.edge_index, data.batch, data.num_graphs)

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

            x_aug, _ = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)
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
            # Assuming model.loss_cal_reweighted and model.reweighted_l2_loss are defined for reweighted modes.
            if args.mode == 'normal':
                # Standard InfoNCE (Self-Pos / All Negs)
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

            elif args.mode == 'reweighted_l2':
                loss, _, pos_sim, neg_sim = model.reweighted_l2_loss(x, x_aug)
            elif args.mode == 'reweighted_by_angle':
                loss, pos_sim, neg_sim = model.reweighted_by_angle(x, x_aug, deg_boundary=args.rotate_angle_deg)
            elif args.mode == 'rm_FNs_by_ENs':
                loss, en_stats = model.loss_cal_rm_FNs_by_ENs(x, x_aug, labels, epoch, epochs, base_en_ratio=0.3, max_en_ratio=0.6, overlap_threshold=0.5, neg_include_self=args.neg_include_self)
                # loss, fn_by_en_prec, fn_by_en_recall, fn_by_en_f1 = model.loss_cal_rm_FNs_by_ENs(x, x_aug, labels, top_k_en=max(5, int(batch_size * 0.2)), overlap_threshold=0.5, neg_include_self=args.neg_include_self)
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
            if not args.mode == 'rm_FNs_by_ENs':
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
        if not args.mode == 'rm_FNs_by_ENs':
            writer.add_scalar('Similarity/pos_sim', pos_sim_all / len(dataloader), epoch)
            writer.add_scalar('Similarity/neg_sim', neg_sim_all / len(dataloader), epoch)
        else:
            writer.add_scalar('EN_FN_Dynamics/Removal_Recall', en_stats['fn_recall'], epoch)
            writer.add_scalar('EN_FN_Dynamics/Wrong_Rate_TN_Killed', en_stats['fn_wrong_rate'], epoch)
            writer.add_scalar('EN_FN_Dynamics/Removal_Precision', en_stats['fn_precision'], epoch)
            writer.add_scalar('EN_FN_Dynamics/Removal_F1_Score', en_stats['fn_f1'], epoch)
            writer.add_scalar('EN_FN_Dynamics/Current_EN_Ratio', en_stats['curr_en_ratio'], epoch)
            writer.add_scalars('EN_FN_Dynamics/Purity_Check', {
                'Cleaned_FN_Ratio': en_stats['remaining_fn_ratio'],
                'Original_FN_Ratio': en_stats['original_fn_ratio']
            }, epoch)
            # writer.add_scalar('EN_Inference/FN_Precision', prec, epoch)
            # writer.add_scalar('EN_Inference/FN_Recall', recall, epoch)
            # writer.add_scalar('EN_Inference/FN_F1', f1, epoch)

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
            if args.do_hn_analysis: # 增加一個旗標控制是否執行
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
                # plot_kde_unitcircle_kde(
                #     x.detach().cpu().numpy(),                 # (N, D) 的 embeddings
                #     data.y.detach().cpu().numpy(),                 # (N,) 的標籤（int 或可 hash）
                #     classes=None,      # 要畫哪些 class，None=全部
                #     save_path=f'KDE/{args.DS}/{args.mode}/{args.aug}/anchor/epoch_{epoch}_kde_unit_circle',
                # )
                # plot_kde_unitcircle_kde(
                #     x_aug.detach().cpu().numpy(),                 # (N, D) 的 embeddings
                #     data.y.detach().cpu().numpy(),                 # (N,) 的標籤（int 或可 hash）
                #     classes=None,      # 要畫哪些 class，None=全部
                #     save_path=f'KDE/{args.DS}/{args.mode}/{args.aug}/graph_pos/epoch_{epoch}_kde_unit_circle',
                # )

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
    
