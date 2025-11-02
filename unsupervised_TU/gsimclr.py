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

class GcnInfomax(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(GcnInfomax, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)
    # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
    # self.global_d = MIFCNet(self.embedding_dim, mi_units)

    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)

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
    
    g_enc = self.global_d(y)
    l_enc = self.local_d(M)

    mode='fd'
    measure='JSD'
    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)
 
    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
    
    return local_global_loss + PRIOR


class simclr(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(simclr, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))

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
    
    return y

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
  
  def create_pos_and_neg_mask(self, labels):

    labels = labels.view(-1, 1)
    pos_mask = labels.eq(labels.T)
    neg_mask = ~pos_mask
    # remove self-positive pairs
    pos_mask = pos_mask & ~torch.diag(torch.ones(labels.size(0), device=labels.device, dtype=torch.bool))

    return pos_mask, neg_mask

  def get_confusion_matrix(self, labels, sim_matrix, args, epoch=None):
        pos_mask, neg_mask = self.create_pos_and_neg_mask(labels=labels)
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
        pos_mask, neg_mask = self.create_pos_and_neg_mask(labels=labels)

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

        return -(log_pos - log_denom).mean(), FN_matrix

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

    # tensorboard
    writer = SummaryWriter(log_dir=f'logs/KDE/200epochs_log_interval_10/{args.DS}/lr_{args.lr}/tensorboard_{args.similarity_measure}_{args.aug}_{args.mode}_{time.ctime(time.time())}_{args.or_loss}')

    accuracies = {'val':[], 'test':[]}
    epochs = args.epochs
    log_interval = args.log_interval
    batch_size = args.batch_size
    aug_ratio = round(args.aug_ratio * 0.1, 1)
    loss_list = []
    loss_min = float('inf')
    stage_finish_epochs = []
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
    model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

    for epoch in range(0, epochs+1):
        loss_all = 0
        pos_sim_all = 0
        neg_sim_all = 0
        model.train()

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
            x = model(data.x, data.edge_index, data.batch, data.num_graphs)

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

            '''
            print(data.edge_index)
            print(data.edge_index.size())
            print(data_aug.edge_index)
            print(data_aug.edge_index.size())
            print(data.x.size())
            print(data_aug.x.size())
            print(data.batch.size())
            print(data_aug.batch.size())
            pdb.set_trace()
            '''

            x_aug = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)
            if args.mode == 'normal':
                # loss, pos_sim, neg_sim = model.loss_cal_normal(x, x_aug, labels=labels, get_cm=True if epoch % log_interval == 0 else False, epoch=epoch)
                loss, pos_sim, neg_sim = model.loss_cal_normal(x, x_aug, labels=labels, get_cm=False, epoch=epoch)
            elif args.mode == 'cheated':
                # loss, pos_sim, neg_sim = model.loss_cal_cheated(x, x_aug, labels, get_cm=True if epoch % log_interval == 0 else False, epoch=epoch)
                loss, pos_sim, neg_sim = model.loss_cal_cheated(x, x_aug, labels, get_cm=False, epoch=epoch, similarity_measure=args.similarity_measure)
            elif args.mode == 'rm_FN':
                # loss, pos_sim, neg_sim = model.loss_cal_rm_FN_only(x, x_aug, labels, get_cm=True if epoch % log_interval == 0 else False, epoch=epoch)
                loss, pos_sim, neg_sim = model.loss_cal_rm_FN_only(x, x_aug, labels, get_cm=False, epoch=epoch)
            elif args.mode == 'rm_FP':
                # loss, pos_sim, neg_sim = model.loss_cal_rm_FP_only(x, x_aug, labels, get_cm=True if epoch % log_interval == 0 else False, epoch=epoch)
                loss, pos_sim, neg_sim = model.loss_cal_rm_FP_only(x, x_aug, labels, get_cm=False, epoch=epoch)
            elif args.mode == 'reweighted':
                # loss, pos_sim, neg_sim = model.loss_cal_rm_FP_only(x, x_aug, labels, get_cm=True if epoch % log_interval == 0 else False, epoch=epoch)
                loss, _ = model.loss_cal_reweighted(x, x_aug)
                pos_sim, neg_sim = 0, 0
            else:
               raise RuntimeError(f"no mode matching {args.mode}, input should be: normal, cheated, rm_FN")
            
            # scatter plot for theta and l2 norm
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
            pos_sim_all += pos_sim.item() if args.mode != 'reweighted' else 0
            neg_sim_all += neg_sim.item() if args.mode != 'reweighted' else 0
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
        writer.add_scalar('Similarity/pos_sim', pos_sim_all / len(dataloader), epoch)
        writer.add_scalar('Similarity/neg_sim', neg_sim_all / len(dataloader), epoch)

        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader.dataset)))
        # print("pos sim = ", pos_sim_all, "; neg sim = ", neg_sim_all)
        loss_list.append(loss_all / len(dataloader.dataset))
        if epoch % log_interval == 0:
            if args.plot_kde:
                os.makedirs(f'KDE/{args.DS}/{args.mode}/{args.aug}/anchor', exist_ok=True)
                os.makedirs(f'KDE/{args.DS}/{args.mode}/{args.aug}/graph_pos', exist_ok=True)
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
                    save_path=f'KDE/{args.DS}/{args.mode}/{args.aug}/anchor/epoch_{epoch}_kde_unit_circle',
                    plot_scatter=True,
                )
                plot_kde_unitcircle_kde(
                    X_pos, y_pos,
                    classes=None,
                    save_path=f'KDE/{args.DS}/{args.mode}/{args.aug}/graph_pos/epoch_{epoch}_kde_unit_circle',
                    plot_scatter=True,
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
                                             all_real_pos_theta, all_real_neg_l2, all_real_neg_theta, args=args, epoch=epoch)
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            # visualize_embeddings(emb, y, args, epoch, method="t-SNE")
            acc_val, acc = evaluate_embedding(emb, y)
            # singular_values = check_dimensional_collapse(emb)
            # for i, value in enumerate(singular_values):
            #     writer.add_scalar(f'Singular_Values/{epoch}_{args.DS}', np.log10(value), i)

            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)
            # tensorboard
            writer.add_scalar('Accuracy/val', acc_val, epoch)
            writer.add_scalar('Accuracy/test', acc, epoch)

    tpe  = ('local' if args.local else '') + ('prior' if args.prior else '')
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./logs/GCL"):
        os.makedirs("./logs/GCL")
    if not os.path.exists(f"./logs/GCL/{args.DS}"):
        os.makedirs(f"./logs/GCL/{args.DS}")

    with open((f'./logs/GCL/{args.DS}/{args.DS}_{aug_ratio}_'+str(args.seed)), 'a+') as f:
        s1 = json.dumps(stage_finish_epochs)
        s2 = json.dumps(loss_list)
        s3 = json.dumps(accuracies)
        f.write('{},{},{},{},{},{},{},{}\n'.format(args.DS, args.num_gc_layers, epochs, log_interval, lr, s1, s2, s3))
    
