def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

# from torch_geometric.datasets import TUDataset
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import sys
import json
from torch import optim

from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *

from arguments import arg_parse
from check_dim_collapse import check_dimensional_collapse
from ortho_loss import l2_reg_ortho
from torch.utils.tensorboard import SummaryWriter
import time
from utils_plot.plot_parallel_and_radar_online import plot_advanced_diagnosis
from utils_plot.plot_3D_online_scatter import plot_3d_interactive_diagnosis
from make_save_dir import make_save_dir # 引入創建儲存目錄的函數
from save_load_ckpts import load_checkpoint, save_checkpoint # 引入檢查點函數
from unified_loss import unified_loss, get_pair_angles, flexible_hard_mining_loss
from utils import get_soft_metrics, compute_pairwise_differences
from compute_structural_info import compute_adamic_adar_sim, compute_structural_consensus_metrics, compute_geometry_jaccard_sim
import random

def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

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
        
        return stretched_coverage, stats
  
  def loss_cal_structural_PPR_infonce(self, x, x_aug, labels, T=0.2, alpha_ppr=0.8, gamma=3.0):
        batch_size = x.size(0)
        device = x.device
        diag_mask = torch.eye(batch_size, device=device).bool()

        # --- A. 核心邏輯 (PPR -> Target -> Suppression) ---
        cos_sim = torch.mm(F.normalize(x, dim=1), F.normalize(x_aug, dim=1).T)
        # jaccard_mtx = compute_geometry_jaccard_sim(data)
        # jaccard_mtx = compute_adamic_adar_sim(jaccard_mtx)
        
        # PPR 擴散 (作為 Target)
        W = cos_sim.clone()

        # W[W < 0.2] = 0.0 
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
        jaccard_mtx = compute_geometry_jaccard_sim(data)
        jaccard_mtx = compute_adamic_adar_sim(jaccard_mtx)
        
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
  
  def loss_cal_structural_PPR_infonce_oracle(self, x, x_aug, labels, data, T=0.2, alpha_ppr=0.8, gamma=5.0, high_sim_threshold=0.8):
        batch_size = x.size(0)
        device = x.device
        diag_mask = torch.eye(batch_size, device=device).bool()

        cos_sim = torch.mm(F.normalize(x, dim=1), F.normalize(x_aug, dim=1).T)
        jaccard_mtx = compute_geometry_jaccard_sim(data)
        
        W = jaccard_mtx.clone()
        W[W < 0.2] = 0.0 
        D_inv = torch.diag(1.0 / (W.sum(dim=1) + 1e-8))
        P = torch.mm(D_inv, W)
        Target_ppr = torch.eye(batch_size, device=device)
        for _ in range(3):
            Target_ppr = alpha_ppr * torch.eye(batch_size, device=device) + (1 - alpha_ppr) * torch.mm(P, Target_ppr)
        
        Target_ppr = (Target_ppr - Target_ppr.min()) / (Target_ppr.max() - Target_ppr.min() + 1e-8)

        # B. 【核心修改 1】利用 Label 定義「作弊版」FN Confidence
        labels_col = labels.view(-1, 1)
        gt_mask = labels_col.eq(labels_col.T).float().masked_fill(diag_mask, 0.0)
        
        # 強行拉開 Cosine 動態範圍
        cos_sim_norm = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-8)
        
        # 定義高壓區 (原本會被重推的地方)
        high_sim_mask = (cos_sim > high_sim_threshold).float().masked_fill(diag_mask, 0.0)
        
        # 關鍵 FN：真的是同類 (gt) 且 模型覺得很像 (high_sim)
        critical_fn_mask = gt_mask * high_sim_mask
        
        # 我們將 PPR 的預測與 GT 結合：
        # 如果是 Critical FN，信心值強制設為 Target_ppr 與 gt 的結合 (這裡可以根據妳想看 labels 的程度調整)
        # 這裡我們用 GT 導向：只要是 GT_Pos 且 Cos 高，我就認定它是 FN
        # fn_confidence = critical_fn_mask * Target_ppr 
        predicted_fn_confidence = Target_ppr * cos_sim_norm
        predicted_suppression = torch.pow(1.0 - predicted_fn_confidence, gamma)
        predicted_suppression = torch.clamp(predicted_suppression, min=0.01)
        fn_confidence = critical_fn_mask
        
        # C. 計算 Suppression (煞車權重)
        # 只有在 fn_confidence 高的地方，suppression 才會變小 (趨近 0)
        suppression = torch.pow(1.0 - fn_confidence, gamma)
        suppression = torch.clamp(suppression, min=0.01)

        # D. 【核心修改 2】重寫指標邏輯：局部救援評估
        # 我們只關心：在那群「原本會被推開的同類 (critical_fn)」中，我們救了多少？
        
        def get_rescue_metrics(supp_mtx, crit_mask, hn_mask):
            # 救援成功：權重被壓低了 (例如 < 0.5)
            is_rescued = (supp_mtx < 0.5).float()
            
            # FN 救援率 (Survival Rate)
            num_crit = crit_mask.sum() + 1e-8
            rescue_rate = (is_rescued * crit_mask).sum() / num_crit
            
            # HN 誤救率 (Leakage Rate)：本來該推開的異類，卻被妳當成好人救了
            num_hn = hn_mask.sum() + 1e-8
            leakage_rate = (is_rescued * hn_mask).sum() / num_hn
            
            # 平均推力對比
            avg_fn_push = (supp_mtx * crit_mask).sum() / num_crit
            avg_hn_push = (supp_mtx * hn_mask).sum() / num_hn
            
            return rescue_rate.item(), leakage_rate.item(), avg_fn_push.item(), avg_hn_push.item()

        # 這裡的 HN 定義為：Cosine 很高但標籤不同
        true_hn_mask = (1.0 - gt_mask) * high_sim_mask
        
        r_rate, l_rate, fn_push, hn_push = get_rescue_metrics(predicted_suppression, critical_fn_mask, true_hn_mask)

        # E. InfoNCE Loss
        exp_sim = torch.exp(cos_sim / T)
        weighted_neg_sim = exp_sim * suppression
        neg_denom = weighted_neg_sim.masked_fill(diag_mask, 0.0).sum(dim=1)
        loss = -torch.log(exp_sim.diag() / (neg_denom + 1e-8) + 1e-8).mean()

        stats = {
            'FN_Rescue_Rate': r_rate,      # 越高越好 (救到好人)
            'HN_Leakage_Rate': l_rate,     # 越低越好 (沒救錯壞人)
            'Avg_Supp_FN': fn_push,        # 越低越好 (煞車踩多死)
            'Avg_Supp_HN': hn_push,        # 應該接近 1 (壞人要照樣推)
            'Target_F1': get_soft_metrics(Target_ppr, gt_mask)[2] # 保留原本的 Target 品質參考
        }
        
        return loss, Target_ppr, stats


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
    global_jaccard_matrix = compute_geometry_jaccard_sim(raw_data)  

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

    for epoch in range(0, epochs + 1):
        if args.mode == 'focal_infonce' or args.mode == 'PPR_infonce':
            epoch_metrics = {k: 0.0 for k in ['t_p', 't_r', 't_f1', 's_p', 's_r', 's_f1']}
        if args.mode == 'PPR_infonce_oracle':
            epoch_metrics = {k: 0.0 for k in ['FN_Rescue_Rate', 'HN_Leakage_Rate', 'Avg_Supp_FN', 'Avg_Supp_HN', 'Target_F1']}

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
                
                struct_metrics = compute_structural_consensus_metrics(sub_batch)
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

                plot_advanced_diagnosis(sub_labels, base_metrics, epoch, f"{diag_dir}/base_metrics")
                plot_advanced_diagnosis(sub_labels, diff_metrics, epoch, f"{diag_dir}/diff_metrics")
                plot_advanced_diagnosis(sub_labels, plot_metrics, epoch, f"{diag_dir}/all_metrics")
                plot_3d_interactive_diagnosis(sub_labels, plot_metrics, epoch, diag_dir)

        loss_all = 0
        model.train()

        for batch_idx, data in enumerate(dataloader):
            first_batch = (batch_idx == 0)
            last_batch = (batch_idx == len(dataloader) - 1)

            data, data_aug = data
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
                
                jaccard_sim = compute_geometry_jaccard_sim(data.to(device))
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
            elif args.mode == 'PPR_infonce_oracle':
                loss, _, stats = model.loss_cal_structural_PPR_infonce_oracle(x, x_aug, labels, data)
                for k in epoch_metrics:
                    epoch_metrics[k] += stats[k]

            else:
                # Handles all other unmatched modes
                raise RuntimeError(f"no mode matching {args.mode}, input should be: normal, TPs_TNs, etc.")

            oloss = odecay * l2_reg_ortho(model)
            loss_all += loss.item() * data.num_graphs
            if args.or_loss:
                loss += oloss
            loss.backward()
            optimizer.step()

        # tensorboard
        writer.add_scalar('Loss/train', loss_all / len(dataloader.dataset), epoch)
        if args.mode == 'focal_infonce' or args.mode == 'PPR_infonce' or args.mode == 'PPR_infonce_oracle':
            num_batches = len(dataloader)
            for k in epoch_metrics:
                writer.add_scalar(f'Metric_Group/{k}', epoch_metrics[k] / num_batches, epoch)

        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader.dataset)))
        loss_list.append(loss_all / len(dataloader.dataset))

        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            # visualize_embeddings(emb, y, args, epoch, method="t-SNE")
            acc_val, acc = evaluate_embedding(emb, y)
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

    with open((f'{save_dir}/{aug_ratio}_'+str(args.seed)), 'a+') as f:
        s1 = json.dumps(stage_finish_epochs)
        s2 = json.dumps(loss_list)
        s3 = json.dumps(accuracies)
        f.write('{},{},{},{},{},{},{},{}\n'.format(args.DS, args.num_gc_layers, epochs, log_interval, lr, s1, s2, s3))
    
