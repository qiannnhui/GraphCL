import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def get_angle_and_l2_before_norm(x, x_aug):
    cos_sim_matrix = torch.matmul(x, x_aug.T) / (torch.norm(x, dim=1).view(-1, 1) * torch.norm(x_aug, dim=1))  # 計算 Cosine Similarity
    theta_matrix = torch.acos(torch.clamp(cos_sim_matrix, -1.0 + 1e-7, 1.0 - 1e-7))  # 計算角度
    l2_matrix = torch.cdist(x, x_aug, p=2)
    return theta_matrix, l2_matrix

def get_real_pos_and_neg_mask(labels):
    '''Real Positives and Negatives'''

    labels = labels.view(-1, 1)
    pos_mask = labels.eq(labels.T)
    neg_mask = ~pos_mask
    # remove self-positive pairs
    # pos_mask = pos_mask & ~torch.diag(torch.ones(labels.size(0), device=labels.device, dtype=torch.bool))

    return pos_mask, neg_mask


def get_masks_and_theta_l2(x, x_aug, labels=None):
        """
        Calculate theta and l2 norm data for plotting
        """
        # ==== get normalized cosine similarity matrix ====
        x_norm = x / x.norm(dim=1, keepdim=True)
        x_aug_norm = x_aug / x_aug.norm(dim=1, keepdim=True)
        cos_sim_matrix = torch.einsum('ik,jk->ij', x_norm, x_aug_norm)  # (B, B)

        # ==== theta ====
        # theta_matrix = torch.acos(torch.clamp(cos_sim_matrix, -1.0 + 1e-7, 1.0 - 1e-7))
        # theta_matrix_deg = theta_matrix * 180.0 / torch.pi  # to degree

        # # ==== l2 norm matrix ====
        # x_sq = (x_norm ** 2).sum(dim=1, keepdim=True)  # (B, 1)
        # x_aug_sq = (x_aug_norm ** 2).sum(dim=1, keepdim=True).T  # (1, B)
        # l2_matrix = torch.sqrt(x_sq + x_aug_sq - 2 * torch.einsum('ik,jk->ij', x_norm, x_aug_norm) + 1e-8)

        # get theta and l2 before normalized
        theta_matrix, l2_matrix = get_angle_and_l2_before_norm(x, x_aug)
        theta_matrix_deg = theta_matrix * 180.0 / torch.pi  # to degree

        # ==== create GCL pos and neg mask ====
        pos_mask = torch.diag(torch.ones(x.size(0), device=x.device, dtype=torch.bool))  # self-positive pairs, diagonal
        neg_mask = torch.triu(torch.ones_like(cos_sim_matrix), diagonal=1).bool()  # Exclude diagonal, upper triangular part

        # ==== getting data ====
        # pos_theta = theta_matrix_deg[pos_mask].detach().cpu().numpy()
        # pos_l2 = l2_matrix[pos_mask].detach().cpu().numpy()

        # neg_theta = theta_matrix_deg[neg_mask].detach().cpu().numpy()
        # neg_l2 = l2_matrix[neg_mask].detach().cpu().numpy()

        if labels is not None:
            # real positive and negative pairs by labels
            real_pos_mask, real_neg_mask = get_real_pos_and_neg_mask(labels)
            upper_tri_mask = torch.triu(torch.ones_like(cos_sim_matrix), diagonal=0).bool() # get upper triangular part
            real_pos_mask = real_pos_mask & upper_tri_mask
            real_neg_mask = real_neg_mask & upper_tri_mask
            
            # real_pos_theta = theta_matrix_deg[real_pos_mask].detach().cpu().numpy()
            # real_pos_l2 = l2_matrix[real_pos_mask].detach().cpu().numpy()
            # real_neg_theta = theta_matrix_deg[real_neg_mask].detach().cpu().numpy()
            # real_neg_l2 = l2_matrix[real_neg_mask].detach().cpu().numpy()

            return pos_mask, neg_mask, theta_matrix_deg, l2_matrix, real_pos_mask, real_neg_mask
        
        return pos_mask, neg_mask, theta_matrix_deg, l2_matrix

def plot_theta_l2_distribution(x, x_aug, labels=None, args=None, epoch=None):
    """
    Plot theta and l2 norm distribution for each anchor, with FN comparison.
    """
    pos_mask, neg_mask, theta_matrix_deg, l2_matrix, real_pos_mask, real_neg_mask = get_masks_and_theta_l2(x, x_aug, labels)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()  # Flatten the 2D array of axes for easy iteration

    # Iterate over each row to treat it as an anchor graph
    for anchor_id in range(x.size(0)):
        # For each anchor, find corresponding theta and l2 values for real positives/negatives, GCL positives/negatives

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))  # 一行五張子圖
        fig.suptitle(f'Theta vs L2 for Anchor {anchor_id}', fontsize=16)

        # 固定只取 anchor_id 的資料
        anchor_theta = theta_matrix_deg[anchor_id, :]
        anchor_l2 = l2_matrix[anchor_id, :]

        real_pos_theta = anchor_theta[real_pos_mask[anchor_id]].detach().cpu().numpy()
        real_pos_l2 = anchor_l2[real_pos_mask[anchor_id]].detach().cpu().numpy()

        real_neg_theta = anchor_theta[real_neg_mask[anchor_id]].detach().cpu().numpy()
        real_neg_l2 = anchor_l2[real_neg_mask[anchor_id]].detach().cpu().numpy()

        gcl_pos_theta = anchor_theta[pos_mask[anchor_id]].detach().cpu().numpy()
        gcl_pos_l2 = anchor_l2[pos_mask[anchor_id]].detach().cpu().numpy()

        gcl_neg_theta = anchor_theta[neg_mask[anchor_id]].detach().cpu().numpy()
        gcl_neg_l2 = anchor_l2[neg_mask[anchor_id]].detach().cpu().numpy()

        # Subplot 1: Real Pos
        axes[0].scatter(real_pos_l2, real_pos_theta, color='green', alpha=0.6)
        axes[0].set_title("Real Positive")
        
        # Subplot 2: Real Neg
        axes[1].scatter(real_neg_l2, real_neg_theta, color='blue', alpha=0.6)
        axes[1].set_title("Real Negative")

        # Subplot 3: GCL Pos
        axes[2].scatter(gcl_pos_l2, gcl_pos_theta, color='orange', alpha=0.6)
        axes[2].set_title("GCL Positive")

        # Subplot 4: GCL Neg
        axes[3].scatter(gcl_neg_l2, gcl_neg_theta, color='red', alpha=0.6)
        axes[3].set_title("GCL Negative")

        # Subplot 5: All
        axes[4].scatter(real_neg_l2, real_neg_theta, color='blue', alpha=0.6, label='Real Negative')
        axes[4].scatter(gcl_neg_l2, gcl_neg_theta, color='red', alpha=0.6, label='GCL Negative')
        axes[4].scatter(real_pos_l2, real_pos_theta, color='green', alpha=0.6, label='Real Positive')
        axes[4].scatter(gcl_pos_l2, gcl_pos_theta, color='orange', label='GCL Positive')
        axes[4].set_title("All Types")
        axes[4].legend()

        for ax in axes:
            ax.set_xlabel("L2 Norm")
            ax.set_ylabel("Theta (Degrees)")

        os.makedirs(f'./logs/theta_vs_l2/{args.DS}/lr_{args.lr}/epoch_{epoch}', exist_ok=True)
        plt.savefig(f'./logs/theta_vs_l2/{args.DS}/lr_{args.lr}/epoch_{epoch}/anchor_{anchor_id}_theta_l2_distribution.png')
        plt.close()

        # Calculate distance and angle distributions for FN vs GCL positive
        # fn_l2_diff = np.mean(real_neg_l2) - np.mean(gcl_pos_l2)
        # fn_theta_diff = np.mean(real_neg_theta) - np.mean(gcl_pos_theta)

        # print(f'Anchor {anchor_id}: FN L2 difference: {fn_l2_diff:.4f}, FN Theta difference: {fn_theta_diff:.4f}')
        # === 計算基準 ===
        gcl_pos_l2_mean = np.mean(gcl_pos_l2)
        gcl_pos_theta_mean = np.mean(gcl_pos_theta)

        # === False Negatives 比 GCL Pos 大的數量 ===
        fn_l2_count = np.sum(real_neg_l2 > gcl_pos_l2_mean)
        fn_theta_count = np.sum(real_neg_theta > gcl_pos_theta_mean)

        # === True Negatives (GCL Negatives) 比 GCL Pos 大的數量 ===
        tn_l2_count = np.sum(gcl_neg_l2 > gcl_pos_l2_mean)
        tn_theta_count = np.sum(gcl_neg_theta > gcl_pos_theta_mean)

        print(f"Anchor {anchor_id} 😚😚😚：")
        print(f"🔵 False Negatives: {fn_l2_count} l2 > GCL pos, {fn_theta_count} theta > GCL pos")
        print(f"🔴 True Negatives : {tn_l2_count} l2 > GCL pos, {tn_theta_count} theta > GCL pos\n")
