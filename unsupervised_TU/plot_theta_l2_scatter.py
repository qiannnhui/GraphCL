import os
import torch
import numpy as np

def create_pos_and_neg_mask(labels):

    labels = labels.view(-1, 1)
    pos_mask = labels.eq(labels.T)
    neg_mask = ~pos_mask
    # remove self-positive pairs
    # pos_mask = pos_mask & ~torch.diag(torch.ones(labels.size(0), device=labels.device, dtype=torch.bool))

    return pos_mask, neg_mask


def plot_theta_l2(x, x_aug, labels=None):
        """
        Calculate theta and l2 norm data for plotting
        """
        # ==== get cosine similarity matrix ====
        # x_norm = x / x.norm(dim=1, keepdim=True)
        # x_aug_norm = x_aug / x_aug.norm(dim=1, keepdim=True)
        x_norm = x
        x_aug_norm = x_aug
        cos_sim_matrix = torch.einsum('ik,jk->ij', x_norm, x_aug_norm)  # (B, B)

        # ==== create pos and neg mask ====
        pos_mask = torch.diag(torch.ones(x.size(0), device=x.device, dtype=torch.bool))  # self-positive pairs, diagonal
        neg_mask = torch.triu(torch.ones_like(cos_sim_matrix), diagonal=1).bool()  # Exclude diagonal, upper triangular part

        # ==== theta ====
        theta_matrix = torch.acos(torch.clamp(cos_sim_matrix, -1.0 + 1e-7, 1.0 - 1e-7))
        theta_matrix_deg = theta_matrix * 180.0 / torch.pi  # to degree

        # ==== l2 norm matrix ====
        x_sq = (x_norm ** 2).sum(dim=1, keepdim=True)  # (B, 1)
        x_aug_sq = (x_aug_norm ** 2).sum(dim=1, keepdim=True).T  # (1, B)
        l2_matrix = torch.sqrt(x_sq + x_aug_sq - 2 * torch.einsum('ik,jk->ij', x_norm, x_aug_norm) + 1e-8)

        # ==== getting data ====
        pos_theta = theta_matrix_deg[pos_mask].detach().cpu().numpy()
        pos_l2 = l2_matrix[pos_mask].detach().cpu().numpy()

        neg_theta = theta_matrix_deg[neg_mask].detach().cpu().numpy()
        neg_l2 = l2_matrix[neg_mask].detach().cpu().numpy()

        if labels is not None:
            # real positive and negative pairs by labels
            real_pos_mask, real_neg_mask = create_pos_and_neg_mask(labels)
            upper_tri_mask = torch.triu(torch.ones_like(cos_sim_matrix), diagonal=0).bool() # get upper triangular part
            real_pos_mask = real_pos_mask & upper_tri_mask
            real_neg_mask = real_neg_mask & upper_tri_mask
            
            real_pos_theta = theta_matrix_deg[real_pos_mask].detach().cpu().numpy()
            real_pos_l2 = l2_matrix[real_pos_mask].detach().cpu().numpy()
            real_neg_theta = theta_matrix_deg[real_neg_mask].detach().cpu().numpy()
            real_neg_l2 = l2_matrix[real_neg_mask].detach().cpu().numpy()

            return pos_l2, pos_theta, neg_l2, neg_theta, real_pos_l2, real_pos_theta, real_neg_l2, real_neg_theta
        
        return pos_l2, pos_theta, neg_l2, neg_theta


def plot_theta_l2_epoch(all_pos_l2, all_pos_theta, all_neg_l2, all_neg_theta, real_pos_l2=None, real_pos_theta=None,
                        real_neg_l2=None, real_neg_theta=None, args=None, epoch=None, similarity_measure="cosine"):
        """
        Plot theta vs l2 norm for all accumulated data
        """
        pos_l2 = np.concatenate(all_pos_l2)
        pos_theta = np.concatenate(all_pos_theta)
        neg_l2 = np.concatenate(all_neg_l2)
        neg_theta = np.concatenate(all_neg_theta)
        
        if real_pos_l2 is not None:
            real_pos_l2 = np.concatenate(real_pos_l2)
            real_pos_theta = np.concatenate(real_pos_theta)
            real_neg_l2 = np.concatenate(real_neg_l2)
            real_neg_theta = np.concatenate(real_neg_theta)

        pos_avg_theta = pos_theta.mean()
        pos_avg_l2 = pos_l2.mean()
        neg_avg_theta = neg_theta.mean()
        neg_avg_l2 = neg_l2.mean()

        result = {
            'pos_avg_theta': pos_avg_theta,
            'pos_avg_l2': pos_avg_l2,
            'neg_avg_theta': neg_avg_theta,
            'neg_avg_l2': neg_avg_l2
        }

        # ==== Plotting ====
        import matplotlib.pyplot as plt
        fontsize = 40
        plt.figure(figsize=(100, 25))
        # plt.axis('equal')  # Set equal scaling for both axes

        # Plot for Positive Pairs
        plt.subplot(131)  # (1 row, 3 columns, 1st plot)
        plt.scatter(pos_l2, pos_theta, color='deepskyblue', marker='^', s=400, label='Positive Pairs')
        if real_pos_l2 is not None:
            plt.scatter(real_pos_l2, real_pos_theta, color='green', label='Real Positive Pairs', alpha=0.6)
        plt.xlabel('L2 Norm', fontsize=fontsize)
        plt.ylabel('Theta (degrees)', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title(f'Positive Pairs (Epoch {epoch})', fontsize=fontsize)
        plt.grid(True)
        plt.legend(fontsize=fontsize)

        # Plot for Negative Pairs
        plt.subplot(132)  # (1 row, 3 columns, 2nd plot)
        plt.scatter(neg_l2, neg_theta, color='orange', marker='^', s=400, label='Negative Pairs', alpha=0.6)
        if real_neg_l2 is not None:
            plt.scatter(real_neg_l2, real_neg_theta, color='red', label='Real Negative Pairs', alpha=0.6)
        plt.xlabel('L2 Norm', fontsize=fontsize)
        plt.ylabel('Theta (degrees)', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title(f'Negative Pairs (Epoch {epoch})', fontsize=fontsize)
        plt.grid(True)
        plt.legend(fontsize=fontsize)

        # Plot for Both Positive & Negative Pairs
        plt.subplot(133)  # (1 row, 3 columns, 3rd plot)
        plt.scatter(pos_l2, pos_theta, color='deepskyblue', marker='^', s=400, label='Positive Pairs')
        plt.scatter(neg_l2, neg_theta, color='orange', label='Negative Pairs', alpha=0.6)
        if real_pos_l2 is not None:
            plt.scatter(real_pos_l2, real_pos_theta, color='green', label='Real Positive Pairs', alpha=0.6)
            plt.scatter(real_neg_l2, real_neg_theta, color='red', label='Real Negative Pairs', alpha=0.6)
        plt.xlabel('L2 Norm', fontsize=fontsize)
        plt.ylabel('Theta (degrees)', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title(f'Theta vs L2 Norm with {similarity_measure} (Epoch {epoch})', fontsize=fontsize)
        plt.grid(True)
        plt.legend(fontsize=fontsize)

        # Save the figure with all plots
        os.makedirs(f'./logs/theta_vs_l2/SimGCL/{args.DS}/lr_{args.lr}/{similarity_measure}', exist_ok=True)
        plt.tight_layout()  # Makes sure everything fits without overlap
        plt.savefig(f'./logs/theta_vs_l2/SimGCL/{args.DS}/lr_{args.lr}/{similarity_measure}/epoch_{epoch}_theta_vs_l2_{args.aug}.png')
        plt.close()  # Close the figure to free memory

        return result