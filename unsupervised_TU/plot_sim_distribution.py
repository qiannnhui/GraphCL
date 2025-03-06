import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
def get_sim_mask(sim_matrix, pos_mask, neg_mask):

    device = sim_matrix.device
    batch_size = sim_matrix.size(0)
    
    # Create a mask for the upper triangle (i < j) to avoid duplicate samples
    mask = torch.triu(torch.ones(batch_size, batch_size), diagonal=1).bool().to(device)

    # Get the positive pairs (same label)
    pos_mask = pos_mask.masked_select(mask)  # only upper triangle, no duplicates
    neg_mask = neg_mask.masked_select(mask)

    pos_sim = sim_matrix.masked_select(mask) * pos_mask.float()
    neg_sim = sim_matrix.masked_select(mask) * neg_mask.float()

    return pos_sim, neg_sim


def plot_similarity_distribution(sim_matrix, pos_mask, neg_mask, args, epoch):

    pos_sim, neg_sim = get_sim_mask(sim_matrix, pos_mask, neg_mask)
    dir = os.path.join("logs", "sim_dist", args.DS, args.lr, args.mode, args.aug)

    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)  # The exist_ok=True will prevent error if directory exists

    # Convert tensors to numpy for seaborn to use
    pos_sim_np = pos_sim.cpu().detach().numpy()
    neg_sim_np = neg_sim.cpu().detach().numpy()

    # Create two subplots for positive and negative pairs
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # KDE plot for positive pairs (red)
    sns.kdeplot(pos_sim_np, shade=True, color="red", label="Positive Pairs", ax=axes[0])
    axes[0].set_title("Cosine Similarity Distribution - Positive Pairs")
    axes[0].set_xlabel("Cosine Similarity / tau")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # KDE plot for negative pairs (purple)
    sns.kdeplot(neg_sim_np, shade=True, color="magenta", label="Negative Pairs", ax=axes[1])
    axes[1].set_title("Cosine Similarity Distribution - Negative Pairs")
    axes[1].set_xlabel("Cosine Similarity / tau")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(dir, f"epoch_{epoch}.png"))

    # # Plot the distributions
    # plt.figure(figsize=(8, 6))
    # plt.hist(pos_sim.cpu().detach().numpy(), bins=50, alpha=0.7, density=False, label="Positive Pairs")
    # plt.hist(neg_sim.cpu().detach().numpy(), bins=50, alpha=0.7, density=False, label="Negative Pairs")
    # plt.legend()
    # plt.title('Similarity Distribution (Positive vs Negative)')
    # plt.xlabel('Similarity')
    # plt.ylabel('Frequency')
    # plt.savefig(f'{file_name}_similarity_distribution.png')
