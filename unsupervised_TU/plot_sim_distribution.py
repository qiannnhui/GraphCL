import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from arguments import arg_parse

args = arg_parse()

# def sort_similarity_and_labels(sim_matrix, pos_mask):
#     # 將相似度矩陣展平（flatten），並排序
#     sim_matrix_flatten = sim_matrix.view(-1)
#     pos_mask_flatten = pos_mask.view(-1)

#     # 排序並獲得排序的索引
#     sorted_indices = torch.argsort(sim_matrix_flatten, descending=True)

#     # 根據排序的索引重新排序相似度和對應的 labels
#     sorted_sim_matrix = sim_matrix_flatten[sorted_indices]
#     sorted_pos_mask = pos_mask_flatten[sorted_indices]

#     return sorted_sim_matrix, sorted_pos_mask

# def plot_similarity_distribution(sim_matrix, pos_mask, file_name='_'):
#     # 顯示相似度分佈
#     sorted_sim_matrix, sorted_pos_mask = sort_similarity_and_labels(sim_matrix, pos_mask)
#     pos_sim = sorted_sim_matrix[sorted_pos_mask == 1]
#     neg_sim = sorted_sim_matrix[sorted_pos_mask == 0]

#     plt.figure(figsize=(8, 6))
#     plt.hist(pos_sim.cpu().detach().numpy(), bins=100, alpha=0.7, label="Positive Pairs")
#     plt.hist(neg_sim.cpu().detach().numpy(), bins=100, alpha=0.7, label="Negative Pairs")
#     plt.legend()
#     plt.title('Similarity Distribution (Positive vs Negative)')
#     plt.xlabel('Similarity')
#     plt.ylabel('Frequency')
#     plt.savefig(f'{args.DS}_{file_name}')

import matplotlib.pyplot as plt

def plot_similarity_distribution(sim_matrix, pos_mask, file_name='_'):
    device = sim_matrix.device
    batch_size = sim_matrix.size(0)
    
    # Create a mask for the upper triangle (i < j) to avoid duplicate samples
    mask = torch.triu(torch.ones(batch_size, batch_size), diagonal=1).bool().to(device)

    # Get the positive pairs (same label)
    pos_mask = pos_mask.masked_select(mask)  # only upper triangle, no duplicates

    # Get the negative pairs (different label)
    neg_mask = ~pos_mask  # inverse of positive mask
    pos_sim = sim_matrix.masked_select(mask) * pos_mask.float()
    neg_sim = sim_matrix.masked_select(mask) * neg_mask.float()

    # Plot the distributions
    plt.figure(figsize=(8, 6))
    plt.hist(pos_sim.cpu().detach().numpy(), bins=50, alpha=0.7, density=False, label="Positive Pairs")
    plt.hist(neg_sim.cpu().detach().numpy(), bins=50, alpha=0.7, density=False, label="Negative Pairs")
    plt.legend()
    plt.title('Similarity Distribution (Positive vs Negative)')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    plt.savefig(f'{file_name}_similarity_distribution.png')
