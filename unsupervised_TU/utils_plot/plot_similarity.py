import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from arguments import arg_parse

args = arg_parse()

def plot_similarity_matrix(sim_matrix, pos_mask, file_name='normalized'):
    # 將 sim_matrix 和 pos_mask 轉換為 NumPy 陣列
    sim_matrix_np = sim_matrix.cpu().detach().numpy()
    pos_mask_np = pos_mask.cpu().detach().numpy()

    # 使用 Seaborn 畫出相似度矩陣
    plt.figure(figsize=(160, 120))
    ax = sns.heatmap(sim_matrix_np, annot=True, fmt='.2f', cmap='Blues', 
                     cbar_kws={'label': 'Similarity'}, linewidths=0.5)

    # 在正樣本的位置上加上背景色
    for i in range(sim_matrix_np.shape[0]):
        for j in range(sim_matrix_np.shape[1]):
            if pos_mask_np[i, j]:  # 只在正樣本對的位置上顯示顏色
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='yellow', alpha=0.3))

    # 設置標題和標籤
    plt.title('Similarity Matrix with Positive Pairs Highlighted')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.savefig(f'{args.DS}_{file_name}')