from typing import List
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as colors

def plot_sim_per_epoch(args, all_epoch_sims: List[np.ndarray], save_dir=None, title_suffix=""):
    # 設置輸出目錄
    if save_dir is None:
        save_dir = f'./result/{args.DS}/reweighted_{args.reweighted_loss}/theta_loss_{args.theta_loss}/dist_{args.dist}/sims'
    os.makedirs(save_dir, exist_ok=True)
    
    # 從 args 中獲取 epoch 和 log_interval
    epochs = np.arange(0, len(all_epoch_sims) * args.log_interval, args.log_interval)
    num_epochs = len(all_epoch_sims)
    
    # =========================================================================
    # 步驟 1: 計算統計數據 (Q1, Q2, Q3, Mean, Count)
    # 這些數據將用於 Scatter Plot 和 Statistics Table
    # =========================================================================
    q1_list, q2_list, q3_list, mean_list, count_list = [], [], [], [], []
    stats_list = [] # 用於統計表格的詳細數據
    
    for i, arr in enumerate(all_epoch_sims):
        epoch_stats = {'Epoch': epochs[i]}
        if len(arr) > 0:
            q1 = np.percentile(arr, 25)
            q2 = np.percentile(arr, 50)
            q3 = np.percentile(arr, 75)
            mean = np.mean(arr)
            count = len(arr)
        else:
            q1, q2, q3, mean, count = np.nan, np.nan, np.nan, np.nan, 0
        
        q1_list.append(q1)
        q2_list.append(q2)
        q3_list.append(q3)
        mean_list.append(mean)
        count_list.append(count)
        
        epoch_stats['Q1'] = q1
        epoch_stats['Median (Q2)'] = q2
        epoch_stats['Q3'] = q3
        epoch_stats['Mean'] = mean
        epoch_stats['Count'] = count
        stats_list.append(epoch_stats)
        
    # =========================================================================
    # 步驟 2: 繪製 Scatter Plot + Quartiles
    # =========================================================================
    plt.figure(figsize=(10, 6))

    # 繪製散點圖
    for i, arr in enumerate(all_epoch_sims):
        if len(arr) > 0:
            epoch_val = epochs[i]
            # 增加水平抖動 (jitter) 以避免數據點重疊
            jitter = np.random.uniform(-0.3, 0.3, size=arr.shape)
            plt.scatter(np.full_like(arr, epoch_val) + jitter, arr, 
                        alpha=0.1, s=6, color='gray')

    # 繪製 Q1 和 Q3 之間的陰影區域 (IQR)
    # 將 NaN 值替換為 Q2 (中位數) 以防止 fill_between 錯誤
    q1_plot = np.nan_to_num(q1_list, nan=np.array(q2_list).mean() if np.isnan(q2_list).all() else np.nanmin(q2_list))
    q3_plot = np.nan_to_num(q3_list, nan=np.array(q2_list).mean() if np.isnan(q2_list).all() else np.nanmax(q2_list))
    plt.fill_between(epochs, q1_plot, q3_plot, color='skyblue', alpha=0.3, label='IQR (Q1–Q3)')

    # 繪製 Q2 (中位數) 線
    plt.plot(epochs, q2_list, color='blue', linewidth=2, label='Median (Q2)')
    # 繪製 Mean (平均值) 線
    plt.plot(epochs, mean_list, color='orange', linestyle='--', linewidth=2, label='Mean')

    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.title(f'Similarity Distribution {title_suffix} (Scatter + Quartiles)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(-1.1, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sim_distribution_scatter_quartiles.png'), dpi=300)
    plt.close()


    # =========================================================================
    # 步驟 3: 繪製 Heatmap (相似度分佈密度)
    # =========================================================================
    sim_min, sim_max = -1.0, 1.0
    num_bins = 100 
    bins = np.linspace(sim_min, sim_max, num_bins + 1)
    heatmap_data = np.zeros((num_bins, num_epochs))
    
    for i, arr in enumerate(all_epoch_sims):
        if len(arr) > 0:
            counts, _ = np.histogram(arr, bins=bins, range=(sim_min, sim_max))
            heatmap_data[:, i] = counts
    
    # 正規化每個 Epoch 的分佈
    column_sums = heatmap_data.sum(axis=0)
    normalized_data = np.divide(heatmap_data, column_sums, 
                                out=np.zeros_like(heatmap_data), where=column_sums != 0)
    
    plt.figure(figsize=(14, 9))
    epoch_edges = np.append(epochs, epochs[-1] + args.log_interval)
    vmax_val = np.percentile(normalized_data[normalized_data > 0], 99) if normalized_data.max() > 0 else 0.1
    
    plt.pcolormesh(epoch_edges, bins, normalized_data, 
                   cmap='jet',
                   shading='flat',
                   norm=colors.PowerNorm(gamma=0.5, vmax=vmax_val)
                  )
    
    cbar = plt.colorbar(label='Normalized Density (Proportion in Epoch)')
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Cosine Similarity', fontsize=14)
    plt.title(f'Similarity Distribution Heatmap {title_suffix}', fontsize=16) 
    
    plt.xticks(epochs)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(alpha=0.3, color='white', linestyle='--')
    plt.tight_layout()
    
    heatmap_filename = f'sim_distribution_heatmap.png' 
    plt.savefig(os.path.join(save_dir, heatmap_filename), dpi=300)
    plt.close()


    # =========================================================================
    # 步驟 4: 繪製 Statistics Table (每個 Epoch 顯示 Q1, Q2, Q3, Mean, Count)
    # =========================================================================
    
    # 準備表格數據
    stats_rows_labels = ['Epoch', 'Q1', 'Median (Q2)', 'Q3', 'Mean', 'Count']
    stats_data_table = []
    
    for row_data in stats_list:
        formatted_row = [
            f'{row_data["Epoch"]}',
            f'{row_data["Q1"]:.4f}' if not np.isnan(row_data["Q1"]) else '--',
            f'{row_data["Median (Q2)"]:.4f}' if not np.isnan(row_data["Median (Q2)"]) else '--',
            f'{row_data["Q3"]:.4f}' if not np.isnan(row_data["Q3"]) else '--',
            f'{row_data["Mean"]:.4f}' if not np.isnan(row_data["Mean"]) else '--',
            f'{row_data["Count"]}'
        ]
        stats_data_table.append(formatted_row)


    plt.figure(figsize=(10, 0.5 + num_epochs * 0.4))
    ax = plt.gca()
    ax.axis('off')
    
    # 創建表格
    table = ax.table(cellText=stats_data_table, 
                     colLabels=stats_rows_labels,
                     loc='center',
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.title(f'Similarity Stats {title_suffix}', fontsize=14) 
    plt.tight_layout()
    
    stats_filename = f'sim_distribution_stats_per_epoch.png' 
    plt.savefig(os.path.join(save_dir, stats_filename), dpi=300)
    plt.close()
