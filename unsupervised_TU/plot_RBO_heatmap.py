import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

def plot_sorted_rbo_heatmap(weights, labels, epoch, save_path):
    """
    weights: (B, B) Tensor, RBO 計算出的 negative weights
    labels: (B,) Tensor, 真實標籤
    epoch: 當前 epoch 數
    save_path: 儲存檔案的路徑
    """
    # 1. 搬移到 CPU 並轉換為 Numpy
    # weights_np = weights.detach().cpu().numpy()
    # labels_np = labels.detach().cpu().numpy()
    weights_np = weights
    labels_np = labels
    
    # 2. 取得排序索引 (依照 Label 從小到大排序)
    sort_idx = np.argsort(labels_np)
    sorted_weights = weights_np[sort_idx][:, sort_idx]
    sorted_labels = labels_np[sort_idx]
    
    # 3. 找出 Label 變化的邊界位置
    # np.unique(..., return_index=True) 會回傳每個類別第一次出現的 index
    unique_labels, first_indices = np.unique(sorted_labels, return_index=True)
    
    # 4. 繪圖
    plt.figure(figsize=(10, 8))
    
    # 使用熱力圖，cmap 可以選 'YlGnBu' 或 'viridis'
    # 顏色越深代表權重越高 (TN)，顏色越淺代表權重越低 (FN)
    ax = sns.heatmap(sorted_weights, cmap='YlGnBu', xticklabels=False, yticklabels=False)
    
    # 5. 在畫面上標示 Label 邊界
    # 我們在每個類別的起始點畫線 (跳過第一個，因為那是 0)
    for idx in first_indices[1:]:
        # 畫縱線
        plt.axvline(x=idx, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        # 畫橫線
        plt.axhline(y=idx, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # 6. 在軸上標示類別名稱 (選配，標在區塊中間)
    mid_points = []
    indices_with_end = np.append(first_indices, len(sorted_labels))
    for i in range(len(indices_with_end)-1):
        mid_points.append((indices_with_end[i] + indices_with_end[i+1]) / 2)
    
    plt.xticks(mid_points, unique_labels)
    plt.yticks(mid_points, unique_labels)
    plt.xlabel("Samples (grouped by Label)")
    plt.ylabel("Anchors (grouped by Label)")
    plt.title(f"RBO Weights Heatmap - Epoch {epoch}\n(Red lines = Class Boundaries)")
    
    # 儲存圖片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# --- 使用方式範例 (放在你的 training loop 裡面) ---
# if epoch % 50 == 0:
#     # 假設你算出的 weight 是 normalized_weights
#     plot_sorted_rbo_heatmap(
#         weights=normalized_weights, 
#         labels=labels, 
#         epoch=epoch, 
#         save_path=f'./logs/rbo_heatmap_epoch_{epoch}.png'
#     )