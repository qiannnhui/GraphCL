import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_rbo_cosine_diagnosis(sim_matrix, coverage, labels, epoch, save_dir):
    """
    診斷 Cosine Similarity 與 RBO Coverage 的關係，並標註 Label 一致性。
    
    Args:
        sim_matrix: (B, B) torch.Tensor, 原始 Cosine Similarity (非 exp 後的)
        coverage: (B, B) torch.Tensor, RBO Coverage 矩陣
        labels: (B,) torch.Tensor, 類別標籤
        epoch: int, 當前 Epoch
        save_dir: str, 儲存路徑
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 確保資料在 CPU 且為 numpy
    sim_matrix = sim_matrix.detach().cpu().numpy()
    coverage = coverage.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    batch_size = sim_matrix.shape[0]
    
    # 建立 Label 一致性矩陣 (True if same label)
    label_match = (labels[:, None] == labels[None, :])
    
    # 只取上三角矩陣 (不含對角線)
    upper_idx = np.triu_indices(batch_size, k=1)
    
    x_cosine = sim_matrix[upper_idx]
    y_coverage = coverage[upper_idx]
    is_same_label = label_match[upper_idx]
    
    # 分離 正樣本對 (Same Label) 與 負樣本對 (Diff Label)
    pos_x = x_cosine[is_same_label]
    pos_y = y_coverage[is_same_label]
    
    neg_x = x_cosine[~is_same_label]
    neg_y = y_coverage[~is_same_label]

    # --- 開始繪圖 ---
    plt.figure(figsize=(12, 8))
    
    # 繪製負樣本 (紅色) - 先畫負樣本讓正樣本在上面，比較好觀察
    plt.scatter(neg_x, neg_y, color='#d62728', alpha=0.3, s=20, label='Diff Label (Neg Pairs)')
    
    # 繪製正樣本 (綠色)
    plt.scatter(pos_x, pos_y, color='#2ca02c', alpha=0.6, s=25, label='Same Label (Pos Pairs)')
    
    # 輔助線：標註可能的閾值
    plt.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='High Cosine Threshold')
    plt.axhline(y=0.4, color='blue', linestyle='--', alpha=0.5, label='RBO Coverage Threshold')

    plt.xlabel('Cosine Similarity (Feature Space)', fontsize=14)
    plt.ylabel('RBO Coverage (Structural Consistency)', fontsize=14)
    plt.title(f'Diagnosis Plot: Epoch {epoch}\n(Green: Same Label, Red: Diff Label)', fontsize=16)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 儲存
    save_path = os.path.join(save_dir, f'diagnosis_epoch_{epoch}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    # --- 打印簡單統計 ---
    # print(f"[Diagnosis Epoch {epoch}] Plot saved to {save_path}")
    # if len(pos_x) > 0:
    #     print(f"  - Avg Pos (Same Label) Cosine: {pos_x.mean():.4f}, Coverage: {pos_y.mean():.4f}")
    # if len(neg_x) > 0:
    #     print(f"  - Avg Neg (Diff Label) Cosine: {neg_x.mean():.4f}, Coverage: {neg_y.mean():.4f}")