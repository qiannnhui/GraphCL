import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from collections import defaultdict

# 假設您在主腳本中定義了以下函數/變數，這裡為分析提供一個佔位符
# from utils import create_pos_and_neg_mask 
# dataloader_eval # 驗證/測試資料載入器
# model # 訓練好的模型

def get_graph_structural_features(data_list):
    """
    計算圖結構特徵，如節點數、邊數、Fiedler Value (最小非零特徵值)
    Args:
        data_list (list of torch_geometric.data.Data): 批次中的圖資料對象列表
    Returns:
        dict: 包含所有圖特徵的字典
    """
    features = defaultdict(list)
    for data in data_list:
        try:
            # 1. 節點/邊數
            features['num_nodes'].append(data.num_nodes)
            features['num_edges'].append(data.num_edges)
            
            # 2. Fiedler Value (圖拉普拉斯矩陣的第二小特徵值)
            G = to_networkx(data, to_undirected=True)
            if G.number_of_nodes() < 2:
                 # 孤立點或單點圖，無法計算 Fiedler Value
                features['fiedler_value'].append(0.0)
                continue

            # 確保圖是連通的
            if not nx.is_connected(G):
                # 對於不連通的圖，Fiedler Value 為 0
                features['fiedler_value'].append(0.0) 
            else:
                L = nx.laplacian_matrix(G).todense()
                # 計算特徵值 (eigenvalues)
                eigenvalues = np.linalg.eigvalsh(L)
                # Fiedler Value 是第二小的特徵值 (最小特徵值總為 0)
                # 由於浮點數誤差，我們檢查非零的最小特徵值
                non_zero_eig = eigenvalues[eigenvalues > 1e-6] 
                if len(non_zero_eig) > 0:
                    features['fiedler_value'].append(non_zero_eig.min())
                else:
                    features['fiedler_value'].append(0.0) 
                
        except Exception as e:
            # print(f"Error processing graph: {e}")
            features['fiedler_value'].append(0.0)
            features['num_nodes'].append(data.num_nodes)
            features['num_edges'].append(data.num_edges)

    for key in features:
        features[key] = np.array(features[key])

    return features

def analyze_high_similarity_negatives(model, dataloader_eval, device, args, similarity_threshold=0.8, epoch=None):
    """
    分析高相似度負樣本 (Hard Negatives) 的特性。

    Args:
        model (nn.Module): 訓練好的 SimCLR 模型。
        dataloader_eval (DataLoader): 用於評估的資料載入器。
        device (torch.device): 運行的設備 (cpu/cuda)。
        args (Namespace): 包含訓練參數的命名空間。
        similarity_threshold (float): 判斷為高相似度的門檻。
        epoch (int, optional): 當前訓練週期，用於儲存檔案名。
    """
    model.eval()
    all_x = []
    all_labels = []
    all_data = [] # 儲存原始圖資料用於結構分析
    
    # 1. 獲取所有嵌入和標籤
    with torch.no_grad():
        for batch, data in enumerate(dataloader_eval):
            data, _ = data
            data = data.to(device)
            # 假設 x_aug 是 x 的複製，我們只使用 x
            x, _ = model(data.x, data.edge_index, data.batch)
            
            all_x.append(x.cpu())
            # print("data.y = ", data.y)
            all_labels.append(data.y.cpu())
            
            # 將批次中的圖物件轉換為列表，以便後續分析
            all_data.extend(data.to_data_list()) 

    x_embeddings = torch.cat(all_x, dim=0).to(device)
    labels = torch.cat(all_labels, dim=0).to(device)
    # print("labels = ", labels)
    
    N = x_embeddings.size(0)
    
    # 2. 計算餘弦相似度矩陣
    x_norm = F.normalize(x_embeddings, dim=1)
    # 使用點積計算餘弦相似度矩陣 (S_ij = cos(x_i, x_j))
    sim_matrix = torch.einsum('ik,jk->ij', x_norm, x_norm)
    
    # 3. 識別高相似度對 (排除對角線 i vs i)
    # (i, j) 且 i != j
    upper_tri_mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool().to(device)
    high_sim_mask = (sim_matrix > similarity_threshold) & upper_tri_mask
    
    labels_col = labels.view(-1, 1) # 變成 (N, 1) 的列向量
    pos_mask_all = labels_col.eq(labels_col.T) # (i, j) 標籤相同
    # False Positives / Hard Negatives (HN): 高相似度 且 標籤不同
    FP_HN_mask = high_sim_mask & (~pos_mask_all)
    
    # True Positives (TP): 高相似度 且 標籤相同
    TP_mask = high_sim_mask & pos_mask_all
    
    num_high_sim_pairs = high_sim_mask.sum().item()
    num_fp_hn = FP_HN_mask.sum().item()
    num_tp = TP_mask.sum().item()
    
    # if num_high_sim_pairs == 0:
    #     print("\n--- 高相似度負樣本分析 ---")
    #     print(f"在相似度門檻 > {similarity_threshold} 時，沒有找到任何高相似度配對。")
    #     return

    # 5. 結果總結
    # fp_ratio = num_fp_hn / num_high_sim_pairs
    # print("\n" + "="*50)
    # print(f"🌟 相似度 > {similarity_threshold} 的配對分析 (Epoch: {epoch}) 🌟")
    # print(f"總高相似度配對數: {num_high_sim_pairs}")
    # print(f"其中 False Positives (Hard Negatives, 標籤不同) 數: {num_fp_hn}")
    # print(f"其中 True Positives (標籤相同) 數: {num_tp}")
    # print(f"FP / (FP + TP) 比例: {fp_ratio:.4f}")
    # print("="*50)

    # # 6. 結構特徵分析
    # if num_fp_hn > 0:
    #     # 提取 FP/HN 對的索引
    #     fp_hn_indices = FP_HN_mask.nonzero() # (M, 2)
        
    #     # 提取 TP 對的索引 (只取與 FP/HN 數量相近的樣本，以進行公平比較)
    #     num_tp_sample = min(num_tp, num_fp_hn) # 確保數量平衡
    #     tp_indices = TP_mask.nonzero()
    #     if tp_indices.size(0) > num_tp_sample:
    #         # 隨機採樣
    #         perm = torch.randperm(tp_indices.size(0))[:num_tp_sample]
    #         tp_indices = tp_indices[perm]

    #     # 準備進行結構分析的圖列表
    #     # FP/HN Analysis (Anchor: i, Negative: j)
    #     fp_hn_anchors_data = [all_data[i.item()] for i in fp_hn_indices[:, 0]]
    #     fp_hn_negatives_data = [all_data[j.item()] for j in fp_hn_indices[:, 1]]

    #     # TP Analysis (Anchor: i, Positive: j)
    #     tp_anchors_data = [all_data[i.item()] for i in tp_indices[:, 0]]
    #     tp_positives_data = [all_data[j.item()] for j in tp_indices[:, 1]]

    #     # 執行結構特徵計算
    #     fp_hn_anchor_features = get_graph_structural_features(fp_hn_anchors_data)
    #     fp_hn_negative_features = get_graph_structural_features(fp_hn_negatives_data)
    #     tp_anchor_features = get_graph_structural_features(tp_anchors_data)
    #     tp_positive_features = get_graph_structural_features(tp_positives_data)

    #     print("\n--- 結構特徵平均值分析 ---")
    #     print(f"FP/HN 樣本數: {len(fp_hn_anchors_data)}")
    #     print(f"TP 樣本數: {len(tp_anchors_data)}")
        
    #     feature_keys = ['num_nodes', 'num_edges', 'fiedler_value']
        
    #     for key in feature_keys:
    #         # 檢查是否有 NaN 或 Inf，並處理
    #         def safe_mean(arr):
    #             arr = arr[np.isfinite(arr)]
    #             return arr.mean() if len(arr) > 0 else np.nan

    #         fp_hn_anc_mean = safe_mean(fp_hn_anchor_features[key])
    #         fp_hn_neg_mean = safe_mean(fp_hn_negative_features[key])
    #         tp_anc_mean = safe_mean(tp_anchor_features[key])
    #         tp_pos_mean = safe_mean(tp_positive_features[key])

    #         print(f"\n{key} (平均值):")
    #         print(f"  高相似度 HN Anchor: {fp_hn_anc_mean:.4f}")
    #         print(f"  高相似度 HN Negative: {fp_hn_neg_mean:.4f}")
    #         print(f"  高相似度 TP Anchor: {tp_anc_mean:.4f}")
    #         print(f"  高相似度 TP Positive: {tp_pos_mean:.4f}")

    # print("\n" + "="*50)
    return num_high_sim_pairs, num_fp_hn, num_tp
    
