import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx
import torch_geometric.utils # for degree calculation
from sklearn.linear_model import LogisticRegression
import math
from aug import TUDataset_aug as TUDataset
from gsimclr import simclr
from arguments import arg_parse
from make_save_dir import make_save_dir
from save_load_ckpts import load_checkpoint
from torch_geometric.nn import global_add_pool # 確保已導入


class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes, lr_weights, lr_bias):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        
        # 創建一個線性層
        self.linear = nn.Linear(in_features, num_classes)
        
        # 將 Scikit-learn 的權重 W 和偏置 b 轉換並載入
        # 注意：Scikit-learn 的 W 是 (num_classes, in_features)，PyTorch 也是 (out, in)
        self.linear.weight.data.copy_(torch.from_numpy(lr_weights).float())
        self.linear.bias.data.copy_(torch.from_numpy(lr_bias).float())
        
    def forward(self, y):
        # y 是圖嵌入 (B, D)
        return self.linear(y) # (B, num_classes) -> Logits
    

def get_saliency_nodes(model, data, clf, target_class_idx, k=5):
    """
    使用 Grad-CAM 思想計算節點的重要性（Logit 相對於 M 的梯度 L2 範數）。
    
    Args:
        model (simclr): GNN 模型。
        data (Data): 單個圖數據。
        clf (LogisticRegression): 訓練好的 Scikit-learn 分類器。
        target_class_idx (int): 目標類別的索引 (e.g., 0, 1, 2...)。
        k (int): 突出顯示的節點數量。
    
    Returns:
        numpy.ndarray: 最重要的 k 個節點的索引。
    """
    device = next(model.parameters()).device
    data = data.to(device)
    
    # --- 1. 建立 PyTorch 分類頭 (集成 LR 權重) ---
    in_features = model.embedding_dim
    num_classes = len(clf.classes_)
    
    # 獲取目標類別的權重和偏置
    W = clf.coef_
    b = clf.intercept_
    
    classifier_head = ClassifierHead(in_features, num_classes, W, b).to(device)
    
    # --- 2. 獲取節點嵌入 M (啟用梯度追蹤) ---
    model.eval()
    if data.x is None:
        data.x = torch.ones((data.batch.shape[0], 1)).to(device)
        
    with torch.enable_grad():
        # 這裡需要確保您的 simclr forward 可以返回 M (節點嵌入)
        data.x.requires_grad_(True)
        y_dummy, M = model(data.x, data.edge_index, data.batch, 1) 
        # M.requires_grad_(True)
        
        # --- 3. 聚合 M 為圖嵌入 y ---
        y = global_add_pool(M, data.batch) 
        
        # --- 4. 計算目標 Logit ---
        logits = classifier_head(y) # (1, num_classes)
        target_logit = logits[0, target_class_idx]
        
        # --- 5. 反傳遞：計算 Logit 相對於 M 的梯度 ---
        # $\frac{\partial \text{Logit}_c}{\partial \mathbf{M}}$
        # target_logit.backward()
        grad_output = torch.autograd.grad(
            outputs=target_logit, 
            inputs=M, # 想要計算 M 的梯度
            retain_graph=False, # 可以設為 True 以防萬一，但通常 False 即可
            allow_unused=True # 允許 M 實際上可能沒有貢獻
        )[0]
        
        if grad_output is None:
            # 如果 M.grad 仍為 None，可能是因為 M 沒有直接連接到 Logit (模型內部有 detach)
            print("Warning: Gradient of M is None. Check model internal detach.")
            return np.array([])
        
        # 梯度值 (重要性得分)
        # Saliency Score $\mathbf{S}_i = || \frac{\partial \text{Logit}_c}{\partial \mathbf{M}_i} ||_2$
        # saliency_map = M.grad.abs().sum(dim=1) # 也可以用 norm 或 sum(abs)
        saliency_map = grad_output.abs().sum(dim=1) # <--- 使用 grad_output        
        # --- 6. 選取 Top-K 節點 ---
        if saliency_map.numel() == 0:
            return np.array([])
            
        k = min(k, saliency_map.numel())
        highlight_nodes = saliency_map.topk(k).indices.cpu().numpy()

    return highlight_nodes

NODE_MARKERS = ['o', 's', '^', 'D', 'p', 'h', 'v'] # 圓形, 方形, 三角形, 菱形, 五角星, 六角星, 倒三角

def get_node_features_and_types(data):
    """
    從 PyG Data 中提取節點類型 ID。
    
    假設：data.x 是 One-hot 編碼或離散類別 ID。
    """
    if data.x is None:
        # 如果沒有特徵，所有節點都是同一類型
        return torch.zeros(data.num_nodes, dtype=torch.long).cpu().numpy()
    
    # 對於 One-hot 編碼 (例如 MUTAG)，類型 ID 是非零元素的索引
    if data.x.dim() == 2 and data.x.shape[1] > 1:
        # 假設是 One-hot，取最大值所在的索引作為類型 ID
        node_types = torch.argmax(data.x, dim=1).cpu().numpy()
    else:
        # 否則假設是單維類別 ID 或常數特徵 (都視為類型 0)
        node_types = torch.zeros(data.num_nodes, dtype=torch.long).cpu().numpy()
        
    return node_types

# 修改 draw_graph_with_highlight 函數

def draw_graph_with_highlight(data, graph_id, title, highlight_nodes=None, ax=None):
    
    # --- 1. 結構提取和清理 (保持不變) ---
    G = to_networkx(data, to_undirected=True) 
    G_simple = nx.Graph(G) 
    G_simple.remove_edges_from(nx.selfloop_edges(G_simple))
    G = G_simple 
    num_nodes = data.num_nodes
    
    # --- 2. 獲取節點類型和高亮標記 ---
    node_types = get_node_features_and_types(data) # 獲取每個節點的類型 ID
    unique_types = np.unique(node_types)
    
    # 節點的基礎屬性（顏色和尺寸）
    base_node_colors = ['#1f78b4'] * num_nodes
    base_node_sizes = [100] * num_nodes
    
    # 處理高亮節點 (Saliency Map 輸出)
    highlight_indices = []
    if highlight_nodes is not None:
        if isinstance(highlight_nodes, np.ndarray):
            highlight_nodes = highlight_nodes.tolist()
        highlight_indices = [int(i) for i in highlight_nodes if i < num_nodes]
    
    # 計算佈局 (Layout)
    N = G.number_of_nodes()
    LAYOUT_K = 1.0 / np.sqrt(N) # 預設值是 1/sqrt(N)。我們可以使用一個更大的乘數，例如 1.5 或 2.0
    LAYOUT_K = max(LAYOUT_K, 0.2) # 設置最小 k 值，避免小圖過於緊湊
    
    pos = nx.spring_layout(
        G, 
        seed=42, 
        k=LAYOUT_K, # 使用計算出的或更大的 k
        iterations=100 # 將迭代次數從 50 增加到 100，以獲得更穩定的非重疊佈局
    )
    if ax is None:
        # 增加 figsize 以便容納圖例 (Legend)
        fig, ax = plt.subplots(figsize=(10, 8)) 
    
    # --- 3. 繪製邊 (只需要繪製一次) ---
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=0.5)
    
    # --- 4. 迴圈繪製不同類型的節點 ---
    all_markers = []
    for type_id in unique_types:
        # 選取該類型節點的索引
        type_indices = np.where(node_types == type_id)[0]
        
        # 篩選出該類型節點的座標、顏色、尺寸
        node_coords = {n: pos[n] for n in type_indices}
        
        # 為當前類型計算顏色和尺寸
        current_colors = [base_node_colors[n] for n in type_indices]
        current_sizes = [base_node_sizes[n] for n in type_indices]
        
        # 應用 Saliency Map 高亮
        for idx, n in enumerate(type_indices):
            if n in highlight_indices:
                current_colors[idx] = 'red' # Saliency 標記為紅色
                current_sizes[idx] = 300   # 放大
                
        # 選擇符號
        marker_symbol = NODE_MARKERS[type_id % len(NODE_MARKERS)]
        
        # 繪製節點
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=type_indices.tolist(), 
            node_color=current_colors, 
            node_size=current_sizes,
            node_shape=marker_symbol, # 使用不同的符號
            ax=ax,
            label=f'Type {type_id}', # 用於圖例
            linewidths=1.0,
            edgecolors='k' # 黑色邊框
        )

    # 繪製節點 ID 標籤 (Labeling)
    if G.number_of_nodes() < 50: # 節點少於 50 個時才畫標籤，避免混亂
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)


    ax.set_title(f"{title}\nGraph ID: {graph_id} (Nodes: {num_nodes})", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 顯示圖例 (Type 0, Type 1, Type 2...)
    ax.legend(scatterpoints=1, markerscale=1, loc='upper left', bbox_to_anchor=(1.0, 1.0))


def load_and_classify(args):
    """載入模型並執行分類，返回資料集、嵌入和預測標籤。"""
    
    # --- 1. 資料載入 ---
    path = osp.join(args.path, args.DS)
    dataset_eval = TUDataset(path, name=args.DS, aug='none').shuffle() 
    dataloader_eval = DataLoader(dataset_eval, batch_size=64) # 使用小 batch_size
    
    # --- 2. 模型載入 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        dataset_num_features = dataset_eval.get_num_feature()
    except:
        dataset_num_features = 1

    model = simclr(args.hidden_dim, args.num_gc_layers, shuffle_DBN=args.shuffle_DBN, dataset_num_features=dataset_num_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    save_dir = make_save_dir(base="./result/GCL", args=args)
    ckpt_filename = f'{save_dir}/ckpts.pth.tar'

    if not os.path.exists(ckpt_filename):
        print(f"Error: Checkpoint file not found at {ckpt_filename}. Cannot proceed.")
        return None, None, None

    # 載入檢查點
    _, _ = load_checkpoint(ckpt_filename, model, optimizer, device)
    model.eval()
    
    # --- 3. 獲取嵌入和預測標籤 ---
    print("Generating embeddings and performing classification...")
    emb, y_true = model.encoder.get_embeddings(dataloader_eval) 
    
    if emb.size == 0:
        print("Embeddings generation failed.")
        return dataset_eval, np.array([]), np.array([])
        
    # 為了視覺化，我們訓練一個分類器來獲取預測標籤
    n_samples = len(y_true)
    if n_samples == 0:
        return dataset_eval, np.array([]), np.array([])
        
    # 使用 80/20 分割來訓練分類器
    split_idx = max(1, int(n_samples * 0.8)) 
    
    X_train, X_test = emb[:split_idx], emb[split_idx:]
    y_train, y_test = y_true[:split_idx], y_true[split_idx:]
    
    clf = LogisticRegression(solver='liblinear', multi_class='auto', random_state=42, max_iter=300)
    clf.fit(X_train, y_train)
    
    # 對所有嵌入進行預測
    y_pred = clf.predict(emb)
    
    print(f"Classification accuracy on held-out test set: {clf.score(X_test, y_test):.4f}")
    
    return dataset_eval, y_true, y_pred, model, clf


def visualize_graphs_by_prediction(dataset, y_pred, y_true, model, clf, save_dir, num_samples_per_class=4):
    """
    根據模型的預測標籤 (y_pred) 抽取樣本並繪製圖結構。
    """
    if not dataset or y_pred is None or len(y_pred) == 0:
        print("Visualization skipped: no data or predictions available.")
        return

    unique_classes = np.unique(y_pred)
    print(f"Total predicted classes to visualize: {len(unique_classes)}")
    
    class_indices = {c: np.where(y_pred == c)[0] for c in unique_classes}

    rows = len(unique_classes)
    cols = num_samples_per_class
    max_rows = min(rows, 8) 
    
    # fig, axes = plt.subplots(max_rows, cols, figsize=(4 * cols, 4 * max_rows))
    WIDTH_MULTIPLIER = 7.0 # 增大寬度以容納圖例
    HEIGHT_MULTIPLIER = 6.0 # 增大高度
    
    fig, axes = plt.subplots(
        max_rows, 
        cols, 
        figsize=(WIDTH_MULTIPLIER * cols, HEIGHT_MULTIPLIER * max_rows) # 增大整個圖像大小
    )
    
    if max_rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif max_rows == 1 or cols == 1:
        axes = axes.reshape(max_rows, cols)
    
    # 處理 rows > max_rows 時，axes 結構不匹配的問題
    if rows > 1 and max_rows < rows and cols > 1:
        axes = axes.reshape(max_rows, cols)

    for i, cls in enumerate(unique_classes):
        if i >= max_rows:
            print(f"Skipping visualization for predicted class {cls} due to row limit ({max_rows}).")
            break
            
        indices = class_indices[cls]
        
        # 隨機抽取樣本
        sampled_indices = random.sample(
            indices.tolist(), 
            min(len(indices), num_samples_per_class)
        )
        
        print(f"Predicted Class {cls}: Sampled graph IDs {sampled_indices}")

        for j, graph_idx in enumerate(sampled_indices):
            data, _ = dataset[graph_idx]
            ax = axes[i, j]
            
            # --- 節點突出顯示邏輯 (這裡使用度中心性最高的 5 個節點作為範例) ---
            highlight_nodes = get_saliency_nodes(
                model=model,
                data=data,
                clf=clf,
                target_class_idx=cls, # 使用預測類別作為目標
                k=5
            )
            # highlight_nodes = None
            # if data.edge_index.numel() > 0:
            #     degrees = torch_geometric.utils.degree(data.edge_index[0], num_nodes=data.num_nodes)
            #     top_k = min(5, data.num_nodes)
            #     # 確保 topk 不會崩潰
            #     if top_k > 0:
            #         highlight_nodes = degrees.topk(top_k).indices.cpu().numpy() 
            # --------------------------------------------------------------------
            true_cls = y_true[graph_idx] 
            title_text = f"PRED: {cls} | TRUE: {true_cls}"
            if cls != true_cls:
                 title_text += " (MISCLASSIFIED)"
            
            draw_graph_with_highlight(
                data, 
                graph_idx,
                title=title_text,
                highlight_nodes=highlight_nodes,
                ax=ax
            )
        
        # 隱藏多餘的子圖
        for j in range(len(sampled_indices), cols):
            if i < axes.shape[0] and j < axes.shape[1]:
                 fig.delaxes(axes[i, j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    final_save_path = osp.join(save_dir, 'graph_structure_by_prediction.png')
    fig.suptitle(f"Sampled Graph Structures Grouped by Model Prediction ({dataset.name})", fontsize=16)
    plt.savefig(final_save_path)
    plt.close()
    print(f"\nGraph structure visualization based on model prediction saved to {final_save_path}")

# ====================================================================
# III. 主執行邏輯
# ====================================================================

if __name__ == '__main__':

    args = arg_parse()
    
    dataset_eval, y_true, y_pred, model, clf = load_and_classify(args)
    
    if dataset_eval is None or y_pred is None or len(y_pred) == 0:
        print("Process aborted.")
    else:
        # 2. 定位儲存路徑
        save_dir = make_save_dir(base="./result/GCL", args=args)
        os.makedirs(save_dir, exist_ok=True)
        
        # 3. 繪製圖結構
        visualize_graphs_by_prediction(
            dataset=dataset_eval, 
            y_pred=y_pred,
            y_true=y_true,
            model=model,
            clf=clf,
            save_dir=save_dir,
            num_samples_per_class=4 
        )