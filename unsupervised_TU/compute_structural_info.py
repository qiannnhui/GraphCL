import torch
from torch_geometric.utils import degree
import torch.nn.functional as F

def compute_adamic_adar_sim(adj_matrix):
        """
        計算 Batch 內部的 Adamic-Adar 相似度
        adj_matrix: [B, B] 的初始相似度矩陣 (例如 Jaccard)
        """
        degree = adj_matrix.sum(dim=1)
        
        weights = 1.0 / torch.log(degree + 1.1) 
        weights[torch.isinf(weights)] = 0
        weights[torch.isnan(weights)] = 0
        
        weighted_adj = adj_matrix * weights.unsqueeze(0) # 廣播相乘
        aa_matrix = torch.mm(weighted_adj, adj_matrix.t())
        
        aa_min, aa_max = aa_matrix.min(), aa_matrix.max()
        aa_norm = (aa_matrix - aa_min) / (aa_max - aa_min + 1e-8)
        
        return aa_norm

def compute_geometry_jaccard_sim(self, data, max_deg=20):
        edge_index = data.edge_index
        batch = data.batch
        num_graphs = data.num_graphs
        deg = degree(edge_index[0], dtype=torch.long)
        deg_clamped = deg.clamp(max=max_deg)
        deg_onehot = F.one_hot(deg_clamped, num_classes=max_deg + 1).float()
        
        graph_hist = torch.zeros((num_graphs, max_deg + 1), device=data.device)
        graph_hist.scatter_add_(0, batch.view(-1, 1).expand(-1, max_deg + 1), deg_onehot)

        A = graph_hist.unsqueeze(1) 
        B = graph_hist.unsqueeze(0) 
        intersection = torch.min(A, B).sum(dim=-1)
        union = torch.max(A, B).sum(dim=-1)
        return intersection / (union + 1e-8)

def compute_structural_consensus_metrics(data, alpha=0.15, ppr_steps=10, heat_t=1.0, walk_len=5, num_walk_samples=30):
        """
        高效稀疏運算版本：整合 PPR, Heat Kernel, High-Pass 與 Anonymous Walk
        解決維度衝突 (Node vs Edge) 與 OOM 問題
        """
        from torch_geometric.nn import global_mean_pool
        from torch_geometric.utils import degree
        from collections import Counter
        
        # 處理 random_walk 函式庫版本相容性
        try:
            from torch_cluster import random_walk
        except ImportError:
            from torch_geometric.utils import random_walk

        edge_index = data.edge_index
        num_nodes = data.num_nodes
        batch = data.batch # 標記每個節點屬於哪張圖 [N]
        num_graphs = data.num_graphs
        device = data.device

        # --- 1. 譜域預處理：計算歸一化係數 ---
        row, col = edge_index
        deg = degree(col, num_nodes)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        deg_inv_sqrt = deg_inv_sqrt.view(-1, 1) # 強制轉為 [N, 1] 以利廣播計算

        # 定義 SpMV (Sparse Matrix-Vector Multiplication) 運算：D^-1/2 * A * D^-1/2 * x
        def sparse_op(input_x):
            # input_x shape: [N, 1]
            # (1) 先做對源節點的歸一化
            x_scaled = input_x * deg_inv_sqrt # [N, 1]
            # (2) 透過邊傳遞訊息 (A * x)
            norm_src = x_scaled[row] # [Edges, 1]
            out = torch.zeros((num_nodes, input_x.size(1)), device=device)
            out.scatter_add_(0, col.unsqueeze(-1), norm_src) # [N, 1]
            # (3) 最後做對目標節點的歸一化
            return out * deg_inv_sqrt

        # 初始信號：全 1 向量 [N, 1]
        x_ones = torch.ones((num_nodes, 1), device=device)

        # --- 2. 譜域指標計算 ---
        # (A) PPR 疊代 (Power Iteration)
        x_ppr = x_ones.clone()
        for _ in range(ppr_steps):
            x_ppr = (1 - alpha) * sparse_op(x_ppr) + alpha * x_ones
        # 聚合至圖級特徵 [B, 1]
        ppr_repr = global_mean_pool(x_ppr, batch)

        # (B) High-Pass (L*x = x - S*x)
        Sx = sparse_op(x_ones)
        Lx = x_ones - Sx
        # Lx 代表結構變化的能量，聚合至圖級 [B, 1]
        high_pass_repr = global_mean_pool(Lx, batch)

        # (C) Heat Kernel (近似 exp(-tL)x ≈ x - tLx)
        x_heat = x_ones - heat_t * Lx
        heat_repr = global_mean_pool(x_heat, batch)

        # 輔助函式：計算 [B, B] 相似度矩陣並歸一化
        def get_sim_matrix(repr_vec):
            # repr_vec shape: [B, 1]
            sim = torch.mm(repr_vec, repr_vec.t())
            c_min, c_max = sim.min(), sim.max()
            return (sim - c_min) / (c_max - c_min + 1e-8)

        sub_ppr = get_sim_matrix(ppr_repr)
        sub_high = get_sim_matrix(high_pass_repr)
        sub_heat = get_sim_matrix(heat_repr)
        sub_aa = compute_adamic_adar_sim(sub_ppr)

        # --- 3. Anonymous Walk 計算 ---
        # 為每張圖隨機選擇起點
        subset_indices = []
        for i in range(num_graphs):
            nodes_in_g = (batch == i).nonzero(as_tuple=True)[0]
            if len(nodes_in_g) > 0:
                # 隨機抽樣，若節點數不足則允許重複
                idx = nodes_in_g[torch.randint(0, len(nodes_in_g), (num_walk_samples,), device=device)]
                subset_indices.append(idx)
        
        if len(subset_indices) > 0:
            start_nodes = torch.cat(subset_indices)
            # 使用 GPU 進行平行隨機走訪
            walks = random_walk(row, col, start_nodes, walk_length=walk_len-1)
            
            # 模式統計 (CPU 處理)
            walks_cpu = walks.cpu().numpy()
            graph_patterns = []
            for i in range(num_graphs):
                g_walks = walks_cpu[i*num_walk_samples : (i+1)*num_walk_samples]
                counts = Counter()
                for w in g_walks:
                    # 匿名化編碼 (例如: [102, 45, 102] -> [0, 1, 0])
                    d = {}
                    p = tuple([d.setdefault(node, len(d)) for node in w])
                    counts[p] += 1
                graph_patterns.append(counts)

            # 計算圖與圖之間的 Anonymous Walk 相似度 [B, B]
            sub_anon = torch.zeros((num_graphs, num_graphs), device=device)
            for i in range(num_graphs):
                for j in range(i, num_graphs):
                    c1, c2 = graph_patterns[i], graph_patterns[j]
                    intersection = sum((c1 & c2).values())
                    union = sum((c1 | c2).values())
                    val = intersection / (union + 1e-8)
                    sub_anon[i, j] = sub_anon[j, i] = val
        else:
            sub_anon = torch.zeros((num_graphs, num_graphs), device=device)

        return {
            'PPR': sub_ppr,
            'Heat_Kernel': sub_heat,
            'High_Pass': sub_high,
            'Anonymous_Walk': sub_anon,
            'Adamic_Adar': sub_aa
        }