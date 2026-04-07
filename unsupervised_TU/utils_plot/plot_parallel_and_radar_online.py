import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

def plot_advanced_diagnosis(labels, metrics_dict, epoch, save_path, n_samples=500):
        # todos: upper-triangle only

        # 1. 建立 Mask 移除對角線
        B = labels.size(0)
        mask = ~torch.eye(B, dtype=torch.bool, device=labels.device)
        labels_col = labels.view(-1, 1)
        gt_mask = labels_col.eq(labels_col.T) & mask
        
        # 2. 拉平所有指標並放入 DataFrame
        data_for_df = {}
        for name, mtx in metrics_dict.items():
            data_for_df[name] = mtx[mask].cpu().numpy()
        
        flat_gt = gt_mask[mask].cpu().numpy()
        data_for_df['Label'] = flat_gt.astype(int)
        for name in data_for_df:
            if name != 'Label':
                # 確保數值在 0~1 之間，並處理掉可能的 nan
                data_for_df[name] = np.nan_to_num(np.clip(data_for_df[name], 0, 1))
        
        full_df = pd.DataFrame(data_for_df)

        # 3. 平衡抽樣
        pos_indices = full_df[full_df['Label'] == 1].index
        neg_indices = full_df[full_df['Label'] == 0].index
        n_pos = min(n_samples // 2, len(pos_indices))
        n_neg = n_samples - n_pos
        
        sel_idx = np.concatenate([
            np.random.choice(pos_indices, n_pos, replace=False),
            np.random.choice(neg_indices, n_neg, replace=False)
        ])
        df = full_df.loc[sel_idx].reset_index(drop=True)

        # --- Plot A: Parallel Plot (現在會顯示所有指標) ---
        fig_parallel = px.parallel_coordinates(
            df, color="Label",
            dimensions=list(metrics_dict.keys()), # 自動包含所有傳入的指標
            color_continuous_scale=[(0, 'red'), (1, 'green')],
            title=f"Global Parallel Flow - Epoch {epoch}"
        )
        fig_parallel.write_html(f"{save_path}/parallel_E{epoch}.html")

        # --- Plot B: Radar Plot (看各指標平均) ---
        categories = list(metrics_dict.keys())
        avg_df = df.groupby('Label')[categories].mean().reset_index()
        fig_radar = go.Figure()
        for _, row in avg_df.iterrows():
            name = "Same Label" if row['Label'] == 1 else "Diff Label"
            fig_radar.add_trace(go.Scatterpolar(r=row[categories].values, theta=categories, fill='toself', name=name))
        fig_radar.write_html(f"{save_path}/radar_E{epoch}.html")