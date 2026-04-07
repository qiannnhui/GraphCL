import torch
import pandas as pd
import plotly.graph_objects as go

def plot_3d_interactive_diagnosis(labels, metrics_dict, epoch, save_path, n_samples=1000):
        """
        實作完全自由選擇 X, Y, Z 軸的 3D 互動圖
        """

        # 1. 準備數據
        B = labels.size(0)
        mask = ~torch.eye(B, dtype=torch.bool, device=labels.device)
        labels_col = labels.view(-1, 1)
        gt_label = (labels_col.eq(labels_col.T) & mask)[mask].cpu().numpy()
        
        df_data = {}
        for name, mtx in metrics_dict.items():
            # 確保資料是 numpy array
            val = mtx[mask].cpu().numpy()
            df_data[name] = val
        
        df = pd.DataFrame(df_data)
        df['Label'] = gt_label
        df = df.sample(n=min(len(df), n_samples)) # 採樣
        
        # 取得所有可選的指標名稱
        columns = [c for c in df.columns if c != 'Label']
        
        # 定義顏色與標籤
        colors = df['Label'].map({True: 'green', False: 'red'})
        hover_text = df['Label'].map({True: 'Same Class', False: 'Diff Class'})

        # 2. 建立基礎 Figure (預設顯示前三維)
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=df[columns[0]], 
            y=df[columns[1]], 
            z=df[columns[2]],
            mode='markers',
            marker=dict(size=3, color=colors, opacity=0.6),
            text=hover_text,
            name="Data Points"
        ))

        # 3. 建立三個獨立的下拉選單 (X, Y, Z)
        def create_menu(target_axis, button_index):
            return dict(
                buttons=[
                    dict(
                        label=col,
                        method="update",
                        args=[{target_axis: [df[col]]}, 
                              {"scene": {f"{target_axis}axis": {"title": col}}}]
                    ) for col in columns
                ],
                direction="down",
                showactive=True,
                x=0.1 + (button_index * 0.25), # 平排選單
                y=1.15,
                xanchor="left",
                yanchor="top"
            )

        # 4. 更新佈局
        fig.update_layout(
            updatemenus=[
                create_menu("x", 0),
                create_menu("y", 1),
                create_menu("z", 2)
            ],
            annotations=[
                dict(text="Select X:", x=0.05, y=1.12, xref="paper", yref="paper", showarrow=False),
                dict(text="Select Y:", x=0.30, y=1.12, xref="paper", yref="paper", showarrow=False),
                dict(text="Select Z:", x=0.55, y=1.12, xref="paper", yref="paper", showarrow=False),
            ],
            title=f"3D Flexible Diagnostic Axis - Epoch {epoch}",
            scene=dict(
                xaxis_title=columns[0],
                yaxis_title=columns[1],
                zaxis_title=columns[2]
            ),
            margin=dict(l=0, r=0, b=0, t=60)
        )

        fig.write_html(f"{save_path}/3D_Flexible_Diagnosis_E{epoch}.html")
