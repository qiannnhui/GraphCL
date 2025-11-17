# -*- coding: utf-8 -*-
"""
plot_kde_unitcircle.py
以 KDE 繪製 embeddings 投影到單位圓上的角度分佈（Overall + per-class）。
"""

import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from typing import Optional, Sequence, Tuple, Union
from matplotlib.colors import PowerNorm, Normalize
import os

# -------- 輔助：容忍 torch / CUDA 張量 --------
def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _labels_1d(y):
    y = _to_numpy(y)
    if y.ndim == 1:
        return y
    if y.ndim == 2 and y.shape[1] == 1:
        return y.reshape(-1)
    if y.ndim == 2 and y.shape[1] > 1:
        return y.argmax(axis=1)
    return y.reshape(-1)

# -------- PCA(確定性) 降到 2D --------
def _pca_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.shape[1] == 2:
        return X
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD full => deterministic
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T

def _umap_2d(X: np.ndarray) -> np.ndarray:
    try:
        from umap import UMAP
    except ImportError:
        print("UMAP required. Please install with 'pip install umap-learn'. Falling back to PCA.")
        return _pca_2d(X)
        
    X = np.asarray(X, dtype=float)
    if X.shape[1] <= 2:
        return X

    # 建議：先用 PCA 預降維到 50 維，以提高 UMAP 穩定性和速度
    if X.shape[1] > 50:
        from sklearn.decomposition import PCA
        X = PCA(n_components=50, random_state=42).fit_transform(X)
        
    # UMAP 參數通常需要調整，這裡使用通用設置
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    return reducer.fit_transform(X)

# -------- t-SNE 降到 2D --------
def _tsne_2d(X: np.ndarray) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except ImportError:
        print("TSNE required. Please install scikit-learn. Falling back to PCA.")
        return _pca_2d(X)
        
    X = np.asarray(X, dtype=float)
    if X.shape[1] <= 2:
        return X

    # 建議：先用 PCA 預降維到 50 維，以提高 t-SNE 穩定性和速度
    if X.shape[1] > 50:
        X = PCA(n_components=50, random_state=42).fit_transform(X)

    # t-SNE 參數通常需要調整，這裡使用通用設置
    # 注意：t-SNE 對大型數據集非常慢
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    return tsne.fit_transform(X)

# -------- 單位圓投影 + 角度 --------
def _unit_and_theta(X2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(X2, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    U = X2 / norm
    theta = np.mod(np.arctan2(U[:, 1], U[:, 0]), 2 * np.pi)  # [0, 2π)
    return U, theta

# -------- 1D KDE over theta --------
def _kde_theta(theta: np.ndarray, grid_bins: int = 720, bw: Optional[float] = None):
    from scipy.stats import gaussian_kde
    # 角度是週期變量：做循環擴展避免邊界效應
    t = np.r_[theta, theta + 2*np.pi, theta - 2*np.pi]
    kde = gaussian_kde(t, bw_method=bw)
    grid = np.linspace(0, 2*np.pi, grid_bins, endpoint=False)
    dens = kde(grid)
    dens /= dens.max() if dens.max() > 0 else 1.0
    return grid, dens

# -------- 把角度密度「塗」成圓環影像 --------
def _ring_image(theta_grid: np.ndarray,
                density: np.ndarray,
                img_size: int = 420,
                sigma_r: float = 0.16) -> np.ndarray:
    """
    將 θ 的密度映射到半徑=1 的圓環；徑向用高斯讓環有厚度。
    """
    H = W = img_size
    margin = 1.15
    xs = np.linspace(-margin, margin, W)
    ys = np.linspace(-margin, margin, H)
    Xg, Yg = np.meshgrid(xs, ys)
    R = np.sqrt(Xg**2 + Yg**2)
    Theta = np.mod(np.arctan2(Yg, Xg), 2*np.pi)

    bins = len(theta_grid)
    idx = (Theta / (2*np.pi) * bins).astype(int)
    idx = np.clip(idx, 0, bins - 1)

    radial = np.exp(-0.5 * ((R - 1.0) / sigma_r)**2)
    img = density[idx] * radial
    img[R > 1.35] = 0.0
    return img

def plot_embedding_scatter(
    X2,
    y=None,
    figsize=(6, 6),
    alpha=0.7,
    s=20,
    title="Embedding Scatter (2D)",
    cmap="tab10",
    save_path=None,
):
    X2 = _to_numpy(X2)
    assert X2.shape[1] == 2, "X2 must be 2D embeddings."

    plt.figure(figsize=figsize)

    if y is None:
        plt.scatter(X2[:, 0], X2[:, 1], s=s, alpha=alpha, color='steelblue')
    else:
        y = _labels_1d(y)
        classes = np.unique(y)
        cmap_obj = plt.get_cmap(cmap)
        for i, c in enumerate(classes):
            mask = (y == c)
            plt.scatter(
                X2[mask, 0], X2[mask, 1],
                s=s, alpha=alpha,
                color=cmap_obj(i / len(classes)),
                label=f"Class {c}"
            )
        plt.legend(loc="best", frameon=False)

    plt.xlabel("Component 1", fontsize=12)
    plt.ylabel("Component 2", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(alpha=0.3)
    plt.axis("equal")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_kde_unitcircle_kde(
    X: Union[np.ndarray, "torch.Tensor"],
    y: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    classes: Optional[Sequence] = None,
    theta_bins: int = 720,
    img_size: int = 420,
    sigma_r: float = 0.16,
    kde_bw: Optional[float] = None,
    ncols: int = 4,
    cmap: str = "inferno",
    titlesize: int = 28,
    overall_title: str = "Uniformity\nFeature Distribution",
    figsize_per_panel: float = 3.1,
    save_path: Optional[str] = None,
    show: bool = True,
    plot_scatter: bool = False,
    reduction_method: str = "pca", # <--- 新增參數, 可選 "pca", "umap", "tsne"
):
    """
    X: (N,D) embeddings；y: (N,) 整數或可 hash 的標籤（None 時只畫 Overall）。
    classes: 要畫哪些類別（None=自動由 y 的 unique 排序）
    sigma_r: 圓環厚度（徑向高斯的標準差）
    kde_bw: KDE 的 bandwidth（None=自動）
    """
    if save_path is None:
        save_path = f"./{reduction_method}/embedding_kde_unit_circle.png"
    os.makedirs(save_path.rsplit("/", 1)[0], exist_ok=True)

    X = _to_numpy(X)
    # X2 = _pca_2d(X)
    if reduction_method == "umap":
        X2 = _umap_2d(X)
        method_name = "UMAP"
    elif reduction_method == "tsne":
        X2 = _tsne_2d(X)
        method_name = "t-SNE"
    elif reduction_method == "pca":
        X2 = _pca_2d(X)
        method_name = "PCA"
    else:
        raise ValueError(f"Unknown reduction method: {reduction_method}. Choose 'pca', 'umap', or 'tsne'.")
    U, theta_all = _unit_and_theta(X2)
    if plot_scatter:
        scatter_path = save_path.replace("_kde_unit_circle", f"_{method_name.lower()}_scatter")
        plot_embedding_scatter(X2, y=y, title=f"Embedding Scatter ({method_name})", save_path=scatter_path)

    # 先做 Overall
    t_grid_all, dens_all = _kde_theta(theta_all, grid_bins=theta_bins, bw=kde_bw)
    img_overall = _ring_image(t_grid_all, dens_all, img_size=img_size, sigma_r=sigma_r)
    overall_title = f"{method_name} - {overall_title}"

    panels = [(overall_title, img_overall)]

    # 再做 per-class
    if y is not None:
        y = _labels_1d(y)
        if classes is None:
            classes = list(np.unique(y))
        else:
            classes = list(_to_numpy(classes).reshape(-1))
        for c in classes:
            mask = (y == c)
            theta_c = theta_all[np.asarray(mask, dtype=bool).ravel()]
            if len(theta_c) == 0:
                # 空類別，放空白
                panels.append((f"Class {c}", np.zeros((img_size, img_size))))
                continue
            t_grid, dens = _kde_theta(theta_c, grid_bins=theta_bins, bw=kde_bw)
            img = _ring_image(t_grid, dens, img_size=img_size, sigma_r=sigma_r)
            panels.append((f"Class {c}", img))
    
    n = len(panels)
    ncols = min(ncols, n)
    nrows = ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_panel*ncols, figsize_per_panel*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    k = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if k < n:
                title, img = panels[k]
                ax.imshow(img, origin="lower",
                          extent=[-1.15, 1.15, -1.15, 1.15],
                          cmap=cmap, norm=PowerNorm(5))
                # 外框圓
                circle = plt.Circle((0, 0), 1.0, fill=False, linewidth=1.0, alpha=0.45, color="k")
                ax.add_patch(circle)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
                ax.set_title(title, fontsize=titlesize, pad=10)
            else:
                fig.delaxes(ax)
            k += 1

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)