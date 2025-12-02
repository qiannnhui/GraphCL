import torch
import math
from typing import Literal, Tuple

def get_plane_rotation_matrix(D: int, angle_rad: float, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    根據兩個正交的單位向量 u, v 和角度 angle_rad，建立 D x D 的平面旋轉矩陣。
    
    使用 Givens 旋轉的推廣形式：R = I + (cos(theta) - 1) * (u*u^T + v*v^T) + sin(theta) * (v*u^T - u*v^T)
    
    Args:
        D (int): 維度 D。
        angle_rad (float): 旋轉角度 (弧度)。
        u (torch.Tensor): 第一個單位向量 (D, 1)。
        v (torch.Tensor): 第二個單位向量 (D, 1)。
        
    Returns:
        torch.Tensor: D x D 的旋轉矩陣 R。
    """
    device = u.device
    I = torch.eye(D, device=device)
    cos_theta = torch.cos(angle_rad)
    sin_theta = torch.sin(angle_rad)
    
    # 創建投影項 P_uv = u*u^T + v*v^T (投影到平面 P_uv)
    P_uv = torch.matmul(u, u.T) + torch.matmul(v, v.T)
    
    # 創建斜對稱項 A_vu = v*u^T - u*v^T (用於平面內的旋轉)
    A_vu = torch.matmul(v, u.T) - torch.matmul(u, v.T)
    
    # R = I + (cos(theta) - 1) * P_uv + sin(theta) * A_vu
    R = I + (cos_theta - 1.0) * P_uv + sin_theta * A_vu
    
    return R

def rotate_embedding_high_dim_by_angle(
    x: torch.Tensor, 
    angle_degree: float, 
    random_plane: bool = True
) -> torch.Tensor:
    """
    對輸入向量 x 進行指定角度的高維平面旋轉。
    
    Args:
        x (torch.Tensor): 輸入向量，形狀為 (batch_size, D)。
        angle_degree (float): 指定的旋轉角度（度）。
        random_plane (bool): True 表示隨機選取旋轉平面，False 則使用前兩維 (u=[1,0,..], v=[0,1,..])。
                                        
    Returns:
        torch.Tensor: 旋轉後的向量 x_rotated，形狀與 x 相同。
    """
    
    device = x.device
    D = x.size(1) # 維度
    batch_size = x.size(0)
    angle_rad = torch.deg2rad(torch.tensor(angle_degree, device=device))
    
    if D < 2:
        raise ValueError("Rotation requires dimension D >= 2.")
    
    # 1. 決定旋轉平面 (由兩個正交單位向量 u 和 v 定義)
    if random_plane:
        # A. 隨機選取平面 (使用兩個隨機正交向量)
        
        # 1.1 生成兩個隨機標準常態分佈向量
        r1 = torch.randn(D, 1, device=device)
        r2 = torch.randn(D, 1, device=device)
        
        # 1.2 Gram-Schmidt 過程得到兩個正交單位向量 u, v
        # u = r1 / ||r1||
        u = r1 / torch.linalg.norm(r1)
        
        # r2_proj_u = (r2 . u) * u
        r2_proj_u = torch.matmul(r2.T, u) * u
        
        # v_raw = r2 - r2_proj_u
        v_raw = r2 - r2_proj_u
        
        # v = v_raw / ||v_raw||
        # 檢查 v_raw 範數是否接近零 (如果 r1, r2 恰好共線)
        if torch.linalg.norm(v_raw) < 1e-6:
             # 如果共線，則使用第一個標準基 E1=[1,0...] 和第二個標準基 E2=[0,1...]
             # 或者我們可以簡單地重新生成 r1, r2，但為了避免迴圈，我們使用標準基
             u = torch.zeros(D, 1, device=device); u[0] = 1.0
             v = torch.zeros(D, 1, device=device); v[1] = 1.0
        else:
             v = v_raw / torch.linalg.norm(v_raw)

    else:
        # B. 固定使用前兩個維度作為平面 (即 X-Y 平面)
        u = torch.zeros(D, 1, device=device); u[0] = 1.0
        v = torch.zeros(D, 1, device=device); v[1] = 1.0

    # 2. 建立 D x D 旋轉矩陣 R
    R_matrix = get_plane_rotation_matrix(D, angle_rad, u, v)
    
    # 3. 執行旋轉
    # 旋轉公式: x_rotated = x @ R_matrix (我們對整個 batch 使用同一個 R)
    x_rotated = torch.matmul(x, R_matrix)
    
    return x_rotated

# 範例使用：
# D = 128
# x = torch.randn(64, D, device='cuda') # 64個樣本，每個128維

# # 範例 1: 隨機選取平面，旋轉 30 度
# x_aug_random_plane = rotate_embedding_high_dim_by_angle(x, angle_degree=30.0, random_plane=True)

# # 範例 2: 固定在前兩維 (X-Y平面) 旋轉 90 度 (與您最初的 2D 旋轉類似)
# x_aug_fixed_plane = rotate_embedding_high_dim_by_angle(x, angle_degree=90.0, random_plane=False)



import torch
import torch.linalg
from typing import Tuple

def get_random_orthogonal_matrix(D: int, device: torch.device) -> torch.Tensor:
    """
    生成一個 D x D 的隨機正交矩陣 (旋轉矩陣)，使用 Haar 測度。
    
    這是通過 QR 分解一個標準常態分佈 (Standard Normal Distribution) 的矩陣來實現的。
    
    Args:
        D (int): 矩陣的維度。
        device (torch.device): 矩陣所在的設備 (e.g., 'cuda', 'cpu')。
        
    Returns:
        torch.Tensor: 一個形狀為 (D, D) 的隨機正交矩陣 R。
    """
    # 1. 生成一個 D x D 的隨機矩陣，其元素來自標準常態分佈 N(0, 1)
    # 這是生成 Haar 測度下隨機正交矩陣的標準方法 (Mezzadri's Algorithm)
    G = torch.randn(D, D, device=device)
    
    # 2. 執行 QR 分解: G = Q * R (這裡的 R 是上三角矩陣，Q 是正交矩陣)
    Q, R_upper = torch.linalg.qr(G)
    
    # 3. 調整符號:
    # QR 分解的結果 Q 不一定是完全隨機的，我們需要通過 R 的對角線元素符號來調整 Q，
    # 確保 Q 的分佈是均勻的 (Haar measure)。
    # Mezzadri's Algorithm 建議使用 diag(R) 的符號。
    diag_sign = torch.diag(R_upper).sign()
    
    # 對於非零的對角線元素，我們將其符號應用到 Q 的對應列上。
    # 為了避免浮點誤差，我們確保所有元素都是 1 或 -1。
    diag_sign[diag_sign == 0] = 1.0 # 避免 0 
    
    # Q_final = Q * diag(diag_sign)
    Q_final = Q * diag_sign.unsqueeze(0)
    
    return Q_final # Q_final 即為隨機正交矩陣 R

def rotate_embedding_high_dim(
    x: torch.Tensor, 
    R: torch.Tensor = None,
    rotation_type: Literal['random', 'fixed'] = 'random'
) -> torch.Tensor:
    """
    使用一個 D x D 的正交矩陣對整個 D 維向量 x 進行旋轉。
    
    Args:
        x (torch.Tensor): 輸入向量，形狀為 (batch_size, D)。
        R (torch.Tensor, optional): 如果 rotation_type='fixed'，則必須提供一個 (D, D) 的正交矩陣。
        rotation_type (str): 'random' 使用隨機生成的新矩陣，'fixed' 使用提供的 R 矩陣。
                                        
    Returns:
        torch.Tensor: 旋轉後的向量 x_rotated，形狀與 x 相同。
    """
    
    device = x.device
    D = x.size(1) # 維度
    batch_size = x.size(0)
    
    if rotation_type == 'random':
        # 1. 隨機生成一個 D x D 的正交矩陣 R
        R_matrix = get_random_orthogonal_matrix(D, device)
    elif rotation_type == 'fixed':
        if R is None or R.shape != (D, D):
            raise ValueError("When rotation_type='fixed', R must be provided and have shape (D, D).")
        R_matrix = R
    else:
        raise ValueError(f"Unknown rotation_type: {rotation_type}")
    
    # 2. 執行旋轉
    # 旋轉公式: x_rotated = x @ R_matrix
    # 這是因為 x 的形狀是 (B, D)，R_matrix 的形狀是 (D, D)。
    # torch.matmul( (B, D), (D, D) ) -> (B, D)
    x_rotated = torch.matmul(x, R_matrix)
    
    return x_rotated

# 您可以在您的訓練迴圈或資料增強步驟中這樣使用它：
# # 假設 x 是你的 (B, D) 嵌入向量
# x = torch.randn(64, 128) 
# x_aug = rotate_embedding_high_dim(x, rotation_type='random') 

# # 驗證長度不變 (旋轉特性)
# assert torch.allclose(x.norm(dim=1), x_aug.norm(dim=1))



# import torch
# import math
# from typing import Literal, Tuple

# （保留原有的 get_anchor_aug_theta_degree, create_tptn_masks, get_similarity_matrix, sample_pairs_by_label_mask 和 unified_loss 函數）

def rotate_embedding_targeted_angle(
    x: torch.Tensor, 
    angle_degree: float
) -> torch.Tensor:
    """
    對批次中的每個向量 x_i 進行旋轉，使得旋轉後的向量 x_aug_i 與 x_i 之間的夾角
    精確等於目標角度 angle_degree。旋轉平面是隨機選擇的。
    
    Args:
        x (torch.Tensor): 輸入向量，形狀為 (batch_size, D)。
        angle_degree (float): 指定的目標旋轉角度（度）。
                                        
    Returns:
        torch.Tensor: 旋轉後的向量 x_rotated，形狀與 x 相同。
    """
    
    device = x.device
    B, D = x.shape
    
    # 1. 角度轉換
    angle_rad = torch.deg2rad(torch.tensor(angle_degree, device=device))
    
    # 2. 為每個 x_i 尋找一個隨機的正交單位向量 y_i
    
    # 2.1. 隨機生成一個與 x 同形狀的參考向量 P (B, D)
    P = torch.randn_like(x)
    
    # 2.2. 使用 Gram-Schmidt 過程求出正交分量 (y_raw)
    
    # 計算 x 的範數 (B, 1)
    x_norm = x.norm(dim=1, keepdim=True) + 1e-8 # 避免除以零
    x_unit = x / x_norm # x 的單位向量 (B, D)
    
    # P 在 x 上的投影: proj_x P = (P . x_unit) * x_unit
    # P . x_unit: 內積 (B, D) * (B, D) -> sum(dim=1) -> (B, 1)
    proj_x_P_scalar = (P * x_unit).sum(dim=1, keepdim=True) # (B, 1)
    proj_x_P = proj_x_P_scalar * x_unit                    # (B, D)
    
    # 正交分量: y_raw = P - proj_x_P
    y_raw = P - proj_x_P # (B, D)
    
    # 2.3. 將 y_raw 單位化得到 y_unit (與 x_unit 正交的單位向量)
    y_unit_norm = y_raw.norm(dim=1, keepdim=True) + 1e-8
    y_unit = y_raw / y_unit_norm # (B, D)
    
    # 3. 在 x_unit 和 y_unit 定義的平面上，將 x 旋轉 angle_rad
    
    # 旋轉公式: x_aug = cos(theta) * x_unit + sin(theta) * y_unit
    # 這是標準的 2D 向量旋轉公式，但現在基於高維的正交基 {x_unit, y_unit} 進行
    # 為了保持原長度，我們必須將單位向量結果乘以原長度 x_norm
    
    cos_theta = torch.cos(angle_rad)
    sin_theta = torch.sin(angle_rad)
    
    # (B, D) = (1) * (B, D) + (1) * (B, D)
    x_aug_unit = cos_theta * x_unit + sin_theta * y_unit
    
    # 4. 恢復原長度
    # x_aug = x_aug_unit * ||x||
    x_aug = x_aug_unit * x_norm
    
    return x_aug