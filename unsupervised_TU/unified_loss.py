import torch
from typing import Literal

def get_all_pair_theta_degree(x, x_aug):
    # Calculate the angle in degrees between x and x_aug (positive anchor and its augmentation)
    x_norm = x / x.norm(dim=1, keepdim=True)
    x_aug_norm = x_aug / x_aug.norm(dim=1, keepdim=True)
    # cos_theta = (x_norm * x_aug_norm).sum(dim=1).clamp(-1.0, 1.0) # Original line for self-pairs
    cos_theta = torch.matmul(x_norm, x_aug_norm.T).clamp(-1.0, 1.0) # All pairwise cosine similarities
    theta_rad = torch.acos(cos_theta)
    theta_deg = torch.rad2deg(theta_rad)
    return theta_deg.detach()

def create_tptn_masks(labels, device):
    labels = labels.view(-1, 1)
    pos_all_mask = labels.eq(labels.T)
    diag_mask = torch.diag(torch.ones(labels.size(0), device=device, dtype=torch.bool))
    TP_mask = pos_all_mask & ~diag_mask
    TN_mask = ~pos_all_mask
    return TP_mask, TN_mask, diag_mask

def get_similarity_matrix(x, x_aug, similarity_measure="cosine", T=0.2):
        
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1) 

        if similarity_measure == "cosine":
            # Cosine similarity
            cos_sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
            cos_sim_matrix = torch.clamp(cos_sim_matrix, -1.0, 1.0)  # Clamp to avoid invalid values
            sim_matrix = torch.exp(cos_sim_matrix / T)

        elif similarity_measure == "l2":
            # L2 norm similarity
            l2_matrix = torch.cdist(x, x_aug, p=2)  # Pairwise L2 distance
            l2_matrix = l2_matrix / l2_matrix.max()  # Normalize L2 norm to [0, 1]
            sim_matrix = torch.exp(-l2_matrix / T)  # Convert distance to similarity

        elif similarity_measure == "cosine+l2":
            # Compute cosine similarity
            cos_sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
            cos_sim_matrix = torch.clamp(cos_sim_matrix, -1.0, 1.0)  # Clamp to avoid invalid values

            # Compute L2 norm
            l2_matrix = torch.cdist(x, x_aug, p=2)  # Pairwise L2 distance
            l2_matrix = l2_matrix / l2_matrix.max()  # Normalize L2 norm to [0, 1]

            # Combine cosine similarity and L2 norm
            combined_matrix = cos_sim_matrix - l2_matrix  # Example: subtract L2 norm from cosine similarity
            sim_matrix = torch.exp(combined_matrix / T)  # Convert to similarity

        else:
            raise ValueError(f"Unsupported similarity measure: {similarity_measure}")
        
        return sim_matrix

def sample_pairs_by_label_mask(
    labels: torch.Tensor,
    device: torch.device,
    sample_type: Literal['TP', 'TN'],
    num_samples: int
) -> torch.Tensor:
    """
    Sample number of positive (TP) or negative (TN) pairs based on labels.
    Args:
        labels (torch.Tensor): Tensor of shape (batch_size,) containing class labels.
        device (torch.device): Device to perform computations on.
        sample_type (str): 'TP' for true positives, 'TN' for true negatives.
        num_samples (int): Number of samples to draw for each anchor.
    Returns:
        torch.Tensor: A boolean mask of shape (batch_size, batch_size) indicating sampled pairs.
    """
    
    batch_size = labels.size(0)
    labels = labels.view(-1, 1)
    sample_mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
    
    TP_mask, TN_mask, _ = create_tptn_masks(labels, device)
    if sample_type == 'TP':
        potential_mask = TP_mask
    elif sample_type == 'TN':
        potential_mask = TN_mask
    else:
        raise ValueError(f"Unknown sample_type: {sample_type}")
    
    # sampling
    for i in range(batch_size):
        potential_indices = potential_mask[i, :].nonzero(as_tuple=True)[0]
        if potential_indices.numel() > 0:
            num_to_sample = min(num_samples, potential_indices.numel())
            if num_to_sample > 0:
                sampled_indices = potential_indices[torch.randperm(potential_indices.numel())[:num_to_sample]]
                sample_mask[i, sampled_indices] = True
        
    return sample_mask

def get_hard_sets(sim_matrix, labels, hard_sim_threshold):
    """
    根據餘弦相似度門檻和標籤，劃分樣本為 TP, FN, EN, HN, SP (五個子集)。
    - EP: Easy Positive (標籤相同 AND 相似度 >= 門檻 AND i != j)
    - HP: Hard Positive (標籤相同 AND 相似度 < 門檻 AND i != j)
    - EN: Easy Negative
    - HN: Hard Negative
    - SP: Self Positive (i == j)
    """
    B = sim_matrix.size(0)
    device = sim_matrix.device
    
    labels_col = labels.view(-1, 1)
    pos_label_mask = labels_col.eq(labels_col.T)
    neg_label_mask = ~pos_label_mask

    hard_sim_mask = sim_matrix >= hard_sim_threshold
    easy_sim_mask = sim_matrix < hard_sim_threshold
    diag_mask = torch.diag(torch.ones(B, device=device, dtype=torch.bool))
    
    SP_mask = diag_mask
    EP_mask = pos_label_mask & hard_sim_mask & ~diag_mask
    HP_mask = pos_label_mask & easy_sim_mask & ~diag_mask
    HN_mask = neg_label_mask & hard_sim_mask
    EN_mask = neg_label_mask & easy_sim_mask
    
    return {
        'SP': SP_mask,
        'EP': EP_mask,
        'HP': HP_mask,
        'HN': HN_mask,
        'EN': EN_mask,
    }

def get_pair_angles(
    x: torch.Tensor, 
    x_aug: torch.Tensor, 
    labels: torch.Tensor, 
    pair_type: Literal['TP', 'TN', 'Self', 'ALL_NoneSelf'] = 'Self') -> torch.Tensor:
    device = x.device
    theta_matrix = get_all_pair_theta_degree(x, x_aug)
    
    TP_mask, TN_mask, diag_mask = create_tptn_masks(labels, device)
    
    if pair_type == 'TP':
        mask = TP_mask
    elif pair_type == 'TN':
        mask = TN_mask
    elif pair_type == 'Self':
        mask = diag_mask
    elif pair_type == 'ALL_NoneSelf':
        mask = ~diag_mask
    else:
        raise ValueError(f"Unknown pair_type: {pair_type}. Must be 'TP', 'TN', 'ALL_NoneSelf, or 'Self'.")
        
    angles = theta_matrix[mask].detach()
    
    return angles

def unified_loss(x: torch.Tensor, x_aug: torch.Tensor, labels: torch.Tensor, T: float = 0.2,
                     sim_measure: str = "cosine",
                     pos_strategy: Literal['normal', 'sum_tp', 'sum_fp', 'sum_all_pos', 'sum_sample_tp', 'sum_sample_fp'] = 'normal', 
                     neg_strategy: Literal['normal', 'sum_tn', 'sum_fn', 'sum_sample_tn', 'sum_sample_fn', 'normalized_normal'] = 'normal',
                     pos_num_samples: int = 10,
                     neg_num_samples: int = 10,
                     neg_include_self: bool = False):
    
    device = x.device
    sim_matrix = get_similarity_matrix(x=x, x_aug=x_aug, similarity_measure=sim_measure, T=T)
    TP_mask, TN_mask, diag_mask = create_tptn_masks(labels, device)
    # Positives
    if pos_strategy == 'normal':
        # Standard InfoNCE Positive: P = S(x_i, x_i+)
        P_term = sim_matrix.diag()

    elif pos_strategy == 'sum_tp':
        # Sum of Labeled Pairs: P = Sum(TPs)
        P_term = (sim_matrix * TP_mask).sum(dim=1)

    elif pos_strategy == 'sum_fp':
        # Sum of Labeled Pairs: P = Sum(FPs/TNs)
        P_term = (sim_matrix * TN_mask).sum(dim=1)

    elif pos_strategy == 'sum_all_pos':
        # Sum of Labeled Pairs: P = Sum(TPs + Self)
        P_term = (sim_matrix * (TP_mask | diag_mask)).sum(dim=1)

    elif pos_strategy == 'sum_sample_tp':
        # Sum of Sampled Pairs: P = Sum(K random TPs)
        tp_mask_sample = sample_pairs_by_label_mask(labels, device, 'TP', pos_num_samples)
        P_term = (sim_matrix * tp_mask_sample).sum(dim=1)

    elif pos_strategy == 'sum_sample_fp':
        # Sum of Sampled Pairs: P = Sum(K random FPs/TNs)
        tn_mask_sample = sample_pairs_by_label_mask(labels, device, 'TN', pos_num_samples)
        P_term = (sim_matrix * tn_mask_sample).sum(dim=1)

    else:
        # Default case for error handling
        raise ValueError(f"Unknown pos_strategy: {pos_strategy}")
    
    # Negatives
    if neg_strategy == 'normal':
        # Standard InfoNCE Negative: N = Total Sum - P(Self)
        N_sum = sim_matrix.sum(dim=1) - sim_matrix.diag()

    elif neg_strategy == 'sum_tn':
        # Sum of Labeled Pairs: N = Sum(TNs)
        N_sum = (sim_matrix * TN_mask).sum(dim=1)

    elif neg_strategy == 'sum_fn':
        # Sum of Labeled Pairs: N = Sum(FNs/TPs)
        N_sum = (sim_matrix * TP_mask).sum(dim=1)

    elif neg_strategy == 'sum_sample_tn':
        # Sum of Sampled Pairs: N = Sum(K random TNs)
        tn_mask_sample = sample_pairs_by_label_mask(labels, device, 'TN', neg_num_samples)
        N_sum = (sim_matrix * tn_mask_sample).sum(dim=1)

    elif neg_strategy == 'sum_sample_fn':
        # Sum of Sampled Pairs: N = Sum(K random FNs/TPs)
        tp_mask_sample = sample_pairs_by_label_mask(labels, device, 'TP', neg_num_samples)
        N_sum = (sim_matrix * tp_mask_sample).sum(dim=1)

    elif neg_strategy == 'normalized_normal':
        # Normalized Sum: N = (Total Sum - P(Self)) / N
        N_sum_unnormalized = sim_matrix.sum(dim=1) - sim_matrix.diag()
        batch_size = labels.size(0)
        N_sum = N_sum_unnormalized / batch_size

    # # Normalized Sum of Labeled/Sampled Pairs (將所有註解掉的邏輯也轉換為 elif)
    # elif neg_strategy == 'normalized_sum_tn':
    #     # N = Avg(TNs)
    #     N_sum_unnormalized = (sim_matrix * TN_mask).sum(dim=1)
    #     total_tn_counts = TN_mask.sum(dim=1).clamp(min=1)
    #     N_sum = N_sum_unnormalized / total_tn_counts

    # elif neg_strategy == 'normalized_sum_fn':
    #     # N = Avg(FNs)
    #     N_sum_unnormalized = (sim_matrix * TP_mask).sum(dim=1)
    #     total_tp_counts = TP_mask.sum(dim=1).clamp(min=1)
    #     N_sum = N_sum_unnormalized / total_tp_counts

    # elif neg_strategy == 'normalized_sample_tn':
    #     # N = Estimated Sum(TNs)
    #     tn_mask_sample = sample_pairs_by_label_mask(labels, device, 'TN', neg_num_samples)
    #     actual_sampled_counts = tn_mask_sample.sum(dim=1).clamp(min=1)
    #     total_tn_counts = TN_mask.sum(dim=1)
    #     N_sum_sample = (sim_matrix * tn_mask_sample).sum(dim=1)
    #     N_sum = (N_sum_sample / actual_sampled_counts) * total_tn_counts
        
    # elif neg_strategy == 'normalized_sample_fn':
    #     # N = Estimated Sum(FNs)
    #     tp_mask_sample = sample_pairs_by_label_mask(labels, device, 'TP', neg_num_samples)
    #     actual_sampled_counts = tp_mask_sample.sum(dim=1).clamp(min=1)
    #     total_tp_counts = TP_mask.sum(dim=1)
    #     N_sum_sample = (sim_matrix * tp_mask_sample).sum(dim=1)
    #     N_sum = (N_sum_sample / actual_sampled_counts) * total_tp_counts

    else:
        # Default case for error handling
        raise ValueError(f"Unknown neg_strategy: {neg_strategy}")
    
    if neg_include_self:
        N_sum = N_sum + sim_matrix.diag()

    # Loss calculation    
    loss = P_term / (N_sum + 1e-8)
    loss = -torch.log(loss + 1e-8).mean()
    
    pos_sim = P_term.mean()
    neg_sim = N_sum.mean()

    return loss, pos_sim, neg_sim


def flexible_hard_mining_loss(
    x: torch.Tensor, 
    x_aug: torch.Tensor, 
    labels: torch.Tensor, 
    hard_sim_threshold: float,
    num_sets: str,
    den_sets: str,
    T: float = 0.2,
    sim_measure: str = "cosine",
    neg_include_self: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    基於相似度門檻劃分樣本集 (SP, EP, HP, HN, EN)，並根據 num_sets/den_sets 構建 InfoNCE 損失。
    
    Args:
        x, x_aug, labels, T, sim_measure: 標準 InfoNCE 參數。
        hard_sim_threshold (float): Hard/Easy 相似度門檻。
        num_sets (str): 分子包含的集合 (e.g., 'SP,EP')。
        den_sets (str): 分母包含的集合 (e.g., 'HN,EN,SP')。
    """
    eps = 1e-8
    device = x.device
    
    # --- 1. 獲取相似度矩陣 (使用餘弦相似度進行劃分) ---
    cos_sim_matrix = get_similarity_matrix(x=x, x_aug=x_aug, similarity_measure="cosine", T=1.0) # T=1.0 確保 cos_sim_matrix 是原始餘弦相似度
    
    # --- 2. 劃分 Hard/Easy 集合 ---
    labels_copy = labels.detach().clone() # 確保不影響原標籤張量
    masks = get_hard_sets(
        sim_matrix=cos_sim_matrix,
        labels=labels_copy, 
        hard_sim_threshold=hard_sim_threshold
    )

    # --- 3. 獲取指數相似度矩陣 (用於損失計算) ---
    # 根據指定的 sim_measure 和 T 重新計算 sim_matrix (即 exp(logit))
    sim_matrix = get_similarity_matrix(x=x, x_aug=x_aug, similarity_measure=sim_measure, T=T)
    
    # --- 4. 構建分子 (Numerator Mask) ---
    numerator_mask = torch.zeros_like(masks['SP'])
    pos_set_names = num_sets.split(',') 
    
    for name in pos_set_names:
        name = name.strip().upper()
        if name in masks:
            numerator_mask |= masks[name]
        elif name == 'ALL_POS':
             # 所有正標籤樣本 (i vs i', i vs j' label matched)
             numerator_mask |= (masks['SP'] | masks['EP'] | masks['HP'])
        else:
             print(f"Warning: Unknown numerator set '{name}'")
    
    # 計算分子總和
    P_term = (sim_matrix * numerator_mask.float()).sum(dim=1)

    # --- 5. 構建分母 (Denominator Mask) ---
    denominator_mask = torch.zeros_like(masks['SP'])
    neg_set_names = den_sets.split(',') 
    
    for name in neg_set_names:
        name = name.strip().upper()
        if name in masks:
            denominator_mask |= masks[name]
        elif name == 'ALL_NEG':
            # 所有負標籤樣本 (i vs j' label mismatched)
            denominator_mask |= (masks['HN'] | masks['EN'])
        elif name == 'ALL_OTHERS':
            # 所有非自我對比的樣本 (Standard InfoNCE Denominator)
            denominator_mask |= (~masks['SP'])
        else:
             print(f"Warning: Unknown denominator set '{name}'")
             
    # 由於 InfoNCE 的分母通常是 **Positive Term + Negative Terms** 的總和，
    # 這裡我們將自定義分母視為 Negative Terms 的總和 (N_sum)。

    # 計算分母總和
    N_sum = (sim_matrix * denominator_mask.float()).sum(dim=1)

    # 處理 neg_include_self (如果分母集不包含 SP，但用戶要求加入)
    if neg_include_self and 'SP' not in den_sets.upper().split(','):
        N_sum += sim_matrix.diag()

    # --- 6. 損失計算 ---
    
    # 為了遵循標準的 InfoNCE 結構 L = - log ( P_term / (P_term + N_sum) )
    # 我們將 P_term 視為錨點 i 的正項，N_sum 視為所有負項的總和。
    
    # Logit: P_term / (P_term + N_sum)
    loss_logit = P_term / (N_sum + eps)
    # loss_logit = P_term / (P_term + N_sum + eps)
    
    loss = -torch.log(loss_logit + eps).mean()
    
    pos_sim_avg = P_term.mean()
    neg_sim_avg = N_sum.mean()
    
    # 打印集合大小以供調試
    print(f"|SP|={masks['SP'].sum().item()}, |EP|={masks['EP'].sum().item()}, |HP|={masks['HP'].sum().item()}, |HN|={masks['HN'].sum().item()}, |EN|={masks['EN'].sum().item()}")

    return loss, pos_sim_avg, neg_sim_avg