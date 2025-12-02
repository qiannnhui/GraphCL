import torch
from typing import Literal

def get_anchor_aug_theta_degree(x, x_aug):
    # Calculate the angle in degrees between x and x_aug (positive anchor and its augmentation)
    x_norm = x / x.norm(dim=1, keepdim=True)
    x_aug_norm = x_aug / x_aug.norm(dim=1, keepdim=True)
    cos_theta = (x_norm * x_aug_norm).sum(dim=1).clamp(-1.0, 1.0)
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

    theta_degree = get_anchor_aug_theta_degree(x, x_aug)
    return loss, pos_sim, neg_sim, theta_degree
