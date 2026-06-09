import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from core.encoders import *
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *


class GlobalDiscriminator(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        
        self.l0 = nn.Linear(32, 32)
        self.l1 = nn.Linear(32, 32)

        self.l2 = nn.Linear(512, 1)
    def forward(self, y, M, data):

        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        # h0 = Variable(data['feats'].float()).cuda()
        batch_num_nodes = data['num_nodes'].int().numpy()
        M, _ = self.encoder(M, adj, batch_num_nodes)
        # h = F.relu(self.c0(M))
        # h = self.c1(h)
        # h = h.view(y.shape[0], -1)
        h = torch.cat((y, M), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)

class PriorDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = nn.Linear(input_dim, input_dim)
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))

class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1)
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1, stride=1, padding=0)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding=0)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class DecorrelatedShuffledBatchNorm1d(nn.Module):
    """
    Decorrelated and Shuffled Batch Normalization (DSBN).
    Computes DBN statistics (Mean and Covariance Inverse) on a shuffled batch 
    to prevent information leakage in contrastive learning.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        # Running Stats for Evaluation
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_cov_sqrt_inv', torch.eye(num_features))
        
        # Learnable Parameters (Gamma/Weight, Beta/Bias)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x shape: (B, D)
        
        if self.training:
            # --- 1. Shuffling (for Decorrelation with respect to positive pairs) ---
            batch_size = x.size(0)
            perm = torch.randperm(batch_size, device=x.device)
            x_shuffled = x[perm]
            
            # --- 2. DBN on Shuffled Batch (Compute Whitening Matrix) ---
            
            # Centering (Subtract Batch Mean of the shuffled batch)
            mean = x_shuffled.mean(dim=0)
            x_centered = x_shuffled - mean
            
            # Covariance Matrix (C)
            B = x_shuffled.size(0)
            # DBN's C is computed on the shuffled batch
            cov_matrix = (x_centered.T @ x_centered) / B
            
            # Whitening: Compute Inverse Square Root of Covariance (C^{-1/2}) using SVD
            U, S, _ = torch.svd(cov_matrix + self.eps * torch.eye(self.num_features, device=x.device))
            S_inv_sqrt = torch.diag(1.0 / torch.sqrt(S))
            cov_sqrt_inv = U @ S_inv_sqrt @ U.T
            
            # Decorrelation (Whitening)
            x_whitened = x_centered @ cov_sqrt_inv
            
            # --- 3. Update Running Stats (Using Shuffled Stats) ---
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                self.running_cov_sqrt_inv.mul_(1 - self.momentum).add_(cov_sqrt_inv, alpha=self.momentum)
                
            # --- 4. Restore Order ---
            out = x_whitened[torch.argsort(perm)] # Normalize the feature i, but using shuffled stats
            
        else: # Evaluation mode (using running stats, no shuffling)
            
            x_centered = x - self.running_mean
            x_whitened = x_centered @ self.running_cov_sqrt_inv
            out = x_whitened
        
        # --- 5. Affine Transformation ---
        if self.affine:
            out = out * self.weight + self.bias
            
        return out
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, mask=None, num_slots=None):
        # inputs: [B, N, D]
        # mask: [B, N] boolean mask (True for valid nodes)
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
        slots = mu + sigma * torch.randn(mu.shape, device=inputs.device)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            # q: [B, n_s, D]
            # k: [B, N, D]
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            # dots: [B, n_s, N]
            
            # attention is over slots
            attn_logits = dots
            
            attn = attn_logits.softmax(dim=1) + self.eps # [B, n_s, N]
            
            if mask is not None:
                attn = attn.masked_fill(~mask.unsqueeze(1), 0.0)
                
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, attn
