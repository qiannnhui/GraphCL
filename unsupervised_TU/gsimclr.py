def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
# from core.encoders import *

# from torch_geometric.datasets import TUDataset
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *
from plot_theta_l2_scatter import plot_theta_l2, plot_theta_l2_epoch
from plot_theta_l2_single_anchor_distribution import plot_theta_l2_distribution

from arguments import arg_parse
from torch_geometric.transforms import Constant


class GcnInfomax(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(GcnInfomax, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)
    # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
    # self.global_d = MIFCNet(self.embedding_dim, mi_units)

    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index, batch, num_graphs):

    # batch_size = data.num_graphs
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)
    
    g_enc = self.global_d(y)
    l_enc = self.local_d(M)

    mode='fd'
    measure='JSD'
    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)
 
    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
    
    return local_global_loss + PRIOR


class simclr(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(simclr, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index, batch, num_graphs):

    # batch_size = data.num_graphs
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)
    
    y = self.proj_head(y)
    
    return y

  def loss_cal(self, x, x_aug):

    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()

    return loss

  def reweight_loss(self, z_a, z_b, T=0.2, eps=1e-6):
    """
    • 正例 (i,i) 權重 = 0 ─ 僅影響分母
    • easy / hard / false 根據與正例 (i,i) 的 angle / distance 來定義
    """
    B = z_a.size(0)
    z_a_n, z_b_n = F.normalize(z_a, dim=1), F.normalize(z_b, dim=1)
    sim = (z_a_n @ z_b_n.T).clamp(-1 + 1e-7, 1 - 1e-7)  # (B,B)
    
    angle_mat = torch.acos(sim)                         # 弧度
    dist_mat = torch.cdist(z_a, z_b, p=2)               # L2 距離

    # === 每個 positive pair (i,i) 的角度與距離 ===
    pos_angle = angle_mat.diagonal().unsqueeze(1)       # (B,1)
    pos_dist = dist_mat.diagonal().unsqueeze(1)         # (B,1)

    # === 與正例比距離/角度差異來定義幾何關係 ===
    close_theta = angle_mat <= pos_angle                # 角度更小
    close_dist  = dist_mat  <= pos_dist                 # 距離更近

    false_mask =  close_theta &  close_dist             # (更靠近正例)
    easy_mask  = ~close_theta & ~close_dist             # (明顯遠離)
    hard_mask  =  close_theta ^  close_dist             # (模糊地帶)
    inter_1 = (false_mask & easy_mask).any()
    inter_2 = (false_mask & hard_mask).any()
    inter_3 = (easy_mask  & hard_mask).any()
    assert not (inter_1 or inter_2 or inter_3), "Mask overlap detected!"
    # === 權重矩陣 W (for denominator) ===
    W = torch.full_like(sim, 3.23455410304745)               # 初始化為 easy
    # W[hard_mask]  = 3.23455410304745                         # 中等懲罰
    W[false_mask] = 0.0                        # 高懲罰
    W.fill_diagonal_(0.0)                               # 正例不進分母

    # === 分子：正例 logit ===
    log_pos = (sim.diagonal() / T)                      # (B,)

    # === 分母：負例加權 + softmax ===
    logit_neg = torch.log(W + eps) + sim / T
    log_denom = torch.logsumexp(logit_neg, dim=1)       # (B,)

    return -(log_pos - log_denom).mean()

import random
def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def gen_simgcl_aug(emb):
    random_noise = torch.rand(emb.size()).to(emb.device)
    emb += torch.sign(emb) * torch.nn.functional.normalize(random_noise, p=2, dim=1) * 0.1 # 0.1 is the noise scale(self.eps)
    return emb

if __name__ == '__main__':
    
    args = arg_parse()
    setup_seed(args.seed)

    accuracies = {'val':[], 'test':[]}
    epochs = args.epochs
    log_interval = args.log_interval
    batch_size = args.batch_size
    patience = 2
    min_delta = 0.01
    aug_ratio = round(args.aug_ratio * 0.1, 1)
    loss_list = []
    loss_min = float('inf')
    stage_finish_epochs = []
    lr = args.lr
    DS = args.DS
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    path = osp.join(args.path, DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    dataset = TUDataset(path, name=DS, aug=args.aug, aug_ratio=aug_ratio).shuffle()
    dataset_eval = TUDataset(path, name=DS, aug='none').shuffle()
    print(len(dataset))
    print(dataset.get_num_feature())
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    # model.eval()
    # emb, y = model.encoder.get_embeddings(dataloader_eval)
    # print(emb.shape, y.shape)

    """
    acc_val, acc = evaluate_embedding(emb, y)
    accuracies['val'].append(acc_val)
    accuracies['test'].append(acc)
    """
    
    best_acc_val = 0
    for epoch in range(0, epochs+1):
        loss_all = 0
        model.train()

        if args.plot_theta_l2 and epoch % 50 == 0:
            all_pos_l2, all_pos_theta = [], []
            all_neg_l2, all_neg_theta = [], []
            all_real_pos_l2, all_real_pos_theta = [], []
            all_real_neg_l2, all_real_neg_theta = [], []

        for data in dataloader:

            # print('start')
            data, data_aug = data
            optimizer.zero_grad()

            
            node_num, _ = data.x.size()
            data = data.to(device)
            x = model(data.x, data.edge_index, data.batch, data.num_graphs)

            # if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4':
            #     # node_num_aug, _ = data_aug.x.size()
            #     edge_idx = data_aug.edge_index.numpy()
            #     _, edge_num = edge_idx.shape
            #     idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

            #     node_num_aug = len(idx_not_missing)
            #     data_aug.x = data_aug.x[idx_not_missing]

                

            #     data_aug.batch = data.batch[idx_not_missing]
            #     idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
            #     edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
            #     data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

            # data_aug = data_aug.to(device)

            '''
            print(data.edge_index)
            print(data.edge_index.size())
            print(data_aug.edge_index)
            print(data_aug.edge_index.size())
            print(data.x.size())
            print(data_aug.x.size())
            print(data.batch.size())
            print(data_aug.batch.size())
            pdb.set_trace()
            '''

            # x_aug = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)
            x_aug = gen_simgcl_aug(x)
            loss = model.loss_cal(x, x_aug) if args.loss == 'InfoNCE' else model.reweight_loss(x, x_aug)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
            if args.plot_theta_l2 and epoch % 50 == 0:
                # pos_l2, pos_theta, neg_l2, neg_theta = plot_theta_l2(x_anchor, x_graph_pos)
                pos_l2, pos_theta, neg_l2, neg_theta, real_pos_l2, real_pos_theta, real_neg_l2, real_neg_theta = plot_theta_l2(x, x_aug, data.y)
                all_pos_l2.append(pos_l2)
                all_pos_theta.append(pos_theta)
                all_neg_l2.append(neg_l2)
                all_neg_theta.append(neg_theta)
                all_real_pos_l2.append(real_pos_l2)
                all_real_pos_theta.append(real_pos_theta)
                all_real_neg_l2.append(real_neg_l2)
                all_real_neg_theta.append(real_neg_theta)
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader.dataset)))
        loss_list.append(loss_all / len(dataloader.dataset))

        if epoch % log_interval == 0:
            if args.plot_theta_l2 and epoch % 50 == 0:
                result = plot_theta_l2_epoch(all_pos_l2, all_pos_theta, all_neg_l2, all_neg_theta, all_real_pos_l2,
                                             all_real_pos_theta, all_real_neg_l2, all_real_neg_theta, args=args, epoch=epoch)
            if args.plot_theta_l2_distribution and epoch % 100 == 0:
                single_anchor_result = plot_theta_l2_distribution(x, x_aug, data.y, args=args, epoch=epoch)

            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            acc_val, acc = evaluate_embedding(emb, y)
            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                print(f"Epoch {epoch}: new best val accuracy: {best_acc_val:.4f}, saving model...")
                # os.makedirs(f'./logs/ckpt/{args.DS}', exist_ok=True)
                # torch.save(model.state_dict(), f'./logs/ckpt/{args.DS}/best_model_{aug_ratio}_{args.seed}.pth')

            

    tpe  = ('local' if args.local else '') + ('prior' if args.prior else '')
    # os.makedirs(f'./logs/single_anchor_dist/GCL/{args.DS}/{args.DS}_{aug_ratio}_{args.seed}', exist_ok=True)
    os.makedirs(f'./results/SimGCL/{args.DS}', exist_ok=True)

    # with open((f'./logs/single_anchor_dist/GCL/{args.DS}/{args.DS}_{aug_ratio}_'+str(args.seed)), 'a+') as f:
    with open(f'./results/SimGCL/{args.DS}/{args.loss}_SimGCL.log', 'a') as f:
        # s1 = json.dumps(stage_finish_epochs)
        # s2 = json.dumps(loss_list)
        # s3 = json.dumps(accuracies)
        # # s4 = json.dumps(result) if args.plot_theta_l2 else ''
        # f.write('{},{},{},{},{},{},{},{}\n'.format(args.DS, args.num_gc_layers, epochs, log_interval, lr, s1, s2, s3))
        # json.dump(single_anchor_result, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
        # s4 = json.dumps(single_anchor_result) if args.plot_theta_l2_distribution else ''
        f.write('\nFinal Test Accuracy: {} \n'.format(accuracies['test'][-1]))
        f.write('Best Test Accuracy: {} \n'.format(max(accuracies['test'])))
        f.write('Final Val Accuracy: {} \n'.format(accuracies['val'][-1]))
        f.write('Best Val Accuracy: {} \n'.format(max(accuracies['val'])))
        if args.plot_theta_l2:
            f.write('result: {}\n'.format(result))
        f.close()    
