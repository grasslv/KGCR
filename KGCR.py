import math
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from torch_scatter import scatter_mean
from torch_geometric.utils import remove_self_loops, add_self_loops, degree, scatter_, dropout_adj, softmax
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter_


##########################################################################
# 边列表GCN
# class GCN(MessagePassing):
#     def __init__(self, in_channels, out_channels, edge_index, num_node, aggr='add', bias=True):
#         super(GCN, self).__init__(aggr)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.edge_index = edge_index
#         self.num_node = num_node
#         row, col = edge_index
#         deg = degree(row, num_node)
#         deg_inv_sqrt = deg.pow(-0.5)
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
#         self.norm = norm.view(-1, 1)

#     def forward(self, x):
#         return self.propagate(edge_index=self.edge_index, size=None, x=x, edge_weight=None, res_n_id=None)

#     def message(self, x_i, x_j):
#         return self.norm*x_j

#     def update(self, aggr_out, x):
#         return aggr_out

# 稀疏矩阵GCG
class GCN(MessagePassing):
    def __init__(self, in_channels, out_channels, sparse_matrix, num_node, aggr='add', bias=True):
        super(GCN, self).__init__(aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sparse_matrix = sparse_matrix
        row, col = sparse_matrix.indices()
        deg = degree(row, num_node)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        self.norm = norm.view(-1, 1)

    def forward(self, x):
        return self.propagate(edge_index=self.sparse_matrix.indices(),
                              size=self.sparse_matrix.size(),
                              x=x,
                              edge_weight=self.sparse_matrix.values())

    def message(self, x_i, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out, x):
        return aggr_out


class KGCR(torch.nn.Module):
    # TODO: 1. KGCR 方法部分不用大概，GCN都是现成的，需要加那里的图卷积就直接加就好了，不需要这么复杂
    # TODO: 2. Sparse 矩阵那里直接from_dense，不需要这么大费周章
    # ia:u-u
    # ua:i-i
    def __init__(self, num_u, num_i, num_a, ui_data, ia_data, ua_data, reg_weight, dim_E, alpha, beta, margin):
        super(KGCR, self).__init__()
        print('KGCR')
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.num_u = num_u
        self.num_i = num_i
        self.dim_E = dim_E
        self.reg_weight = reg_weight

        # u-i ----- u-i
        # i-a ----- u-u
        # u-a ----- i-i
        # 边列表
        # self.ui_edge_index = self.edge_index_gen(ui_data)
        # self.ia_edge_index = self.edge_index_gen(ia_data)
        # self.ua_edge_index = self.edge_index_gen(ua_data)
        # 稀疏矩阵
        self.ui_sparse_matrix = self.sparse_matrix_gen(ui_data, num_u, num_i)
        self.ia_sparse_matrix = self.sparse_matrix_gen(ia_data, num_u, num_u)
        self.ua_sparse_matrix = self.sparse_matrix_gen(ua_data, num_i, num_i)

        self.id_embedding = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_u + num_i, dim_E))))
        self.user_pre = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_u, dim_E))))
        self.item_pre = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_i, dim_E))))
        self.attribute = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_a, dim_E))))

        # 边列表
        # self.UI_GCN = GCN(dim_E, dim_E, self.ui_edge_index, num_u+num_i)         
        # self.IA_GCN = GCN(dim_E, dim_E, self.ia_edge_index, num_u+num_i)         
        # self.UA_GCN = GCN(dim_E, dim_E, self.ua_edge_index, num_i+num_i)  
        # 稀疏矩阵
        self.UI_GCN = GCN(dim_E, dim_E, self.ui_sparse_matrix, num_u + num_i)
        self.IA_GCN = GCN(dim_E, dim_E, self.ia_sparse_matrix, num_u + num_u)
        self.UA_GCN = GCN(dim_E, dim_E, self.ua_sparse_matrix, num_i + num_i)

        self.u_result = nn.init.xavier_normal_(torch.rand((num_u, dim_E))).cuda()
        self.hat_u_result = nn.init.xavier_normal_(torch.rand((num_u, dim_E))).cuda()
        self.i_result = nn.init.xavier_normal_(torch.rand((num_i, dim_E))).cuda()
        self.hat_i_result = nn.init.xavier_normal_(torch.rand((num_i, dim_E))).cuda()
        self.result = nn.init.xavier_normal_(torch.rand((num_u + num_i, dim_E))).cuda()

        # TODO: 没用的变量删除
        self.trans_mlp = nn.Linear(dim_E, dim_E, bias=False)
        # self.margin_critic = nn.TripletMarginLoss()

    # 边列表
    # def edge_index_gen(self, data):
    #     temp = torch.LongTensor(data).t()
    #     edge_index = torch.LongTensor(data).t().contiguous().cuda()
    #     edge_index = torch.cat((edge_index, edge_index[[1,0]]), dim=1)
    #     edge_index, _ = remove_self_loops(edge_index)
    #     return edge_index 
    # 稀疏矩阵
    def sparse_matrix_gen(self, data, num_row, num_col):
        # 获取源节点和目标节点
        row = torch.tensor(data[:, 0], dtype=torch.long).cuda()
        col = torch.tensor(data[:, 1], dtype=torch.long).cuda()
        values = torch.ones(data.shape[0]).cuda()  # 默认权重为 1
        # 创建稀疏矩阵
        num_node = num_col + num_row
        sparse_matrix = torch.sparse_coo_tensor(torch.stack([row, col]), values, (num_node, num_node)).cuda()
        sparse_matrix = sparse_matrix.coalesce()
        return sparse_matrix

    def forward(self, user_tensor, item_tensor):
        ############################################################################
        # u-a ---- i-i
        # ua_rep_0 = torch.cat((self.user_pre, self.attribute), dim=0)
        ua_rep_0 = torch.cat((self.item_pre, self.item_pre), dim=0)
        ua_rep_1 = self.UA_GCN(ua_rep_0)
        # ua_rep_2 = self.UA_GCN(ua_rep_1)
        ua_rep = (ua_rep_0 + ua_rep_1) / 2
        ############################################################################
        user_tensor = user_tensor.view(-1)
        item_tensor = item_tensor.view(-1)

        # u-i
        ui_rep_1 = self.UI_GCN(self.id_embedding)
        ui_rep_2 = self.UI_GCN(ui_rep_1)
        ui_rep_3 = self.UI_GCN(ui_rep_2)
        ui_rep = (self.id_embedding + ui_rep_1 + ui_rep_2 + ui_rep_3) / 4

        ############################################################################
        # i-a ---- u-u
        # ia_rep_0 = torch.cat((self.item_pre, self.attribute), dim=0)
        ia_rep_0 = torch.cat((self.user_pre, self.user_pre), dim=0)
        ia_rep_1 = self.IA_GCN(ia_rep_0)
        # ia_rep_2 = self.IA_GCN(ia_rep_1)
        ia_rep = (ia_rep_0 + ia_rep_1) / 2

        u_id_embed = ui_rep[user_tensor]
        i_id_embed = ui_rep[item_tensor]
        cf_score = torch.sum(u_id_embed * i_id_embed, dim=1).view(-1, 2)
        cf_pos_score = cf_score[:, 0]
        cf_neg_score = cf_score[:, 1]
        ############################################################################

        # u_rep = ua_rep[user_tensor]
        # i_rep = ia_rep[item_tensor-self.num_u]
        u_rep = ia_rep[user_tensor]
        i_rep = ui_rep[item_tensor - self.num_u]
        kg_score = torch.sum(u_rep * i_rep, dim=1).view(-1, 2)
        kg_pos_score = kg_score[:, 0]
        kg_neg_score = kg_score[:, 1]

        ############################################################################

        # all_hat_u_rep = torch.sparse.mm(self.att_weight, self.attribute)

        # 边列表
        # all_hat_u_rep = torch.tensor(scatter_('mean', ua_rep_0[self.ua_edge_index[1]], self.ua_edge_index[0]))

        # all_hat_u_rep = torch.tensor(scatter_('mean', ua_rep_0[self.ua_sparse_matrix.indices()[1]], self.ua_sparse_matrix.indices()[0]))
        # 稀疏矩阵
        # all_hat_u_rep = scatter_('mean', ua_rep_0[self.ua_sparse_matrix.indices()[1]], self.ua_sparse_matrix.indices()[0]).clone().detach().requires_grad_(True)
        all_hat_u_rep = scatter_('mean', ia_rep_0[self.ia_sparse_matrix.indices()[1]],
                                 self.ia_sparse_matrix.indices()[0]).clone().detach().requires_grad_(True)

        hat_u_rep = all_hat_u_rep[user_tensor]

        # all_hat_i_pre = ia_rep      #self.trans_mlp(ia_rep)#torch.tensor(scatter_('mean', ia_rep_0[self.ia_edge_index[1]], self.ia_edge_index[0]))#.cuda()
        # all_hat_i_pre = ia_rep_1
        all_hat_i_pre = ui_rep
        hat_i_rep = all_hat_i_pre[item_tensor - self.num_u]

        hat_score = torch.sum(hat_u_rep * hat_i_rep, dim=1).view(-1, 2)
        hat_pos_score = hat_score[:, 0]
        hat_neg_score = hat_score[:, 1]

        ############################################################################

        pos_score = cf_pos_score
        kg_pos_score = kg_pos_score * torch.sigmoid(hat_pos_score)
        neg_score = cf_neg_score
        kg_neg_score = kg_neg_score * torch.sigmoid(hat_neg_score)

        ############################################################################
        # print('-'*20)
        # print(cf_pos_score.max().item(), cf_neg_score.max().item(), cf_pos_score.mean().item(), cf_neg_score.mean().item(), cf_pos_score.min().item(), cf_neg_score.min().item())
        # print(kg_pos_score.max().item(), kg_neg_score.max().item(), kg_pos_score.mean().item(), kg_neg_score.mean().item(), kg_pos_score.min().item(), kg_neg_score.min().item())
        # print(hat_pos_score.max().item(), hat_neg_score.max().item(), hat_pos_score.mean().item(), hat_neg_score.mean().item(), hat_pos_score.min().item(), hat_neg_score.min().item())

        score = torch.sigmoid(pos_score + kg_pos_score - neg_score - kg_neg_score)
        clamped_score = torch.clamp(score, min=1e-6)
        loss1 = -torch.mean(torch.log(clamped_score))
        # loss1 = -torch.mean(torch.log(torch.sigmoid(pos_score+kg_pos_score - neg_score-kg_neg_score)))

        # loss2 = -torch.mean(torch.log(torch.sigmoid(pos_score+hat_pos_score - neg_score-hat_neg_score)))
        # loss2 = -torch.mean(torch.log(torch.sigmoid(hat_pos_score - hat_neg_score)))
        # loss2 = torch.mean(F.relu(torch.sigmoid(hat_neg_score)-torch.sigmoid(hat_pos_score) + self.margin))
        zeros = torch.zeros(hat_pos_score.size()).cuda()

        sigmod_hat_pos = torch.sigmoid(hat_pos_score)
        sigmod_hat_neg = torch.sigmoid(hat_neg_score)
        clapmed_hat_pos_score = torch.clamp(sigmod_hat_pos, min=1e-6)
        clapmed_hat_neg_score = torch.clamp(sigmod_hat_neg, min=1e-6)
        loss2 = torch.mean(torch.max(clapmed_hat_pos_score - clapmed_hat_neg_score - self.margin, zeros))
        # loss2 = torch.mean(torch.max(torch.sigmoid(hat_pos_score)-torch.sigmoid(hat_neg_score)-self.margin, zeros))        

        loss = loss1 + self.alpha * loss2

        reg_loss = (self.id_embedding ** 2).mean() + (self.user_pre ** 2).mean() + (
                    self.item_pre ** 2).mean()  # + (self.attribute**2).mean()
        ############################################################################

        # self.ua_rep = ua_rep
        # self.ia_rep = ia_rep
        # self.result = ui_rep
        # self.u_result = ua_rep[:self.num_u]
        # self.hat_u_result = all_hat_u_rep
        # self.hat_i_result = all_hat_i_pre[:self.num_i]
        # self.i_result = ia_rep[:self.num_i]
        self.ua_rep = ua_rep
        self.ia_rep = ia_rep
        self.result = ui_rep
        self.u_result = ia_rep[:self.num_u]
        self.hat_u_result = all_hat_u_rep
        self.hat_i_result = all_hat_i_pre[-self.num_i:]
        self.i_result = ui_rep[-self.num_i:]
        return loss, reg_loss

    def loss(self, user_tensor, item_tensor):
        bpr_loss, reg_loss = self.forward(user_tensor, item_tensor)
        reg_loss = self.reg_weight * reg_loss
        return reg_loss + bpr_loss, bpr_loss, reg_loss
