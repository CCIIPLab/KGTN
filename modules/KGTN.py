
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum
import torch_sparse
import scipy.sparse as sp
from reckit import randint_choice
from utils.parser import parse_args

from utils.sample import _relation_aware_edge_sampling, _mae_edge_mask_adapt_mixed, norm_attn_computer
args = parse_args()




class GTLayer(nn.Module):
    def __init__(self):
        super(GTLayer, self).__init__()
        init = nn.init.xavier_uniform_

        self.qTrans = nn.Parameter(init(torch.empty(args.latdim, args.latdim)))
        self.kTrans = nn.Parameter(init(torch.empty(args.latdim, args.latdim)))
        self.vTrans = nn.Parameter(init(torch.empty(args.latdim, args.latdim)))

    def forward(self, adj, embeds):
        indices = adj._indices()
        rows, cols = indices[0, :], indices[1, :]
        rowEmbeds = embeds[rows]
        colEmbeds = embeds[cols]

        qEmbeds = (rowEmbeds @ self.qTrans).view([-1, args.head, args.latdim // args.head])
        kEmbeds = (colEmbeds @ self.kTrans).view([-1, args.head, args.latdim // args.head])
        vEmbeds = (colEmbeds @ self.vTrans).view([-1, args.head, args.latdim // args.head])

        att = torch.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = torch.clamp(att, -10.0, 10.0)
        expAtt = torch.exp(att)
        tem = torch.zeros([adj.shape[0], args.head]).cuda()
        attNorm = (tem.index_add_(0, rows, expAtt))[rows]
        att = expAtt / (attNorm + 1e-8)  # eh

        resEmbeds = torch.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, args.latdim])
        tem = torch.zeros([adj.shape[0], args.latdim]).cuda()
        resEmbeds = tem.index_add_(0, rows, resEmbeds)  # nd
        return resEmbeds

class KGlayer(nn.Module):
    def __init__(self):
        super(KGlayer, self).__init__()
        init = nn.init.xavier_uniform_

        self.qTrans = nn.Parameter(init(torch.empty(args.latdim, args.latdim)))
        self.kTrans = nn.Parameter(init(torch.empty(args.latdim, args.latdim)))
        self.vTrans = nn.Parameter(init(torch.empty(args.latdim, args.latdim)))

    def forward(self, rows, colEmbeds, embeds, shape):

        rowEmbeds = embeds[rows]
        # colEmbeds = embeds[cols]

        qEmbeds = (rowEmbeds @ self.qTrans).view([-1, args.head, args.latdim // args.head])
        kEmbeds = (colEmbeds @ self.kTrans).view([-1, args.head, args.latdim // args.head])
        vEmbeds = (colEmbeds @ self.vTrans).view([-1, args.head, args.latdim // args.head])

        att = torch.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = torch.clamp(att, -10.0, 10.0)
        expAtt = torch.exp(att)
        tem = torch.zeros([shape, args.head]).cuda()
        attNorm = (tem.index_add_(0, rows, expAtt))[rows]
        att = expAtt / (attNorm + 1e-8)  # eh

        resEmbeds = torch.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, args.latdim])
        tem = torch.zeros([shape, args.latdim]).cuda()
        resEmbeds = tem.index_add_(0, rows, resEmbeds)  # nd
        return resEmbeds



class Aggregator(nn.Module):
    def __init__(self, n_users):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.KGLayers = nn.Sequential(*[KGlayer() for i in range(args.gt_layer)])

    def forward(self, entity_emb, user_emb,
                edge_index, edge_type, interact_mat,
                weight, layer):

        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]

        # ------------calculate attention weights ---------------

        # graph transformer in KG layer
        kg_entity_emb = self.KGLayers[layer](head, neigh_relation_emb, entity_emb, n_entities)

        # attention in KG aggregation
        # neigh_relation_emb_weight = self.calculate_sim_hrt(entity_emb[head], entity_emb[tail],
        #                                                    weight[edge_type - 1])
        neigh_relation_emb_weight = self.calculate_sim_hrt(kg_entity_emb[head], kg_entity_emb[tail], weight[edge_type - 1])

        neigh_relation_emb_weight = neigh_relation_emb_weight.expand(neigh_relation_emb.shape[0],
                                                                     neigh_relation_emb.shape[1])
        # neigh_relation_emb_tmp = torch.matmul(neigh_relation_emb_weight, neigh_relation_emb)
        neigh_relation_emb_weight = scatter_softmax(neigh_relation_emb_weight, index=head, dim=0)
        neigh_relation_emb = torch.mul(neigh_relation_emb_weight, entity_emb[tail])
        entity_agg = scatter_sum(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        # entity_agg = kg_entity_emb

        # user_agg = torch.sparse.mm(interact_mat, entity_emb)
        # # user_agg = user_agg + user_emb * user_agg
        # score = torch.mm(user_emb, weight.t())
        # score = torch.softmax(score, dim=-1)
        # user_agg = user_agg + (torch.mm(score, weight)) * user_agg

        return entity_agg

    def calculate_sim_hrt(self, entity_emb_head, entity_emb_tail, relation_emb):

        tail_relation_emb = entity_emb_tail * relation_emb
        tail_relation_emb = tail_relation_emb.norm(dim=1, p=2, keepdim=True)
        head_relation_emb = entity_emb_head * relation_emb
        head_relation_emb = head_relation_emb.norm(dim=1, p=2, keepdim=True)
        att_weights = torch.matmul(head_relation_emb.unsqueeze(dim=1), tail_relation_emb.unsqueeze(dim=2)).squeeze(dim=-1)
        att_weights = att_weights ** 2
        return att_weights

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users, n_items, n_dim,
                  n_relations, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.topk = 10
        self.lambda_coeff = 0.5
        self.temperature = 0.2
        self.device = torch.device("cuda:" + str(0))
        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout
        self.prepare_adjmat()
        self.prepare_mask_sample(n_dim)
        self.gtLayers = nn.Sequential(*[GTLayer() for i in range(args.gt_layer)])
        self.kgLayers = nn.Sequential(*[KGlayer() for i in range(args.gt_layer)])

    def prepare_mask_sample(self, hidden):
        self.nblayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.nblayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

        self.selflayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.selflayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

        self.attentions_0 = nn.Sequential(nn.Linear(2 * hidden, 1))
        self.attentions_1 = nn.Sequential(nn.Linear(2 * hidden, 1))

    def prepare_adjmat(self):
        self.plain_adj = self.create_adj_mat(self.interact_mat)
        self.all_h_list, self.all_t_list, self.all_v_list = self.load_adjacency_list_data(self.plain_adj)
        self.A_in_shape = self.plain_adj.tocoo().shape
        self.A_indices = torch.tensor([self.all_h_list, self.all_t_list], dtype=torch.long).cuda()
        self.D_indices = torch.tensor(
            [list(range(self.n_users + self.n_items)), list(range(self.n_users + self.n_items))],
            dtype=torch.long).cuda()
        self.all_h_list = torch.LongTensor(self.all_h_list).cuda()
        self.all_t_list = torch.LongTensor(self.all_t_list).cuda()
        self.G_indices, self.G_values = self._cal_sparse_adj()

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def _cal_sparse_adj(self):

        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).cuda()

        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values, sparse_sizes=self.A_in_shape).cuda()
        D_values = A_tensor.sum(dim=1).pow(-0.5)

        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])

        return G_indices, G_values

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def create_adj_mat(self, intermat):

        rows = intermat.tocoo().row
        cols = intermat.tocoo().col
        new_rows = np.concatenate([rows, cols + self.n_users], axis=0)
        new_cols = np.concatenate([cols + self.n_users, rows], axis=0)
        adj_mat = sp.coo_matrix((np.ones(len(new_rows)), (new_rows, new_cols)), shape=[self.n_users + self.n_items, self.n_users + self.n_items]).tocsr().tocoo()
        adj_mat = adj_mat.todok()
        return adj_mat.tocsr()

    def load_adjacency_list_data(self, adj_mat):
        tmp = adj_mat.tocoo()
        all_h_list = list(tmp.row)
        all_t_list = list(tmp.col)
        all_v_list = list(tmp.data)

        return all_h_list, all_t_list, all_v_list

    def _adaptive_mask(self, head_embeddings, tail_embeddings):
        head_embeddings = torch.nn.functional.normalize(head_embeddings)
        tail_embeddings = torch.nn.functional.normalize(tail_embeddings)
        edge_alpha = (torch.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2

        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=edge_alpha, sparse_sizes=self.A_in_shape).cuda()
        # D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).view(-1)

        G_indices = torch.stack([self.all_h_list, self.all_t_list], dim=0)
        G_values = D_scores_inv[self.all_h_list] * edge_alpha

        return G_indices, G_values

    def sample_kg(self, edge_index, edge_type, entity_emb):

        # 1. graph sprasification;
        edge_index, edge_type = _relation_aware_edge_sampling(
            edge_index, edge_type, self.n_relations, self.node_dropout_rate)
        # 2. compute rationale scores;
        edge_attn_score, edge_attn_logits = norm_attn_computer(
           entity_emb, edge_index, edge_type)
        mae_msize = int(0.5*(edge_attn_score.size(0)))

        # for adaptive UI MAE
        # item_attn_mean_1 = scatter_mean(edge_attn_score, edge_index[0], dim=0, dim_size=self.n_items)
        # item_attn_mean_1[item_attn_mean_1 == 0.] = 1.
        # item_attn_mean_2 = scatter_mean(edge_attn_score, edge_index[1], dim=0, dim_size=self.n_items)
        # item_attn_mean_2[item_attn_mean_2 == 0.] = 1.
        # item_attn_mean = (0.5 * item_attn_mean_1 + 0.5 * item_attn_mean_2)[:self.n_items]

        # for adaptive MAE training
        # std = torch.std(edge_attn_score).detach()

        noise = -torch.log(-torch.log(torch.rand_like(edge_attn_score)))
        edge_attn_score = edge_attn_score + noise
        topk_v, topk_attn_edge_id = torch.topk(
            edge_attn_logits, mae_msize, sorted=False)
        top_attn_edge_type = edge_type[topk_attn_edge_id]

        enc_edge_index, enc_edge_type, masked_edge_index, masked_edge_type, mask_bool = _mae_edge_mask_adapt_mixed(
            edge_index, edge_type, topk_attn_edge_id)
        return enc_edge_index, enc_edge_type

    def forward(self, user_emb, entity_emb, user_intent, item_intent, edge_index, edge_type,
                interact_mat, training, mess_dropout=True, node_dropout=False):

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        all_embeddings = []
        self.gnn_embeddings = []
        self.int_embeddings = []
        self.gaa_embeddings = []
        self.iaa_embeddings = []
        all_embeddings.append(torch.cat([user_emb, entity_emb], dim=0))

        # KG中的edge sample应该放在意图建模之后，目前先写在整体卷积之外看看效果
        # edge_index, edge_type = self.sample_kg(edge_index, edge_type, entity_emb)

        # for i in range(len(self.convs)):
        #     entity_emb = self.convs[i](entity_emb, user_emb,
        #                                          edge_index, edge_type, interact_mat,
        #                                          self.weight, i)
        #     tmp_all_embeddings = torch.cat([user_emb, entity_emb], dim=0)
        #     all_embeddings_new = self.graph_distangle(tmp_all_embeddings, user_intent, item_intent, i, training)
        #     user_emb, entity_emb = torch.split(all_embeddings_new, [self.n_users, self.n_items], 0)
        #     entity_res_emb = torch.add(entity_res_emb, entity_emb)
        #     user_res_emb = torch.add(user_res_emb, user_emb)
        for i in range(len(self.convs)):


            tmp_all_embeddings = torch.cat([user_emb, entity_emb], dim=0)
            all_embeddings_new = self.graph_distangle(tmp_all_embeddings, user_intent, item_intent, i, training)
            user_emb, entity_emb = torch.split(all_embeddings_new, [self.n_users, self.n_items], 0)

            sampled_edge_index, sampled_edge_type = self.sample_kg(edge_index, edge_type, entity_emb)
            entity_emb = self.convs[i](entity_emb, user_emb,
                                       sampled_edge_index, sampled_edge_type, interact_mat,
                                       self.weight, i)
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        # for i in range(len(self.convs)):
        #
        #     # tmp_all_embeddings = torch.cat([user_emb, entity_emb], dim=0)
        #     # all_embeddings_new = self.graph_distangle(tmp_all_embeddings, user_intent, item_intent, i, training)
        #     # user_emb, entity_emb = torch.split(all_embeddings_new, [self.n_users, self.n_items], 0)
        #
        #     # user_intent_emb, entity_intent_emb = user_emb, entity_emb
        #
        #     # user_intent_emb, entity_intent_emb = torch.split(self.iaa_embeddings[i]+self.int_embeddings[i], [self.n_users, self.n_items], 0)
        #
        #     entity_emb = self.convs[i](entity_emb, user_emb, edge_index, edge_type, interact_mat, self.weight, i)
        #     tmp_all_embeddings = torch.cat([user_emb, entity_emb], dim=0)
        #     all_embeddings_new = self.graph_distangle(tmp_all_embeddings, user_intent, item_intent, i, training)
        #     user_emb, entity_emb = torch.split(all_embeddings_new, [self.n_users, self.n_items], 0)
        #
        #     entity_res_emb = torch.add(entity_res_emb, entity_emb)
        #     user_res_emb = torch.add(user_res_emb, user_emb)
        #
        disentangle_dict = [self.gnn_embeddings, self.int_embeddings, self.gaa_embeddings, self.iaa_embeddings]
        return entity_res_emb, user_res_emb, disentangle_dict

    def graph_distangle(self, all_embeddings, user_intent, item_intent, layer, training_flag):
        # Graph-based Message Passing
        gnn_layer_embeddings = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1],
                                                 all_embeddings)
        # Intent-aware Information Aggregation
        u_embeddings, i_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
        u_int_embeddings = torch.softmax(u_embeddings @ user_intent, dim=1) @ user_intent.T
        i_int_embeddings = torch.softmax(i_embeddings @ item_intent, dim=1) @ item_intent.T
        int_layer_embeddings = torch.cat([u_int_embeddings, i_int_embeddings], dim=0)

        # 先过一层transformer了再进行mask，transformer+mask
        orign_graph = torch.sparse.FloatTensor(self.G_indices, self.G_values, self.A_in_shape)
        tmp_int_layer_embeddings = self.gtLayers[layer](orign_graph, int_layer_embeddings)
        # Adaptive Augmentation
        gnn_head_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.all_h_list)
        gnn_tail_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.all_t_list)
        int_head_embeddings = torch.index_select(tmp_int_layer_embeddings, 0, self.all_h_list)
        int_tail_embeddings = torch.index_select(tmp_int_layer_embeddings, 0, self.all_t_list)

        G_graph_indices, G_graph_values = self._adaptive_mask(gnn_head_embeddings, gnn_tail_embeddings)
        G_inten_indices, G_inten_values = self._adaptive_mask(int_head_embeddings, int_tail_embeddings)
        # G_graph_indices, G_graph_values = self._transformer_mask(gnn_head_embeddings, gnn_tail_embeddings, gnn_layer_embeddings)
        # G_inten_indices, G_inten_values = self._transformer_mask(int_head_embeddings, int_tail_embeddings, int_layer_embeddings)

        # 替换为graph transformer 的编码方式尝试
        gaa_layer_embeddings = torch_sparse.spmm(G_graph_indices, G_graph_values, self.A_in_shape[0],
                                                 self.A_in_shape[1], all_embeddings)
        iaa_layer_embeddings = torch_sparse.spmm(G_inten_indices, G_inten_values, self.A_in_shape[0],
                                                 self.A_in_shape[1], all_embeddings)
        # orign_graph = torch.sparse.FloatTensor(G_graph_indices, G_graph_values, self.A_in_shape)
        # gaa_layer_embeddings = self.gtLayers[layer](orign_graph, all_embeddings)
        # intent_graph = torch.sparse.FloatTensor(G_inten_indices, G_inten_values, self.A_in_shape)
        # iaa_layer_embeddings = self.gtLayers[layer](intent_graph, all_embeddings)

        self.gnn_embeddings.append(gnn_layer_embeddings)
        self.int_embeddings.append(int_layer_embeddings)
        self.gaa_embeddings.append(gaa_layer_embeddings)
        self.iaa_embeddings.append(iaa_layer_embeddings)

        all_embedding_layer = gnn_layer_embeddings + int_layer_embeddings + gaa_layer_embeddings + iaa_layer_embeddings\
                              + all_embeddings
        return all_embedding_layer


class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")

        self.adj_mat = adj_mat
        self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)

        self.n_intent = 128
        self.temp = 0.3
        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.user_intent = nn.Parameter(self.user_intent)
        self.item_intent = nn.Parameter(self.item_intent)
        self.gcn = self._init_model()
        self.lightgcn_layer = 2
        self.n_item_layer = 1
        self.alpha = 0.2
        self.fc1 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)
        self.user_intent = initializer(torch.empty(self.emb_size, self.n_intent))
        self.item_intent = initializer(torch.empty(self.emb_size, self.n_intent))

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_items = self.n_entities,
                         n_dim = self.emb_size,
                         n_relations=self.n_relations,
                         interact_mat=self.adj_mat,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(
        self,
        batch=None,
        training=True):
        user = batch['users']
        item = batch['items']
        labels = batch['labels']
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        entity_gcn_emb, user_gcn_emb, disentanle_dict = self.gcn(user_emb,
                                 item_emb,
                                 self.user_intent,
                                 self.item_intent,
                                 self.edge_index,
                                 self.edge_type,
                                 self.interact_mat,
                                 training,
                                 mess_dropout=self.mess_dropout,
                                 node_dropout=self.node_dropout
                                 )
        u_e = user_gcn_emb[user]
        i_e = entity_gcn_emb[item]

        disentanle_loss = self.cal_distangle_loss(user, item, disentanle_dict)
        loss_contrast = disentanle_loss
        return self.create_bpr_loss(u_e, i_e, labels, loss_contrast)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def calculate_loss(self, A_embedding, B_embedding):
        # first calculate the sim rec
        tau = 0.6    # default = 0.8
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.fc1(A_embedding)
        B_embedding = self.fc1(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        # refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        # between_sim_1 = f(self.sim(B_embedding, A_embedding))
        # loss_2 = -torch.log(
        #     between_sim_1.diag()
        #     / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        # ret = (loss_1 + loss_2) * 0.5
        ret = loss_1
        ret = ret.mean()
        return ret


    def create_bpr_loss(self, users, items, labels, loss_contrast):
        batch_size = users.shape[0]
        scores = (items * users).sum(dim=1)
        scores = torch.sigmoid(scores)
        criteria = nn.BCELoss()
        bce_loss = criteria(scores, labels.float())
        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return bce_loss + emb_loss + 0.001*loss_contrast, scores, bce_loss, emb_loss

    def cal_distangle_loss(self, users, items,disentanle_dict):
        gnn_emb, int_emb, gaa_emb, iaa_emb = disentanle_dict

        cl_loss = 0.0
        def cal_loss(emb1, emb2):
            pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.temp)
            neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.temp), axis=1)
            loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            loss /= pos_score.shape[0]
            return loss

        for i in range(len(gnn_emb)):
            u_gnn_embs, i_gnn_embs = torch.split(gnn_emb[i], [self.n_users, self.n_entities], 0)
            u_int_embs, i_int_embs = torch.split(int_emb[i], [self.n_users, self.n_entities], 0)
            u_gaa_embs, i_gaa_embs = torch.split(gaa_emb[i], [self.n_users, self.n_entities], 0)
            u_iaa_embs, i_iaa_embs = torch.split(iaa_emb[i], [self.n_users, self.n_entities], 0)

            u_gnn_embs = F.normalize(u_gnn_embs[users], dim=1)
            u_int_embs = F.normalize(u_int_embs[users], dim=1)
            u_gaa_embs = F.normalize(u_gaa_embs[users], dim=1)
            u_iaa_embs = F.normalize(u_iaa_embs[users], dim=1)

            i_gnn_embs = F.normalize(i_gnn_embs[items], dim=1)
            i_int_embs = F.normalize(i_int_embs[items], dim=1)
            i_gaa_embs = F.normalize(i_gaa_embs[items], dim=1)
            i_iaa_embs = F.normalize(i_iaa_embs[items], dim=1)

            cl_loss += cal_loss(u_gnn_embs, u_int_embs)
            cl_loss += cal_loss(u_gnn_embs, u_gaa_embs)
            cl_loss += cal_loss(u_gnn_embs, u_iaa_embs)

            cl_loss += cal_loss(i_gnn_embs, i_int_embs)
            cl_loss += cal_loss(i_gnn_embs, i_gaa_embs)
            cl_loss += cal_loss(i_gnn_embs, i_iaa_embs)

        return cl_loss



