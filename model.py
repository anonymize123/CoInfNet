import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Sequential, GATConv
from new_gat import TopNGAT
from torch.nn import ReLU, Linear, ELU


class ModelV1(nn.Module):

    def __init__(self, args, in_dims, hidden_dims, out_dims, edge_class,
                 max_len, node_num, temp):
        super(ModelV1, self).__init__()

        self.not_use_tpe = args.no_tpe
        self.not_edge_emb = args.no_edge_emb
        self.delete_graph2 = args.delete_graph2
        self.use_graph1 = args.use_graph1
        self.use_same_graph = args.use_same_graph
        self.use_gru = args.use_gru
        self.use_gat1 = args.use_gat1

        if self.use_same_graph is not None:
            self.use_graph1 = True

        self.read_in = Linear(in_dims, hidden_dims)

        self.temporal_encoder = TimeEncoder(hidden_dims)

        if not self.use_gat1:
            self.graph_0 = Sequential('x, edge_index', [
                (GCNConv(hidden_dims, hidden_dims), 'x, edge_index -> x'),
                ReLU(inplace=True),
                (GCNConv(hidden_dims, hidden_dims), 'x, edge_index -> x'),
                ReLU(inplace=True),
            ])
        else:
            self.graph_0 = Sequential('x, edge_index', [
                (GATConv(hidden_dims, hidden_dims, 4), 'x, edge_index -> x'),
                ReLU(inplace=True),
                (GATConv(hidden_dims * 4, hidden_dims, 4), 'x, edge_index -> x'),
                ReLU(inplace=True),
                (Linear(hidden_dims * 4, hidden_dims), 'x -> x'),
            ])

        self.graph_1 = GraphModel(hidden_dims, node_num, use_gat=args.use_gat)

        self.lin_invite = Linear(hidden_dims * 2, hidden_dims)
        self.team_read_out = Linear(hidden_dims, edge_class)

        # self.sequence_model = ModelSequence(hidden_dims, max_len, args.no_edge_emb)
        self.sequence_model1 = ModelSequence1(hidden_dims, max_len, args.no_edge_emb)
        self.gru = nn.LSTM(hidden_dims * 3, hidden_dims, num_layers=1, batch_first=False)

        self.graph_2 = GraphModel(hidden_dims, node_num, temp, args.use_gat)

        self.read_out = Linear(hidden_dims, out_dims)

        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(3), requires_grad=True)
        self.fuse_weight.data.fill_(float(0.5))

        self.fusion_layer = AttentionPool(hidden_dims, hidden_dims)

    def reset_parameters(self):
        self.read_in.reset_parameters()
        self.lin_invite.reset_parameters()
        self.team_read_out.reset_parameters()
        self.read_out.reset_parameters()

    def forward(self, x, edge_index, invitations, offline_days):

        ###################################### read in  ######################################
        x_in = F.dropout(F.relu(self.read_in(x)), p=0.2, training=self.training)

        if not self.not_use_tpe:
            x_in = self.temporal_encoder(x_in, offline_days)

        x_zeros = torch.zeros((1, x_in.size(1)), dtype=x_in.dtype, device=x_in.device)
        x_cat = torch.cat([x_in, x_zeros], dim=0)  # [n+1, hid_dim]

        invitor_idxs = invitations[:, :, 0]  # [batch_size, max_len]
        invitee_idxs = invitations[:, :, 1]  # [batch_size, max_len]

        x_invitor = x_cat[invitor_idxs]  # [batch_size, max_len, hid_dim]
        x_invitee = x_cat[invitee_idxs]  # [batch_size, max_len, hid_dim]
        if not self.use_gru:
            invite_node_emb = torch.cat([x_invitor, x_invitee[:, -1, :].unsqueeze(1)], dim=1) \
                .transpose(0, 1).contiguous()
        else:
            invite_node_emb = torch.cat([x_invitor, x_invitee], dim=-1).transpose(0, 1).contiguous()
        x0 = x_invitee[:, -1, :]  # [batch_size, hid_dim] 聚合邻居信息前

        ###################################### graph_1 ######################################

        if self.use_graph1:
            if self.use_same_graph:
                x_1 = self.graph_2(x_in, edge_index)
            else:
                x_1 = self.graph_1(x_in, edge_index)
            x_1 = torch.cat([x_1, x_zeros], dim=0)
            x_1 = x_1[invitee_idxs][:, -1, :]

        ###################################### graph_0 ######################################

        if not self.not_edge_emb:
            x_graph = self.graph_0(x_in, edge_index)
            x_graph = torch.cat([x_graph, x_zeros], dim=0)

        ###################################### team emb ######################################

            x_invitor = x_graph[invitor_idxs]  # [batch_size, max_len, hid_dim]
            x_invitee = x_graph[invitee_idxs]  # [batch_size, max_len, hid_dim]
            invite_emb = torch.cat([x_invitee, x_invitor], dim=-1)  # [batch_size, max_len, hid_dim * 2]
            invite_edge_emb = F.dropout(F.relu(self.lin_invite(invite_emb)), p=0.2, training=self.training) \
                .transpose(0, 1).contiguous()

            team_result = self.team_read_out(invite_edge_emb)
        else:
            invite_edge_emb = None
            team_result = None

        ###################################### sequence ######################################

        if not self.use_gru:
            hx = self.sequence_model1(invite_edge_emb, invite_node_emb)  # [batch_size, hid_dim]
        else:
            gru_in_feat = torch.cat([invite_edge_emb, invite_node_emb], dim=-1)  # [max_len, batch_size, hid_dim * 3]
            _, hx = self.gru(gru_in_feat)
            hx = hx[0][-1, :, :]

        ###################################### graph_2 #########################################

        if not self.delete_graph2:
            x_cat[invitee_idxs[:, -1]] = hx
            x_graph_2 = self.graph_2(x_cat, edge_index)
            x2 = x_graph_2[invitee_idxs[:, -1]]

        ###################################### read out ######################################

            hx = self.fuse_weight[0] * x0 + self.fuse_weight[1] * x2 + hx
            # hx = self.fusion_layer([x0, x2, hx])
        else:
            hx = self.fuse_weight[0] * x0 + hx
            # hx = self.fusion_layer([x0, hx])

        if self.use_graph1:
            hx = self.fuse_weight[2] * x_1 + hx

        return self.read_out(hx), team_result


class TimeEncoder(nn.Module):

    def __init__(self, hidden_dims, factor=5):
        super(TimeEncoder, self).__init__()

        self.hidden_dim = hidden_dims
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.hidden_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.hidden_dim).float())

    def forward(self, x, ts):

        ts = ts.unsqueeze(-1)  # [N, 1]
        map_ts = ts * self.basis_freq.view(1, -1)
        map_ts += self.phase.view(1, -1)
        harmonic = torch.cos(map_ts)

        return x + harmonic


class ModelSequence(nn.Module):

    def __init__(self, hidden_dims, max_len, no_use_edge_emb):
        super(ModelSequence, self).__init__()

        self.hid_dim = hidden_dims
        self.max_len = max_len

        self.no_edge_emb = no_use_edge_emb

        if self.no_edge_emb:
            self.lin_xi = Linear(hidden_dims * 3, hidden_dims)
        else:
            self.lin_xi = Linear(hidden_dims * 4, hidden_dims)

        self.lin_u = Linear(hidden_dims * 3, hidden_dims)
        self.lin_r = Linear(hidden_dims * 3, hidden_dims)
        self.lin_k = Linear(hidden_dims * 3, hidden_dims)

    def reset_parameters(self):
        self.lin_xi.reset_parameters()
        self.lin_u.reset_parameters()
        self.lin_r.reset_parameters()
        self.lin_k.reset_parameters()

    def forward(self, edge_feats, node_feats):  # [L, N, d], [L+1, N, d]

        # assert self.max_len == edge_feats.shape[0]

        x0 = node_feats[-1]   # (N, d)

        xs = node_feats[:-1]  # (L, N, d)

        num_layers = xs.shape[0]
        del node_feats
        hx = torch.zeros(xs.shape[1], xs.shape[-1],
                         dtype=xs.dtype, device=xs.device)  # (N, d)

        for l in range(num_layers):

            if self.no_edge_emb:
                x_1 = torch.cat([hx, x0, xs[l]], dim=-1)  # (N, 3d)
            else:
                x_1 = torch.cat([hx, edge_feats[l], x0, xs[l]], dim=-1)  # (N, 4d)
            xi = torch.sigmoid(self.lin_xi(x_1))  # (N, d)

            x0_h = (1 - xi) * x0
            xt_h = xi * xs[l]

            x_2 = torch.cat([hx, x0_h, xt_h], dim=-1)  # (N, 3d)

            u = torch.sigmoid(self.lin_u(x_2))  # (N, d)
            r = torch.sigmoid(self.lin_r(x_2))  # (N, d)

            hx_h = r * hx

            x_3 = torch.cat([x0_h, xt_h, hx_h], dim=-1)  # (N, 3d)
            k = torch.tanh(self.lin_k(x_3))   # (N, d)

            hx = (1 - u) * k + u * hx

        return hx


class GraphModel(nn.Module):

    def __init__(self, hidden_dims, n, temp=1.0, use_gat=True):
        super(GraphModel, self).__init__()

        if use_gat:
            self.gat = GATConv(hidden_dims, hidden_dims, 4)
        else:
            self.gat = TopNGAT(hidden_dims, hidden_dims, 4, n+1, temp)
        self.relu = ELU(inplace=True)

        self.lin = Linear(hidden_dims * 4, hidden_dims)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index):

        x = self.gat(x, edge_index)
        x = self.relu(x)

        return self.lin(x)


class AttentionPool(nn.Module):

    def __init__(self, in_dim, hid_dim, heads=4, num_layers=2):
        super(AttentionPool, self).__init__()

        self.num_layers = num_layers

        self.w_k = Linear(in_dim, hid_dim * heads)
        self.w_q = Linear(in_dim, hid_dim * heads)
        self.w_v = Linear(in_dim, hid_dim * heads)

    def reset_parameters(self):
        self.w_k.reset_parameters()
        self.w_q.reset_parameters()
        self.w_v.reset_parameters()

    def forward(self, xs):

        xs = torch.stack(xs, dim=2).transpose(1, 2)  # [n, in_dim, l] -> [n, l, in_dim]

        k = self.w_k(xs)  # [n, l, hid_dim * heads]
        q = self.w_q(xs)
        v = self.w_v(xs)

        attn = F.leaky_relu(torch.matmul(k, q.transpose(1, 2)))  # [n, l, l]

        alpha = F.softmax(attn, dim=-1)  # [n, l, l]

        attn_out = torch.matmul(alpha, v)  # [n, l, hid_dim * heads]

        return attn_out[:, -1, :].squeeze(1)  # [n, hid_dim * heads]


class ModelSequence1(nn.Module):

    def __init__(self, hidden_dims, max_len, no_use_edge_emb=False):
        super(ModelSequence1, self).__init__()

        self.hid_dim = hidden_dims
        self.max_len = max_len

        self.no_edge_emb = no_use_edge_emb

        if self.no_edge_emb:
            self.lin_xi = Linear(hidden_dims * 3, hidden_dims)
        else:
            self.lin_xi = Linear(hidden_dims * 4, hidden_dims)

        self.lin_u = Linear(hidden_dims * 3, hidden_dims)
        self.lin_r = Linear(hidden_dims * 3, hidden_dims)
        self.lin_k = Linear(hidden_dims * 3, hidden_dims)
        self.lin_node = Linear(hidden_dims * 2, hidden_dims)
        self.lin_edge = Linear(hidden_dims, hidden_dims)

    def reset_parameters(self):
        self.lin_xi.reset_parameters()
        self.lin_u.reset_parameters()
        self.lin_r.reset_parameters()
        self.lin_k.reset_parameters()
        self.lin_node.reset_parameters()
        self.lin_edge.reset_parameters()

    def forward(self, edge_feats, node_feats):  # [L, N, d], [L+1, N, d]

        # assert self.max_len == edge_feats.shape[0]

        x0 = node_feats[-1]   # (N, d)

        xs = node_feats[:-1]  # (L, N, d)

        num_layers = xs.shape[0]
        del node_feats
        hx = torch.zeros(xs.shape[1], xs.shape[-1],
                         dtype=xs.dtype, device=xs.device)  # (N, d)

        for l in range(num_layers):

            if self.no_edge_emb:
                x_node = F.dropout(F.relu(self.lin_node(torch.cat([x0, xs[l]], dim=-1))),
                                   p=0.2, training=self.training)
                x0_h = x_node
                xt_h = x_node
            else:
                x_1 = torch.cat([hx, edge_feats[l], x0, xs[l]], dim=-1)  # (N, 4d)
                xi = torch.sigmoid(self.lin_xi(x_1))  # (N, d)

                x_node = F.dropout(F.relu(self.lin_node(torch.cat([x0, xs[l]], dim=-1))), p=0.2, training=self.training)
                x_edge = F.dropout(F.relu(self.lin_edge(edge_feats[l])), p=0.2, training=self.training)

                x0_h = (1 - xi) * x_node
                xt_h = xi * x_edge

            x_2 = torch.cat([hx, x0_h, xt_h], dim=-1)  # (N, 3d)

            u = torch.sigmoid(self.lin_u(x_2))  # (N, d)
            r = torch.sigmoid(self.lin_r(x_2))  # (N, d)

            hx_h = r * hx

            x_3 = torch.cat([x0_h, xt_h, hx_h], dim=-1)  # (N, 3d)
            k = torch.tanh(self.lin_k(x_3))   # (N, d)

            hx = (1 - u) * k + u * hx

        return hx
