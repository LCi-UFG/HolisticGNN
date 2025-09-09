import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import get_activation
from encoding import (
    RWPEncoder, 
    CentralityEncoder, 
    PEARLEncoder
    )


class BondFastAttention(nn.Module):
    def __init__(
        self, 
        hidden_size, 
        heads,
        dropout, 
        activation=F.relu):

        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.att_size = hidden_size // heads
        self.scale_factor = self.att_size ** -0.5
        self.weight_alpha = nn.Parameter(
            torch.randn(self.att_size)
            )
        self.weight_beta = nn.Parameter(
            torch.randn(self.att_size)
            )
        self.weight_r = nn.Linear(
            self.att_size, self.att_size,
            bias=False
            )
        self.W_b_q = nn.Linear(
            hidden_size,
            heads * self.att_size,
            bias=False
            )
        self.W_b_k = nn.Linear(
            hidden_size,
            heads * self.att_size,
            bias=False
            )
        self.W_b_v = nn.Linear(
            hidden_size,
            heads * self.att_size,
            bias=False
            )
        self.W_b_o = nn.Linear(
            heads * self.att_size,
            hidden_size
            )
        self.act_func = activation
        self.dropout_layer = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self, 
        edge_attr, 
        batch_scopes):

        segments = [edge_attr[b:b + l]
            for b, l in batch_scopes]
        lengths = [seg.size(0) for seg in segments]
        B = len(segments)
        padded = torch.nn.utils.rnn.pad_sequence(
            segments, batch_first=True
            )
        B, L, _ = padded.size()
        b_q = self.W_b_q(padded)
        b_q = b_q.view(B, L, self.heads,
            self.att_size).transpose(1, 2)
        b_k = self.W_b_k(padded)
        b_k = b_k.view(B, L, self.heads,
            self.att_size).transpose(1, 2)
        b_v = self.W_b_v(padded)
        b_v = b_v.view(B, L, self.heads,
            self.att_size).transpose(1, 2)
        alpha = (b_q * self.weight_alpha
            * self.scale_factor
            )
        alpha = F.softmax(alpha, dim=-1)
        global_q = (alpha * b_q).sum(dim=2)
        gq = global_q.unsqueeze(2)
        gq = gq.expand(-1, -1, L, -1)
        p = gq * b_k
        beta = p * self.weight_beta * self.scale_factor
        beta = F.softmax(beta, dim=-1)
        global_k = (beta * p).sum(dim=2)
        gk = global_k.unsqueeze(2)
        gk = gk.expand(-1, -1, L, -1)
        kv = gk * b_v
        kv_out = self.weight_r(kv)
        att = self.act_func(kv_out + b_q)
        att = self.dropout_layer(att)
        att = att.transpose(1, 2)
        att = att.contiguous().view(B, L,
            self.heads * self.att_size
            )
        att = self.W_b_o(att)
        att = self.norm(att)
        outputs = [att[b, :lengths[b]]
                   for b in range(B)]
        bond_vecs = torch.cat(outputs, dim=0)

        return bond_vecs


class MultiAtomAttention(nn.Module):
    def __init__(
        self,
        node_feat_dim,
        dropout,
        activation,
        device,
        heads,
        f_scale=1.0):

        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.dropout = dropout
        self.device = device
        self.heads = heads
        self.head_dim = node_feat_dim // heads
        self.scale_factor = self.head_dim ** -0.5
        self.act_func = get_activation(activation)
        self.f_scale = f_scale
        self.W_att_q = nn.Linear(
            node_feat_dim,
            heads * self.head_dim,
            bias=False
            )
        self.W_att_k = nn.Linear(
            node_feat_dim,
            heads * self.head_dim,
            bias=False
            )
        self.W_att_v = nn.Linear(
            node_feat_dim,
            heads * self.head_dim,
            bias=False
            )
        self.W_att_o = nn.Linear(
            heads * self.head_dim,
            node_feat_dim
            )
        self.norm = nn.LayerNorm(node_feat_dim)
        self.rwpe_encoder = RWPEncoder()
        self.central_encoder = CentralityEncoder()
        self.pearl_encoder = PEARLEncoder()
        self.last_attn_weights = None

    def forward(
        self,
        x,
        edge_index,
        batch_index,
        laplacian_matrix,
        adjacency_matrix):

        num_nodes = x.size(0)
        q = self.W_att_q(x)
        q = q.view(num_nodes, self.heads,
            self.head_dim).transpose(0, 1)
        k = self.W_att_k(x)
        k = k.view(num_nodes, self.heads,
            self.head_dim)
        k = k.transpose(0, 1).transpose(1, 2)
        v = self.W_att_v(x)
        v = v.view(num_nodes, self.heads,
            self.head_dim).transpose(0, 1)
        rwpe_enc = self.rwpe_encoder(
            edge_index, num_nodes
            )
        rwpe_score = rwpe_enc.sum(dim=1)
        rwpe_bias = (
            rwpe_score.unsqueeze(0).unsqueeze(2)
            + rwpe_score.unsqueeze(0).unsqueeze(1)
            )
        cent_enc = self.central_encoder([edge_index])
        cent_score = cent_enc.sum(dim=1)
        cent_bias = (
            cent_score.unsqueeze(0).unsqueeze(2)
            + cent_score.unsqueeze(0).unsqueeze(1)
            )
        pearl_bias = self.pearl_encoder(
            laplacian_matrix,
            adjacency_matrix,
            edge_index
            )
        att_bias = torch.zeros(
            self.heads,
            num_nodes,
            num_nodes,
            device=x.device
            )
        att_bias[0:2] = rwpe_bias
        att_bias[2:4] = cent_bias
        att_bias[4:6] = pearl_bias
        attn_scores = torch.matmul(q, k)
        attn_scores = (attn_scores
            + self.f_scale * att_bias
            )
        attn_scores = attn_scores * self.scale_factor
        graph_mask = (batch_index.unsqueeze(0)
            == batch_index.unsqueeze(1)
            )
        attn_scores = attn_scores.masked_fill(
            ~graph_mask.unsqueeze(0), float('-inf')
            )
        attn_weights = F.softmax(
            attn_scores, dim=-1
            )
        attn_output = torch.matmul(attn_weights, v)
        attn_output = self.act_func(attn_output)
        attn_output = F.dropout(
            attn_output,
            p=self.dropout,
            training=self.training
            )
        attn_output = attn_output.transpose(
            0, 1).contiguous().view(
            num_nodes, self.heads * self.head_dim
            )
        attn_output = self.W_att_o(attn_output)
        attn_output = self.norm(
            attn_output.unsqueeze(0)).squeeze(0)
        attn_weights = attn_weights.mean(dim=0)
        self.last_attn_weights = attn_weights

        return attn_output, attn_weights
    

class SublayerConnection(nn.Module):

    def __init__(self, dropout):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, original, attention):
        return original + self.dropout(attention)