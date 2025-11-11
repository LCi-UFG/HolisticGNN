import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import (
    degree, 
    softmax
    )
from torch_geometric.nn import (
    MessagePassing, 
    GATv2Conv, 
    GINEConv,
    GraphNorm
    )

from utils import device
from dirichlet import DMPGM
from attention import (
    BondFastAttention, 
    SublayerConnection
    )

class AttentiveLayer(MessagePassing):
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        edge_dim, 
        dropout_rate):
        
        super(AttentiveLayer, self).__init__(aggr='add')
        self.node_proj = nn.Linear(input_dim, output_dim
            ) if input_dim != output_dim else nn.Identity()
        self.layer_norm = GraphNorm(output_dim)
        self.edge_encoder = nn.Linear(
            edge_dim, output_dim
            )
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            )
        attn_input_dim = 3 * output_dim
        self.attn_mlp = nn.Sequential(
            nn.Linear(attn_input_dim, 1),
            nn.LeakyReLU(0.1)
            )
        self.gru = nn.GRUCell(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.last_attn_weights = None     

    def message(
        self, 
        x_i, x_j, 
        edge_attr, 
        index, ptr, 
        size_i):

        attn_input = torch.cat(
            [x_i, x_j, edge_attr], dim=-1
            )
        attn_scores = self.attn_mlp(attn_input)
        attn_weights = softmax(
            attn_scores, index, ptr, size_i
            )
        self.last_attn_weights = attn_weights.squeeze(-1)  
        message = self.msg_mlp(
            torch.cat([x_j, edge_attr], dim=-1)
            )
        
        return message * attn_weights
    
    def forward(
        self, x, 
        edge_index, 
        edge_attr,
        batch):
        
        x_proj = self.node_proj(x)
        edge_enc = self.edge_encoder(edge_attr)
        aggr_out = self.propagate(
            edge_index=edge_index, 
            x=x_proj, 
            edge_attr=edge_enc
            )
        out = self.gru(aggr_out, x_proj)
        out = self.layer_norm(out + x_proj, batch)
        out = self.dropout(out)

        return out                        


class GATLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        edge_dim,
        heads,
        dropout,
        concat=True):

        super(GATLayer, self).__init__()
        self.concat = concat
        self.heads = heads
        self.conv = GATv2Conv(
            in_channels=input_dim,
            out_channels=(
                output_dim // heads
                if concat else output_dim
                ),
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=concat
            )
        self.res_connection = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
            )
        self.norm = GraphNorm(output_dim)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.last_attn_weights = None

    def forward(
        self, x,
        edge_index,
        edge_attr,
        batch):

        residual = self.res_connection(x)
        out, (ei, aw) = self.conv(
            x, edge_index,
            edge_attr=edge_attr,
            return_attention_weights=True
            )
        weights = aw.mean(dim=-1)
        mask = ei[0] != ei[1]
        self.last_attn_weights = weights[mask]

        out = out + residual
        out = self.norm(out, batch)
        out = self.activation(out)
        out = self.dropout(out)

        return out


class GINLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        edge_dim,
        num_lin_layers=2):

        super(GINLayer, self).__init__()

        layers = [nn.Linear(input_dim, output_dim), nn.ReLU()]
        for _ in range(num_lin_layers - 1):
            layers += [nn.Linear(
                output_dim, output_dim), nn.ReLU()]
        self.conv = GINEConv(
            nn.Sequential(*layers),
            edge_dim=edge_dim
            )
        self.conv_norm = GraphNorm(output_dim)
        self.feature_projection = nn.Linear(
            output_dim, output_dim)
        self.control_gate = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.Sigmoid()
            )
        self.post_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
            )
        self.post_norm = GraphNorm(output_dim)

    def forward(
        self, x, 
        edge_index, 
        edge_attr, 
        batch):

        x = self.conv(x, edge_index, edge_attr)
        x = self.conv_norm(x, batch)
        proj = self.feature_projection(x)
        gate = self.control_gate(torch.cat([proj, x], dim=-1))
        out = gate * proj + (1 - gate) * x
        out = self.post_mlp(out)
        out = self.post_norm(out, batch)

        return out

    
class MPNNLayer(MessagePassing):
    def __init__(
        self,
        input_dim,
        output_dim,
        edge_dim,
        dropout_rate):

        super(MPNNLayer, self).__init__(aggr='add')
 
        self.message_proj = nn.Sequential(
            nn.Linear(input_dim + edge_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            )
        self.node_proj = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
            )
        self.update_net = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim, output_dim)
            )
        self.norm = GraphNorm(output_dim)

    def message(self, x_j, edge_attr):
        m = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_proj(m)

    def update(self, aggr_out, x, batch):
        h = self.node_proj(x)
        cat = torch.cat([aggr_out, h], dim=-1)
        new_h = self.update_net(cat)
        out = new_h + h
        out = self.norm(out, batch)
        return out

    def forward(self, x, edge_index, edge_attr, batch):
        return self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr,
            batch=batch
            )


class EEGNNLayer(MessagePassing):
    def __init__(
        self,
        input_dim,
        output_dim,
        edge_dim,
        dropout_rate,
        num_components,
        concentration,
        prior_concentration,
        mcmc_iters):

        super().__init__(aggr='add')
        self._gen_args = {
            'num_components': num_components,
            'concentration': concentration,
            'prior_concentration': prior_concentration,
            'mcmc_iters': mcmc_iters,
            'device': device
            }
        self.generator = None
        self.message_proj = nn.Sequential(
            nn.Linear(input_dim + edge_dim + 1, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            )
        self.node_proj = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim else nn.Identity()
            )
        self.gru = nn.GRUCell(
            output_dim, output_dim
            )
        
    def forward(
        self, x, 
        edge_index, 
        edge_attr):

        N = x.size(0)
        if self.generator is None or self.generator.N != N:
            self.generator = DMPGM(
                num_nodes=N,
                **self._gen_args
                )
        self.generator.mcmc_update(edge_index)
        edge_index_hat, weight_hat = self.generator.virtual_graph()
        edge_index_hat = edge_index_hat.to(device)
        weight_hat     = weight_hat.to(device)
        orig = {(u.item(), v.item()): edge_attr[k]
            for k, (u, v) in enumerate(edge_index.t())
            }
        feats = []
        for u, v in edge_index_hat.t().tolist():
            feats.append(orig.get((u, v),
                    orig.get((v, u), torch.zeros(
                    edge_attr.size(1), device=device))
                    )
                )
        edge_feats = torch.stack(feats, dim=0)
        norm_feat  = weight_hat.unsqueeze(-1)
        edge_attr_hat = torch.cat(
            [edge_feats, norm_feat], dim=1
            )
        aggr = self.propagate(
            edge_index_hat,
            x=x,
            edge_attr=edge_attr_hat
            )
        return self.gru(aggr, self.node_proj(x))

    def message(self, x_j, edge_attr):

        return self.message_proj(
            torch.cat([x_j, edge_attr], dim=-1)
            )

class tMPNNLayer(MessagePassing):
    def __init__(
        self,
        input_dim,
        output_dim,
        edge_dim,
        dropout_rate=0.2,
        heads=6):

        super().__init__(aggr='add')
        self.heads = heads
        self.edge_proj_dim = edge_dim
        if edge_dim % heads != 0:
            self.edge_proj_dim = heads * (
                edge_dim // heads + 1
            )
            self.edge_proj_adjust = nn.Linear(
                edge_dim, self.edge_proj_dim
            )
        else:
            self.edge_proj_adjust = None

        self.input_adjust = nn.Linear(
            input_dim, output_dim
        )
        self.node_proj   = nn.Linear(
            output_dim, output_dim
        )
        self.edge_proj   = nn.Linear(
            self.edge_proj_dim, output_dim
        )
        self.bond_attn      = BondFastAttention(
            hidden_size=output_dim,
            heads=heads,
            dropout=dropout_rate,
            activation=F.relu
        )
        self.bond_residual  = SublayerConnection(
            dropout=dropout_rate
        )
        self.residual = (nn.Linear(
            input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )
        self.layer_norm  = GraphNorm(output_dim)
        self.att_dropout = nn.Dropout(dropout_rate)

    def construct_b_scope(self, batch):
        sb, idx = torch.sort(batch)
        cnt     = torch.bincount(sb)
        scopes, start = [], 0
        for c in cnt:
            scopes.append((start, c.item()))
            start += c.item()
        return scopes, idx

    def message(
        self, x_j,
        edge_attr,
        batch=None):

        xj = self.input_adjust(x_j)
        if self.edge_proj_adjust:
            edge_attr = self.edge_proj_adjust(edge_attr)
        bond_message = self.edge_proj(edge_attr)
        if batch is not None:
            scopes, idx = self.construct_b_scope(batch)
            bond_message = bond_message[idx]
        else:
            scopes = [(0, bond_message.size(0))]
        attn_out = self.bond_attn(bond_message, scopes)
        return self.bond_residual(
            self.node_proj(xj), attn_out
        ).view(-1, attn_out.size(-1))

    def forward(
        self, x,
        edge_index,
        edge_attr,
        batch=None):

        x_in = x.to(device)
        edge_attr = edge_attr.to(device)
        edge_index= edge_index.to(device)
        row, _ = edge_index
        deg = degree(row, x.size(0), 
                    dtype=x.dtype).to(device)
        inv = deg.pow(-0.5)
        inv[torch.isinf(inv)] = 0
        x = x * inv.unsqueeze(-1)
        batch_edge = batch[row] if batch is not None else None
        out = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr,
            batch=batch_edge
        )
        out = out + self.residual(x_in)
        out = self.layer_norm(out, batch)
        out = self.att_dropout(out)
    
        return out
