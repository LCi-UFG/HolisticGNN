import torch
import torch.nn as nn
from torch_geometric.utils import (
    to_dense_adj, 
    degree
    )
from utils import device


class RWPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_steps = 10
        self.hidden_dim = 64
        self.encoding_dim = 64

        self.mlp = nn.Sequential(
            nn.Linear(self.num_steps, 
                self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 
                self.encoding_dim)
            )
        
    def forward(
        self,
        edge_index,
        num_nodes):

        edge_index = edge_index.to(device)
        adj = to_dense_adj(edge_index,
                max_num_nodes=num_nodes)[0].to(device)
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1e-10)
        P = adj / deg
        probs = [torch.eye(num_nodes, device=device)]
        for _ in range(1, self.num_steps):
            probs.append(torch.matmul(probs[-1], P))
        rw_matrix = torch.stack(probs, dim=-1)
        rw_diag = rw_matrix[
            torch.arange(num_nodes),
            torch.arange(num_nodes), :
            ]
        rw_encoded = self.mlp(rw_diag)
        rw_encoded = torch.nn.functional.normalize(
            rw_encoded, p=2, dim=1
            )
        rw_bias = rw_encoded @ rw_encoded.T

        return rw_bias.unsqueeze(0)


class CentralityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_in_deg = 64
        self.max_out_deg = 64
        self.node_feat_dim = 32

        self.in_embed = nn.Parameter(
            torch.randn(
                self.max_in_deg, self.node_feat_dim)
                )
        self.out_embed = nn.Parameter(
            torch.randn(
                self.max_out_deg, self.node_feat_dim)
                )
        
    def forward(
        self,
        edge_index_list):

        all_encodings = []
        for edge_index in edge_index_list:
            num_nodes = edge_index.max().item() + 1
            deg_in = degree(
                edge_index[1],
                num_nodes=num_nodes
                ).long()
            deg_out = degree(
                edge_index[0],
                num_nodes=num_nodes
                ).long()
            deg_in = deg_in.clamp(
                max=self.in_embed.size(0) - 1
                )
            deg_out = deg_out.clamp(
                max=self.out_embed.size(0) - 1
                )
            encoding = (
                self.in_embed[deg_in] + self.out_embed[deg_out]
                )
            all_encodings.append(encoding)

        return torch.cat(all_encodings, dim=0)


class SwiGLU(nn.Module):
    def __init__(self, input_dim):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(
            input_dim, input_dim)
        self.fc2 = nn.Linear(
            input_dim, input_dim)

    def forward(self, x):
        swish_part = self.fc1(x) * torch.sigmoid(self.fc1(x))
        gate = torch.sigmoid(self.fc2(x))

        return swish_part * gate


def filter_graph(
    laplacian_matrix,
    initial_signal,
    num_filter_steps):

    out = initial_signal
    signal_list = [out.unsqueeze(-1)]
    for _ in range(num_filter_steps - 1):
        out = laplacian_matrix @ out
        signal_list.append(out.unsqueeze(-1))

    return torch.cat(
        signal_list, dim=-1
        )


class PEARLEncoder(nn.Module):
    def __init__(self, phi=nn.Identity(), basis=None):
        super().__init__()
        self.num_filter_steps = 16
        self.num_lin_layers    = 2
        self.lin_hidden_dim    = 16
        self.lin_out_dim       = 16

        self.phi   = phi
        self.basis = basis

        if self.num_lin_layers > 0:
            if self.num_lin_layers == 1:
                assert self.lin_hidden_dim == self.lin_out_dim
            self.layers = nn.ModuleList([
                nn.Linear(
                    self.num_filter_steps if i == 0
                        else self.lin_hidden_dim,
                    self.lin_hidden_dim if i < self.num_lin_layers - 1
                        else self.lin_out_dim
                    )
                for i in range(self.num_lin_layers)
                ])
            self.norms = nn.ModuleList([
                nn.BatchNorm1d(
                    self.lin_hidden_dim if i < self.num_lin_layers - 1
                        else self.lin_out_dim
                    )
                for i in range(self.num_lin_layers)
                ])
        self.activation = SwiGLU(
            self.lin_hidden_dim
            )

    def forward(
        self,
        laplacian_matrix,
        adjacency_matrix,
        edge_index):

        filtered = filter_graph(
            laplacian_matrix,
            adjacency_matrix,
            self.num_filter_steps
            )
        if self.basis is None:
            out = self.phi(filtered)
        else:
            out = self.phi(filtered, 
            edge_index, self.basis
            )

        return torch.mean(out, dim=-1)

    def out_dims(self):
        return self.phi.out_dims