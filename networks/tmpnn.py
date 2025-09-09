import torch
import torch.nn as nn
from torch_geometric.utils import (
    get_laplacian, 
    to_dense_adj
    )
from torch_geometric.nn import Set2Set

from utils import device
from loss import SupervisedUncertainty
from attention import (
    MultiAtomAttention, 
    SublayerConnection
    )
from layers import tMPNNLayer
from activation import get_activation


class tMPNNet(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        agg_hidden_dims,
        num_agg_layers,
        lin_hidden_dims,
        num_lin_layers,
        activation,
        dropout_rate,
        num_tasks,
        latent_dim,
        task_type='classification',
        embedding_dim=None):

        super(tMPNNet, self).__init__()

        self.uncertainty = SupervisedUncertainty(
            num_tasks=num_tasks, 
            task_type=task_type
            )

        heads = 6
        self.agg_layers = nn.ModuleList()

        for i in range(num_agg_layers):
            in_dim  = node_dim if i == 0 else agg_hidden_dims[i-1]
            out_dim = agg_hidden_dims[i]
            self.agg_layers.append(
                tMPNNLayer(in_dim, out_dim, 
                    edge_dim, 
                    dropout_rate, 
                    heads)
                    )
        self.atom_attention = MultiAtomAttention(
            node_feat_dim=agg_hidden_dims[-1],
            dropout=dropout_rate,
            activation=activation,
            device=device,
            heads=heads
            )
        self.atom_residual = SublayerConnection(
            dropout=dropout_rate
            )
        self.set2set = Set2Set(
            agg_hidden_dims[-1], processing_steps=3
            )
        self.lin_layers = nn.ModuleList()
        for i in range(num_lin_layers):
            in_dim  = 2 * agg_hidden_dims[-1
                    ] if i == 0 else lin_hidden_dims[i-1]
            out_dim = lin_hidden_dims[i]
            self.lin_layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    get_activation(activation),
                    nn.Dropout(dropout_rate)
                    )
                )
        self.embedding_dim = (
            lin_hidden_dims[-1]
            if embedding_dim is None
            else embedding_dim
            )
        self.embedding_layer = nn.Linear(
            lin_hidden_dims[-1], 
            self.embedding_dim
            )

        self.output_layer = nn.Linear(
            self.embedding_dim, num_tasks)

    def forward(
        self,
        data,
        save_embeddings=False,
        return_penultimate=False):

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr  = data.edge_attr.to(device)
        batch = data.batch.to(device)

        for layer in self.agg_layers:
            x = layer(x, edge_index, edge_attr, batch)

        lap_idx, lap_w = get_laplacian(
            edge_index,
            torch.ones(edge_index.size(1), device=device),
            normalization='sym'
            )
        laplacian_matrix = to_dense_adj(
            lap_idx, edge_attr=lap_w,
            max_num_nodes=x.size(0)
            )[0]
        adjacency_matrix = to_dense_adj(
            edge_index,
            batch=batch,
            max_num_nodes=x.size(0)
            )[0]
        att_out, _ = self.atom_attention(
            x, edge_index, batch,
            laplacian_matrix,
            adjacency_matrix
            )
        att_out = self.atom_residual(x, att_out)
        x = self.set2set(att_out, batch)

        for layer in self.lin_layers:
            x = layer(x)

        embeddings = self.embedding_layer(x)
        penultimate = embeddings.clone()
        out = self.output_layer(embeddings)

        if save_embeddings:
            self.saved_embeddings.append(
                penultimate.detach().cpu())
        if return_penultimate:
            return penultimate
        
        return out
        