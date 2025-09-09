import torch
import torch.nn as nn
from torch_geometric.nn import (
    JumpingKnowledge,
    global_add_pool
    )

from loss import SupervisedUncertainty
from layers import GATLayer
from activation import get_activation


class GATNet(nn.Module):
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
        heads,
        num_tasks,
        task_type='classification'):

        super(GATNet, self).__init__()

        self.uncertainty = SupervisedUncertainty(
            num_tasks=num_tasks,
            task_type=task_type
        )
        self.agg_layers = nn.ModuleList()
        dims = [
            agg_hidden_dims[i] *
            (1 if i == num_agg_layers-1 else heads)
            for i in range(num_agg_layers)
        ]
        input_dims = [node_dim] + dims[:-1]
        for i in range(num_agg_layers):
            self.agg_layers.append(
                GATLayer(
                    input_dims[i],
                    dims[i],
                    edge_dim,
                    heads,
                    dropout_rate,
                    concat=(i != num_agg_layers-1)
                )
            )
        self.norm_layers = nn.ModuleList(
            [layer.norm for layer in self.agg_layers]
        )
        self.jk = JumpingKnowledge(mode='cat')
        self.virtualnode_embedding = nn.ParameterList([
            nn.Parameter(torch.zeros(1, d_in))
            for d_in in input_dims
        ])
        self.virtualnode_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_out, d_in),
                nn.LayerNorm(d_in),
                nn.ReLU(),
                nn.Linear(d_in, d_in)
            )
            for d_out, d_in in zip(dims, input_dims)
        ])
        self.lin_layers = nn.ModuleList()
        for i in range(num_lin_layers):
            in_d = (
                sum(dims)
                if i == 0
                else lin_hidden_dims[i-1]
            )
            out_d = lin_hidden_dims[i]
            self.lin_layers += [
                nn.Linear(in_d, out_d),
                nn.LayerNorm(out_d),
                get_activation(activation),
                nn.Dropout(dropout_rate)
            ]
        self.embedding_dim = lin_hidden_dims[-1]
        self.embedding_layer = nn.Linear(
            lin_hidden_dims[-1],
            self.embedding_dim
        )
        self.output_layer = nn.Linear(
            self.embedding_dim,
            num_tasks
        )

    def forward(
        self,
        data,
        save_embeddings=False,
        return_penultimate=False):
        
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch
        )
        num_graphs = batch.max().item() + 1
        v_nodes = [
            emb.expand(num_graphs, -1)
            for emb in self.virtualnode_embedding
        ]
        xs = []
        for i, layer in enumerate(self.agg_layers):
            v_expand = v_nodes[i][batch]
            x = x + v_expand
            x = layer(x, edge_index, edge_attr, batch)
            xs.append(x)
            pooled = global_add_pool(x, batch)
            delta = self.virtualnode_mlp[i](pooled)
            v_nodes[i] = v_nodes[i] + delta
        x = self.jk(xs)
        x = global_add_pool(x, batch)
        for layer in self.lin_layers:
            x = layer(x)
        embeddings = self.embedding_layer(x)
        penultimate = embeddings.clone()
        out = self.output_layer(embeddings)
        if save_embeddings:
            self.saved_embeddings.append(
                penultimate.detach().cpu()
            )
        if return_penultimate:
            return penultimate
        return out