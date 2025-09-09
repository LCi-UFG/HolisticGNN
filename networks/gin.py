import torch
import torch.nn as nn
from torch_geometric.nn import (
    global_add_pool, 
    JumpingKnowledge
    )

from loss import SupervisedUncertainty
from layers import GINLayer
from activation import get_activation


class GINet(nn.Module):
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
        eps,
        num_tasks,
        task_type='classification'
    ):

        super(GINet, self).__init__()

        self.uncertainty = SupervisedUncertainty(
            num_tasks=num_tasks,
            task_type=task_type
        )
        self.jump = JumpingKnowledge(mode='cat')
        self._jk_out_dim = sum(agg_hidden_dims)
        input_dims = [node_dim] + agg_hidden_dims[:-1]
        self.virtualnode_embedding = nn.ParameterList([
            nn.Parameter(torch.zeros(1, d))
            for d in input_dims
        ])
        self.virtualnode_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(agg_hidden_dims[i], input_dims[i]),
                nn.LayerNorm(input_dims[i]),
                nn.ReLU(),
                nn.Linear(input_dims[i], input_dims[i])
            )
            for i in range(num_agg_layers)
        ])
        self.agg_layers = nn.ModuleList([
            GINLayer(
                input_dims[i],
                agg_hidden_dims[i],
                edge_dim,
                num_lin_layers,
                eps
            )
            for i in range(num_agg_layers)
        ])
        self.lin_layers = nn.ModuleList()
        for i, out_dim in enumerate(lin_hidden_dims):
            in_dim = (
                self._jk_out_dim
                if i == 0 else lin_hidden_dims[i - 1]
            )
            self.lin_layers += [
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
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
        return_penultimate=False
    ):
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
        x = self.jump(xs)
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