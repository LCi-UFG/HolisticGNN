import torch.nn as nn
from torch_geometric.nn import global_add_pool
    
from loss import SupervisedUncertainty
from layers import AttentiveLayer
from activation import get_activation


import torch.nn as nn
from torch_geometric.nn import global_add_pool
from layers import AttentiveLayer
from activation import get_activation


class AttentiveNet(nn.Module):
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
        num_timesteps,
        num_tasks,
        task_type='classification'):

        super(AttentiveNet, self).__init__()

        self.uncertainty = SupervisedUncertainty(
            num_tasks=num_tasks,
            task_type=task_type
        )

        self.saved_embeddings = []

        self.num_timesteps = num_timesteps
        self.agg_layers = nn.ModuleList()
        for i in range(num_agg_layers):
            in_dim = node_dim if i == 0 else agg_hidden_dims[i - 1]
            out_dim = agg_hidden_dims[i]
            self.agg_layers.append(
                AttentiveLayer(
                    in_dim, 
                    out_dim, 
                    edge_dim, 
                    dropout_rate
                )
            )

        last_dim = agg_hidden_dims[-1]
        self.readout_mlp = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            get_activation(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(last_dim, 1),
        )

        self.mol_gru = nn.GRUCell(last_dim, last_dim)
        self.lin_layers = nn.ModuleList()
        for i in range(num_lin_layers):
            in_dim = last_dim if i == 0 else lin_hidden_dims[i - 1]
            out_dim = lin_hidden_dims[i]
            self.lin_layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    get_activation(activation),
                    nn.Dropout(dropout_rate),
                )
            )
            
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
        self, data, 
        save_embeddings=False, 
        return_penultimate=False):

        x, edge_index, edge_attr, batch = (
            data.x, 
            data.edge_index, 
            data.edge_attr, 
            data.batch
        )
        for layer in self.agg_layers:
            x = layer(x, edge_index, edge_attr, batch)

        out = global_add_pool(x, batch).relu_()

        for _ in range(self.num_timesteps):
            out = self.mol_gru(out, out).relu_()

        for lin in self.lin_layers:
            out = lin(out)

        embeddings = self.embedding_layer(out)
        penultimate = embeddings.clone()
        out = self.output_layer(embeddings)

        if save_embeddings:
            self.saved_embeddings.append(
                penultimate.detach().cpu()
            )
        if return_penultimate:
            return penultimate

        return out