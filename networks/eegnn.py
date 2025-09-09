import torch.nn as nn
from torch_geometric.nn import global_add_pool

from loss import SupervisedUncertainty
from layers import EEGNNLayer
from activation import get_activation


class EEGNNet(nn.Module):
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
        num_components,
        concentration,
        prior_concentration,
        mcmc_iters,
        num_tasks,
        latent_dim=None,
        task_type='classification',
        embedding_dim=None):

        super(EEGNNet, self).__init__()
        
        self.uncertainty = SupervisedUncertainty(
            num_tasks=num_tasks,
            task_type=task_type
            )

        self.agg_layers = nn.ModuleList([
            EEGNNLayer(
                input_dim=(node_dim if i == 0 else agg_hidden_dims[i-1]),
                output_dim=agg_hidden_dims[i],
                edge_dim=edge_dim,
                dropout_rate=dropout_rate,
                num_components=num_components,
                concentration=concentration,
                prior_concentration=prior_concentration,
                mcmc_iters=mcmc_iters)
            for i in range(num_agg_layers)]
            )
        self.lin_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    (agg_hidden_dims[-1] 
                     if i == 0 else lin_hidden_dims[i-1]),
                    lin_hidden_dims[i]),
                get_activation(activation),
                nn.Dropout(dropout_rate))
            for i in range(num_lin_layers)]
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
            x = layer(x, edge_index, edge_attr)
        x = global_add_pool(x, batch)

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