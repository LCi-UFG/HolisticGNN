import torch.optim as optim

from attentive import AttentiveNet
from gat import GATNet
from gin import GINet
from mpnn import MPNNet
from eegnn import EEGNNet
from tmpnn import tMPNNet

def configure_optimizer(trial, model):
    
    optimizer_name = trial.suggest_categorical(
        'optimizer', [
            'Adam', 'RMSprop', 'SGD'])
    weight_decay = trial.suggest_float(
        'weight_decay', 1e-7, 1e-3)
    lr = 0.0001
    optimizer = getattr(optim, optimizer_name)(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
        )
    return optimizer


def configure_attentive(
    trial, 
    node_dim, 
    edge_dim, 
    num_tasks):
    
    agg_hidden_dims = [
        trial.suggest_int(
            f'agg_hidden_dim_{i+1}', 10, 500) 
        for i in range(
            trial.suggest_int(
                'num_agg_layers', 2, 6)
            )
        ]
    lin_hidden_dims = [
        trial.suggest_int(
            f'lin_hidden_dim_{i+1}', 10, 500) 
        for i in range(
            trial.suggest_int(
                'num_lin_layers', 2, 4)
            )
        ]
    activation_choice = trial.suggest_categorical(
        'activation', ['relu', 'leakyrelu', 
            'elu', 'gelu', 'selu'
            ]
        )
    dropout_rate = trial.suggest_float(
        'dropout_rate', 0.2, 0.6
        )
    num_timesteps = trial.suggest_int(
        'num_timesteps', 1, 3
        )
    
    model = AttentiveNet(
        node_dim, 
        edge_dim, 
        agg_hidden_dims, 
        len(agg_hidden_dims), 
        lin_hidden_dims, 
        len(lin_hidden_dims), 
        activation_choice, 
        dropout_rate,
        num_timesteps, 
        num_tasks
        )
    
    return model


def configure_gat(
    trial, 
    node_dim, 
    edge_dim, 
    num_tasks):

    agg_hidden_dims = [
        trial.suggest_int(
            f'agg_hidden_dim_{i + 1}', 10, 500)
        for i in range(
            trial.suggest_int(
                'num_agg_layers', 2, 6)
            )
        ]
    lin_hidden_dims = [
        trial.suggest_int(
            f'lin_hidden_dim_{i + 1}', 10, 500)
        for i in range(
            trial.suggest_int(
                'num_lin_layers', 2, 4)
            )
        ]
    activation_choice = trial.suggest_categorical(
        'activation', ['relu', 'leakyrelu', 
            'elu', 'gelu', 'selu'
            ]
        )
    dropout_rate = trial.suggest_float(
        'dropout_rate', 0.2, 0.6
        )
    heads = trial.suggest_int(
        'heads', 1, 12
        )

    model = GATNet(
        node_dim,
        edge_dim,
        agg_hidden_dims,
        len(agg_hidden_dims),
        lin_hidden_dims,
        len(lin_hidden_dims),
        activation_choice,
        dropout_rate,
        heads,
        num_tasks
        )

    return model


def configure_gin(
    trial, 
    node_dim, 
    edge_dim, 
    num_tasks):

    agg_hidden_dims = [
        trial.suggest_int(
            f'agg_hidden_dim_{i+1}', 10, 500)
        for i in range(
            trial.suggest_int(
                'num_agg_layers', 2, 6)
            )
        ]
    lin_hidden_dims = [
        trial.suggest_int(
            f'lin_hidden_dim_{i+1}', 10, 500)
        for i in range(
            trial.suggest_int(
                'num_lin_layers', 2, 4)
            )
        ]
    activation_choice = trial.suggest_categorical(
        'activation', ['relu', 'leakyrelu', 
            'elu', 'gelu', 'selu'
            ]
        )
    dropout_rate = trial.suggest_float(
        'dropout_rate', 0.2, 0.6
        )
    eps = trial.suggest_float(
        'eps', 0, 1
        )

    model = GINet(
        node_dim,
        edge_dim,
        agg_hidden_dims,
        len(agg_hidden_dims),
        lin_hidden_dims,
        len(lin_hidden_dims),
        activation_choice,
        dropout_rate,
        eps,
        num_tasks
        )

    return model


def configure_mpnn(
    trial, 
    node_dim, 
    edge_dim, 
    num_tasks):
    
    agg_hidden_dims = [
        trial.suggest_int(
            f'agg_hidden_dim_{i+1}', 10, 500) 
        for i in range(
            trial.suggest_int(
                'num_agg_layers', 2, 6)
            )
        ]
    lin_hidden_dims = [
        trial.suggest_int(
            f'lin_hidden_dim_{i+1}', 10, 500) 
        for i in range(
            trial.suggest_int(
                'num_lin_layers', 2, 4)
            )
        ]
    activation_choice = trial.suggest_categorical(
        'activation', ['relu', 'leakyrelu', 
            'elu', 'gelu', 'selu'
            ]
        )
    dropout_rate = trial.suggest_float(
        'dropout_rate', 0.2, 0.6
        )
    
    model = MPNNet(
        node_dim, 
        edge_dim, 
        agg_hidden_dims, 
        len(agg_hidden_dims), 
        lin_hidden_dims, 
        len(lin_hidden_dims), 
        activation_choice, 
        dropout_rate, 
        num_tasks
        )
    
    return model


def configure_eegnn(
    trial, 
    node_dim, 
    edge_dim, 
    num_tasks,  
    latent_dim=None):
    
    agg_hidden_dims = [
        trial.suggest_int(f'agg_hidden_dim_{i+1}', 10, 500) 
        for i in range(
            trial.suggest_int('num_agg_layers', 2, 6))
            ]
    lin_hidden_dims = [
        trial.suggest_int(f'lin_hidden_dim_{i+1}', 10, 500) 
        for i in range(
            trial.suggest_int('num_lin_layers', 2, 4))
            ]
    activation_choice = trial.suggest_categorical(
        'activation', ['relu', 'leakyrelu', 'elu', 'gelu', 'selu']
        )
    dropout_rate = trial.suggest_float(
        'dropout_rate', 0.2, 0.6
        )
    num_components = trial.suggest_int(
        'num_components', 7, 12
        )
    concentration = trial.suggest_float(
        'concentration', 0.1, 5.0
        )
    prior_concentration = trial.suggest_float(
        'prior_concentration', 0.01, 10.0
        )
    mcmc_iters = trial.suggest_int(
        'mcmc_iters', 1, 3
        )
    # latent_dim = trial.suggest_int(
    #     'latent_dim', 10, 500) if use_projector else None
    
    model = EEGNNet(
        node_dim, 
        edge_dim, 
        agg_hidden_dims, 
        len(agg_hidden_dims), 
        lin_hidden_dims, 
        len(lin_hidden_dims), 
        activation_choice, 
        dropout_rate,
        num_components,
        concentration,
        prior_concentration,
        mcmc_iters,
        num_tasks,  
        latent_dim
        )
    
    return model


def configure_tmpnn(
    trial, 
    node_dim, 
    edge_dim, 
    num_tasks,  
    latent_dim=None):

    agg_hidden_dims = [
        trial.suggest_categorical(f'agg_hidden_dim_{i+1}', 
        [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 
         144, 180, 192, 240, 300, 324, 348, 372, 
         396, 420, 444, 456, 480, 516]) 
        for i in range(
            trial.suggest_int('num_agg_layers', 2, 6)
            )
        ]
    lin_hidden_dims = [
        trial.suggest_int(f'lin_hidden_dim_{i+1}', 10, 500) 
        for i in range(
            trial.suggest_int('num_lin_layers', 2, 4)
            )
        ]
    activation_choice = trial.suggest_categorical(
        'activation', ['relu', 'leakyrelu', 'elu', 'gelu', 'selu']
        )
    dropout_rate = trial.suggest_float(
        'dropout_rate', 0.2, 0.6
        )
    # latent_dim = trial.suggest_int(
    #     'latent_dim', 10, 500) if use_projector else None

    model = tMPNNet(
        node_dim,
        edge_dim,
        agg_hidden_dims,
        len(agg_hidden_dims),
        lin_hidden_dims,
        len(lin_hidden_dims),
        activation_choice,
        dropout_rate,
        num_tasks,
        latent_dim
        )

    return model