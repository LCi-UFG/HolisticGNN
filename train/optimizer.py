import torch.optim as optim

from utils import device, set_seed
from attentive import AttentiveNet
from gat import GATNet
from gin import GINet
from mpnn import MPNNet
from eegnn import EEGNNet
from tmpnn import tMPNNet

from hyperparams import (
    configure_optimizer,
    configure_attentive,
    configure_gat,
    configure_gin,
    configure_mpnn,
    configure_eegnn,
    configure_tmpnn,
    ) 
from resets import (
    attentive_resets,
    gat_resets,
    gin_resets,
    mpnn_resets,
    eegnn_resets,
    tmpnn_resets
    )
from trainer import (
    get_loss, 
    train_model
    )


def objective(
    trial, 
    node_dim, 
    edge_dim, 
    train_loader, 
    val_loader, 
    num_tasks=None, 
    architecture_type='mpnn', 
    task_type='classification', 
    use_uncertainty=False):

    set_seed(42)
    if architecture_type == 'attentive':
        model = configure_attentive(
            trial, 
            node_dim, 
            edge_dim, 
            num_tasks
            )
    elif architecture_type == 'gat':
        model = configure_gat(
            trial, 
            node_dim, 
            edge_dim, 
            num_tasks
            )
    elif architecture_type == 'gin':
        model = configure_gin(
            trial, 
            node_dim, 
            edge_dim, 
            num_tasks
            )
    elif architecture_type == 'mpnn':
        model = configure_mpnn(
            trial, 
            node_dim, 
            edge_dim, 
            num_tasks
            )
    elif architecture_type == 'eegnn':
        model = configure_eegnn(
            trial, 
            node_dim, 
            edge_dim, 
            num_tasks, 
            )
    elif architecture_type == 'tmpnn':
        model = configure_tmpnn(
            trial, 
            node_dim, 
            edge_dim, 
            num_tasks, 
            )
    else:
        raise ValueError(
            f"Invalid architecture: {architecture_type}")

    model.to(device)

    if architecture_type == 'attentive':
        attentive_resets(model)
    elif architecture_type == 'gat':
        gat_resets(model)
    elif architecture_type == 'gin':
        gin_resets(model)
    elif architecture_type == 'mpnn':
        mpnn_resets(model)
    elif architecture_type == 'eegnn':
        eegnn_resets(model)
    elif architecture_type == 'tmpnn':
        tmpnn_resets(model)

    loss_fn = get_loss(
        model,
        task_type, 
        num_tasks, 
        use_uncertainty
        )
    optimizer = configure_optimizer(
        trial, model
        )
    _, min_val_loss, _, _ = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        loss_fn, 
        num_epochs=2000,  
        patience=5, 
        delta=0.01, 
        window_size=5, 
        best_model=False,
        enable_pruning=True
        )
    
    return min_val_loss


def retrain(
    best_params, 
    node_dim, 
    edge_dim, 
    train_loader, 
    val_loader, 
    num_tasks,
    architecture_type='mpnn',
    task_type='classification', 
    use_uncertainty=False):

    set_seed(42)
    agg_hidden_dims = [
        best_params[f'agg_hidden_dim_{i+1}'] 
        for i in range(
            best_params['num_agg_layers'])
        ]
    lin_hidden_dims = [
        best_params[f'lin_hidden_dim_{i+1}'] 
        for i in range(
            best_params['num_lin_layers'])
        ] 
    if architecture_type == 'attentive':
        model = AttentiveNet(
            node_dim, 
            edge_dim, 
            agg_hidden_dims, 
            best_params['num_agg_layers'], 
            lin_hidden_dims, 
            best_params['num_lin_layers'], 
            best_params['activation'], 
            best_params['dropout_rate'], 
            best_params['num_timesteps'],
            num_tasks
            )
    
    elif architecture_type == 'gat':
        model = GATNet(
            node_dim, 
            edge_dim, 
            agg_hidden_dims, 
            best_params['num_agg_layers'], 
            lin_hidden_dims, 
            best_params['num_lin_layers'], 
            best_params['activation'], 
            best_params['dropout_rate'],
            best_params['heads'], 
            num_tasks
            )
    
    elif architecture_type == 'gin':
        model = GINet(
            node_dim, 
            edge_dim, 
            agg_hidden_dims, 
            best_params['num_agg_layers'], 
            lin_hidden_dims, 
            best_params['num_lin_layers'], 
            best_params['activation'], 
            best_params['dropout_rate'],
            best_params['eps'],
            num_tasks
            )
       
    elif architecture_type == 'mpnn':
        model = MPNNet(
            node_dim, 
            edge_dim, 
            agg_hidden_dims, 
            best_params['num_agg_layers'], 
            lin_hidden_dims, 
            best_params['num_lin_layers'], 
            best_params['activation'], 
            best_params['dropout_rate'], 
            num_tasks
            )
    
    elif architecture_type == 'eegnn':
        model = EEGNNet(
            node_dim, 
            edge_dim, 
            agg_hidden_dims, 
            best_params['num_agg_layers'], 
            lin_hidden_dims, 
            best_params['num_lin_layers'], 
            best_params['activation'], 
            best_params['dropout_rate'], 
            best_params['num_components'],
            best_params['concentration'],
            best_params['prior_concentration'],
            best_params['mcmc_iters'],
            num_tasks,  
            )
    
    elif architecture_type == 'tmpnn':
        model = tMPNNet(
            node_dim, 
            edge_dim, 
            agg_hidden_dims, 
            best_params['num_agg_layers'], 
            lin_hidden_dims, 
            best_params['num_lin_layers'], 
            best_params['activation'], 
            best_params['dropout_rate'],
            num_tasks,
            )
    
    else:
        raise ValueError(
            f"Unknown architecture: {architecture_type}"
            )
    
    model.to(device)

    if architecture_type == 'attentive':
        attentive_resets(model)
    elif architecture_type == 'gat':
        gat_resets(model)
    elif architecture_type == 'gin':
        gin_resets(model)
    elif architecture_type == 'mpnn':
        mpnn_resets(model)
    elif architecture_type == 'eegnn':
        eegnn_resets(model)
    elif architecture_type == 'tmpnn':
        tmpnn_resets(model)

    optimizer = getattr(optim, best_params['optimizer'])(
        model.parameters(), lr=0.0001, 
        weight_decay=best_params['weight_decay']
        )
    loss_fn = get_loss(
        model,
        task_type, 
        num_tasks, 
        use_uncertainty
        ) 
    best_val_loss, min_val_loss, train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        loss_fn, 
        num_epochs=2000,  
        patience=5, 
        delta=0.01, 
        window_size=5, 
        best_model=True,
        enable_pruning=True
        )

    return model, best_val_loss, min_val_loss,  train_losses, val_losses