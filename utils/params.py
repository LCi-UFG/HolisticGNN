
import os
import json
import torch
import optuna
import pandas as pd

from attentive import AttentiveNet
from gat import GATNet
from gin import GINet
from mpnn import MPNNet
from eegnn import EEGNNet
from tmpnn import tMPNNet


def initialize_optuna():

    OUTPUT_DIR = "../output/optimization/"
    DB_FILE = os.path.join(
        OUTPUT_DIR, 'optuna_study.db'
        )
    STORAGE_NAME = f"sqlite:///{DB_FILE}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    study_name = "optimization_study"

    try:
        existing_studies = optuna.study.get_all_study_summaries(
            storage=STORAGE_NAME)
        study_names = [
            s.study_name for s in existing_studies]
        if study_name in study_names:
            print(f"Study '{study_name}' found in the database")
            study = optuna.load_study(
                study_name=study_name,
                storage=STORAGE_NAME)
        else:
            print(f"Creating a new study")
            study = optuna.create_study(
                study_name=study_name,
                direction='minimize',
                storage=STORAGE_NAME)
    except Exception as e:
        print(f"Error occurred while accessing the study: {e}")
        raise

    return study


def load_params(hyperparams_path):
    with open(hyperparams_path, 'r') as file:
        hyperparameters = json.load(file)
    return hyperparameters


def load_model(
    model_path, 
    architecture_type, 
    params, 
    node_dim, 
    edge_dim, 
    num_tasks):

    agg_hidden_dims = [
        params[f'agg_hidden_dim_{i+1}'] 
        for i in range(params['num_agg_layers'])
        ]
    lin_hidden_dims = [
        params[f'lin_hidden_dim_{i+1}'] 
        for i in range(params['num_lin_layers'])
        ]

    if architecture_type == 'attentive':
        model = AttentiveNet(
            node_dim=node_dim,
            edge_dim=edge_dim,
            agg_hidden_dims=agg_hidden_dims,
            num_agg_layers=params['num_agg_layers'],
            lin_hidden_dims=lin_hidden_dims,
            num_lin_layers=params['num_lin_layers'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate'],
            num_timesteps=params['num_timesteps'],
            num_tasks=num_tasks
            )
    elif architecture_type == 'gat':
        model = GATNet(
            node_dim=node_dim,
            edge_dim=edge_dim,
            agg_hidden_dims=agg_hidden_dims,
            num_agg_layers=params['num_agg_layers'],
            lin_hidden_dims=lin_hidden_dims,
            num_lin_layers=params['num_lin_layers'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate'],
            heads=params['heads'],
            num_tasks=num_tasks
            )
    elif architecture_type == 'gin':
        model = GINet(
            node_dim=node_dim,
            edge_dim=edge_dim,
            agg_hidden_dims=agg_hidden_dims,
            num_agg_layers=params['num_agg_layers'],
            lin_hidden_dims=lin_hidden_dims,
            num_lin_layers=params['num_lin_layers'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate'],
            eps=params['eps'],
            num_tasks=num_tasks
            )
    elif architecture_type == 'mpnn':
        model = MPNNet(
            node_dim=node_dim,
            edge_dim=edge_dim,
            agg_hidden_dims=agg_hidden_dims,
            num_agg_layers=params['num_agg_layers'],
            lin_hidden_dims=lin_hidden_dims,
            num_lin_layers=params['num_lin_layers'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate'],
            num_tasks=num_tasks
            )
    elif architecture_type == 'eegnn':
        model = EEGNNet(
            node_dim=node_dim,
            edge_dim=edge_dim,
            agg_hidden_dims=agg_hidden_dims,
            num_agg_layers=params['num_agg_layers'],
            lin_hidden_dims=lin_hidden_dims,
            num_lin_layers=params['num_lin_layers'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate'],
            num_components=params['num_components'],
            concentration=params['concentration'],
            prior_concentration=params['prior_concentration'],
            mcmc_iters=params['mcmc_iters'],
            num_tasks=num_tasks,  
            use_projector=False,  
            latent_dim=None 
            )
    elif architecture_type == 'tmpnn':
        model = tMPNNet(
            node_dim=node_dim,
            edge_dim=edge_dim,
            agg_hidden_dims=agg_hidden_dims,
            num_agg_layers=params['num_agg_layers'],
            lin_hidden_dims=lin_hidden_dims,
            num_lin_layers=params['num_lin_layers'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate'],
            num_tasks=num_tasks,  
            use_projector=False,  
            latent_dim=None 
            )
    else:
        raise ValueError("Invalid architecture type")
    
    model.load_state_dict(
        torch.load(model_path, weights_only=True),
        strict=False
        )
    
    return model


def load_embeddings(directory_path, epoch):
    target = f"embeddings_epoch_{epoch+1}"
    for filename in os.listdir(directory_path):
        if target in filename:
            data = torch.load(os.path.join(
                directory_path, filename))
            embeddings = data['embeddings']
            labels = data.get('labels', None)
            return embeddings, labels
    raise FileNotFoundError(
        f"No file found for epoch {epoch+1} in {directory_path}"
    )