import torch.nn as nn
from initialization import (
    attentive_weights,
    eegnn_weights,
    gat_weights,
    gin_weights,
    mpnn_weights,
    tmpnn_weights
    )
from utils import set_seed
         

def attentive_resets(model, seed=42):
    set_seed(seed)
    for layer in model.agg_layers:
        attentive_weights(layer.node_proj)
        attentive_weights(layer.layer_norm)
        attentive_weights(layer.edge_encoder)
        attentive_weights(layer.msg_mlp)
        attentive_weights(layer.attn_mlp)
        if hasattr(layer, 'gru') and hasattr(
            layer.gru, 'reset_parameters'):
            layer.gru.reset_parameters()
    attentive_weights(model.readout_mlp)
    if hasattr(model.mol_gru, 'reset_parameters'):
        model.mol_gru.reset_parameters()
    for lin_seq in model.lin_layers:
        attentive_weights(lin_seq)
    attentive_weights(model.embedding_layer)
    attentive_weights(model.output_layer)


def gat_resets(model, seed=42):
    set_seed(seed)
    for layer in model.agg_layers:
        gat_weights(layer.conv)
        gat_weights(layer.res_connection)
    for norm in model.norm_layers:
        gat_weights(norm)
    for m in model.lin_layers:
        gat_weights(m)
    gat_weights(model.embedding_layer)
    gat_weights(model.output_layer)


def gin_resets(model, seed=42):
    set_seed(seed)
    for layer in model.agg_layers:
        gin_weights(layer.conv)
        gin_weights(layer.control_gate)
    for m in model.lin_layers:
        gin_weights(m)
    gin_weights(model.embedding_layer)
    gin_weights(model.output_layer)


def mpnn_resets(model, seed=42):
    set_seed(seed)
    for layer in model.agg_layers:
        mpnn_weights(layer)
    for lin_seq in model.lin_layers:
        mpnn_weights(lin_seq)
    mpnn_weights(model.embedding_layer)
    mpnn_weights(model.output_layer)

def eegnn_resets(model, seed=42):
    set_seed(seed)
    for layer in model.agg_layers:
        eegnn_weights(layer)
    for lin_seq in model.lin_layers:
        eegnn_weights(lin_seq)
    eegnn_weights(model.embedding_layer)
    eegnn_weights(model.output_layer)


def tmpnn_resets(model, seed=42):
    set_seed(seed)
    for layer in model.agg_layers:
        if hasattr(layer, 'tmpnn_resets'):
            layer.tmpnn_resets()
        else:
            tmpnn_weights(layer)
    if hasattr(model, 'atom_attention'):
        for attr in [
            'W_att_q', 
            'W_att_k', 
            'W_att_v', 
            'W_att_o']:
            weight_module = getattr(
                model.atom_attention, attr, None)
            if weight_module is not None:
                tmpnn_weights(weight_module)
        if hasattr(model.atom_attention, 'norm'):
            nn.init.ones_(
                model.atom_attention.norm.weight.data)
            nn.init.zeros_(
                model.atom_attention.norm.bias.data)
        if hasattr(model.atom_attention, 'central_encoder'):
            central = model.atom_attention.central_encoder
            if hasattr(central, 'in_embed'):
                central.in_embed.data.normal_(
                    mean=0.0, std=0.02)
            if hasattr(central, 'out_embed'):
                central.out_embed.data.normal_(
                    mean=0.0, std=0.02)
    for lin_seq in model.lin_layers:
        tmpnn_weights(lin_seq)
    tmpnn_weights(model.embedding_layer)
    tmpnn_weights(model.output_layer)
