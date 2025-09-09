import torch.nn as nn
from torch_geometric.nn import GATv2Conv 


def attentive_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Sequential):
        for sublayer in m:
            attentive_weights(sublayer)
    elif hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def gat_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, GATv2Conv):
        m.reset_parameters()
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Sequential):
        for sub in m:
            gat_weights(sub)
    elif hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def gin_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Sequential):
        for sub in m:
            gin_weights(sub)
    elif hasattr(m, 'reset_parameters'):
        m.reset_parameters()
        

def mpnn_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Sequential):
        for sub in m:
            mpnn_weights(sub)
    elif hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def eegnn_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Sequential):
        for sub in m:
            eegnn_weights(sub)
    elif hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def tmpnn_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.MultiheadAttention):
        nn.init.xavier_uniform_(m.in_proj_weight.data)
        if m.in_proj_bias is not None:
            m.in_proj_bias.data.zero_()
        if hasattr(m, 'out_proj'):
            nn.init.xavier_uniform_(m.out_proj.weight.data)
            if m.out_proj.bias is not None:
                m.out_proj.bias.data.zero_()
    elif isinstance(m, nn.LayerNorm
            ) or isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Sequential):
        for sub in m:
            tmpnn_weights(sub)
    elif hasattr(m, 'reset_parameters'):
        m.reset_parameters()