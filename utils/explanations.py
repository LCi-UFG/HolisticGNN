import os
import os
import copy
import random
import numpy as np
import torch
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import (
    get_laplacian,
    to_dense_adj,
    to_scipy_sparse_matrix
    )
from torch.quasirandom import SobolEngine

from utils import (
    decode_one_hot, 
    one_hot
    )
from rules import ATOMIC_NUMBER
from augmentations import (
    atomic_rules,
    charge_rules,
    hybridization_rules
    )
from predictor import predict


def mask_atoms(
    data_raw, idx, 
    seed=None):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    data = copy.deepcopy(data_raw)
    atoms = ATOMIC_NUMBER()
    n_atoms = len(atoms)

    d_end = n_atoms + 8
    c_start, c_end = d_end, d_end + 4
    h_start, h_end = c_end, c_end + 5

    old_atomic = data.x[idx, :n_atoms]
    new_atomic = atomic_rules(old_atomic)
    data.x[idx, :n_atoms] = torch.tensor(
        new_atomic, 
        dtype=torch.float,
        device=data.x.device
        )
    atom_type = decode_one_hot(
        new_atomic, atoms
        )
    old_charge = data.x[
        idx, c_start:c_end
        ]
    new_charge = charge_rules(
        atom_type, old_charge
        )
    data.x[idx, c_start:c_end] = torch.tensor(
        one_hot(new_charge, [-1, 0, 1, 2]),
        dtype=torch.float, 
        device=data.x.device
        )
    old_hybrid = data.x[idx, h_start:h_end]
    new_hybrid = hybridization_rules(
        atom_type, old_hybrid
        )
    data.x[idx, h_start:h_end] = torch.tensor(
        one_hot(new_hybrid, list(range(5))),
        dtype=torch.float, device=data.x.device
        )

    return data


def leave1atom(
    model, raw, device,
    num_perturb=10,
    task_idx=0):

    base = copy.deepcopy(raw)
    base.y = torch.zeros(
        (1,), dtype=torch.float, device=device
        )
    p0_all, _, _ = predict(
        model,
        DataLoader([base], batch_size=1),
        device,
        return_embeddings=False
        )
    p0 = p0_all[0, task_idx]
    n = raw.x.size(0)
    mean = np.zeros(n)
    std = np.zeros(n)
    pos = np.zeros(n)
    neg = np.zeros(n)
    trim = 0.1

    for i in range(n):
        sobol = SobolEngine(1, scramble=True)
        seq = sobol.draw(num_perturb).squeeze()
        deltas = []

        for v in seq:
            seed = int(v.item() * (2**32 - 1))
            masked = mask_atoms(raw, i, seed=seed)
            masked.y = torch.zeros(
                (1,), dtype=torch.float, 
                device=device
                )
            pred2_all, _, _ = predict(
                model, DataLoader([masked], batch_size=1),
                device, return_embeddings=False
                )
            p2 = pred2_all[0, task_idx]
            deltas.append(p0 - p2)

        arr = np.array(deltas, dtype=float)
        k = int(len(arr) * trim)
        if len(arr) > 2 * k:
            arr = np.sort(arr)[k:-k]

        mean[i] = arr.mean()
        std[i] = arr.std(ddof=0)
        pos[i] = np.mean(arr > 0)
        neg[i] = np.mean(arr < 0)

    return mean, std, pos, neg


def cluster_predictions(
    scores, 
    edge_index):

    n = scores.shape[0]
    pos = np.where(scores >= 0)[0]
    neg = np.where(scores < 0)[0]
    edges = edge_index.cpu().numpy().T.tolist()
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    clusters = []

    for idx in (pos, neg):
        if idx.size:
            for comp in nx.connected_components(
                G.subgraph(idx)):
                clusters.append(list(comp)
                )

    return clusters


def plot_contours(
    index, smiles, 
    importance, 
    out_path):

    DrawingOptions.atomLabelFontSize = 35
    DrawingOptions.dotsPerAngstrom = 60
    DrawingOptions.useBWAtomPalette = True

    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    rdDepictor.StraightenDepiction(mol)

    fig = SimilarityMaps.GetSimilarityMapFromWeights(
        mol, importance,
        alpha=0.5, 
        contourLines=6,
        colorMap='bwr', 
        size=(200, 200)
        )
    fig.patch.set_facecolor('white')
    for ax in fig.axes:
        ax.set_facecolor('white')
        ax.axis('off')
        for line in ax.get_lines():
            line.set_color('black')
            line.set_linewidth(2)
        for coll in ax.collections:
            coll.set_edgecolor('black')
            coll.set_facecolor(coll.get_facecolor())
        for txt in ax.texts:
            txt.set_color('black')
            txt.set_fontweight('bold')
            txt.set_path_effects([
                pe.Stroke(
                    linewidth=5,
                    foreground='white'),
                pe.Normal()
                ]
            )
    os.makedirs(out_path, exist_ok=True)
    fig.savefig(
        f"{out_path}/{index}.svg",
        bbox_inches='tight',
        pad_inches=0.6,
        dpi=300,
        facecolor='white'
        )
    
    plt.close(fig)


def view_explanations(
    model, 
    loader, 
    device, 
    out_path,
    num_perturb=10,
    cluster_alpha=0.5,
    smooth_alpha=0.5,
    task_idx=0):

    model.to(device)
    model.eval()
    index = 1

    with torch.no_grad():
        for batch in loader:
            raw = copy.deepcopy(batch)
            smiles = raw.smiles
            batch = batch.to(device)
            bi = batch.batch
            src, dst = batch.edge_index
            counts = torch.bincount(bi)
            counts = counts.cpu().numpy()
            start = 0

            for i, c in enumerate(counts):
                n = int(c)
                mask = (bi[src] == i) & (bi[dst] == i)
                ei = batch.edge_index[:, mask].cpu().clone()
                ei[0] -= start
                ei[1] -= start
                ea = batch.edge_attr[mask].cpu().clone()

                data = Data(
                    x=raw.x[start:start+n].cpu().clone(),
                    edge_index=ei,
                    edge_attr=ea,
                    smiles=smiles[i]
                    )
                mean, _, pos, neg = leave1atom(
                    model, data, device,
                    num_perturb=num_perturb,
                    task_idx=task_idx
                    )
                weight = pos - neg
                node_imp = mean * weight

                if n < 5:
                    imp = node_imp.copy()
                else:
                    clusters = cluster_predictions(
                        node_imp, ei
                        )
                    imp = np.zeros(n)
                    for comp in clusters:
                        vals = node_imp[comp]
                        m = vals.mean()
                        imp[comp] = m + cluster_alpha * (
                            vals - m
                        )
                    A = to_scipy_sparse_matrix(
                        ei, num_nodes=n
                        )
                    deg = np.array(
                        A.sum(axis=1)
                    ).flatten()
                    Dinv = sp.diags(
                        1.0 / (deg + 1e-8)
                        )
                    smooth = Dinv.dot(
                        A.dot(imp)
                        )
                    imp = (
                        smooth_alpha * imp +
                        (1 - smooth_alpha) * smooth
                        )
                imp = (imp - imp.mean()) / (
                    imp.std() + 1e-8
                    )
                plot_contours(
                    index, data.smiles,
                    imp, out_path
                    )
                
                start += n
                index += 1


def cluster_attentions(
    weights, 
    edge_index):

    n = weights.size(0)
    raw = weights.sum(0).cpu().numpy() / n
    centered = raw - raw.mean()
    attn = np.zeros(n)
    edges = edge_index.cpu().numpy().T.tolist()
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    pos = np.where(centered >= 0)[0]
    neg = np.where(centered < 0)[0]

    for idx in [pos, neg]:
        if idx.size:
            for comp in nx.connected_components(
                G.subgraph(idx)):
                m = list(comp)
                attn[m] = centered[m].mean()

    return attn


def plot_attentions(
    out_path, 
    index, 
    smiles, 
    attn):

    DrawingOptions.atomLabelFontSize = 35
    DrawingOptions.dotsPerAngstrom = 60
    DrawingOptions.useBWAtomPalette = True
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    rdDepictor.StraightenDepiction(mol)
    fig = SimilarityMaps.GetSimilarityMapFromWeights(
        mol, attn,
        alpha=0.5,
        contourLines=6,
        colorMap='bwr',
        size=(200, 200)
        )
    fig.patch.set_facecolor('white')
    for ax in fig.axes:
        ax.set_facecolor('white')
        ax.axis('off')
        for line in ax.get_lines():
            line.set_color('black')
            line.set_linewidth(2)
        for coll in ax.collections:
            coll.set_edgecolor('black')
            coll.set_facecolor(
                coll.get_facecolor()
                )
        for txt in ax.texts:
            txt.set_color('black')
            txt.set_fontweight('bold')
            txt.set_path_effects([
                pe.Stroke(
                    linewidth=5,
                    foreground='white'),
                pe.Normal()
                ]
            )
    os.makedirs(out_path, exist_ok=True)
    fig.savefig(
        f"{out_path}/{index}.svg",
        bbox_inches='tight',
        pad_inches=0.6,
        dpi=300,
        facecolor='white'
        )
    
    plt.close(fig)


def view_attentions(
    model, 
    loader, 
    device, 
    out_path,
    cluster_alpha=0.5,
    smooth_alpha=0.5):

    model.to(device)
    model.eval()
    index = 1

    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)
            smiles = batch_data.smiles
            x = batch_data.x
            edge_index = batch_data.edge_index
            edge_attr = batch_data.edge_attr
            batch_idx = batch_data.batch

            for layer in model.agg_layers:
                x = layer(x, edge_index, edge_attr)
            lap_idx, lap_w = get_laplacian(
                edge_index,
                edge_weight=torch.ones(
                    edge_index.size(1), device=device),
                normalization='sym'
                )
            laplacian_matrix = to_dense_adj(
                lap_idx, edge_attr=lap_w,
                max_num_nodes=x.size(0))[0]
            
            adjacency_matrix = torch.eye(
                x.size(0), device=device
                )
            if hasattr(model, 'atom_attention'):
                att_out, attn = model.atom_attention(
                    x,
                    edge_index,
                    batch_idx,
                    laplacian_matrix,
                    adjacency_matrix
                    )
            else:
                alpha = model.agg_layers[-1].last_attn_weights
                N = x.size(0)
                att = torch.zeros(N, N, device=device)
                src, dst = edge_index
                att[src, dst] = alpha
                attn = (att + att.t()) * 0.5

            attn = attn.detach().cpu()
            counts = torch.bincount(batch_idx).cpu().numpy()
            start = 0

            for i, c in enumerate(counts):
                n = int(c)
                sub_attn = attn[start:start + n, start:start + n]
                mask = (
                    (batch_idx[edge_index[0]] == i) &
                    (batch_idx[edge_index[1]] == i)
                    )
                sub_ei = edge_index[:, mask].cpu().clone()
                sub_ei[0] -= start
                sub_ei[1] -= start

                Amean = cluster_attentions(sub_attn, sub_ei)
                if n < 5:
                    att_vals = Amean.copy()
                else:
                    G = nx.Graph()
                    G.add_nodes_from(range(n))
                    G.add_edges_from(sub_ei.T.numpy().tolist())
                    comps = list(nx.connected_components(G))
                    att_vals = np.zeros(n)
                    for comp in comps:
                        m = list(comp)
                        vals = Amean[m]
                        avg = vals.mean()
                        att_vals[m] = avg + cluster_alpha * (vals - avg)
                    A = sp.coo_matrix(
                        (np.ones(sub_ei.size(1)), (
                            sub_ei[0], sub_ei[1])),
                        shape=(n, n)).tocsr()
                    deg = np.array(A.sum(axis=1)).flatten()
                    Dinv = sp.diags(1.0 / (deg + 1e-8))
                    smooth = Dinv @ A @ att_vals
                    att_vals = smooth_alpha * att_vals + (
                        1 - smooth_alpha) * smooth

                att_vals = (att_vals - att_vals.mean()
                        ) / (att_vals.std() + 1e-8)
                plot_attentions(out_path, index, 
                        smiles[i], att_vals
                        )

                start += n
                index += 1