import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch


def predict(
    model, 
    data_loader, 
    device, 
    task_type='classification', 
    return_embeddings=False):

    model.eval()
    torch.set_grad_enabled(False)
    all_predictions = []
    all_labels = []
    embeddings = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            inputs = batch.to(device)
            if return_embeddings:
                emb = model(
                    inputs, return_embeddings=True
                    )
                embeddings.extend(emb.cpu().numpy())
            outputs = model(inputs)
            if task_type == 'classification':
                predictions = torch.sigmoid(outputs)
            elif task_type == 'regression':
                predictions = outputs
            else:
                raise ValueError(
                    "`Please use classification or regression"
                    )
            all_predictions.extend(
                predictions.cpu().numpy()
                )
            labels = inputs.y.cpu().numpy()
            all_labels.extend(labels)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    embeddings = np.array(embeddings
            ) if return_embeddings else None

    return all_predictions, all_labels, embeddings