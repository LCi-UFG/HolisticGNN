import os
import json
import torch


def save_model(
    model, 
    out_path, 
    filename):

    os.makedirs(out_path, exist_ok=True)
    if not filename.endswith(".pth"):
        filename += ".pth"
    filepath = os.path.join(out_path, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def save_params(
    params, 
    out_path, 
    filename):

    if not filename.endswith(".json"):
        filename += ".json"
    filepath = os.path.join(out_path, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Best parameters saved to {filepath}")


def save_thresholds(
    best_thresholds, 
    out_path='../output/calibration/thresholds.json'):

    try:
        dir_path = os.path.dirname(out_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        absolute_path = os.path.abspath(out_path)
        with open(absolute_path, 'w') as file:
            json.dump(best_thresholds, file, indent=4)
    except Exception as e:
        print(f"Failed to save best thresholds: {e}")


def save_embeddings(
    model,
    data_loader,
    epoch,
    out_path="../output/embeddings"):

    os.makedirs(out_path, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            batches = batch if isinstance(
                batch, list) else [batch]
            for b in batches:
                b = b.to(device)
                emb = model(
                    b,
                    save_embeddings=False,
                    return_penultimate=not getattr(
                        model, 'use_projector', False)
                ).cpu()
                all_embeddings.append(emb)

                if hasattr(b, 'y') and b.y is not None:
                    all_labels.append(b.y.cpu())

    embeddings_tensor = torch.cat(
        all_embeddings, dim=0)
    payload = {'embeddings': embeddings_tensor}

    if all_labels:
        labels_tensor = torch.cat(all_labels, dim=0)
        payload['labels'] = labels_tensor

    filename = os.path.join(
        out_path, f"embeddings_epoch_{epoch+1}.pt"
        )
    torch.save(payload, filename)