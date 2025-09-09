import copy
import torch
import optuna
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    LambdaLR
    )

from utils import (
    device, 
    clip_gradients
    )

from loss import (
    MaskedLoss,
    prepare_masked_data
    )

from save import save_embeddings


def get_loss(
    model,
    task_type,
    num_tasks,
    use_uncertainty):

    if not use_uncertainty:
        return MaskedLoss(
            task_type=task_type,
            num_tasks=num_tasks,
            )
    def loss_fn(y_pred, y_true):
        y_t, y_p, mask = prepare_masked_data(
            y_true, y_pred)
        return model.uncertainty(y_p, y_t, mask)

    return loss_fn



def train_epoch(
    model, 
    optimizer, 
    data_loader, 
    loss_fn, 
    max_grad_norm=1.0, 
    clip_method='norm'):

    model.train()
    total_loss = 0
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()

        if isinstance(data, Batch):
            data = data.to(device)
            labels = data.y.to(device
                ) if hasattr(data, 'y') else None
            out = model(data)
            loss = loss_fn(out, labels)
        else:
            if isinstance(data, (list, tuple)) and len(data) == 2:
                batch_1, batch_2 = data
                batch_1 = batch_1.to(device)
                batch_2 = batch_2.to(device)
                zis = model(batch_1)
                zjs = model(batch_2)
                smiles = batch_1.smiles
                
                loss = loss_fn(zis, zjs, smiles)
            else:
                continue

        loss.backward()
        clip_gradients(model, 
            max_grad_norm, 
            method=clip_method
            )
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate(
    model, 
    data_loader, 
    loss_fn):

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if data is None:
                continue

            if isinstance(data, Batch): 
                data = data.to(device)
                labels = data.y.to(device
                    ) if hasattr(data, 'y') else None
                out = model(data)
                loss = loss_fn(out, labels)
            else:  
                if isinstance(data, (list, tuple)) and len(data) == 2:
                    batch_1, batch_2 = data
                    batch_1 = batch_1.to(device)
                    batch_2 = batch_2.to(device)
                    zis = model(batch_1)
                    zjs = model(batch_2)
                    smiles = batch_1.smiles
                    
                    loss = loss_fn(zis, zjs, smiles)
                else:
                    continue
            
            total_loss += loss.item()

    return total_loss / len(data_loader)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    num_epochs,
    patience,
    delta,
    window_size,
    best_model=True,
    warm_up_epochs=5,
    T_max=50,
    eta_min=0.001,
    enable_pruning=True):

    def warm_up_lr(epoch, max_lr=0.1):
        if epoch < warm_up_epochs:
            return (max_lr / warm_up_epochs) * epoch
        else:
            return max_lr

    lr_scheduler_warm_up = LambdaLR(
        optimizer,
        lr_lambda=warm_up_lr
        )
    scheduler_cosine = CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=eta_min
        )
    best_val_loss = float('inf')
    min_val_loss = float('inf')
    best_model_state = None
    best_epoch = None
    epochs_no_improve = 0
    val_loss_window = []
    train_losses = []
    val_losses = []
    improvement_threshold = 0.03
    initial_val_loss = None

    for epoch in range(num_epochs):
        avg_train_loss = train_epoch(
            model,
            optimizer,
            train_loader,
            loss_fn
            )
        train_losses.append(avg_train_loss)

        avg_val_loss = evaluate(
            model,
            val_loader,
            loss_fn
            )
        val_losses.append(avg_val_loss)
        val_loss_window.append(avg_val_loss)

        if len(val_loss_window) > window_size:
            val_loss_window.pop(0)

        avg_val_loss_window = (
            sum(val_loss_window)
            / len(val_loss_window)
            )
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {avg_train_loss:.4f} - "
            f"Val: {avg_val_loss:.4f} - "
            f"Win: {avg_val_loss_window:.4f}"
            )

        if avg_val_loss_window < best_val_loss - delta:
            best_val_loss = avg_val_loss_window
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(
                model.state_dict()
                )
            best_epoch = epoch + 1

        if epochs_no_improve >= patience:
            print("Early stopping")
            break

        if best_model:
            out_path = "../output/embeddings"
            save_embeddings(
                model,
                train_loader,
                epoch,
                out_path
                )

        if enable_pruning:
            if val_losses[0] < 0.6:
                print(
                    f"Pruned due to low initial val loss: "
                    f"{val_losses[0]:.4f}"
                    )   
                raise optuna.exceptions.TrialPruned()
            if epoch == 0:
                initial_val_loss = avg_val_loss
            elif epoch == 10:
                if (initial_val_loss is not None
                        and initial_val_loss > 0):
                    improvement = (
                        initial_val_loss
                        - avg_val_loss
                    ) / initial_val_loss
                    if improvement < improvement_threshold:
                        print(
                            f"Pruned due to low "
                            f"improvement: "
                            f"{improvement:.2%}"
                        )
                        raise optuna.exceptions.TrialPruned()

        if epoch < warm_up_epochs:
            lr_scheduler_warm_up.step()
        else:
            scheduler_cosine.step()

    if best_model_state is not None:
        print(
            f"Restoring best model from "
            f"epoch {best_epoch} with "
            f"val_loss {min_val_loss:.4f}"
            )
        model.load_state_dict(best_model_state)

    return best_val_loss, min_val_loss, train_losses, val_losses