import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch


def prepare_masked_data(
    y_true, y_pred):

    if isinstance(y_true, (Batch, Data)):
        y_true = y_true.y
    if isinstance(y_pred, (Batch, Data)):
        raise ValueError("y_pred must be a tensor")
    if not torch.is_tensor(y_true):
        raise ValueError("y_true must be a tensor")
    if not torch.is_tensor(y_pred):
        raise ValueError("y_pred must be a tensor")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must match shape"
        )
    y_true = y_true.to(dtype=y_pred.dtype)
    mask = ~torch.isnan(y_true) 
    y_true = torch.nan_to_num(y_true, nan=0.0)
    y_pred = torch.nan_to_num(y_pred, nan=0.0)
    return y_true, y_pred, mask


class SupervisedUncertainty(nn.Module):
    def __init__(self, num_tasks,
                 task_type='classification'):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_type = task_type
        self.log_vars = nn.Parameter(
            torch.zeros(num_tasks)
            )
    def compute_task_loss(self, pred, true):
        if self.task_type == 'classification':
            return F.binary_cross_entropy_with_logits(
                pred, true, reduction='none'
                )
        return F.mse_loss(
            pred, true, reduction='none'
            )
    def forward(self, y_pred, y_true, mask):
        losses = self.compute_task_loss(
            y_pred, y_true
            )
        alphas = self.log_vars.view(1, -1
        ).to(losses.device)  # align devices
        if self.task_type == 'classification':
            weighted = (
                torch.exp(-alphas) * losses + alphas
                )
        else:
            weighted = (
                0.5 * torch.exp(-alphas) * losses
                + 0.5 * alphas
                )
        denom = mask.sum()
        if denom == 0:
            return losses.new_tensor(
                0.0, requires_grad=True
                )
        return (
            weighted * mask.float()
        ).sum() / denom


def MaskedLoss(
    task_type='classification',
    num_tasks=None):

    if num_tasks is None:
        raise ValueError("num_tasks must be provided")

    def masked_loss_function(y_pred, y_true):
        y_t, y_p, mask = prepare_masked_data(
            y_true, y_pred
            )
        if task_type == 'classification':
            loss = F.binary_cross_entropy_with_logits(
                y_p, y_t, reduction='none'
                )
        else:
            loss = F.mse_loss(
                y_p, y_t, reduction='none'
                )
        denom = mask.sum()
        if denom == 0:
            return loss.new_tensor(
                0.0, requires_grad=True
                )
        return (loss * mask.float()).sum() / denom

    return masked_loss_function
