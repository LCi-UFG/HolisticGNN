import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    confusion_matrix, 
    roc_auc_score, 
    f1_score
    )


class ClassificationMetrics:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @staticmethod
    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def recall(y_true, y_pred):
        return recall_score(y_true, y_pred)

    @staticmethod
    def specificity(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)

    @staticmethod
    def g_mean(sensitivity, specificity):
        return np.sqrt(sensitivity * specificity)

    @staticmethod
    def f1(y_true, y_pred):
        return f1_score(y_true, y_pred)

    @staticmethod
    def auc(y_true, y_prob):
        return roc_auc_score(y_true, y_prob)

    def calculate_metrics(
        self, 
        probabilities, 
        y_true, 
        threshold=0.5):
        
        metrics = []
        num_tasks = y_true.shape[1
                ] if y_true.ndim > 1 else 1
        for i in range(num_tasks):
            mask = ~np.isnan(y_true[:, i]
                ) if num_tasks > 1 else ~np.isnan(y_true)
            y_true_task = y_true[mask
                ] if num_tasks == 1 else y_true[:, i][mask]
            y_pred_task = (probabilities[mask] > 
                threshold).astype(int) if num_tasks == 1 else (
                probabilities[:, i][mask] > threshold).astype(int)
            y_prob_task = probabilities[mask
                ] if num_tasks == 1 else probabilities[:, i][mask]
            if len(y_true_task) == 0:
                continue
            accuracy = self.accuracy(y_true_task, y_pred_task)
            recall = self.recall(y_true_task, y_pred_task)
            specificity = self.specificity(y_true_task, y_pred_task)
            g_mean = self.g_mean(recall, specificity)
            f1 = self.f1(y_true_task, y_pred_task)
            auc = self.auc(y_true_task, y_prob_task)
            metrics.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'recall': recall,
                'specificity': specificity,
                'g_mean': g_mean,
                'f1': f1,
                'auc': auc
            })
        return metrics
    
    def calibration(
        self, 
        threshold, 
        probabilities, 
        y_true):

        return self.calculate_metrics(
            probabilities, y_true, threshold
            )