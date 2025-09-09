import numpy as np
from IPython.display import (
    Markdown, 
    display
    )

from metrics import ClassificationMetrics
from predictor import predict 
from save import save_thresholds


def threshold_moving(
    calculator, 
    y_pred_train, 
    y_true_train, 
    y_pred_val, 
    y_true_val):

    threshold_range = np.arange(0.0, 1.0, 0.01)
    train_results = []
    val_results = []

    for t in threshold_range:
        train_results.append(
            calculator.calibration(
                t, y_pred_train, y_true_train
                )
            )
        val_results.append(
            calculator.calibration(
                t, y_pred_val, y_true_val
                )
            )
    return train_results, val_results


def best_thresholds(
    train_results, 
    val_results, 
    num_tasks):

    train_metrics, val_metrics = [], []
    avg_thresholds = {'train': [], 'val': []}
    for task_idx in range(num_tasks):
        best_train = None
        best_val = None
        for train_res, val_res in zip(train_results, val_results):
            train_task = train_res[task_idx]
            val_task = val_res[task_idx]
            if best_train is None or train_task['g_mean'] > best_train['g_mean']:
                best_train = train_task
            if best_val is None or val_task['g_mean'] > best_val['g_mean']:
                best_val = val_task
        avg_thresholds['train'].append(best_train['threshold'])
        avg_thresholds['val'].append(best_val['threshold'])
        train_metrics.append({
            'task': task_idx + 1,
            'best_threshold': best_train['threshold'],
            'best_metrics': best_train}
            )
        val_metrics.append({
            'task': task_idx + 1,
            'best_threshold': best_val['threshold'],
            'best_metrics': best_val}
            )
    return train_metrics, val_metrics, avg_thresholds


def apply_thresholds(
    test_pred, 
    test_true, 
    val_thresholds, 
    calculator):

    test_metrics = []
    for task_idx, threshold in enumerate(val_thresholds):
        mask = ~np.isnan(test_true[:, task_idx])
        task_true = test_true[:, task_idx][mask]
        task_pred = (test_pred[:, task_idx] > threshold).astype(int)[mask]
        task_prob = test_pred[:, task_idx][mask]
        metrics = {
            'accuracy': calculator.accuracy(task_true, task_pred),
            'recall': calculator.recall(task_true, task_pred),
            'specificity': calculator.specificity(task_true, task_pred),
            'g_mean': calculator.g_mean(
                calculator.recall(task_true, task_pred),
                calculator.specificity(task_true, task_pred)),
            'f1': calculator.f1(task_true, task_pred),
            'auc': calculator.auc(task_true, task_prob)
            }
        test_metrics.append({
            'task': task_idx + 1,
            'best_threshold': threshold,
            'best_metrics': metrics}
            )
    return test_metrics


def standard_threshold(
    train_pred, 
    train_true, 
    val_pred, 
    val_true, 
    test_pred, 
    test_true, 
    threshold, 
    calculator):

    train_metrics = calculator.calculate_metrics(
        train_pred, 
        train_true, 
        threshold
        )
    val_metrics = calculator.calculate_metrics(
        val_pred, 
        val_true, 
        threshold
        )
    test_metrics = calculator.calculate_metrics(
        test_pred, 
        test_true, 
        threshold
        )
    train_metrics = [
            {'task': idx + 1, 
            'best_threshold': threshold, 
            'best_metrics': m}
            for idx, m in enumerate(train_metrics)
            ]
    val_metrics = [
            {'task': idx + 1, 
            'best_threshold': threshold, 
            'best_metrics': m}
            for idx, m in enumerate(val_metrics)
            ]
    test_metrics = [
            {'task': idx + 1, 
            'best_threshold': threshold, 
            'best_metrics': m}
            for idx, m in enumerate(test_metrics)
            ]
    return train_metrics, val_metrics, test_metrics


def classification_markdown(
    train_metrics, 
    val_metrics, 
    test_metrics,
    global_train_metrics, 
    global_val_metrics, 
    global_test_metrics,
    avg_train_threshold, 
    avg_val_threshold, 
    avg_test_threshold):

    def format_row(label, set_name, threshold, metrics):
        return (
            f"{label} | {set_name:<10} | {threshold:<9.4f} | "
            f"{metrics['accuracy']:<8.4f} | {metrics['recall']:<6.4f} | "
            f"{metrics['specificity']:<11.4f} | {metrics['f1']:<6.4f} | "
            f"{metrics['g_mean']:<6.4f} | {metrics['auc']:<4.4f}\n"
            )

    output = (
        "Task  | Set   | Threshold | Accuracy | Recall | Specificity | F1   | G-mean | AUC\n"
        "------|-------|-----------|----------|--------|-------------|------|--------|-----\n"
        )
    for idx, (train, val, test) in enumerate(
        zip(train_metrics, val_metrics, test_metrics)):
        task_label = f'Task {idx + 1}'
        output += format_row(
            task_label, 
            'Training', 
            train['best_threshold'], 
            train['best_metrics']
            )
        output += format_row(
            task_label, 
            'Validation', 
            val['best_threshold'], 
            val['best_metrics']
            )
        output += format_row(
            task_label, 
            'Test', 
            test['best_threshold'], 
            test['best_metrics']
            )
    output += format_row(
        'Global', 
        'Training', 
        avg_train_threshold, 
        global_train_metrics
        )
    output += format_row(
        'Global', 
        'Validation', 
        avg_val_threshold, 
        global_val_metrics
        )
    output += format_row(
        'Global', 
        'Test', 
        avg_test_threshold, 
        global_test_metrics
        )
    display(Markdown(output))


class ClassificationEvaluator:
    def __init__(
        self,
        model,
        device,
        train_loader,
        val_loader,
        test_loader,
        calibration=True):

        self.calculator = ClassificationMetrics(model, device)
        self.best_thresholds = {}
        self.evaluate(
            train_loader,
            val_loader,
            test_loader,
            calibration
            )
    def evaluate(
        self,
        train_loader,
        val_loader,
        test_loader,
        calibration=True):

        train_pred, train_true, _ = predict(
            self.calculator.model,
            train_loader,
            self.calculator.device
            )
        val_pred, val_true, _ = predict(
            self.calculator.model,
            val_loader,
            self.calculator.device
            )
        test_pred, test_true, _ = predict(
            self.calculator.model,
            test_loader,
            self.calculator.device
            )
        self.evaluate_metrics(
            train_pred,
            train_true,
            val_pred,
            val_true,
            test_pred,
            test_true,
            calibration
            )
    def evaluate_metrics(
        self,
        train_pred,
        train_true,
        val_pred,
        val_true,
        test_pred,
        test_true,
        calibration):

        if calibration:
            train_results, val_results = threshold_moving(
                self.calculator,
                train_pred,
                train_true,
                val_pred,
                val_true
                )
            train_metrics, val_metrics, avg_thresholds = best_thresholds(
                train_results,
                val_results,
                train_true.shape[1]
                )
            test_metrics = apply_thresholds(
                test_pred,
                test_true,
                avg_thresholds['val'],
                self.calculator
                )
            avg_thresholds['test'] = avg_thresholds['val']
            self.best_thresholds.update(avg_thresholds)
            save_thresholds(self.best_thresholds)
        else:
            default_thresh = 0.5
            train_metrics, val_metrics, test_metrics = standard_threshold(
                train_pred,
                train_true,
                val_pred,
                val_true,
                test_pred,
                test_true,
                default_thresh,
                self.calculator
                )
            avg_thresholds = {
                'train': [default_thresh] * train_true.shape[1],
                'val':   [default_thresh] * val_true.shape[1],
                'test':  [default_thresh] * test_true.shape[1]
                }
            self.best_thresholds.update(avg_thresholds)

        train_avg_thresh = np.mean(avg_thresholds['train'])
        val_avg_thresh   = np.mean(avg_thresholds['val'])
        test_avg_thresh  = np.mean(avg_thresholds['test'])
        train_mask = ~np.isnan(train_true)
        val_mask   = ~np.isnan(val_true)
        test_mask  = ~np.isnan(test_true)
        train_pred_global = train_pred[train_mask].reshape(-1, 1)
        train_true_global = train_true[train_mask].reshape(-1, 1)
        val_pred_global   = val_pred[val_mask].reshape(-1, 1)
        val_true_global   = val_true[val_mask].reshape(-1, 1)
        test_pred_global  = test_pred[test_mask].reshape(-1, 1)
        test_true_global  = test_true[test_mask].reshape(-1, 1)
        global_train = self.calculator.calculate_metrics(
            train_pred_global,
            train_true_global,
            train_avg_thresh
            )[0]
        global_val = self.calculator.calculate_metrics(
            val_pred_global,
            val_true_global,
            val_avg_thresh
            )[0]
        global_test = self.calculator.calculate_metrics(
            test_pred_global,
            test_true_global,
            test_avg_thresh
            )[0]
        classification_markdown(
            train_metrics,
            val_metrics,
            test_metrics,
            global_train,
            global_val,
            global_test,
            train_avg_thresh,
            val_avg_thresh,
            test_avg_thresh
            )