import numpy as np
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)
    )

def directional_accuracy(y_true, y_pred):
    direction_true = np.sign(np.diff(y_true))
    direction_pred = np.sign(np.diff(y_pred))
    return np.mean(direction_true == direction_pred)

def threshold_accuracy(y_true, y_pred, threshold=100):
    y_true_bin = (np.array(y_true) > threshold).astype(int)
    y_pred_bin = (np.array(y_pred) > threshold).astype(int)
    return accuracy_score(y_true_bin, y_pred_bin)

def evaluate_all_metrics(y_true, y_pred, threshold=100, runtime=None):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'sMAPE': symmetric_mean_absolute_percentage_error(y_true, y_pred),
        'Directional_Accuracy': directional_accuracy(y_true, y_pred),
        f'Threshold_Accuracy(>{threshold})': threshold_accuracy(y_true, y_pred, threshold),
        'Runtime_Seconds': runtime if runtime is not None else -1
    }
