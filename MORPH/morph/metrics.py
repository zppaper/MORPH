import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def compute_metrics(y_true, y_probs):
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return [np.nan, np.nan]
    try:
        auc = roc_auc_score(y_true, y_probs)
        aupr = average_precision_score(y_true, y_probs)
    except:
        (auc, aupr) = (np.nan, np.nan)
    return [auc, aupr]

def nanstd_ddof1(values, axis=0):
    values = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(values, axis=axis, keepdims=True)
    sq_diff = (values - mean) ** 2
    valid_count = np.sum(~np.isnan(values), axis=axis)
    denom = np.where(valid_count > 1, valid_count - 1, np.nan)
    return np.sqrt(np.nansum(sq_diff, axis=axis) / denom)
