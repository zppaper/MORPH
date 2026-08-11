import random
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def set_seeds(seed):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true, y_score):
    """Return ROC-AUC and AUPR for binary predictions."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if np.unique(y_true).size < 2:
        return {"auc": np.nan, "aupr": np.nan}
    return {
        "auc": roc_auc_score(y_true, y_score),
        "aupr": average_precision_score(y_true, y_score),
    }
