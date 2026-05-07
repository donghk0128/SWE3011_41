import numpy as np
from sklearn.metrics import roc_curve

def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[idx]
    threshold = thresholds[idx]

    return eer, threshold