import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, balanced_accuracy_score

def compute_metrics(y_true, fake_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    fake_score = np.asarray(fake_score).astype(float)
    fpr, tpr, thresholds = roc_curve(y_true, fake_score, pos_label=1)
    fnr = 1.0 - tpr
    i = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[i] + fnr[i]) / 2.0)
    eer_threshold = float(thresholds[i])
    pred = (fake_score >= threshold).astype(int)
    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": float(roc_auc_score(y_true, fake_score)),
        "f1": float(f1_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
    }