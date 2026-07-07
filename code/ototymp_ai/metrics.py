from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from statsmodels.stats.proportion import proportion_confint


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    lo, hi = proportion_confint(int(k), int(n), alpha=alpha, method="wilson")
    return float(lo), float(hi)


def percentile_ci(values, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    return tuple(float(x) for x in np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def one_vs_rest_counts(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> dict[str, int]:
    yt = (y_true == label).astype(int)
    yp = (y_pred == label).astype(int)
    return {
        "tp": int(((yt == 1) & (yp == 1)).sum()),
        "tn": int(((yt == 0) & (yp == 0)).sum()),
        "fp": int(((yt == 0) & (yp == 1)).sum()),
        "fn": int(((yt == 1) & (yp == 0)).sum()),
    }


def multiclass_metrics(y_true: np.ndarray, probs: np.ndarray, labels: list[str]) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    y_pred = probs.argmax(axis=1) if len(probs) else np.array([], dtype=int)
    n_classes = len(labels)

    out = {
        "overall_acc": float((y_pred == y_true).mean()) if len(y_true) else np.nan,
        "num_samples": int(len(y_true)),
        "label_list": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist(),
        "per_class_accuracy": {},
        "per_class_sensitivity": {},
        "per_class_specificity": {},
        "per_class_auroc": {},
    }

    aucs = []
    for i, name in enumerate(labels):
        idx = y_true == i
        out["per_class_accuracy"][name] = float((y_pred[idx] == i).mean()) if idx.any() else None

        c = one_vs_rest_counts(y_true, y_pred, i)
        out["per_class_sensitivity"][name] = safe_div(c["tp"], c["tp"] + c["fn"])
        out["per_class_specificity"][name] = safe_div(c["tn"], c["tn"] + c["fp"])

        y_bin = (y_true == i).astype(int)
        if len(np.unique(y_bin)) < 2:
            auc = None
        else:
            auc = float(roc_auc_score(y_bin, probs[:, i]))
            aucs.append(auc)
        out["per_class_auroc"][name] = auc

    out["macro_auroc"] = float(np.mean(aucs)) if aucs else None
    return out


def bootstrap_ci(df: pd.DataFrame, metric_func: Callable[[pd.DataFrame], float], n_boot: int = 1000, seed: int = 2026):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(df)
    if n == 0:
        return np.nan, np.nan
    for _ in range(n_boot):
        sample = df.iloc[rng.integers(0, n, size=n)]
        value = metric_func(sample)
        if np.isfinite(value):
            vals.append(float(value))
    return percentile_ci(vals)

