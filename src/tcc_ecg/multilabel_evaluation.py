"""Metricas para a formulacao multilabel do TCC II."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def optimize_thresholds_on_validation(
    y_true: np.ndarray,
    y_score: np.ndarray,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Escolhe thresholds por classe usando somente o conjunto de validacao."""
    _validate_shapes(y_true, y_score)
    candidates = grid if grid is not None else np.linspace(0.1, 0.9, 33)
    thresholds = np.full(y_true.shape[1], 0.5, dtype="float32")
    for index in range(y_true.shape[1]):
        if np.unique(y_true[:, index]).size < 2:
            continue
        scores = [
            f1_score(y_true[:, index], y_score[:, index] >= threshold, zero_division=0)
            for threshold in candidates
        ]
        thresholds[index] = float(candidates[int(np.argmax(scores))])
    return thresholds


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: float | Sequence[float] = 0.5,
) -> dict[str, float]:
    """Calcula metricas multilabel globais sem ajustar parametros no teste."""
    _validate_shapes(y_true, y_score)
    threshold_array = np.broadcast_to(np.asarray(thresholds, dtype="float32"), (y_true.shape[1],))
    y_pred = (y_score >= threshold_array[None, :]).astype("int8")
    valid_auc = _valid_binary_columns(y_true)
    macro_auroc = float(roc_auc_score(y_true[:, valid_auc], y_score[:, valid_auc], average="macro")) if valid_auc.any() else float("nan")
    macro_auprc = float(average_precision_score(y_true[:, valid_auc], y_score[:, valid_auc], average="macro")) if valid_auc.any() else float("nan")
    return {
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }


def multilabel_report_frame(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Sequence[str],
    thresholds: float | Sequence[float] = 0.5,
) -> pd.DataFrame:
    """Retorna metricas por classe com suporte e prevalencia."""
    _validate_shapes(y_true, y_score)
    if len(class_names) != y_true.shape[1]:
        raise ValueError("class_names deve ter uma entrada por coluna de rotulo.")
    threshold_array = np.broadcast_to(np.asarray(thresholds, dtype="float32"), (y_true.shape[1],))
    rows = []
    for index, class_name in enumerate(class_names):
        truth = y_true[:, index]
        score = y_score[:, index]
        pred = (score >= threshold_array[index]).astype("int8")
        has_both = np.unique(truth).size == 2
        rows.append(
            {
                "class": class_name,
                "support_positive": int(truth.sum()),
                "prevalence": float(truth.mean()),
                "threshold": float(threshold_array[index]),
                "auroc": float(roc_auc_score(truth, score)) if has_both else float("nan"),
                "auprc": float(average_precision_score(truth, score)) if truth.sum() else float("nan"),
                "precision": float(precision_score(truth, pred, zero_division=0)),
                "recall": float(recall_score(truth, pred, zero_division=0)),
                "f1": float(f1_score(truth, pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def _valid_binary_columns(y_true: np.ndarray) -> np.ndarray:
    return np.asarray([np.unique(y_true[:, index]).size == 2 for index in range(y_true.shape[1])])


def _validate_shapes(y_true: np.ndarray, y_score: np.ndarray) -> None:
    if y_true.ndim != 2 or y_score.ndim != 2 or y_true.shape != y_score.shape:
        raise ValueError("y_true e y_score devem ter o mesmo shape bidimensional.")

