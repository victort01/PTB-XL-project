"""Metricas e tabelas de avaliacao."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from tcc_ecg.paths import resolve_project_path
from tcc_ecg.utils import save_table


def compute_classification_metrics(
    y_true,
    y_pred,
    model_name: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Calcula metricas principais, priorizando F1 macro para dados desbalanceados."""
    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if model_name is not None:
        metrics["model"] = model_name
    if split is not None:
        metrics["split"] = split
    return metrics


def classification_report_frame(y_true, y_pred, target_names: list[str]) -> pd.DataFrame:
    """Retorna classification report em formato tabular."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(target_names))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T.reset_index(names="class")


def save_metrics_tables(metrics: pd.DataFrame, config: dict[str, Any], stem: str = "model_metrics") -> None:
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    save_table(metrics, tables_dir / f"{stem}.csv", tables_dir / f"{stem}.tex")


def select_best_model(metrics: pd.DataFrame, split: str = "validation") -> pd.Series:
    """Seleciona melhor modelo por F1 macro no split informado."""
    candidates = metrics.loc[metrics["split"].eq(split)].copy()
    if candidates.empty:
        candidates = metrics.copy()
    return candidates.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]


def save_classification_report_tables(
    y_true,
    y_pred,
    target_names: list[str],
    config: dict[str, Any],
    stem: str,
) -> pd.DataFrame:
    report = classification_report_frame(y_true, y_pred, target_names)
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    save_table(report, tables_dir / f"{stem}.csv", tables_dir / f"{stem}.tex")
    return report


def metrics_to_latex_ready(metrics: pd.DataFrame) -> pd.DataFrame:
    """Arredonda metricas para tabelas do TCC."""
    result = metrics.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].round(4)
    return result
