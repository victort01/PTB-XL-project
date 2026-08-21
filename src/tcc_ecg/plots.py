"""Graficos gerados para EDA, avaliacao e interpretabilidade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from tcc_ecg.paths import ensure_dir, resolve_project_path


def save_figure(fig: plt.Figure, path: str | Path, also_pdf: bool = True, dpi: int = 160) -> None:
    """Salva figura em PNG e opcionalmente PDF."""
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(
    counts: pd.Series | pd.DataFrame,
    title: str,
    output_path: str | Path,
) -> None:
    if isinstance(counts, pd.Series):
        df = counts.rename_axis("classe").reset_index(name="n")
    else:
        df = counts.copy()
        if df.shape[1] >= 2:
            df = df.rename(columns={df.columns[0]: "classe", df.columns[1]: "n"})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=df, x="classe", y="n", ax=ax, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("Classe")
    ax.set_ylabel("Quantidade de registros")
    save_figure(fig, output_path)


def plot_age_distribution(metadata: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(metadata["age_clean"], bins=30, kde=False, ax=ax, color="#59A14F")
    anon_count = int(metadata["age_is_anon_90_plus"].sum())
    ax.set_title(f"Distribuicao de idade limpa (idade 300 anonimizada: n={anon_count})")
    ax.set_xlabel("Idade limpa")
    ax.set_ylabel("Quantidade de registros")
    save_figure(fig, output_path)


def plot_records_by_fold(metadata: pd.DataFrame, output_path: str | Path) -> None:
    counts = metadata["strat_fold"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="#F28E2B")
    ax.set_title("Quantidade de registros por fold")
    ax.set_xlabel("Fold PTB-XL")
    ax.set_ylabel("Quantidade de registros")
    save_figure(fig, output_path)


def plot_example_signals_by_class(
    metadata: pd.DataFrame,
    config: dict[str, Any],
    output_path: str | Path,
    lead_index: int = 1,
) -> None:
    """Plota um exemplo simples de ECG por classe para EDA."""
    from tcc_ecg.features import load_signal, record_path_for_row

    classes = list(config["labels"]["superclasses"])
    fig, axes = plt.subplots(len(classes), 1, figsize=(10, 8), sharex=True)
    for ax, klass in zip(axes, classes, strict=False):
        subset = metadata.loc[metadata["target"].eq(klass)]
        if subset.empty:
            ax.set_title(f"{klass} - sem exemplo disponivel")
            ax.axis("off")
            continue
        row = subset.iloc[0]
        signal = load_signal(record_path_for_row(row, config))
        ax.plot(signal[:, lead_index], linewidth=0.8)
        ax.set_title(f"Exemplo {klass} - derivacao II")
        ax.set_ylabel("mV")
    axes[-1].set_xlabel("Amostras")
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_metrics_comparison(metrics: pd.DataFrame, output_path: str | Path) -> None:
    plot_df = metrics.loc[metrics["split"].eq("test")].copy()
    keep = ["model", "accuracy", "f1_macro", "f1_weighted", "balanced_accuracy"]
    plot_df = plot_df[[col for col in keep if col in plot_df.columns]]
    melted = plot_df.melt(id_vars="model", var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="model", y="value", hue="metric", ax=ax)
    ax.set_title("Comparacao das metricas principais no teste")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Valor")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Metrica", loc="lower right")
    save_figure(fig, output_path)


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: list[str],
    output_path: str | Path,
    normalize: str | None = None,
    title: str | None = None,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize=normalize)
    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    display.plot(ax=ax, cmap="Blues", values_format=".2f" if normalize else "d", colorbar=False)
    ax.set_title(title or "Matriz de confusao")
    save_figure(fig, output_path)


def output_figure_path(config: dict[str, Any], filename: str) -> Path:
    figures_dir = resolve_project_path(config["outputs"]["figures_dir"], config.get("project_root"))
    return figures_dir / filename
