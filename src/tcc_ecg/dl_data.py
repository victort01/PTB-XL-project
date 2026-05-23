"""Dados, splits e artefatos de desbalanceamento para deep learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tcc_ecg.data import final_labeled_records
from tcc_ecg.deep_learning import build_resnet1d_cache, compute_train_channel_normalization
from tcc_ecg.models import split_by_folds
from tcc_ecg.paths import ensure_dir, resolve_project_path
from tcc_ecg.plots import save_figure
from tcc_ecg.utils import save_table


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depende do ambiente local
        raise ImportError("PyTorch nao esta instalado. Rode: python -m pip install -e .[dl]") from exc
    return torch


class ECGMemmapAugmentedDataset:
    """Dataset com normalizacao do treino e aumentacoes leves apenas no treino."""

    def __init__(
        self,
        signals_path: str | Path,
        labels: np.ndarray,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        train: bool,
        augmentations: dict[str, Any] | None = None,
    ) -> None:
        self.signals = np.load(signals_path, mmap_mode="r")
        self.labels = labels.astype("int64")
        self.indices = indices.astype("int64")
        self.mean = mean.astype("float32")
        self.std = std.astype("float32")
        self.train = train
        self.augmentations = augmentations or {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        torch = _import_torch()
        idx = int(self.indices[item])
        x = self.signals[idx].astype("float32")
        x = (x - self.mean[None, :]) / self.std[None, :]
        if self.train and self.augmentations:
            x = apply_ecg_augmentations(x, self.augmentations)
        return torch.from_numpy(np.ascontiguousarray(x.T)), torch.tensor(int(self.labels[idx]), dtype=torch.long)


def apply_ecg_augmentations(signal: np.ndarray, augmentations: dict[str, Any]) -> np.ndarray:
    """Aumentacoes simples para ECG, sem distorcoes agressivas."""
    x = signal.copy()
    noise_std = float(augmentations.get("noise_std", 0.0))
    if noise_std > 0:
        x += np.random.normal(0.0, noise_std, size=x.shape).astype("float32")

    scale_min = float(augmentations.get("scale_min", 1.0))
    scale_max = float(augmentations.get("scale_max", 1.0))
    if scale_min != 1.0 or scale_max != 1.0:
        x *= np.random.uniform(scale_min, scale_max, size=(1, x.shape[1])).astype("float32")

    max_shift = int(augmentations.get("time_shift", 0))
    if max_shift > 0:
        x = np.roll(x, int(np.random.randint(-max_shift, max_shift + 1)), axis=0)

    channel_dropout_prob = float(augmentations.get("channel_dropout_prob", 0.0))
    if channel_dropout_prob > 0:
        mask = np.random.random(x.shape[1]) < channel_dropout_prob
        x[:, mask] = 0.0
    return x.astype("float32")


def class_counts(labels: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(labels.astype(int), minlength=n_classes)


def compute_class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Calcula pesos balanceados usando somente labels do treino."""
    counts = class_counts(labels, n_classes).astype("float64")
    weights = counts.sum() / np.maximum(counts, 1) / n_classes
    weights[counts == 0] = 0.0
    return weights.astype("float32")


def sample_weights_for_labels(labels: np.ndarray, n_classes: int) -> np.ndarray:
    weights = compute_class_weights(labels, n_classes)
    return weights[labels.astype(int)].astype("float64")


def cache_config_for_frequency(config: dict[str, Any], frequency: int, cache_cfg: dict[str, Any]) -> dict[str, Any]:
    """Monta config temporaria para reaproveitar o cache de sinais brutos."""
    cache_config = dict(config)
    cache_config["data"] = dict(config["data"])
    cache_config["data"]["signal_frequency"] = int(frequency)
    cache_config["deep_learning"] = dict(config.get("deep_learning", {}))
    cache_config["deep_learning"]["resnet1d"] = {
        "cache_dtype": cache_cfg.get("cache_dtype", "float16"),
        "force_rebuild_cache": bool(cache_cfg.get("force_rebuild_cache", False)),
        "max_records": cache_cfg.get("max_records"),
    }
    return cache_config


def prepare_raw_signal_splits(
    metadata: pd.DataFrame,
    config: dict[str, Any],
    frequency: int,
    batch_size: int,
    cache_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Prepara cache, splits oficiais e normalizacao calculada somente no treino."""
    cache_config = cache_config_for_frequency(config, frequency, cache_cfg)
    cache_paths = build_resnet1d_cache(metadata, cache_config)
    labels = np.load(cache_paths["labels"])
    cache_metadata = pd.read_csv(cache_paths["metadata"]).assign(row_idx=np.arange(len(labels)))
    splits = split_by_folds(cache_metadata, config)
    train_idx = splits["train"]["row_idx"].to_numpy(dtype="int64")
    val_idx = splits["validation"]["row_idx"].to_numpy(dtype="int64")
    test_idx = splits["test"]["row_idx"].to_numpy(dtype="int64")
    mean, std = compute_train_channel_normalization(cache_paths["signals"], train_idx, batch_size=batch_size)
    return {
        "cache_paths": cache_paths,
        "labels": labels,
        "cache_metadata": cache_metadata,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "mean": mean,
        "std": std,
    }


def save_normalization(mean: np.ndarray, std: np.ndarray, config: dict[str, Any], stem: str) -> Path:
    processed_dir = resolve_project_path(config["outputs"]["processed_dir"], config.get("project_root"))
    ensure_dir(processed_dir)
    path = processed_dir / f"{stem}_normalization_500hz.npz"
    np.savez(path, mean=mean, std=std)
    return path


def create_weighted_sampler(labels: np.ndarray, train_idx: np.ndarray, n_classes: int, seed: int):
    """Cria WeightedRandomSampler apenas para o DataLoader de treino."""
    torch = _import_torch()
    train_labels = labels[train_idx]
    sample_weights = sample_weights_for_labels(train_labels, n_classes)
    generator = torch.Generator().manual_seed(int(seed))
    return torch.utils.data.WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_idx),
        replacement=True,
        generator=generator,
    )


def generate_balance_and_split_artifacts(metadata: pd.DataFrame, config: dict[str, Any]) -> dict[str, Path]:
    """Gera tabelas/figuras de desbalanceamento sem alterar validacao ou teste."""
    figures_dir = resolve_project_path(config["outputs"]["figures_dir"], config.get("project_root"))
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    records = final_labeled_records(metadata)
    classes = list(config["labels"]["superclasses"])
    split_name = np.select(
        [
            records["strat_fold"].isin(config["folds"]["train"]),
            records["strat_fold"].isin(config["folds"]["validation"]),
            records["strat_fold"].isin(config["folds"]["test"]),
        ],
        ["treino", "validacao", "teste"],
        default="other",
    )
    split_df = records.assign(split=split_name)

    count_table = (
        split_df.groupby(["target", "split"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=classes, columns=["treino", "validacao", "teste"], fill_value=0)
    )
    percent_table = count_table.div(count_table.sum(axis=0), axis=1) * 100
    output = count_table.copy()
    output.columns = [f"{col}_count" for col in output.columns]
    for col in ["treino", "validacao", "teste"]:
        output[f"{col}_percent"] = percent_table[col].round(4)
    output = output.reset_index(names="classe")
    save_table(
        output,
        tables_dir / "class_distribution_by_split.csv",
        tables_dir / "class_distribution_by_split.tex",
    )

    _plot_class_distribution_by_split(count_table, figures_dir / "fig_class_distribution_by_split.png")
    _plot_class_distribution_by_split_percent(percent_table, figures_dir / "fig_class_distribution_by_split_percent.png")
    _plot_class_distribution_by_split_heatmap(percent_table, figures_dir / "fig_class_distribution_by_split_heatmap.png")

    train_counts = count_table["treino"].astype(int)
    smote_counts = pd.Series(train_counts.max(), index=train_counts.index, name="after_smote")
    smote_df = pd.DataFrame({"Classe": train_counts.index, "Antes do SMOTE": train_counts.values, "Depois do SMOTE": smote_counts.values})
    _plot_before_after_smote(smote_df, figures_dir / "fig_train_distribution_before_after_smote.png")

    weights = compute_class_weights(
        split_df.loc[split_df["split"].eq("treino"), "target_id"].astype(int).to_numpy(),
        len(classes),
    )
    weights_df = pd.DataFrame({"classe": classes, "treino_count": train_counts.values, "class_weight": weights})
    save_table(
        weights_df,
        tables_dir / "class_weights_deep_learning.csv",
        tables_dir / "class_weights_deep_learning.tex",
    )
    _plot_class_weights(weights_df, figures_dir / "fig_class_weights_deep_learning.png")

    sampled = _sample_weighted_distribution(
        split_df.loc[split_df["split"].eq("treino"), "target_id"].astype(int).to_numpy(),
        classes,
        seed=int(config["project"]["seed"]),
    )
    sampler_df = pd.DataFrame(
        {
            "classe": classes,
            "treino_original_count": train_counts.values,
            "weighted_sampler_observed_count": sampled,
        }
    )
    save_table(sampler_df, tables_dir / "weighted_sampler_distribution.csv")
    _plot_weighted_sampler_distribution(sampler_df, figures_dir / "fig_weighted_sampler_distribution.png")

    summary = pd.DataFrame(
        [
            {
                "etapa": "Distribuicao original",
                "estrategia": "Manter distribuicao real das classes apos a estrategia multiclasse.",
                "aplicacao": "treino, validacao e teste",
                "observacao": "Usada para descrever o desbalanceamento do problema.",
            },
            {
                "etapa": "Split por folds oficiais",
                "estrategia": "Treino nos folds 1-8, validacao no fold 9 e teste no fold 10.",
                "aplicacao": "separacao deterministica PTB-XL",
                "observacao": "Validacao e teste preservam a distribuicao original.",
            },
            {
                "etapa": "SMOTE em modelos classicos",
                "estrategia": "Sobreamostragem sintetica aplicada somente ao conjunto de treino.",
                "aplicacao": "somente treino",
                "observacao": "Nao aplicado em validacao ou teste.",
            },
            {
                "etapa": "Class weights em deep learning",
                "estrategia": "Pesos calculados a partir das contagens do conjunto de treino.",
                "aplicacao": "somente treino",
                "observacao": "Validacao e teste nao recebem balanceamento artificial.",
            },
            {
                "etapa": "Focal loss em deep learning",
                "estrategia": "Perda ponderada para reduzir dominancia de classes majoritarias.",
                "aplicacao": "somente treino",
                "observacao": "Parametro avaliado por validacao, sem ajuste no teste.",
            },
            {
                "etapa": "WeightedRandomSampler em deep learning",
                "estrategia": "Amostragem ponderada no DataLoader de treino.",
                "aplicacao": "somente treino",
                "observacao": "Validacao e teste usam DataLoader sequencial sem sampler.",
            },
        ]
    )
    save_table(
        summary,
        tables_dir / "balance_strategy_summary.csv",
        tables_dir / "balance_strategy_summary.tex",
    )

    return {
        "class_distribution_by_split": tables_dir / "class_distribution_by_split.csv",
        "class_weights": tables_dir / "class_weights_deep_learning.csv",
        "balance_summary": tables_dir / "balance_strategy_summary.csv",
        "class_distribution_heatmap": figures_dir / "fig_class_distribution_by_split_heatmap.png",
    }


def _plot_class_distribution_by_split(count_table: pd.DataFrame, output_path: Path) -> None:
    plot_df = count_table.reset_index(names="classe").melt(id_vars="classe", var_name="split", value_name="count")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x="classe", y="count", hue="split", ax=ax)
    ax.set_title("Distribuicao de classes por split")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Quantidade de registros")
    save_figure(fig, output_path)


def _plot_class_distribution_by_split_percent(percent_table: pd.DataFrame, output_path: Path) -> None:
    plot_df = percent_table.reset_index(names="classe").melt(id_vars="classe", var_name="split", value_name="percentual")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x="classe", y="percentual", hue="split", ax=ax)
    ax.set_title("Distribuicao percentual das classes por split")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Percentual dentro do split (%)")
    ax.legend(title="Split")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=8, padding=2)
    ax.set_ylim(0, max(float(plot_df["percentual"].max()) * 1.18, 5.0))
    save_figure(fig, output_path)


def _plot_class_distribution_by_split_heatmap(percent_table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.heatmap(percent_table.T, annot=True, fmt=".1f", cmap="Blues", cbar_kws={"label": "%"}, ax=ax)
    ax.set_title("Distribuicao percentual das classes por split")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Split")
    save_figure(fig, output_path)


def _plot_before_after_smote(smote_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = smote_df.melt(id_vars="Classe", var_name="cenario", value_name="count")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x="Classe", y="count", hue="cenario", ax=ax)
    ax.set_title("Distribuicao do treino antes e depois do SMOTE")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Quantidade de registros")
    ax.text(
        0.5,
        -0.2,
        "SMOTE aplicado apenas no treino; validacao e teste preservam a distribuicao original.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    save_figure(fig, output_path)


def _plot_class_weights(weights_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=weights_df, x="classe", y="class_weight", ax=ax, color="#4C78A8")
    ax.set_title("Pesos por classe calculados no treino")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Peso")
    save_figure(fig, output_path)


def _sample_weighted_distribution(train_labels: np.ndarray, classes: list[str], seed: int) -> np.ndarray:
    sample_weights = sample_weights_for_labels(train_labels, len(classes))
    probability = sample_weights / sample_weights.sum()
    rng = np.random.default_rng(seed)
    draws = rng.choice(train_labels, size=min(len(train_labels), 10000), replace=True, p=probability)
    return class_counts(draws, len(classes))


def _plot_weighted_sampler_distribution(sampler_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = sampler_df.melt(id_vars="classe", var_name="distribution", value_name="count")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x="classe", y="count", hue="distribution", ax=ax)
    ax.set_title("Distribuicao original do treino vs WeightedRandomSampler")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Quantidade aproximada")
    save_figure(fig, output_path)
