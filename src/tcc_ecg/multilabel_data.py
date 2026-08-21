"""Cache e datasets de sinais brutos para o protocolo multilabel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcc_ecg.deep_learning import compute_train_channel_normalization
from tcc_ecg.features import extract_signal_features, load_signal, record_path_for_row
from tcc_ecg.multilabel import final_multilabel_records, label_column, multilabel_target_matrix
from tcc_ecg.paths import ensure_dir, resolve_project_path
from tcc_ecg.protocol import split_by_official_folds


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise ImportError("PyTorch nao instalado. Rode: python -m pip install -e .[tcc2]") from exc
    return torch


class MultilabelMemmapDataset:
    """Dataset que aplica estatisticas do treino e aumentacao somente no treino."""

    def __init__(
        self,
        signals_path: str | Path,
        labels: np.ndarray,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        train: bool = False,
        augmentations: dict[str, Any] | None = None,
    ) -> None:
        self.signals = np.load(signals_path, mmap_mode="r")
        self.labels = np.asarray(labels, dtype="float32")
        self.indices = np.asarray(indices, dtype="int64")
        self.mean = np.asarray(mean, dtype="float32")
        self.std = np.asarray(std, dtype="float32")
        self.train = bool(train)
        self.augmentations = augmentations or {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        torch = _import_torch()
        index = int(self.indices[item])
        signal = self.signals[index].astype("float32")
        signal = (signal - self.mean[None, :]) / self.std[None, :]
        if self.train and self.augmentations.get("enabled", False):
            signal = apply_train_augmentations(signal, self.augmentations)
        x = torch.from_numpy(np.ascontiguousarray(signal.T))
        y = torch.from_numpy(self.labels[index].copy())
        return x, y


def apply_train_augmentations(signal: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Aplica perturbacoes leves sem modificar validacao ou teste."""
    output = signal.copy()
    noise_std = float(config.get("noise_std", 0.0))
    if noise_std > 0:
        output += np.random.normal(0.0, noise_std, output.shape).astype("float32")
    scale_min = float(config.get("scale_min", 1.0))
    scale_max = float(config.get("scale_max", 1.0))
    if (scale_min, scale_max) != (1.0, 1.0):
        output *= np.random.uniform(scale_min, scale_max, (1, output.shape[1])).astype("float32")
    shift = int(config.get("time_shift", 0))
    if shift > 0:
        output = np.roll(output, np.random.randint(-shift, shift + 1), axis=0)
    probability = float(config.get("channel_dropout_prob", 0.0))
    if probability > 0:
        output[:, np.random.random(output.shape[1]) < probability] = 0.0
    return output.astype("float32")


def build_multilabel_signal_cache(
    metadata: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Path]:
    """Materializa sinais e rotulos uma vez para evitar releitura por epoca."""
    classes = list(config["labels"]["superclasses"])
    frequency = int(config["data"]["signal_frequency"])
    training = config.get("training", {})
    processed_dir = resolve_project_path(config["outputs"]["processed_dir"], config.get("project_root"))
    ensure_dir(processed_dir)
    stem = f"ptbxl_multilabel_{frequency}hz"
    paths = {
        "signals": processed_dir / f"{stem}_signals.npy",
        "labels": processed_dir / f"{stem}_labels.npy",
        "metadata": processed_dir / f"{stem}_metadata.csv",
    }
    if all(path.exists() for path in paths.values()) and not bool(training.get("force_rebuild_cache", False)):
        return paths

    records = final_multilabel_records(metadata).sort_values(["strat_fold"]).copy()
    max_records = training.get("max_records")
    if max_records:
        records = records.head(int(max_records))
    if records.empty:
        raise ValueError("Nenhum registro multilabel disponivel para o cache.")

    first = load_signal(record_path_for_row(records.iloc[0], config))
    dtype = np.dtype(training.get("cache_dtype", "float16"))
    signals = np.lib.format.open_memmap(
        paths["signals"], mode="w+", dtype=dtype, shape=(len(records), *first.shape)
    )
    for position, (_, row) in enumerate(records.iterrows()):
        signal = load_signal(record_path_for_row(row, config))
        if signal.shape != first.shape:
            raise ValueError(f"Shape inesperado {signal.shape}; esperado {first.shape}.")
        signals[position] = signal.astype(dtype)
    signals.flush()

    labels = multilabel_target_matrix(records, classes)
    np.save(paths["labels"], labels)
    cache_metadata = records.reset_index()
    metadata_columns = ["ecg_id", "strat_fold", *[label_column(name) for name in classes]]
    cache_metadata[metadata_columns].to_csv(paths["metadata"], index=False)
    return paths


def build_multilabel_feature_table(
    metadata: pd.DataFrame,
    config: dict[str, Any],
) -> Path:
    """Extrai atributos por registro sem aprender estatisticas globais."""
    records = final_multilabel_records(metadata).copy()
    max_records = config.get("features", {}).get("max_records")
    if max_records:
        records = records.head(int(max_records))
    classes = list(config["labels"]["superclasses"])
    rows: list[dict[str, Any]] = []
    for ecg_id, row in records.iterrows():
        signal = load_signal(record_path_for_row(row, config))
        features = extract_signal_features(
            signal,
            include_fft_features=bool(config["features"].get("include_fft_features", False)),
            sampling_rate=int(config["data"]["signal_frequency"]),
        )
        if bool(config["features"].get("include_metadata", True)):
            features.update(
                {
                    "age_clean": row.get("age_clean", np.nan),
                    "age_is_anon_90_plus": int(bool(row.get("age_is_anon_90_plus", False))),
                    "sex": pd.to_numeric(row.get("sex", np.nan), errors="coerce"),
                }
            )
        features.update(
            {
                "ecg_id": ecg_id,
                "strat_fold": int(row["strat_fold"]),
                **{label_column(name): int(row[label_column(name)]) for name in classes},
            }
        )
        rows.append(features)
    output = resolve_project_path(config["data"]["classical_features_path"], config.get("project_root"))
    ensure_dir(output.parent)
    pd.DataFrame(rows).to_parquet(output, index=False)
    return output


def prepare_multilabel_signal_splits(
    metadata: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Prepara indices e normalizacao calculada exclusivamente no treino."""
    cache_paths = build_multilabel_signal_cache(metadata, config)
    labels = np.load(cache_paths["labels"])
    cache_metadata = pd.read_csv(cache_paths["metadata"])
    cache_metadata["row_idx"] = np.arange(len(cache_metadata), dtype="int64")
    splits = split_by_official_folds(cache_metadata, config)
    train_indices = splits["train"]["row_idx"].to_numpy(dtype="int64")
    validation_indices = splits["validation"]["row_idx"].to_numpy(dtype="int64")
    test_indices = splits["test"]["row_idx"].to_numpy(dtype="int64")
    mean, std = compute_train_channel_normalization(
        cache_paths["signals"],
        train_indices,
        batch_size=int(config["training"]["batch_size"]),
    )
    normalization_path = Path(config["outputs"]["processed_dir"]) / f"normalization_{config['data']['signal_frequency']}hz_train_only.npz"
    normalization_path = resolve_project_path(normalization_path, config.get("project_root"))
    ensure_dir(normalization_path.parent)
    np.savez(normalization_path, mean=mean, std=std, train_indices=train_indices)
    return {
        "cache_paths": cache_paths,
        "labels": labels,
        "metadata": cache_metadata,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "mean": mean,
        "std": std,
        "normalization_path": normalization_path,
    }


def compute_positive_weights(y_train: np.ndarray) -> np.ndarray:
    """Calcula ``negativos/positivos`` usando somente os rotulos do treino."""
    positives = y_train.sum(axis=0, dtype="float64")
    negatives = len(y_train) - positives
    weights = negatives / np.maximum(positives, 1.0)
    weights[positives == 0] = 0.0
    return weights.astype("float32")
