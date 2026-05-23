"""Carregamento de metadados PTB-XL e preparacao de rotulos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcc_ecg.labels import build_multiclass_target, summarize_label_strategy
from tcc_ecg.paths import ensure_dir, resolve_project_path
from tcc_ecg.utils import save_table


def _project_path_from_config(config: dict[str, Any], key_path: tuple[str, ...]) -> Path:
    current: Any = config
    for key in key_path:
        current = current[key]
    return resolve_project_path(current, config.get("project_root"))


def get_data_dir(config: dict[str, Any]) -> Path:
    return _project_path_from_config(config, ("data", "base_dir"))


def get_signal_frequency(config: dict[str, Any]) -> int:
    """Retorna a frequencia configurada, restrita aos modos suportados do PTB-XL."""
    frequency = int(config["data"].get("signal_frequency", 100))
    if frequency not in {100, 500}:
        raise ValueError("data.signal_frequency deve ser 100 ou 500.")
    return frequency


def get_records_dir_name(config: dict[str, Any]) -> str:
    """Mapeia frequencia configurada para a pasta oficial do PTB-XL."""
    return "records100" if get_signal_frequency(config) == 100 else "records500"


def get_signal_filename_column(config: dict[str, Any]) -> str:
    """Mapeia frequencia configurada para a coluna de arquivo dos metadados."""
    return "filename_lr" if get_signal_frequency(config) == 100 else "filename_hr"


def check_ptbxl_files(config: dict[str, Any]) -> dict[str, Path]:
    """Verifica arquivos essenciais do PTB-XL e retorna seus caminhos."""
    data_dir = get_data_dir(config)
    records_dir = data_dir / get_records_dir_name(config)
    paths = {
        "metadata": data_dir / config["data"]["metadata_file"],
        "scp_statements": data_dir / config["data"]["scp_statements_file"],
        "records_dir": records_dir,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        message = (
            "Arquivos PTB-XL ausentes. Baixe o dataset e posicione os arquivos em "
            f"{data_dir}. Ausentes: {missing}"
        )
        raise FileNotFoundError(message)
    return paths


def load_metadata(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega `ptbxl_database.csv` e `scp_statements.csv`."""
    paths = check_ptbxl_files(config)
    metadata = pd.read_csv(paths["metadata"], index_col="ecg_id")
    scp_statements = pd.read_csv(paths["scp_statements"], index_col=0)
    return metadata, scp_statements


def add_age_features(metadata: pd.DataFrame) -> pd.DataFrame:
    """Cria features de idade sem interpretar `age == 300` como idade real."""
    df = metadata.copy()
    df["age_is_anon_90_plus"] = df["age"].eq(300)
    df["age_clean"] = df["age"].where(~df["age_is_anon_90_plus"], np.nan)
    # No PTB-XL, idade 300 representa anonimização de pacientes >= 90 anos,
    # portanto a imputacao deve ocorrer apenas dentro dos pipelines ajustados no treino.
    return df


def prepare_metadata(config: dict[str, Any], save_summary: bool = True) -> pd.DataFrame:
    """Carrega metadados, trata idade, constroi rotulos e salva resumo do dataset."""
    metadata, scp_statements = load_metadata(config)
    metadata = add_age_features(metadata)
    labeled = build_multiclass_target(
        metadata,
        scp_statements,
        strategy=config["labels"]["strategy"],
        superclasses=config["labels"]["superclasses"],
    )
    if save_summary:
        save_dataset_summary(labeled, config)
    return labeled


def final_labeled_records(metadata: pd.DataFrame) -> pd.DataFrame:
    """Retorna somente registros com target multiclasse definido."""
    return metadata.loc[metadata["target"].notna()].copy()


def save_dataset_summary(metadata: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Salva resumo de classes, folds e registros removidos em CSV/LaTeX."""
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    ensure_dir(tables_dir)

    total = len(metadata)
    kept = int(metadata["target"].notna().sum())
    multilabel = int(metadata["n_diagnostic_superclasses"].gt(1).sum())
    no_label = int(metadata["n_diagnostic_superclasses"].eq(0).sum())
    summary = pd.DataFrame(
        [
            {"metric": "total_records", "value": total},
            {"metric": "records_kept", "value": kept},
            {"metric": "records_without_diagnostic_superclass", "value": no_label},
            {"metric": "records_multilabel_removed", "value": multilabel},
            {"metric": "n_classes_final", "value": metadata["target"].nunique(dropna=True)},
        ]
    )
    save_table(summary, tables_dir / "dataset_summary.csv", tables_dir / "dataset_summary.tex")

    class_counts = (
        metadata.loc[metadata["target"].notna(), "target"]
        .value_counts()
        .rename_axis("target")
        .reset_index(name="n_records")
    )
    save_table(class_counts, tables_dir / "class_counts.csv", tables_dir / "class_counts.tex")

    fold_counts = (
        metadata.loc[metadata["target"].notna()]
        .groupby(["strat_fold", "target"], observed=True)
        .size()
        .reset_index(name="n_records")
    )
    save_table(fold_counts, tables_dir / "fold_counts.csv", tables_dir / "fold_counts.tex")

    label_strategy = summarize_label_strategy(metadata)
    save_table(
        label_strategy,
        tables_dir / "label_strategy_removals.csv",
        tables_dir / "label_strategy_removals.tex",
    )
    return summary
