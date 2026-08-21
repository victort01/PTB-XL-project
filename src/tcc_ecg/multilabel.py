"""Construcao da tarefa multilabel diagnostica do TCC II."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from tcc_ecg.labels import DEFAULT_SUPERCLASSES, map_scp_to_superclasses, parse_scp_codes


def label_column(superclass: str) -> str:
    """Retorna o nome estavel da coluna binaria de uma superclasse."""
    return f"label_{superclass}"


def build_multilabel_targets(
    metadata: pd.DataFrame,
    scp_statements: pd.DataFrame,
    superclasses: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Mantem todos os registros com ao menos uma superclasse diagnostica relevante."""
    classes = list(superclasses or DEFAULT_SUPERCLASSES)
    df = metadata.copy()
    df["scp_codes_parsed"] = df["scp_codes"].apply(parse_scp_codes)
    df["diagnostic_superclass_weights"] = df["scp_codes_parsed"].apply(
        lambda codes: map_scp_to_superclasses(codes, scp_statements, classes)
    )
    df["diagnostic_superclasses"] = df["diagnostic_superclass_weights"].apply(list)
    df["n_diagnostic_superclasses"] = df["diagnostic_superclasses"].str.len()

    for superclass in classes:
        column = label_column(superclass)
        df[column] = df["diagnostic_superclass_weights"].apply(
            lambda weights, name=superclass: np.int8(name in weights)
        )

    label_columns = [label_column(name) for name in classes]
    df["target_multilabel"] = df[label_columns].apply(
        lambda row: row.astype("int8").tolist(), axis=1
    )
    df["label_status_multilabel"] = np.where(
        df["n_diagnostic_superclasses"].gt(0),
        "kept",
        "removed_no_diagnostic_superclass",
    )
    return df


def final_multilabel_records(metadata: pd.DataFrame) -> pd.DataFrame:
    """Retorna registros com pelo menos um rotulo multilabel positivo."""
    if "label_status_multilabel" not in metadata.columns:
        raise KeyError("Metadados ainda nao possuem rotulos multilabel.")
    return metadata.loc[metadata["label_status_multilabel"].eq("kept")].copy()


def multilabel_target_matrix(
    metadata: pd.DataFrame,
    superclasses: Sequence[str] | None = None,
) -> np.ndarray:
    """Converte as colunas binarias para matriz ``n_registros x n_classes``."""
    classes = list(superclasses or DEFAULT_SUPERCLASSES)
    columns = [label_column(name) for name in classes]
    missing = [column for column in columns if column not in metadata.columns]
    if missing:
        raise KeyError(f"Colunas multilabel ausentes: {missing}")
    return metadata[columns].to_numpy(dtype="int8", copy=True)


def summarize_multilabel_targets(
    metadata: pd.DataFrame,
    superclasses: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Resume prevalencia e numero de positivos de cada superclasse."""
    classes = list(superclasses or DEFAULT_SUPERCLASSES)
    records = final_multilabel_records(metadata)
    total = len(records)
    rows = []
    for superclass in classes:
        positives = int(records[label_column(superclass)].sum())
        rows.append(
            {
                "class": superclass,
                "positive_records": positives,
                "prevalence": positives / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)

