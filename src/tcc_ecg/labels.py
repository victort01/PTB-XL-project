"""Construcao dos rotulos diagnosticos multiclasse a partir do PTB-XL."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def parse_scp_codes(value: Any) -> dict[str, float]:
    """Parseia a coluna `scp_codes`, armazenada como string de dicionario."""
    if value is None:
        return {}
    if isinstance(value, float) and np.isnan(value):
        return {}
    if isinstance(value, Mapping):
        return {str(k): float(v) for k, v in value.items()}
    if not isinstance(value, str):
        raise TypeError(f"scp_codes deve ser string ou dict, recebido: {type(value)!r}")

    stripped = value.strip()
    if not stripped:
        return {}
    parsed = ast.literal_eval(stripped)
    if not isinstance(parsed, Mapping):
        raise ValueError(f"scp_codes nao representa um dicionario: {value!r}")
    return {str(k): float(v) for k, v in parsed.items()}


def _is_diagnostic(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "diagnostic"}
    return bool(value)


def map_scp_to_superclasses(
    scp_codes: str | Mapping[str, float],
    scp_statements: pd.DataFrame,
    superclasses: list[str] | None = None,
) -> dict[str, float]:
    """Mapeia codigos SCP para superclasses diagnosticas com maior peso por classe."""
    parsed = parse_scp_codes(scp_codes)
    allowed = superclasses or DEFAULT_SUPERCLASSES
    weights: dict[str, float] = {}

    statements = scp_statements.copy()
    if "diagnostic_class" not in statements.columns:
        raise KeyError("scp_statements.csv precisa conter a coluna 'diagnostic_class'.")

    for code, weight in parsed.items():
        if code not in statements.index:
            continue
        row = statements.loc[code]
        if "diagnostic" in statements.columns and not _is_diagnostic(row["diagnostic"]):
            continue
        diagnostic_class = row["diagnostic_class"]
        if diagnostic_class in allowed:
            weights[diagnostic_class] = max(float(weight), weights.get(diagnostic_class, 0.0))

    return {klass: weights[klass] for klass in allowed if klass in weights}


def build_multiclass_target(
    metadata: pd.DataFrame,
    scp_statements: pd.DataFrame,
    strategy: str = "strict_single_label",
    superclasses: list[str] | None = None,
) -> pd.DataFrame:
    """Adiciona `diagnostic_superclasses`, `target` e `target_id` ao dataframe."""
    allowed = superclasses or DEFAULT_SUPERCLASSES
    class_to_id = {name: idx for idx, name in enumerate(allowed)}

    df = metadata.copy()
    df["scp_codes_parsed"] = df["scp_codes"].apply(parse_scp_codes)
    df["diagnostic_superclass_weights"] = df["scp_codes_parsed"].apply(
        lambda codes: map_scp_to_superclasses(codes, scp_statements, allowed)
    )
    df["diagnostic_superclasses"] = df["diagnostic_superclass_weights"].apply(lambda item: list(item))
    df["n_diagnostic_superclasses"] = df["diagnostic_superclasses"].str.len()

    if strategy == "strict_single_label":
        df["target"] = df["diagnostic_superclasses"].apply(
            lambda classes: classes[0] if len(classes) == 1 else pd.NA
        )
    elif strategy == "primary_by_scp_weight":
        df["target"] = df["diagnostic_superclass_weights"].apply(
            lambda weights: _choose_primary_superclass(weights, allowed)
        )
    else:
        raise ValueError(
            "Estrategia de rotulo desconhecida. Use 'strict_single_label' "
            "ou 'primary_by_scp_weight'."
        )

    df["target_id"] = df["target"].map(class_to_id).astype("Int64")
    df["label_status"] = np.select(
        [
            df["target"].notna(),
            df["n_diagnostic_superclasses"].eq(0),
            df["n_diagnostic_superclasses"].gt(1),
        ],
        ["kept", "removed_no_diagnostic_superclass", "removed_multilabel"],
        default="removed_other",
    )
    return df


def _choose_primary_superclass(weights: dict[str, float], superclasses: list[str]) -> str | pd.NA:
    if not weights:
        return pd.NA
    ordered = sorted(weights.items(), key=lambda item: (-item[1], superclasses.index(item[0])))
    return ordered[0][0]


def summarize_label_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Resume quantos registros foram mantidos ou removidos pela estrategia de rotulo."""
    summary = (
        df["label_status"]
        .value_counts(dropna=False)
        .rename_axis("status")
        .reset_index(name="n_records")
        .sort_values("status")
    )
    summary["percentage"] = summary["n_records"] / len(df) * 100
    return summary
