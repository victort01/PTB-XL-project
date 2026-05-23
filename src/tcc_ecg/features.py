"""Extracao de caracteristicas simples e reproduziveis dos sinais ECG."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from tcc_ecg.data import final_labeled_records, get_data_dir, get_signal_filename_column
from tcc_ecg.paths import ensure_dir, resolve_project_path
from tcc_ecg.utils import save_table

LOGGER = logging.getLogger(__name__)

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ID_COLUMNS = {"ecg_id", "strat_fold", "target", "target_id"}


def extract_signal_features(
    signal: np.ndarray,
    lead_names: list[str] | None = None,
    include_fft_features: bool = False,
    sampling_rate: int = 100,
) -> dict[str, float]:
    """Extrai estatisticas por derivacao como baseline simples para ML classico."""
    values = np.asarray(signal, dtype=float)
    if values.ndim != 2:
        raise ValueError("O sinal deve ter shape (n_timesteps, n_leads).")

    lead_names = lead_names or LEADS[: values.shape[1]]
    if len(lead_names) != values.shape[1]:
        raise ValueError("A quantidade de nomes de derivacoes nao bate com o sinal.")

    features: dict[str, float] = {}
    for idx, lead in enumerate(lead_names):
        x = values[:, idx]
        features[f"{lead}_mean"] = float(np.nanmean(x))
        features[f"{lead}_std"] = float(np.nanstd(x))
        features[f"{lead}_min"] = float(np.nanmin(x))
        features[f"{lead}_max"] = float(np.nanmax(x))
        features[f"{lead}_median"] = float(np.nanmedian(x))
        features[f"{lead}_p25"] = float(np.nanpercentile(x, 25))
        features[f"{lead}_p75"] = float(np.nanpercentile(x, 75))
        features[f"{lead}_ptp"] = float(np.nanmax(x) - np.nanmin(x))
        features[f"{lead}_energy"] = float(np.nansum(x**2))
        features[f"{lead}_rms"] = float(np.sqrt(np.nanmean(x**2)))
        features[f"{lead}_skew"] = float(skew(x, nan_policy="omit"))
        features[f"{lead}_kurtosis"] = float(kurtosis(x, nan_policy="omit"))

        if include_fft_features:
            features.update(_fft_band_features(x, lead, sampling_rate))

    return features


def _fft_band_features(x: np.ndarray, lead: str, sampling_rate: int) -> dict[str, float]:
    spectrum = np.abs(np.fft.rfft(np.nan_to_num(x))) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1 / sampling_rate)
    bands = {
        "fft_0_5_5": (0.5, 5),
        "fft_5_15": (5, 15),
        "fft_15_40": (15, 40),
    }
    return {
        f"{lead}_{name}_energy": float(spectrum[(freqs >= low) & (freqs < high)].sum())
        for name, (low, high) in bands.items()
    }


def load_signal(record_path: str | Path) -> np.ndarray:
    """Carrega um registro WFDB e retorna apenas a matriz do sinal."""
    import wfdb

    signal, _ = wfdb.rdsamp(str(record_path))
    return signal


def record_path_for_row(row: pd.Series, config: dict[str, Any]) -> Path:
    """Resolve o caminho do registro conforme frequencia configurada."""
    data_dir = get_data_dir(config)
    column = get_signal_filename_column(config)
    if column not in row:
        raise KeyError(f"Coluna ausente nos metadados PTB-XL: {column}")
    return data_dir / str(row[column])


def build_feature_table(
    metadata: pd.DataFrame,
    config: dict[str, Any],
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extrai features brutas por sinal e salva features/labels processadas."""
    records = final_labeled_records(metadata)
    max_records = config.get("features", {}).get("max_records")
    if max_records:
        records = records.head(int(max_records))

    rows: list[dict[str, Any]] = []
    for ecg_id, row in records.iterrows():
        record_path = record_path_for_row(row, config)
        try:
            signal = load_signal(record_path)
        except Exception as exc:  # pragma: no cover - depende dos arquivos WFDB reais
            LOGGER.warning("Falha ao carregar ECG %s em %s: %s", ecg_id, record_path, exc)
            continue

        # Esta extracao usa apenas o proprio registro, sem parametros aprendidos globalmente.
        feature_row = extract_signal_features(
            signal,
            include_fft_features=bool(config["features"].get("include_fft_features", False)),
            sampling_rate=int(config["data"]["signal_frequency"]),
        )
        if bool(config["features"].get("include_metadata", True)):
            feature_row.update(
                {
                    "age_clean": row.get("age_clean", np.nan),
                    "age_is_anon_90_plus": int(bool(row.get("age_is_anon_90_plus", False))),
                    "sex": pd.to_numeric(row.get("sex", np.nan), errors="coerce"),
                }
            )
        feature_row.update(
            {
                "ecg_id": ecg_id,
                "strat_fold": int(row["strat_fold"]),
                "target": row["target"],
                "target_id": int(row["target_id"]),
            }
        )
        rows.append(feature_row)

    features = pd.DataFrame(rows)
    labels = features[["ecg_id", "strat_fold", "target", "target_id"]].copy()
    if save:
        save_feature_outputs(features, labels, config)
    return features, labels


def save_feature_outputs(features: pd.DataFrame, labels: pd.DataFrame, config: dict[str, Any]) -> None:
    processed_dir = resolve_project_path(config["outputs"]["processed_dir"], config.get("project_root"))
    ensure_dir(processed_dir)
    frequency = int(config["data"]["signal_frequency"])
    features.to_parquet(processed_dir / "features.parquet", index=False)
    labels.to_parquet(processed_dir / "labels.parquet", index=False)
    features.to_parquet(processed_dir / f"features_{frequency}hz.parquet", index=False)
    labels.to_parquet(processed_dir / f"labels_{frequency}hz.parquet", index=False)
    save_feature_dictionary(features, config)


def save_feature_dictionary(features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Salva dicionario simples das features numericas."""
    rows = []
    for column in get_feature_columns(features):
        if column in {"age_clean", "age_is_anon_90_plus", "sex"}:
            source = "metadata"
        elif "_fft_" in column:
            source = "frequency_domain"
        else:
            source = "signal_statistics"
        rows.append({"feature": column, "source": source, "description": _describe_feature(column)})
    dictionary = pd.DataFrame(rows)
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    save_table(dictionary, tables_dir / "feature_dictionary.csv")
    return dictionary


def get_feature_columns(features: pd.DataFrame) -> list[str]:
    """Retorna colunas usadas como entrada dos modelos."""
    return [col for col in features.columns if col not in ID_COLUMNS]


def _describe_feature(column: str) -> str:
    if column == "age_clean":
        return "Idade com valores 300 substituidos por NaN antes da imputacao no pipeline."
    if column == "age_is_anon_90_plus":
        return "Flag indicando idade anonimizada como 300 no PTB-XL."
    if column == "sex":
        return "Sexo informado no metadado do PTB-XL."
    suffixes = {
        "mean": "media",
        "std": "desvio padrao",
        "min": "minimo",
        "max": "maximo",
        "median": "mediana",
        "p25": "percentil 25",
        "p75": "percentil 75",
        "ptp": "amplitude pico-a-pico",
        "energy": "energia",
        "rms": "raiz media quadratica",
        "skew": "assimetria",
        "kurtosis": "curtose",
    }
    for suffix, description in suffixes.items():
        if column.endswith(f"_{suffix}"):
            return f"{description} da derivacao {column[: -len(suffix) - 1]}"
    return "Caracteristica derivada do sinal ECG."
