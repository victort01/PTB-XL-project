"""Modelos e inventario experimental do TCC II."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from tcc_ecg.preprocessing import build_preprocessor


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    backend: str
    source: str
    frequency: int
    input_representation: str
    status: str


MODEL_SPECS = (
    ModelSpec("logistic_regression", "classical", "local", "scikit-learn", 500, "statistical_features", "ready"),
    ModelSpec("svm", "classical", "local", "scikit-learn", 500, "statistical_features", "ready"),
    ModelSpec("random_forest", "classical", "local", "scikit-learn", 500, "statistical_features", "ready"),
    ModelSpec("lightgbm", "classical", "local", "LightGBM", 500, "statistical_features", "ready"),
    ModelSpec("catboost", "classical", "local", "CatBoost", 500, "statistical_features", "ready"),
    ModelSpec("helme_inception1d", "helme", "external", "helme_benchmark", 100, "raw_signal", "prepared"),
    ModelSpec("helme_xresnet1d101", "helme", "external", "helme_benchmark", 100, "raw_signal", "prepared"),
    ModelSpec("helme_resnet1d_wang", "helme", "external", "helme_benchmark", 100, "raw_signal", "prepared"),
    ModelSpec("helme_fcn_wang", "helme", "external", "helme_benchmark", 100, "raw_signal", "prepared"),
    ModelSpec("helme_lstm", "helme", "external", "helme_benchmark", 100, "raw_signal", "prepared"),
    ModelSpec("helme_lstm_bidir", "helme", "external", "helme_benchmark", 100, "raw_signal", "prepared"),
    ModelSpec("tcn", "temporal_convolution", "local", "project_extension", 500, "raw_signal", "ready"),
    ModelSpec("s4", "state_space", "external", "ssm_ecg", 100, "raw_signal", "prepared"),
    ModelSpec("ecg_jepa", "self_supervised", "external", "ecg_jepa", 100, "raw_signal", "prepared"),
    ModelSpec("cpc", "self_supervised", "external", "ecg_selfsupervised", 100, "raw_signal", "contingency"),
)


def model_inventory_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(spec) for spec in MODEL_SPECS])


def get_model_spec(model_name: str) -> ModelSpec:
    for spec in MODEL_SPECS:
        if spec.name == model_name:
            return spec
    raise KeyError(f"Modelo TCC II desconhecido: {model_name}")


def build_multilabel_classical_pipelines(
    config: dict[str, Any],
    feature_columns: list[str],
) -> dict[str, Pipeline]:
    """Cria classificadores um-contra-rest com preprocessamento ajustado no treino."""
    seed = int(config["project"]["seed"])
    model_config = config["models"]["classical"]

    def pipeline(estimator, scale: bool) -> Pipeline:
        return Pipeline(
            [
                ("preprocessor", build_preprocessor(feature_columns, scale=scale)),
                ("model", OneVsRestClassifier(estimator, n_jobs=None)),
            ]
        )

    pipelines: dict[str, Pipeline] = {
        "logistic_regression": pipeline(
            LogisticRegression(
                max_iter=int(model_config["logistic_regression"]["max_iter"]),
                C=float(model_config["logistic_regression"]["C"]),
                class_weight="balanced",
                random_state=seed,
            ),
            scale=True,
        ),
        "svm": pipeline(
            SGDClassifier(
                loss="hinge",
                alpha=float(model_config["svm"]["alpha"]),
                max_iter=int(model_config["svm"]["max_iter"]),
                class_weight="balanced",
                random_state=seed,
            ),
            scale=True,
        ),
        "random_forest": pipeline(
            RandomForestClassifier(
                n_estimators=int(model_config["random_forest"]["n_estimators"]),
                max_depth=model_config["random_forest"]["max_depth"],
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
            scale=False,
        ),
    }
    try:
        from lightgbm import LGBMClassifier

        pipelines["lightgbm"] = pipeline(
            LGBMClassifier(
                objective="binary",
                n_estimators=int(model_config["lightgbm"]["n_estimators"]),
                learning_rate=float(model_config["lightgbm"]["learning_rate"]),
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            ),
            scale=False,
        )
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier

        pipelines["catboost"] = pipeline(
            CatBoostClassifier(
                loss_function="Logloss",
                auto_class_weights="Balanced",
                iterations=int(model_config["catboost"]["iterations"]),
                learning_rate=float(model_config["catboost"]["learning_rate"]),
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
            ),
            scale=False,
        )
    except ImportError:
        pass
    return pipelines


def build_tcn_model(
    input_channels: int,
    n_classes: int,
    channels: list[int],
    kernel_size: int,
    dropout: float,
):
    """Constroi uma TCN residual dilatada com saida multilabel em logits."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise ImportError("PyTorch nao instalado. Rode: python -m pip install -e .[tcc2]") from exc
    nn = torch.nn

    class TemporalBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
            super().__init__()
            padding = dilation * (kernel_size - 1) // 2
            self.main = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
            self.activation = nn.GELU()

        def forward(self, x):
            return self.activation(self.main(x) + self.shortcut(x))

    class TCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks = []
            current = input_channels
            for index, output_channels in enumerate(channels):
                blocks.append(TemporalBlock(current, int(output_channels), dilation=2**index))
                current = int(output_channels)
            self.encoder = nn.Sequential(*blocks)
            self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(current, n_classes))

        def forward(self, x):
            return self.head(self.encoder(x))

    return TCN()

