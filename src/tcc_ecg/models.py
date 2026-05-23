"""Treinamento dos modelos classicos de Machine Learning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid

from tcc_ecg.balancing import make_smote
from tcc_ecg.evaluation import compute_classification_metrics, save_metrics_tables
from tcc_ecg.features import get_feature_columns
from tcc_ecg.paths import ensure_dir, resolve_project_path
from tcc_ecg.preprocessing import build_preprocessor
from tcc_ecg.utils import save_table

LOGGER = logging.getLogger(__name__)


def split_by_folds(features: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Divide dados conforme folds oficiais do PTB-XL."""
    folds = config["folds"]
    return {
        "train": features.loc[features["strat_fold"].isin(folds["train"])].copy(),
        "validation": features.loc[features["strat_fold"].isin(folds["validation"])].copy(),
        "test": features.loc[features["strat_fold"].isin(folds["test"])].copy(),
    }


def build_classical_model_pipelines(
    config: dict[str, Any],
    feature_columns: list[str],
    use_smote: bool = False,
    y_train=None,
) -> dict[str, ImbPipeline]:
    """Cria pipelines dos modelos classicos com preprocessamento dentro do treino."""
    seed = int(config["project"]["seed"])
    smote = None
    if use_smote and y_train is not None:
        # SMOTE e ajustado somente no treino; validacao e teste permanecem observados.
        smote = make_smote(y_train, seed, int(config["balancing"]["smote_k_neighbors"]))

    pipelines: dict[str, ImbPipeline] = {}

    def steps(scale: bool, estimator):
        pipeline_steps = [("preprocessor", build_preprocessor(feature_columns, scale=scale))]
        if smote is not None:
            pipeline_steps.append(("smote", smote))
        pipeline_steps.append(("model", estimator))
        return pipeline_steps

    # LR e SVM recebem escala porque sao sensiveis a magnitude das features.
    pipelines["logistic_regression"] = ImbPipeline(
        steps(
            True,
            LogisticRegression(
                max_iter=int(config["models"]["logistic_regression"]["max_iter"]),
                class_weight="balanced",
                random_state=seed,
                n_jobs=None,
            ),
        )
    )
    pipelines["svm"] = ImbPipeline(
        steps(
            True,
            SGDClassifier(
                loss="hinge",
                penalty="l2",
                class_weight="balanced",
                random_state=seed,
                max_iter=2000,
                tol=1e-3,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5,
            ),
        )
    )
    pipelines["random_forest"] = ImbPipeline(
        steps(
            False,
            RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
        )
    )

    try:
        from lightgbm import LGBMClassifier

        pipelines["lightgbm"] = ImbPipeline(
            steps(
                False,
                LGBMClassifier(
                    objective="multiclass",
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                    verbose=-1,
                ),
            )
        )
    except ImportError:
        LOGGER.warning("LightGBM nao instalado; modelo lightgbm sera ignorado.")

    try:
        from catboost import CatBoostClassifier

        pipelines["catboost"] = ImbPipeline(
            steps(
                False,
                CatBoostClassifier(
                    loss_function="MultiClass",
                    auto_class_weights="Balanced",
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                ),
            )
        )
    except ImportError:
        LOGGER.warning("CatBoost nao instalado; modelo catboost sera ignorado.")

    return pipelines


def get_param_grid(config: dict[str, Any], model_name: str) -> list[dict[str, Any]]:
    """Converte a configuracao YAML em grade pequena de parametros."""
    model_config = config["models"].get(model_name, {})
    if model_name == "logistic_regression":
        grid = {"model__C": model_config.get("C", [1.0])}
    elif model_name == "svm":
        grid = {"model__alpha": model_config.get("alpha", [0.0001])}
    elif model_name == "random_forest":
        grid = {
            "model__n_estimators": model_config.get("n_estimators", [200]),
            "model__max_depth": model_config.get("max_depth", [None]),
        }
    elif model_name == "lightgbm":
        grid = {
            "model__n_estimators": model_config.get("n_estimators", [300]),
            "model__learning_rate": model_config.get("learning_rate", [0.1]),
        }
    elif model_name == "catboost":
        grid = {
            "model__iterations": model_config.get("iterations", [300]),
            "model__learning_rate": model_config.get("learning_rate", [0.1]),
        }
    else:
        grid = {}
    return list(ParameterGrid(grid or {}))


def train_and_evaluate_models(
    features: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Treina modelos classicos, compara SMOTE e salva metricas/artefatos."""
    clean = features.loc[features["target_id"].notna()].copy()
    clean["target_id"] = clean["target_id"].astype(int)
    feature_columns = get_feature_columns(clean)
    splits = split_by_folds(clean, config)

    X_train, y_train = splits["train"][feature_columns], splits["train"]["target_id"]
    X_val, y_val = splits["validation"][feature_columns], splits["validation"]["target_id"]
    X_test, y_test = splits["test"][feature_columns], splits["test"]["target_id"]
    train_val = pd.concat([splits["train"], splits["validation"]], axis=0)

    scenarios = ["without_smote"]
    if bool(config["balancing"].get("use_smote", False)):
        scenarios.append("with_smote")

    all_metrics: list[dict[str, Any]] = []
    best_params_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    predictions: dict[str, dict[str, Any]] = {}

    for scenario in scenarios:
        use_smote = scenario == "with_smote"
        pipelines = build_classical_model_pipelines(config, feature_columns, use_smote, y_train)
        for model_name, pipeline in pipelines.items():
            LOGGER.info("Treinando %s (%s)", model_name, scenario)
            best_pipeline, best_params, val_pred, val_score = _fit_best_on_validation(
                pipeline, get_param_grid(config, model_name), X_train, y_train, X_val, y_val
            )

            val_metrics = compute_classification_metrics(
                y_val, val_pred, model_name=f"{model_name}_{scenario}", split="validation"
            )
            val_metrics["smote"] = use_smote
            all_metrics.append(val_metrics)

            final_pipeline = clone(best_pipeline).set_params(**best_params)
            final_pipelines = build_classical_model_pipelines(
                config,
                feature_columns,
                use_smote,
                train_val["target_id"].astype(int),
            )
            final_pipeline = final_pipelines[model_name].set_params(**best_params)
            final_pipeline.fit(train_val[feature_columns], train_val["target_id"].astype(int))
            test_pred = final_pipeline.predict(X_test)
            test_metrics = compute_classification_metrics(
                y_test, test_pred, model_name=f"{model_name}_{scenario}", split="test"
            )
            test_metrics["smote"] = use_smote
            all_metrics.append(test_metrics)
            pred_df = _prediction_frame(splits["test"], test_pred)
            _save_predictions(pred_df, f"{model_name}_{scenario}", config)

            fitted_models[f"{model_name}_{scenario}"] = final_pipeline
            predictions[f"{model_name}_{scenario}"] = {
                "y_test": y_test.to_numpy(),
                "y_pred": test_pred,
                "predictions": pred_df,
                "feature_columns": feature_columns,
            }
            best_params_rows.append(
                {
                    "model": f"{model_name}_{scenario}",
                    "validation_f1_macro": val_score,
                    "best_params": best_params,
                }
            )
            _save_model(final_pipeline, f"{model_name}_{scenario}", config)

    metrics_df = pd.DataFrame(all_metrics)
    best_params_df = pd.DataFrame(best_params_rows)
    frequency = int(config["data"]["signal_frequency"])
    metrics_df["signal_frequency"] = frequency
    best_params_df["signal_frequency"] = frequency
    _save_training_tables(metrics_df, best_params_df, config)
    return {
        "metrics": metrics_df,
        "best_params": best_params_df,
        "models": fitted_models,
        "predictions": predictions,
        "feature_columns": feature_columns,
    }


def _fit_best_on_validation(pipeline, param_grid, X_train, y_train, X_val, y_val):
    best_score = -1.0
    best_params: dict[str, Any] = {}
    best_pipeline = None
    best_pred = None

    for params in param_grid or [{}]:
        candidate = clone(pipeline).set_params(**params)
        candidate.fit(X_train, y_train)
        pred = candidate.predict(X_val)
        score = f1_score(y_val, pred, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_params = params
            best_pipeline = candidate
            best_pred = pred

    if best_pipeline is None or best_pred is None:
        raise RuntimeError("Nenhum modelo foi ajustado durante a busca de parametros.")
    return best_pipeline, best_params, best_pred, best_score


def _save_model(model, model_name: str, config: dict[str, Any]) -> Path:
    models_dir = resolve_project_path(config["outputs"]["models_dir"], config.get("project_root"))
    ensure_dir(models_dir)
    frequency = int(config["data"]["signal_frequency"])
    path = models_dir / f"{model_name}.joblib"
    joblib.dump(model, path)
    joblib.dump(model, models_dir / f"{model_name}_{frequency}hz.joblib")
    return path


def _prediction_frame(test_split: pd.DataFrame, y_pred) -> pd.DataFrame:
    columns = [col for col in ["ecg_id", "target", "target_id"] if col in test_split.columns]
    pred_df = test_split[columns].copy() if columns else pd.DataFrame(index=test_split.index)
    if "ecg_id" not in pred_df.columns:
        pred_df = pred_df.reset_index(names="ecg_id")
    pred_df["y_pred"] = y_pred
    return pred_df


def _save_predictions(predictions: pd.DataFrame, model_name: str, config: dict[str, Any]) -> Path:
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    ensure_dir(tables_dir)
    frequency = int(config["data"]["signal_frequency"])
    path = tables_dir / f"test_predictions_{model_name}.csv"
    predictions.to_csv(path, index=False)
    predictions.to_csv(tables_dir / f"test_predictions_{model_name}_{frequency}hz.csv", index=False)
    return path


def _save_training_tables(
    metrics: pd.DataFrame,
    best_params: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    frequency = int(config["data"]["signal_frequency"])
    save_metrics_tables(metrics, config, stem="model_metrics")
    save_metrics_tables(metrics, config, stem=f"model_metrics_{frequency}hz")
    save_table(best_params, tables_dir / "best_params.csv")
    save_table(best_params, tables_dir / f"best_params_{frequency}hz.csv")
    balancing = (
        metrics.loc[metrics["split"].eq("test")]
        .assign(balanceamento=lambda df: df["smote"].map({True: "com SMOTE", False: "sem SMOTE"}))
        .sort_values(["model", "f1_macro"], ascending=[True, False])
    )
    save_table(balancing, tables_dir / "balancing_comparison.csv", tables_dir / "balancing_comparison.tex")
    save_table(
        balancing,
        tables_dir / f"balancing_comparison_{frequency}hz.csv",
        tables_dir / f"balancing_comparison_{frequency}hz.tex",
    )
