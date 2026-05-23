"""Interpretabilidade com SHAP e LIME para modelos tabulares."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tcc_ecg.paths import ensure_dir, resolve_project_path
from tcc_ecg.utils import save_table


def transformed_features_for_model(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Aplica apenas o preprocessador do pipeline e retorna nomes legiveis."""
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        return X.to_numpy(), list(X.columns)
    X_transformed = preprocessor.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    names = list(preprocessor.get_feature_names_out())
    return np.asarray(X_transformed), names


def run_shap_analysis(
    pipeline,
    X: pd.DataFrame,
    config: dict[str, Any],
    max_samples: int = 300,
) -> pd.DataFrame:
    """Gera ranking global de features e graficos SHAP para modelo de arvore."""
    import shap

    output_dir = resolve_project_path(config["outputs"]["figures_dir"], config.get("project_root"))
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    ensure_dir(output_dir)

    sample = X.sample(n=min(max_samples, len(X)), random_state=int(config["project"]["seed"]))
    X_transformed, feature_names = transformed_features_for_model(pipeline, sample)
    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False, max_display=20)
    plt.title("SHAP summary - importancia das features para o modelo")
    plt.savefig(output_dir / "fig_shap_summary.png", dpi=160, bbox_inches="tight")
    plt.savefig(output_dir / "fig_shap_summary.pdf", bbox_inches="tight")
    plt.close()

    importance = _mean_abs_shap(shap_values)
    top_features = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .head(30)
    )
    save_table(top_features, tables_dir / "top_features_shap.csv", tables_dir / "top_features_shap.tex")

    fig, ax = plt.subplots(figsize=(8, 6))
    top_features.head(20).sort_values("mean_abs_shap").plot.barh(
        x="feature", y="mean_abs_shap", ax=ax, legend=False, color="#4C78A8"
    )
    ax.set_title("Top features por importancia SHAP media")
    ax.set_xlabel("|SHAP| medio")
    fig.savefig(output_dir / "fig_shap_bar.png", dpi=160, bbox_inches="tight")
    fig.savefig(output_dir / "fig_shap_bar.pdf", bbox_inches="tight")
    plt.close(fig)
    return top_features


def _mean_abs_shap(shap_values) -> np.ndarray:
    if isinstance(shap_values, list):
        values = np.stack(shap_values, axis=0)
        return np.mean(np.abs(values), axis=(0, 1))
    values = getattr(shap_values, "values", shap_values)
    values = np.asarray(values)
    if values.ndim == 3:
        return np.mean(np.abs(values), axis=(0, 2))
    return np.mean(np.abs(values), axis=0)


def run_lime_examples(
    pipeline,
    X_train: pd.DataFrame,
    X_examples: pd.DataFrame,
    y_true,
    y_pred,
    class_names: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Gera exemplos locais LIME legiveis para predicoes individuais."""
    from lime.lime_tabular import LimeTabularExplainer

    output_dir = resolve_project_path(config["outputs"]["figures_dir"], config.get("project_root"))
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    ensure_dir(output_dir)
    ensure_dir(tables_dir)
    X_train_transformed, feature_names = transformed_features_for_model(pipeline, X_train)
    X_examples_transformed, _ = transformed_features_for_model(pipeline, X_examples)
    model = pipeline.named_steps["model"]
    if not hasattr(model, "predict_proba"):
        raise AttributeError("O modelo escolhido para LIME precisa implementar predict_proba.")

    explainer = LimeTabularExplainer(
        training_data=X_train_transformed,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=int(config["project"]["seed"]),
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    correct = np.flatnonzero(y_true == y_pred)
    errors = np.flatnonzero(y_true != y_pred)
    selected: list[tuple[str, int]] = []

    if len(correct):
        selected.append(("Predicao correta", int(correct[0])))
    if len(errors):
        selected.append(("Predicao incorreta", int(errors[0])))
    else:
        minority_index = _minority_example_index(y_true, y_pred, correct)
        if minority_index is not None:
            selected.append(("Classe minoritaria", minority_index))

    summaries = []
    explanation_frames = []
    for position, (example_name, idx) in enumerate(selected[:2]):
        pred_label = int(y_pred[idx])
        true_label = int(y_true[idx])
        explanation = explainer.explain_instance(
            X_examples_transformed[idx],
            model.predict_proba,
            labels=[pred_label],
            num_features=12,
        )
        explanation_df = _lime_explanation_frame(explanation, pred_label)
        explanation_frames.append((example_name, explanation_df, true_label, pred_label))
        output_name = "fig_lime_correct_example.png" if position == 0 else "fig_lime_error_or_minority_example.png"
        legacy_name = None if position == 0 else "fig_lime_error_example.png"
        _save_lime_bar_figure(
            explanation_df,
            output_dir / output_name,
            title=f"LIME - {example_name.lower()}",
            subtitle=f"real: {class_names[true_label]} | predita: {class_names[pred_label]}",
        )
        if legacy_name:
            _save_lime_bar_figure(
                explanation_df,
                output_dir / legacy_name,
                title=f"LIME - {example_name.lower()}",
                subtitle=f"real: {class_names[true_label]} | predita: {class_names[pred_label]}",
            )
        summaries.append(
            {
                "exemplo": example_name,
                "classe_real": class_names[true_label],
                "classe_predita": class_names[pred_label],
                "resultado_predicao": "correta" if true_label == pred_label else "incorreta",
                "principais_features_positivas": _join_top_features(explanation_df, positive=True),
                "principais_features_negativas": _join_top_features(explanation_df, positive=False),
            }
        )

    if explanation_frames:
        _save_lime_combined_figure(
            explanation_frames,
            class_names,
            output_dir / "fig_lime_examples_combined.png",
        )

    summary = pd.DataFrame(summaries)
    save_table(
        summary,
        tables_dir / "lime_examples_summary.csv",
        tables_dir / "lime_examples_summary.tex",
    )
    return summary


def _minority_example_index(y_true: np.ndarray, y_pred: np.ndarray, candidates: np.ndarray) -> int | None:
    if len(candidates) == 0:
        return None
    counts = pd.Series(y_true).value_counts(ascending=True)
    for klass in counts.index:
        matches = candidates[y_true[candidates] == klass]
        if len(matches):
            return int(matches[0])
    return None


def _lime_explanation_frame(explanation, label: int) -> pd.DataFrame:
    rows = explanation.as_list(label=label)
    frame = pd.DataFrame(rows, columns=["feature", "weight"])
    frame["feature"] = frame["feature"].map(_clean_lime_feature_name)
    frame["abs_weight"] = frame["weight"].abs()
    return frame.sort_values("abs_weight", ascending=False).head(12)


def _clean_lime_feature_name(name: str) -> str:
    return (
        str(name)
        .replace("num__", "")
        .replace("cat__", "")
        .replace("remainder__", "")
    )


def _join_top_features(explanation_df: pd.DataFrame, positive: bool, n: int = 4) -> str:
    subset = explanation_df.loc[explanation_df["weight"].gt(0) if positive else explanation_df["weight"].lt(0)].copy()
    subset = subset.sort_values("weight", ascending=not positive).head(n)
    if subset.empty:
        return ""
    return "; ".join(subset["feature"].astype(str).tolist())


def _save_lime_bar_figure(explanation_df: pd.DataFrame, output_path: Path, title: str, subtitle: str) -> None:
    plot_df = explanation_df.sort_values("weight")
    colors = np.where(plot_df["weight"].ge(0), "#4C78A8", "#E45756")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.barh(plot_df["feature"], plot_df["weight"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_xlabel("Contribuicao local para a classe predita")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _save_lime_combined_figure(
    explanation_frames: list[tuple[str, pd.DataFrame, int, int]],
    class_names: list[str],
    output_path: Path,
) -> None:
    n_examples = len(explanation_frames)
    fig, axes = plt.subplots(1, n_examples, figsize=(8.5 * n_examples, 5.8), squeeze=False)
    for ax, (example_name, explanation_df, true_label, pred_label) in zip(axes[0], explanation_frames, strict=False):
        plot_df = explanation_df.sort_values("weight")
        colors = np.where(plot_df["weight"].ge(0), "#4C78A8", "#E45756")
        ax.barh(plot_df["feature"], plot_df["weight"], color=colors)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(
            f"{example_name}\nreal: {class_names[true_label]} | predita: {class_names[pred_label]}",
            fontsize=11,
        )
        ax.set_xlabel("Contribuicao local")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Exemplos de explicacoes locais com LIME", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
