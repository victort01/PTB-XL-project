"""Interpretabilidade da InceptionTime 1D com Integrated Gradients.

Este script executa uma analise pos-hoc do checkpoint ja treinado da
InceptionTime 1D forte. Ele nao treina, ajusta hiperparametros nem altera o
pipeline: apenas carrega o modelo, escolhe uma amostra do fold de teste e
calcula regioes do sinal que influenciaram a predicao do modelo.

Uso a partir da raiz do projeto:
    python scripts/interpret_inceptiontime_integrated_gradients.py

Opcionalmente, instale Captum para usar a implementacao da biblioteca:
    python -m pip install captum

Se Captum nao estiver instalado, o script usa uma implementacao simples de
Integrated Gradients baseada em autograd do PyTorch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tcc_ecg.config import load_config
from tcc_ecg.data import prepare_metadata
from tcc_ecg.deep_learning import build_resnet1d_cache, compute_train_channel_normalization
from tcc_ecg.deep_learning_strong import (
    build_inceptiontime1d_model,
    get_deep_learning_strong_config,
)
from tcc_ecg.models import split_by_folds
from tcc_ecg.paths import ensure_dir, resolve_project_path

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
PREFERRED_CLASSES = ["MI", "HYP", "STTC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml", help="Caminho do config.yaml.")
    parser.add_argument(
        "--checkpoint",
        default="models/deep_learning_strong_best.pt",
        help="Checkpoint treinado da InceptionTime 1D.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=64,
        help="Numero de passos do Integrated Gradients.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size para predicao no teste.",
    )
    parser.add_argument(
        "--ig-batch-size",
        type=int,
        default=8,
        help="Batch interno para calcular Integrated Gradients sem estourar memoria.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=0.5,
        help="Tamanho da janela temporal destacada no ECG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    strong_config = _load_strong_config(config)
    config["data"]["signal_frequency"] = int(strong_config.get("frequency", 500))

    torch = _import_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_paths = _load_or_build_cache(config, strong_config)
    signals = np.load(cache_paths["signals"], mmap_mode="r")
    labels = np.load(cache_paths["labels"])
    cache_metadata = pd.read_csv(cache_paths["metadata"]).assign(row_idx=np.arange(len(labels)))
    splits = split_by_folds(cache_metadata, config)
    train_idx = splits["train"]["row_idx"].to_numpy(dtype="int64")
    test_idx = splits["test"]["row_idx"].to_numpy(dtype="int64")

    mean, std = _load_or_compute_normalization(config, cache_paths["signals"], train_idx, args.batch_size)
    model = _load_inceptiontime_model(config, strong_config, args.checkpoint, device)

    y_true, y_pred, pred_prob = _predict_test(
        model=model,
        signals=signals,
        labels=labels,
        test_idx=test_idx,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        device=device,
        torch=torch,
    )

    sample = _select_interpretable_sample(
        test_idx=test_idx,
        y_true=y_true,
        y_pred=y_pred,
        pred_prob=pred_prob,
        cache_metadata=cache_metadata,
        class_names=config["labels"]["superclasses"],
    )

    signal_raw = signals[sample["row_idx"]].astype("float32")
    signal_norm = (signal_raw - mean[None, :]) / std[None, :]
    input_tensor = torch.from_numpy(np.ascontiguousarray(signal_norm.T[None, :, :])).to(device)
    target_id = int(sample["pred_id"])

    attributions, method = _integrated_gradients(
        model=model,
        input_tensor=input_tensor,
        target_id=target_id,
        n_steps=args.n_steps,
        internal_batch_size=args.ig_batch_size,
        torch=torch,
    )
    attributions = attributions.detach().cpu().numpy()[0]

    figure_paths, summary = _save_outputs(
        signal_raw=signal_raw,
        attributions=attributions,
        sample=sample,
        class_names=config["labels"]["superclasses"],
        method=method,
        n_steps=args.n_steps,
        frequency=int(strong_config.get("frequency", 500)),
        window_seconds=float(args.window_seconds),
        config=config,
    )

    _write_explanatory_text(config)
    print(f"Figura PNG: {figure_paths['png']}")
    print(f"Figura PDF: {figure_paths['pdf']}")
    print(f"Classe real: {summary['classe_real']}")
    print(f"Classe predita: {summary['classe_predita']}")
    print(f"Deriva\u00e7\u00e3o mais relevante: {summary['derivacao_mais_relevante']}")
    print(
        "Janela temporal mais relevante: "
        f"{summary['janela_inicio_s']:.2f}s - {summary['janela_fim_s']:.2f}s"
    )


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depende do ambiente local
        raise ImportError("PyTorch nao esta instalado. Rode: python -m pip install -e .[dl]") from exc
    return torch


def _load_strong_config(config: dict[str, Any]) -> dict[str, Any]:
    metadata_path = resolve_project_path("models/deep_learning_strong_metadata.json", config.get("project_root"))
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "config" in metadata:
            return metadata["config"]
    return get_deep_learning_strong_config(config)


def _load_or_build_cache(config: dict[str, Any], strong_config: dict[str, Any]) -> dict[str, Path]:
    frequency = int(strong_config.get("frequency", 500))
    processed_dir = resolve_project_path(config["outputs"]["processed_dir"], config.get("project_root"))
    cache_paths = {
        "signals": processed_dir / f"signals_{frequency}hz_resnet1d.npy",
        "labels": processed_dir / f"labels_{frequency}hz_resnet1d.npy",
        "metadata": processed_dir / f"metadata_{frequency}hz_resnet1d.csv",
    }
    if all(path.exists() for path in cache_paths.values()):
        return cache_paths

    # Reusa o construtor de cache do pipeline neural. O cache e gerado a partir
    # dos dados rotulados, sem aprender parametros de validacao ou teste.
    metadata = prepare_metadata(config, save_summary=False)
    cache_config = dict(config)
    cache_config["data"] = dict(config["data"])
    cache_config["data"]["signal_frequency"] = frequency
    cache_config["deep_learning"] = dict(config.get("deep_learning", {}))
    cache_config["deep_learning"]["resnet1d"] = {
        "cache_dtype": "float16",
        "force_rebuild_cache": False,
        "max_records": strong_config.get("max_records"),
    }
    return build_resnet1d_cache(metadata, cache_config)


def _load_or_compute_normalization(
    config: dict[str, Any],
    signals_path: Path,
    train_idx: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    frequency = int(config["data"]["signal_frequency"])
    norm_path = resolve_project_path(
        f"data/processed/deep_learning_strong_normalization_{frequency}hz.npz",
        config.get("project_root"),
    )
    if norm_path.exists():
        data = np.load(norm_path)
        return data["mean"].astype("float32"), data["std"].astype("float32")
    return compute_train_channel_normalization(signals_path, train_idx, batch_size=batch_size)


def _load_inceptiontime_model(
    config: dict[str, Any],
    strong_config: dict[str, Any],
    checkpoint: str | Path,
    device,
):
    torch = _import_torch()
    checkpoint_path = resolve_project_path(checkpoint, config.get("project_root"))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {checkpoint_path}")

    model = build_inceptiontime1d_model(
        input_channels=12,
        n_classes=len(config["labels"]["superclasses"]),
        base_channels=int(strong_config["base_channels"]),
        block_channels=[int(item) for item in strong_config["block_channels"]],
        kernel_sizes=[int(item) for item in strong_config["kernel_sizes"]],
        bottleneck_channels=int(strong_config["bottleneck_channels"]),
        dropout=float(strong_config["dropout"]),
    ).to(device)
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _predict_test(
    model,
    signals: np.ndarray,
    labels: np.ndarray,
    test_idx: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    device,
    torch,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_pred: list[int] = []
    pred_prob: list[float] = []
    with torch.no_grad():
        for start in range(0, len(test_idx), batch_size):
            idx = test_idx[start : start + batch_size]
            batch = signals[idx].astype("float32")
            batch = (batch - mean[None, None, :]) / std[None, None, :]
            batch_tensor = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 2, 1))).to(device)
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            y_true.extend(labels[idx].astype(int).tolist())
            y_pred.extend(pred.detach().cpu().numpy().astype(int).tolist())
            pred_prob.extend(probs.max(dim=1).values.detach().cpu().numpy().astype(float).tolist())
    return np.array(y_true), np.array(y_pred), np.array(pred_prob)


def _select_interpretable_sample(
    test_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pred_prob: np.ndarray,
    cache_metadata: pd.DataFrame,
    class_names: list[str],
) -> dict[str, Any]:
    correct = y_true == y_pred
    for class_name in PREFERRED_CLASSES:
        class_id = class_names.index(class_name)
        candidates = np.where(correct & (y_true == class_id))[0]
        if len(candidates) > 0:
            chosen_pos = candidates[np.argmax(pred_prob[candidates])]
            return _sample_info(chosen_pos, test_idx, y_true, y_pred, pred_prob, cache_metadata, class_names)

    candidates = np.where(correct)[0]
    if len(candidates) > 0:
        chosen_pos = candidates[np.argmax(pred_prob[candidates])]
        return _sample_info(chosen_pos, test_idx, y_true, y_pred, pred_prob, cache_metadata, class_names)

    # Fallback raro: se nao houver acerto no teste carregado, usa a predicao mais confiante.
    chosen_pos = int(np.argmax(pred_prob))
    return _sample_info(chosen_pos, test_idx, y_true, y_pred, pred_prob, cache_metadata, class_names)


def _sample_info(
    position: int,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pred_prob: np.ndarray,
    cache_metadata: pd.DataFrame,
    class_names: list[str],
) -> dict[str, Any]:
    row_idx = int(test_idx[position])
    row = cache_metadata.loc[cache_metadata["row_idx"].eq(row_idx)].iloc[0]
    true_id = int(y_true[position])
    pred_id = int(y_pred[position])
    return {
        "row_idx": row_idx,
        "ecg_id": int(row["ecg_id"]),
        "true_id": true_id,
        "pred_id": pred_id,
        "true_class": class_names[true_id],
        "pred_class": class_names[pred_id],
        "pred_prob": float(pred_prob[position]),
        "is_correct": bool(true_id == pred_id),
    }


def _integrated_gradients(
    model,
    input_tensor,
    target_id: int,
    n_steps: int,
    internal_batch_size: int,
    torch,
):
    baseline = torch.zeros_like(input_tensor)
    try:
        from captum.attr import IntegratedGradients

        ig = IntegratedGradients(model)
        attrs = ig.attribute(
            input_tensor,
            baselines=baseline,
            target=target_id,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
        )
        return attrs, "Captum Integrated Gradients"
    except ImportError:
        return _manual_integrated_gradients(
            model=model,
            input_tensor=input_tensor,
            baseline=baseline,
            target_id=target_id,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            torch=torch,
        ), "Integrated Gradients manual"


def _manual_integrated_gradients(
    model,
    input_tensor,
    baseline,
    target_id: int,
    n_steps: int,
    internal_batch_size: int,
    torch,
):
    model.eval()
    alphas = torch.linspace(0.0, 1.0, steps=n_steps, device=input_tensor.device)
    total_gradients = torch.zeros_like(input_tensor)
    delta = input_tensor - baseline

    for start in range(0, n_steps, internal_batch_size):
        alpha = alphas[start : start + internal_batch_size].view(-1, 1, 1)
        scaled = baseline + alpha * delta
        scaled.requires_grad_(True)
        logits = model(scaled)
        selected = logits[:, target_id].sum()
        gradients = torch.autograd.grad(selected, scaled)[0]
        total_gradients += gradients.sum(dim=0, keepdim=True)

    avg_gradients = total_gradients / float(n_steps)
    return delta * avg_gradients


def _save_outputs(
    signal_raw: np.ndarray,
    attributions: np.ndarray,
    sample: dict[str, Any],
    class_names: list[str],
    method: str,
    n_steps: int,
    frequency: int,
    window_seconds: float,
    config: dict[str, Any],
) -> tuple[dict[str, Path], dict[str, Any]]:
    figures_dir = resolve_project_path("reports/figures/interpretability", config.get("project_root"))
    tables_dir = resolve_project_path("reports/tables/interpretability", config.get("project_root"))
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    attr_abs = np.abs(attributions)
    lead_importance = attr_abs.mean(axis=1)
    lead_idx = int(np.argmax(lead_importance))
    lead_name = LEAD_NAMES[lead_idx]
    sampling_rate = frequency
    time = np.arange(signal_raw.shape[0]) / float(sampling_rate)

    window_samples = max(1, int(window_seconds * sampling_rate))
    lead_attr = attr_abs[lead_idx]
    if window_samples < len(lead_attr):
        rolling = np.convolve(lead_attr, np.ones(window_samples), mode="valid")
        start_idx = int(np.argmax(rolling))
        end_idx = start_idx + window_samples
    else:
        start_idx = 0
        end_idx = len(lead_attr)
    start_s = start_idx / float(sampling_rate)
    end_s = end_idx / float(sampling_rate)

    heatmap = _downsample_attributions(attr_abs, n_bins=250)
    heatmap = heatmap / max(float(np.percentile(heatmap, 99)), 1e-8)
    heatmap = np.clip(heatmap, 0, 1)

    fig, (ax_heat, ax_signal) = plt.subplots(
        2,
        1,
        figsize=(12, 7.4),
        dpi=180,
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.34},
    )
    fig.patch.set_facecolor("white")
    title = "Interpretabilidade da InceptionTime 1D por Integrated Gradients"
    subtitle = (
        f"Classe real: {sample['true_class']} | Classe predita: {sample['pred_class']} "
        f"| Probabilidade: {sample['pred_prob']:.2f}"
    )
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.5, 0.935, subtitle, ha="center", fontsize=10.5, color="#475569")

    im = ax_heat.imshow(
        heatmap,
        aspect="auto",
        cmap="magma",
        extent=[time[0], time[-1], len(LEAD_NAMES) - 0.5, -0.5],
        vmin=0,
        vmax=1,
    )
    ax_heat.set_yticks(np.arange(len(LEAD_NAMES)))
    ax_heat.set_yticklabels(LEAD_NAMES)
    ax_heat.set_xlabel("Tempo (s)")
    ax_heat.set_ylabel("Deriva\u00e7\u00e3o")
    ax_heat.set_title("Mapa de import\u00e2ncia atribu\u00edda por deriva\u00e7\u00e3o e tempo", fontsize=11.5)
    ax_heat.axvspan(start_s, end_s, color="#38bdf8", alpha=0.18, label="janela destacada")
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.015)
    cbar.set_label("Import\u00e2ncia atribu\u00edda", rotation=90)

    signal = signal_raw[:, lead_idx]
    ax_signal.plot(time, signal, color="#172033", linewidth=1.0, label=f"Sinal de ECG - {lead_name}")
    ax_signal.axvspan(start_s, end_s, color="#f97316", alpha=0.24, label="trecho mais influente")
    ax_signal.set_xlabel("Tempo (s)")
    ax_signal.set_ylabel("Sinal de ECG")
    ax_signal.set_title(f"Deriva\u00e7\u00e3o mais relevante: {lead_name}", fontsize=11.5)
    ax_signal.legend(loc="upper left", frameon=False)
    ax_signal.grid(True, color="#e2e8f0", linewidth=0.8)

    note = (
        "Regi\u00f5es destacadas indicam influ\u00eancia na predi\u00e7\u00e3o do modelo; "
        "n\u00e3o representam causalidade ou valida\u00e7\u00e3o cl\u00ednica."
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=9.2, color="#64748b")

    png_path = figures_dir / "inceptiontime_integrated_gradients_example.png"
    pdf_path = figures_dir / "inceptiontime_integrated_gradients_example.pdf"
    fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = {
        "ecg_id": sample["ecg_id"],
        "classe_real": sample["true_class"],
        "classe_predita": sample["pred_class"],
        "predicao_correta": sample["is_correct"],
        "probabilidade_classe_predita": sample["pred_prob"],
        "derivacao_mais_relevante": lead_name,
        "janela_inicio_s": start_s,
        "janela_fim_s": end_s,
        "metodo": method,
        "n_steps": n_steps,
        "frequencia_hz": frequency,
        "observacao": "Atribui\u00e7\u00f5es indicam influ\u00eancia no modelo, n\u00e3o causalidade m\u00e9dica.",
    }
    pd.DataFrame([summary]).to_csv(
        tables_dir / "inceptiontime_ig_example_summary.csv",
        index=False,
        encoding="utf-8",
    )
    return {"png": png_path, "pdf": pdf_path}, summary


def _downsample_attributions(attr_abs: np.ndarray, n_bins: int) -> np.ndarray:
    n_channels, n_steps = attr_abs.shape
    edges = np.linspace(0, n_steps, n_bins + 1, dtype=int)
    output = np.zeros((n_channels, n_bins), dtype="float32")
    for idx in range(n_bins):
        start, end = edges[idx], max(edges[idx + 1], edges[idx] + 1)
        output[:, idx] = attr_abs[:, start:end].mean(axis=1)
    return output


def _write_explanatory_text(config: dict[str, Any]) -> None:
    text_dir = resolve_project_path("reports/interpretability", config.get("project_root"))
    ensure_dir(text_dir)
    text = (
        "Integrated Gradients foi utilizado para analisar uma predi\u00e7\u00e3o da InceptionTime 1D "
        "treinada sobre sinais brutos de ECG. A figura destaca regi\u00f5es temporais e deriva\u00e7\u00f5es "
        "que mais contribu\u00edram para a classe predita pelo modelo. Essa interpreta\u00e7\u00e3o descreve "
        "o comportamento da rede para a amostra analisada e n\u00e3o constitui causalidade m\u00e9dica "
        "nem valida\u00e7\u00e3o cl\u00ednica da predi\u00e7\u00e3o.\n"
    )
    (text_dir / "inceptiontime_integrated_gradients_text.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
