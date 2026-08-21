"""Treino PyTorch pesado, controlado e reprodutivel para ECG bruto."""

from __future__ import annotations

import json
import platform
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from tcc_ecg.dl_data import (
    ECGMemmapAugmentedDataset,
    compute_class_weights,
    create_weighted_sampler,
    generate_balance_and_split_artifacts,
    prepare_raw_signal_splits,
    save_normalization,
)
from tcc_ecg.dl_models import build_heavy_model, count_parameters
from tcc_ecg.evaluation import classification_report_frame, compute_classification_metrics
from tcc_ecg.paths import ensure_dir, resolve_project_path
from tcc_ecg.plots import plot_confusion_matrix, save_figure
from tcc_ecg.utils import save_table


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depende do ambiente local
        raise ImportError("PyTorch nao esta instalado. Rode: python -m pip install -e .[dl]") from exc
    return torch


def get_torch_environment() -> dict[str, Any]:
    """Retorna informacoes resumidas do ambiente PyTorch/hardware."""
    torch = _import_torch()
    cuda_available = bool(torch.cuda.is_available())
    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "device": torch.cuda.get_device_name(0) if cuda_available else "CPU",
        "gpu_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory) if cuda_available else 0,
    }
    return result


def set_torch_seed(seed: int) -> None:
    """Define sementes e reduz fontes de nao determinismo em PyTorch."""
    torch = _import_torch()
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_deep_learning_heavy_config(config: dict[str, Any]) -> dict[str, Any]:
    """Carrega configuracao heavy com defaults e resolve opcoes auto."""
    defaults = {
        "enabled": True,
        "frequency": 500,
        "model_family": "inceptiontime_resnet_ensemble",
        "architectures": ["inceptiontime_deep", "resnet1d_se"],
        "max_runs_cpu": 1,
        "max_runs_cuda": 2,
        "epochs": 120,
        "patience": 20,
        "batch_size": "auto",
        "learning_rate": 0.001,
        "min_learning_rate": 0.000001,
        "weight_decay": 0.0001,
        "optimizer": "adamw",
        "scheduler": "cosine_or_plateau",
        "loss_type": "cross_entropy_weighted",
        "use_focal_loss": True,
        "focal_gamma": 2.0,
        "use_class_weight": True,
        "use_weighted_sampler": True,
        "use_augmentation": True,
        "use_amp": True,
        "gradient_clip_norm": 1.0,
        "num_workers": "auto",
        "seeds": [42, 123, 2025],
        "cache_dtype": "float16",
        "force_rebuild_cache": False,
        "max_records": None,
        "monitor": "val_f1_macro",
        "augmentations": {
            "noise_std": 0.01,
            "scale_min": 0.9,
            "scale_max": 1.1,
            "time_shift": 60,
            "channel_dropout_prob": 0.04,
        },
        "inceptiontime_deep": {
            "base_channels": 96,
            "block_channels": [96, 96, 128, 128, 192, 192, 256, 256],
            "kernel_sizes": [9, 19, 39],
            "bottleneck_channels": 48,
            "dropout": 0.3,
        },
        "resnet1d_se": {
            "base_filters": 64,
            "stage_channels": [64, 64, 128, 128, 256, 256, 512, 512],
            "kernel_size": 9,
            "dropout": 0.3,
            "se_reduction": 8,
        },
    }
    user = config.get("deep_learning_heavy", {})
    merged = {**defaults, **user}
    for nested in ["augmentations", "inceptiontime_deep", "resnet1d_se"]:
        merged[nested] = {**defaults[nested], **user.get(nested, {})}

    torch = _import_torch()
    cuda = bool(torch.cuda.is_available())
    if merged["batch_size"] == "auto":
        merged["batch_size"] = 96 if cuda else 64
    if merged["num_workers"] == "auto":
        merged["num_workers"] = 2 if cuda else 0
    merged["max_runs"] = int(merged["max_runs_cuda"] if cuda else merged["max_runs_cpu"])
    return merged


class FocalLoss:
    """Focal Loss multiclasse com alpha opcional por classe."""

    def __init__(self, alpha=None, gamma: float = 2.0) -> None:
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, logits, target):
        torch = _import_torch()
        ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha[target] * loss
        return loss.mean()


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    f1_macro: float
    recall_macro: float


@dataclass
class Candidate:
    run_name: str
    architecture: str
    seed: int
    checkpoint_path: Path
    best_epoch: int
    val_f1_macro: float
    n_parameters: int
    history: list[dict[str, float]]


def train_deep_learning_heavy(metadata: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Treina arquitetura(s) pesada(s) e avalia teste apenas apos selecao por validacao."""
    torch = _import_torch()

    heavy_config = get_deep_learning_heavy_config(config)
    if not bool(heavy_config["enabled"]):
        raise ValueError("deep_learning_heavy.enabled esta falso.")

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frequency = int(heavy_config["frequency"])
    config["data"]["signal_frequency"] = frequency

    split_data = prepare_raw_signal_splits(
        metadata,
        config,
        frequency=frequency,
        batch_size=int(heavy_config["batch_size"]),
        cache_cfg=heavy_config,
    )
    save_normalization(split_data["mean"], split_data["std"], config, stem="deep_learning_heavy")
    generate_balance_and_split_artifacts(metadata, config)

    labels = split_data["labels"]
    train_idx = split_data["train_idx"]
    val_idx = split_data["val_idx"]
    test_idx = split_data["test_idx"]
    n_classes = len(config["labels"]["superclasses"])
    class_weights_np = compute_class_weights(labels[train_idx], n_classes)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    candidates: list[Candidate] = []
    histories: list[pd.DataFrame] = []
    _reset_progress_log(config)
    run_specs = _run_specs(heavy_config)
    for architecture, seed in run_specs:
        candidate, history_df = _train_single_candidate(
            architecture=architecture,
            seed=seed,
            split_data=split_data,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            class_weights=class_weights,
            heavy_config=heavy_config,
            config=config,
            device=device,
        )
        candidates.append(candidate)
        histories.append(history_df)

    chosen, is_ensemble = _select_candidate_or_ensemble(candidates, split_data, labels, val_idx, heavy_config, config, device)
    if is_ensemble:
        y_true, y_pred = _predict_ensemble(candidates, split_data, labels, test_idx, heavy_config, config, device)
        model_name = "deep_learning_heavy_ensemble"
        best_epoch = int(max(item.best_epoch for item in candidates))
        n_parameters = int(sum(item.n_parameters for item in candidates))
        _save_ensemble_metadata(candidates, heavy_config, config)
    else:
        y_true, y_pred = _predict_single(chosen, split_data, labels, test_idx, heavy_config, config, device)
        model_name = f"deep_learning_heavy_{chosen.architecture}"
        best_epoch = int(chosen.best_epoch)
        n_parameters = int(chosen.n_parameters)
        _copy_best_checkpoint(chosen.checkpoint_path, config)

    metrics = compute_classification_metrics(y_true, y_pred, model_name=model_name, split="test")
    metrics.update(
        {
            "signal_frequency": frequency,
            "smote": False,
            "device": str(device),
            "best_epoch": best_epoch,
            "epochs_trained": int(max(len(frame) for frame in histories)) if histories else 0,
            "training_seconds": float(time.time() - start_time),
            "n_parameters": n_parameters,
            "architecture": "ensemble" if is_ensemble else chosen.architecture,
            "loss_type": "focal" if bool(heavy_config["use_focal_loss"]) else str(heavy_config["loss_type"]),
        }
    )
    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    _save_heavy_outputs(history, metrics, y_true, y_pred, heavy_config, config)
    _update_text_logs(metrics, heavy_config, config)
    return {
        "metrics": metrics,
        "history": history,
        "device": str(device),
        "selected": "ensemble" if is_ensemble else chosen.run_name,
    }


def _run_specs(heavy_config: dict[str, Any]) -> list[tuple[str, int]]:
    specs = [(architecture, int(seed)) for architecture in heavy_config["architectures"] for seed in heavy_config["seeds"]]
    return specs[: int(heavy_config["max_runs"])]


def _make_loaders(split_data, labels, train_idx, val_idx, heavy_config, seed: int, config, use_sampler: bool):
    torch = _import_torch()
    from torch.utils.data import DataLoader

    augmentations = heavy_config["augmentations"] if bool(heavy_config["use_augmentation"]) else {}
    train_dataset = ECGMemmapAugmentedDataset(
        split_data["cache_paths"]["signals"],
        labels,
        train_idx,
        split_data["mean"],
        split_data["std"],
        train=True,
        augmentations=augmentations,
    )
    val_dataset = ECGMemmapAugmentedDataset(
        split_data["cache_paths"]["signals"],
        labels,
        val_idx,
        split_data["mean"],
        split_data["std"],
        train=False,
    )
    sampler = None
    shuffle = True
    if use_sampler:
        sampler = create_weighted_sampler(labels, train_idx, len(config["labels"]["superclasses"]), seed=seed)
        shuffle = False
    loader_kwargs = {
        "batch_size": int(heavy_config["batch_size"]),
        "num_workers": int(heavy_config["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, shuffle=shuffle, sampler=sampler, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def _make_eval_loader(split_data, labels, indices, heavy_config):
    torch = _import_torch()
    from torch.utils.data import DataLoader

    dataset = ECGMemmapAugmentedDataset(
        split_data["cache_paths"]["signals"],
        labels,
        indices,
        split_data["mean"],
        split_data["std"],
        train=False,
    )
    return DataLoader(
        dataset,
        batch_size=int(heavy_config["batch_size"]),
        shuffle=False,
        num_workers=int(heavy_config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )


def _train_single_candidate(
    architecture: str,
    seed: int,
    split_data,
    labels,
    train_idx,
    val_idx,
    class_weights,
    heavy_config,
    config,
    device,
) -> tuple[Candidate, pd.DataFrame]:
    torch = _import_torch()
    set_torch_seed(seed)
    train_loader, val_loader = _make_loaders(
        split_data,
        labels,
        train_idx,
        val_idx,
        heavy_config,
        seed,
        config,
        use_sampler=bool(heavy_config["use_weighted_sampler"]),
    )
    model = build_heavy_model(
        architecture,
        heavy_config,
        input_channels=12,
        n_classes=len(config["labels"]["superclasses"]),
    ).to(device)
    n_parameters = count_parameters(model)
    criterion = _make_criterion(heavy_config, class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(heavy_config["learning_rate"]),
        weight_decay=float(heavy_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=float(heavy_config["min_learning_rate"]),
    )
    use_amp = bool(heavy_config["use_amp"]) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    run_name = f"{architecture}_seed{seed}"
    checkpoint_path = _candidate_checkpoint_path(config, run_name)
    history, best_epoch, best_val_f1 = _fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        checkpoint_path=checkpoint_path,
        config=config,
        heavy_config=heavy_config,
        architecture=architecture,
        seed=seed,
        n_parameters=n_parameters,
        use_amp=use_amp,
    )
    history_df = pd.DataFrame(history)
    history_df["run_name"] = run_name
    history_df["architecture"] = architecture
    history_df["seed"] = seed
    return (
        Candidate(
            run_name=run_name,
            architecture=architecture,
            seed=seed,
            checkpoint_path=checkpoint_path,
            best_epoch=best_epoch,
            val_f1_macro=best_val_f1,
            n_parameters=n_parameters,
            history=history,
        ),
        history_df,
    )


def _make_criterion(heavy_config, class_weights):
    torch = _import_torch()
    weight = class_weights if bool(heavy_config["use_class_weight"]) else None
    if bool(heavy_config["use_focal_loss"]):
        return FocalLoss(alpha=weight, gamma=float(heavy_config["focal_gamma"]))
    if str(heavy_config["loss_type"]) == "cross_entropy_weighted":
        return torch.nn.CrossEntropyLoss(weight=weight)
    return torch.nn.CrossEntropyLoss()


def _fit_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    checkpoint_path: Path,
    config,
    heavy_config,
    architecture: str,
    seed: int,
    n_parameters: int,
    use_amp: bool,
) -> tuple[list[dict[str, float]], int, float]:
    torch = _import_torch()
    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    patience = int(heavy_config["patience"])
    for epoch in range(1, int(heavy_config["epochs"]) + 1):
        train_result = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            gradient_clip=float(heavy_config["gradient_clip_norm"]),
        )
        val_result = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_result.f1_macro)
        row = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "train_accuracy": train_result.accuracy,
            "train_f1_macro": train_result.f1_macro,
            "train_recall_macro": train_result.recall_macro,
            "val_loss": val_result.loss,
            "val_accuracy": val_result.accuracy,
            "val_f1_macro": val_result.f1_macro,
            "val_recall_macro": val_result.recall_macro,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        _append_progress_row(row, config, architecture, seed)
        score = val_result.f1_macro
        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            ensure_dir(checkpoint_path.parent)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": architecture,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_val_f1_macro": float(best_score),
                    "n_parameters": int(n_parameters),
                    "class_names": config["labels"]["superclasses"],
                    "heavy_config": heavy_config,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    return history, best_epoch, float(best_score)


def _run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    use_amp: bool = False,
    gradient_clip: float | None = None,
) -> EpochResult:
    torch = _import_torch()
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_examples = 0
    y_true = []
    y_pred = []
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
            if is_train:
                scaler.scale(loss).backward()
                if gradient_clip and gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss.item()) * y_batch.size(0)
            total_examples += int(y_batch.size(0))
            y_true.append(y_batch.detach().cpu().numpy())
            y_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
    y_true_arr = np.concatenate(y_true)
    y_pred_arr = np.concatenate(y_pred)
    return EpochResult(
        loss=total_loss / total_examples,
        accuracy=accuracy_score(y_true_arr, y_pred_arr),
        f1_macro=f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
        recall_macro=recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
    )


def _select_candidate_or_ensemble(candidates, split_data, labels, val_idx, heavy_config, config, device):
    if len(candidates) < 2:
        return max(candidates, key=lambda item: item.val_f1_macro), False
    val_probs = []
    y_true = None
    for candidate in candidates:
        y_true, probs = _predict_proba_candidate(candidate, split_data, labels, val_idx, heavy_config, config, device)
        val_probs.append(probs)
    ensemble_pred = np.mean(val_probs, axis=0).argmax(axis=1)
    ensemble_f1 = f1_score(y_true, ensemble_pred, average="macro", zero_division=0)
    best_single = max(candidates, key=lambda item: item.val_f1_macro)
    return (best_single, False) if best_single.val_f1_macro >= ensemble_f1 else (best_single, True)


def _predict_single(candidate, split_data, labels, indices, heavy_config, config, device):
    y_true, probs = _predict_proba_candidate(candidate, split_data, labels, indices, heavy_config, config, device)
    return y_true, probs.argmax(axis=1)


def _predict_ensemble(candidates, split_data, labels, indices, heavy_config, config, device):
    probs_list = []
    y_true = None
    for candidate in candidates:
        y_true, probs = _predict_proba_candidate(candidate, split_data, labels, indices, heavy_config, config, device)
        probs_list.append(probs)
    return y_true, np.mean(probs_list, axis=0).argmax(axis=1)


def _predict_proba_candidate(candidate, split_data, labels, indices, heavy_config, config, device):
    torch = _import_torch()
    checkpoint = torch.load(candidate.checkpoint_path, map_location=device)
    model = build_heavy_model(
        candidate.architecture,
        heavy_config,
        input_channels=12,
        n_classes=len(config["labels"]["superclasses"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = _make_eval_loader(split_data, labels, indices, heavy_config)
    model.eval()
    y_true = []
    probs = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch.to(device, non_blocking=True))
            y_true.append(y_batch.numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(y_true), np.concatenate(probs)


def _candidate_checkpoint_path(config: dict[str, Any], run_name: str) -> Path:
    models_dir = resolve_project_path(config["outputs"]["models_dir"], config.get("project_root"))
    return models_dir / f"deep_learning_heavy_{run_name}_best.pt"


def _progress_path(config: dict[str, Any]) -> Path:
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    return tables_dir / "deep_learning_heavy_progress.csv"


def _reset_progress_log(config: dict[str, Any]) -> None:
    path = _progress_path(config)
    ensure_dir(path.parent)
    if path.exists():
        path.unlink()


def _append_progress_row(row: dict[str, float], config: dict[str, Any], architecture: str, seed: int) -> None:
    path = _progress_path(config)
    progress_row = dict(row)
    progress_row["architecture"] = architecture
    progress_row["seed"] = seed
    pd.DataFrame([progress_row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def _copy_best_checkpoint(source: Path, config: dict[str, Any]) -> None:
    models_dir = resolve_project_path(config["outputs"]["models_dir"], config.get("project_root"))
    ensure_dir(models_dir)
    shutil.copy2(source, models_dir / "deep_learning_heavy_best.pt")


def _save_ensemble_metadata(candidates, heavy_config, config) -> None:
    torch = _import_torch()
    models_dir = resolve_project_path(config["outputs"]["models_dir"], config.get("project_root"))
    ensure_dir(models_dir)
    torch.save(
        {
            "checkpoints": [str(item.checkpoint_path) for item in candidates],
            "architectures": [item.architecture for item in candidates],
            "seeds": [item.seed for item in candidates],
            "heavy_config": heavy_config,
            "selection": "media de probabilidades escolhida por validacao",
        },
        models_dir / "deep_learning_heavy_ensemble.pt",
    )


def _save_heavy_outputs(history, metrics, y_true, y_pred, heavy_config, config) -> None:
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    figures_dir = resolve_project_path(config["outputs"]["figures_dir"], config.get("project_root"))
    models_dir = resolve_project_path(config["outputs"]["models_dir"], config.get("project_root"))
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)
    ensure_dir(models_dir)

    metrics_df = pd.DataFrame([metrics])
    save_table(metrics_df, tables_dir / "deep_learning_heavy_metrics.csv", tables_dir / "deep_learning_heavy_metrics.tex")
    report = classification_report_frame(y_true, y_pred, config["labels"]["superclasses"])
    save_table(
        report,
        tables_dir / "deep_learning_heavy_classification_report.csv",
        tables_dir / "deep_learning_heavy_classification_report.tex",
    )
    save_table(history, tables_dir / "deep_learning_heavy_history.csv")
    _plot_history(history, figures_dir / "fig_deep_learning_heavy_training_curves.png")
    plot_confusion_matrix(
        y_true,
        y_pred,
        config["labels"]["superclasses"],
        figures_dir / "fig_deep_learning_heavy_confusion_matrix.png",
        title="Matriz de confusao - deep learning heavy",
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        config["labels"]["superclasses"],
        figures_dir / "fig_deep_learning_heavy_confusion_matrix_normalized.png",
        normalize="true",
        title="Matriz de confusao normalizada - deep learning heavy",
    )
    _write_metadata(metrics, heavy_config, config)
    _update_final_and_dl_comparisons(metrics_df, config)


def _plot_history(history: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for run_name, group in history.groupby("run_name", sort=False):
        ax.plot(group["epoch"], group["train_loss"], label=f"{run_name} treino loss", alpha=0.75)
        ax.plot(group["epoch"], group["val_loss"], label=f"{run_name} val loss", alpha=0.75)
        ax.plot(group["epoch"], group["val_f1_macro"], label=f"{run_name} val F1 macro", linewidth=2)
    ax.set_title("Curvas de treino e validacao - deep learning heavy")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Valor")
    ax.legend(fontsize=8)
    save_figure(fig, output_path)


def _write_metadata(metrics, heavy_config, config) -> None:
    models_dir = resolve_project_path(config["outputs"]["models_dir"], config.get("project_root"))
    metadata = {
        "checkpoint": "deep_learning_heavy_best.pt",
        "framework": "pytorch",
        "metrics": metrics,
        "config": heavy_config,
        "methodology": "folds 1-8 treino, fold 9 validacao, fold 10 teste; teste usado apenas apos selecao por validacao.",
    }
    (models_dir / "deep_learning_heavy_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _update_final_and_dl_comparisons(metrics_df: pd.DataFrame, config: dict[str, Any]) -> None:
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    figures_dir = resolve_project_path(config["outputs"]["figures_dir"], config.get("project_root"))
    final_frames = []
    for path in [tables_dir / "final_model_comparison.csv", tables_dir / "deep_learning_heavy_metrics.csv"]:
        if path.exists():
            final_frames.append(pd.read_csv(path))
    if not final_frames:
        final_frames = [metrics_df]
    final = (
        pd.concat(final_frames, ignore_index=True)
        .loc[lambda df: df["split"].eq("test")]
        .drop_duplicates(subset=["model", "signal_frequency"], keep="last")
        .sort_values("f1_macro", ascending=False)
    )
    save_table(final, tables_dir / "final_model_comparison.csv", tables_dir / "final_model_comparison.tex")

    dl_names = {"deep_learning_baseline", "resnet1d_light", "inceptiontime1d_strong"}
    dl_comparison = final.loc[final["model"].isin(dl_names) | final["model"].str.startswith("deep_learning_heavy")].copy()
    dl_comparison = _merge_runtime_log(dl_comparison, tables_dir)
    columns = [
        "model",
        "signal_frequency",
        "device",
        "training_seconds",
        "n_parameters",
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
    ]
    for col in columns:
        if col not in dl_comparison.columns:
            dl_comparison[col] = np.nan
    dl_comparison = dl_comparison[columns].sort_values("f1_macro", ascending=False)
    save_table(
        dl_comparison,
        tables_dir / "deep_learning_comparison.csv",
        tables_dir / "deep_learning_comparison.tex",
    )
    _plot_deep_learning_comparison(dl_comparison, figures_dir / "fig_deep_learning_comparison.png")


def _merge_runtime_log(metrics: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    runtime_path = tables_dir / "runtime_log.csv"
    if not runtime_path.exists():
        return metrics
    runtime = pd.read_csv(runtime_path)
    step_to_model = {
        "05_deep_learning_baseline": "deep_learning_baseline",
        "05b_deep_learning_resnet1d": "resnet1d_light",
        "05c_deep_learning_strong": "inceptiontime1d_strong",
        "05d_deep_learning_heavy": None,
    }
    rows = []
    for _, row in runtime.iterrows():
        model = step_to_model.get(row.get("step"))
        if model:
            rows.append({"model": model, "runtime_seconds_from_log": row.get("seconds")})
    if not rows:
        return metrics
    runtime_df = pd.DataFrame(rows)
    merged = metrics.merge(runtime_df, on="model", how="left")
    missing = merged["training_seconds"].isna() if "training_seconds" in merged.columns else pd.Series(True, index=merged.index)
    merged.loc[missing, "training_seconds"] = merged.loc[missing, "runtime_seconds_from_log"]
    return merged.drop(columns=[col for col in ["runtime_seconds_from_log"] if col in merged.columns])


def _plot_deep_learning_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    plot_df = metrics[["model", "accuracy", "balanced_accuracy", "f1_macro", "recall_macro"]].copy()
    melted = plot_df.melt(id_vars="model", var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(9, 5))
    import seaborn as sns

    sns.barplot(data=melted, x="model", y="value", hue="metric", ax=ax)
    ax.set_title("Comparacao entre modelos de deep learning")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Valor")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Metrica", loc="lower right")
    save_figure(fig, output_path)


def _update_text_logs(metrics, heavy_config, config) -> None:
    _append_runtime_log(metrics, config)
    _append_experiment_log(metrics, heavy_config)
    _append_result_text(metrics)
    _append_development_text(metrics, heavy_config)


def _append_runtime_log(metrics, config) -> None:
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    path = tables_dir / "runtime_log.csv"
    rows = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["signal_frequency", "step", "seconds", "notes"])
    rows = rows.loc[rows["step"].ne("05d_deep_learning_heavy")] if not rows.empty else rows
    rows = pd.concat(
        [
            rows,
            pd.DataFrame(
                [
                    {
                        "signal_frequency": int(metrics["signal_frequency"]),
                        "step": "05d_deep_learning_heavy",
                        "seconds": float(metrics["training_seconds"]),
                        "notes": f"Measured during heavy PyTorch training on {metrics['device']}.",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    save_table(rows, path)


def _append_experiment_log(metrics, heavy_config) -> None:
    path = resolve_project_path("reports/experiment_log.md")
    ensure_dir(path.parent)
    text = path.read_text(encoding="utf-8") if path.exists() else "# Experiment Log\n\n"
    text += (
        "\n## Deep Learning Heavy\n\n"
        f"- Model: {metrics['model']}\n"
        f"- Frequency: {metrics['signal_frequency']} Hz\n"
        f"- Device: {metrics['device']}\n"
        f"- Architecture: {metrics['architecture']}\n"
        f"- Loss: {metrics['loss_type']}\n"
        f"- Weighted sampler: {heavy_config['use_weighted_sampler']}\n"
        f"- Accuracy: {metrics['accuracy']:.4f}\n"
        f"- F1 macro: {metrics['f1_macro']:.4f}\n"
        f"- Recall macro: {metrics['recall_macro']:.4f}\n"
        f"- Best epoch: {metrics['best_epoch']}\n"
        f"- Training seconds: {metrics['training_seconds']:.2f}\n"
        "- Notes: test fold evaluated only after validation-based checkpoint/ensemble selection.\n"
    )
    path.write_text(text, encoding="utf-8")


def _append_result_text(metrics) -> None:
    path = resolve_project_path("reports/resultados_resumo.md")
    ensure_dir(path.parent)
    text = path.read_text(encoding="utf-8") if path.exists() else "# Resultados - Texto-base\n\n"
    status = "atingiu" if float(metrics["accuracy"]) >= 0.80 else "nao atingiu"
    text += (
        "\n## Deep learning pesado e desbalanceamento\n\n"
        "Foi adicionado um experimento de deep learning mais robusto em PyTorch usando records500 e sinais brutos. "
        "As estrategias de desbalanceamento foram aplicadas somente no treino, incluindo pesos por classe, focal loss, "
        "WeightedRandomSampler e aumentacoes leves. Validacao e teste foram preservados sem balanceamento artificial.\n\n"
        f"No conjunto de teste, o modelo {metrics['model']} obteve acuracia de {metrics['accuracy']:.4f}, "
        f"F1 macro de {metrics['f1_macro']:.4f}, balanced accuracy de {metrics['balanced_accuracy']:.4f} "
        f"e recall macro de {metrics['recall_macro']:.4f}. O modelo {status} 80% de acuracia. "
        "Esses resultados devem ser interpretados como avaliacao experimental, sem validacao clinica definitiva.\n"
    )
    path.write_text(text, encoding="utf-8")


def _append_development_text(metrics, heavy_config) -> None:
    path = resolve_project_path("reports/desenvolvimento_resumo.md")
    ensure_dir(path.parent)
    text = path.read_text(encoding="utf-8") if path.exists() else "# Desenvolvimento - Texto-base\n\n"
    text += (
        "\n## Modelo deep learning pesado\n\n"
        "O pipeline passou a incluir um experimento PyTorch mais robusto para records500, com selecao por validacao "
        "e avaliacao final unica no teste. Em ambientes com GPU, a configuracao permite executar mais de uma arquitetura "
        "e avaliar ensemble por media de probabilidades; em CPU, a execucao e limitada automaticamente para manter custo "
        "computacional controlado. O treinamento usa AdamW, scheduler por validacao, early stopping, checkpoint do melhor "
        "modelo, normalizacao calculada apenas no treino e estrategias de desbalanceamento aplicadas somente no treino.\n"
    )
    path.write_text(text, encoding="utf-8")
