"""Treino de deep learning mais robusto para sinais brutos PTB-XL."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tcc_ecg.deep_learning import (
    _compute_class_weights,
    build_resnet1d_cache,
    compute_train_channel_normalization,
    set_torch_seed,
)
from tcc_ecg.evaluation import classification_report_frame, compute_classification_metrics
from tcc_ecg.models import split_by_folds
from tcc_ecg.paths import ensure_dir, resolve_project_path
from tcc_ecg.plots import plot_confusion_matrix, plot_metrics_comparison, save_figure
from tcc_ecg.utils import save_table


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depende do ambiente local
        raise ImportError("PyTorch nao esta instalado. Rode: python -m pip install -e .[dl]") from exc
    return torch


def get_deep_learning_strong_config(config: dict[str, Any]) -> dict[str, Any]:
    """Configura InceptionTime 1D com defaults fortes, mas controlados."""
    defaults = {
        "model_name": "inceptiontime1d_strong",
        "frequency": 500,
        "batch_size": 64,
        "epochs": 80,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "patience": 18,
        "num_workers": 0,
        "use_amp": True,
        "class_weight": True,
        "loss_type": "focal",
        "focal_gamma": 1.5,
        "scheduler": "reduce_on_plateau",
        "monitor": "val_f1_macro",
        "seed": int(config.get("project", {}).get("seed", 42)),
        "base_channels": 64,
        "block_channels": [64, 64, 128, 128, 256, 256],
        "kernel_sizes": [9, 19, 39],
        "bottleneck_channels": 32,
        "dropout": 0.25,
        "gradient_clip": 1.0,
        "weighted_sampler": False,
        "max_records": None,
        "force_rebuild_cache": False,
        "augmentations": {
            "enabled": True,
            "noise_std": 0.01,
            "scale_min": 0.9,
            "scale_max": 1.1,
            "time_shift": 40,
            "channel_dropout_prob": 0.03,
        },
    }
    user_config = config.get("deep_learning_strong", {})
    merged = {**defaults, **user_config}
    merged["augmentations"] = {**defaults["augmentations"], **user_config.get("augmentations", {})}
    return merged


class ECGStrongDataset:
    """Dataset com normalizacao de treino e augmentations leves apenas no treino."""

    def __init__(
        self,
        signals_path: str | Path,
        labels: np.ndarray,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        train: bool,
        augmentations: dict[str, Any],
    ) -> None:
        self.signals = np.load(signals_path, mmap_mode="r")
        self.labels = labels.astype("int64")
        self.indices = indices.astype("int64")
        self.mean = mean.astype("float32")
        self.std = std.astype("float32")
        self.train = train
        self.augmentations = augmentations

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        torch = _import_torch()
        idx = int(self.indices[item])
        x = self.signals[idx].astype("float32")
        x = (x - self.mean[None, :]) / self.std[None, :]
        if self.train and bool(self.augmentations.get("enabled", True)):
            x = apply_ecg_augmentations(x, self.augmentations)
        x_tensor = torch.from_numpy(np.ascontiguousarray(x.T))
        y_tensor = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x_tensor, y_tensor


def apply_ecg_augmentations(signal: np.ndarray, augmentations: dict[str, Any]) -> np.ndarray:
    """Aumentacoes leves para ECG/séries temporais, aplicadas apenas no treino."""
    x = signal.copy()
    noise_std = float(augmentations.get("noise_std", 0.0))
    if noise_std > 0:
        x += np.random.normal(0.0, noise_std, size=x.shape).astype("float32")

    scale_min = float(augmentations.get("scale_min", 1.0))
    scale_max = float(augmentations.get("scale_max", 1.0))
    if scale_min != 1.0 or scale_max != 1.0:
        x *= np.random.uniform(scale_min, scale_max, size=(1, x.shape[1])).astype("float32")

    max_shift = int(augmentations.get("time_shift", 0))
    if max_shift > 0:
        shift = int(np.random.randint(-max_shift, max_shift + 1))
        x = np.roll(x, shift=shift, axis=0)

    channel_dropout_prob = float(augmentations.get("channel_dropout_prob", 0.0))
    if channel_dropout_prob > 0:
        mask = np.random.random(x.shape[1]) < channel_dropout_prob
        x[:, mask] = 0.0
    return x.astype("float32")


class FocalLoss:
    """Focal Loss multiclasse com alpha opcional por classe."""

    def __init__(self, alpha=None, gamma: float = 1.5) -> None:
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, logits, target):
        torch = _import_torch()
        ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha[target] * loss
        return loss.mean()


def build_inceptiontime1d_model(
    input_channels: int,
    n_classes: int,
    base_channels: int,
    block_channels: list[int],
    kernel_sizes: list[int],
    bottleneck_channels: int,
    dropout: float,
):
    """InceptionTime 1D com kernels multi-escala e conexoes residuais."""
    torch = _import_torch()
    nn = torch.nn

    class InceptionBlock1D(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
            super().__init__()
            branch_channels = max(out_channels // (len(kernel_sizes) + 1), 8)
            bottleneck = min(bottleneck_channels, in_channels)
            self.reduce = (
                nn.Conv1d(in_channels, bottleneck, kernel_size=1, bias=False)
                if in_channels > 1
                else nn.Identity()
            )
            reduced_channels = bottleneck if in_channels > 1 else in_channels
            self.branches = nn.ModuleList(
                [
                    nn.Conv1d(
                        reduced_channels,
                        branch_channels,
                        kernel_size=k,
                        stride=stride,
                        padding=k // 2,
                        bias=False,
                    )
                    for k in kernel_sizes
                ]
            )
            self.pool_branch = nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=stride, padding=1),
                nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
            )
            concat_channels = branch_channels * (len(kernel_sizes) + 1)
            self.bn = nn.BatchNorm1d(concat_channels)
            self.act = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.out_proj = (
                nn.Conv1d(concat_channels, out_channels, kernel_size=1, bias=False)
                if concat_channels != out_channels
                else nn.Identity()
            )
            self.shortcut = (
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels),
                )
                if stride != 1 or in_channels != out_channels
                else nn.Identity()
            )

        def forward(self, x):
            reduced = self.reduce(x)
            branches = [branch(reduced) for branch in self.branches]
            branches.append(self.pool_branch(x))
            out = torch.cat(branches, dim=1)
            out = self.dropout(self.act(self.bn(out)))
            out = self.out_proj(out)
            return self.act(out + self.shortcut(x))

    class InceptionTime1D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(input_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(base_channels),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            )
            blocks = []
            in_channels = base_channels
            for idx, channels in enumerate(block_channels):
                stride = 2 if idx in {2, 4} else 1
                blocks.append(InceptionBlock1D(in_channels, int(channels), stride=stride))
                in_channels = int(channels)
            self.blocks = nn.Sequential(*blocks)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(in_channels, n_classes),
            )

        def forward(self, x):
            return self.head(self.blocks(self.stem(x)))

    return InceptionTime1D()


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    f1_macro: float
    recall_macro: float


def train_deep_learning_strong(metadata: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Treina InceptionTime 1D forte e avalia teste apenas ao final."""
    torch = _import_torch()
    from torch.utils.data import DataLoader, WeightedRandomSampler

    strong_config = get_deep_learning_strong_config(config)
    config["data"]["signal_frequency"] = int(strong_config["frequency"])
    seed = int(strong_config["seed"])
    set_torch_seed(seed)
    start_time = time.time()

    cache_paths = build_resnet1d_cache(metadata, _config_for_cache(config, strong_config))
    labels = np.load(cache_paths["labels"])
    cache_metadata = pd.read_csv(cache_paths["metadata"]).assign(row_idx=np.arange(len(labels)))
    splits = split_by_folds(cache_metadata, config)

    train_idx = splits["train"]["row_idx"].to_numpy(dtype="int64")
    val_idx = splits["validation"]["row_idx"].to_numpy(dtype="int64")
    test_idx = splits["test"]["row_idx"].to_numpy(dtype="int64")
    mean, std = compute_train_channel_normalization(
        cache_paths["signals"], train_idx, batch_size=int(strong_config["batch_size"])
    )
    _save_strong_normalization(mean, std, config)

    train_dataset = ECGStrongDataset(
        cache_paths["signals"],
        labels,
        train_idx,
        mean,
        std,
        train=True,
        augmentations=strong_config["augmentations"],
    )
    val_dataset = ECGStrongDataset(
        cache_paths["signals"], labels, val_idx, mean, std, train=False, augmentations={}
    )
    test_dataset = ECGStrongDataset(
        cache_paths["signals"], labels, test_idx, mean, std, train=False, augmentations={}
    )

    generator = torch.Generator().manual_seed(seed)
    sampler = None
    shuffle = True
    if bool(strong_config["weighted_sampler"]):
        sample_weights = _sample_weights(labels[train_idx], len(config["labels"]["superclasses"]))
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_idx), replacement=True, generator=generator)
        shuffle = False

    loader_kwargs = {
        "batch_size": int(strong_config["batch_size"]),
        "num_workers": int(strong_config["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=shuffle, sampler=sampler, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_inceptiontime1d_model(
        input_channels=12,
        n_classes=len(config["labels"]["superclasses"]),
        base_channels=int(strong_config["base_channels"]),
        block_channels=[int(item) for item in strong_config["block_channels"]],
        kernel_sizes=[int(item) for item in strong_config["kernel_sizes"]],
        bottleneck_channels=int(strong_config["bottleneck_channels"]),
        dropout=float(strong_config["dropout"]),
    ).to(device)

    class_weights = None
    if bool(strong_config["class_weight"]):
        class_weights = torch.tensor(
            _compute_class_weights(labels[train_idx], len(config["labels"]["superclasses"])),
            dtype=torch.float32,
            device=device,
        )
    if strong_config["loss_type"] == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=float(strong_config["focal_gamma"]))
    elif strong_config["loss_type"] == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError("loss_type deve ser 'focal' ou 'cross_entropy'.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(strong_config["learning_rate"]),
        weight_decay=float(strong_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )
    use_amp = bool(strong_config["use_amp"]) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint_path = _strong_checkpoint_path(config)

    history, best_epoch = _fit_strong_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        scaler,
        device,
        checkpoint_path,
        config,
        strong_config,
        use_amp=use_amp,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_true, y_pred = _predict(model, test_loader, device)
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        model_name=str(strong_config["model_name"]),
        split="test",
    )
    metrics["signal_frequency"] = int(strong_config["frequency"])
    metrics["smote"] = False
    metrics["device"] = str(device)
    metrics["best_epoch"] = int(best_epoch)
    metrics["epochs_trained"] = int(len(history))
    metrics["training_seconds"] = float(time.time() - start_time)
    _save_strong_outputs(history, metrics, y_true, y_pred, strong_config, config)
    return {
        "model": model,
        "history": history,
        "metrics": metrics,
        "checkpoint": checkpoint_path,
        "device": str(device),
    }


def _config_for_cache(config: dict[str, Any], strong_config: dict[str, Any]) -> dict[str, Any]:
    cache_config = dict(config)
    cache_config["data"] = dict(config["data"])
    cache_config["data"]["signal_frequency"] = int(strong_config["frequency"])
    cache_config["deep_learning"] = dict(config.get("deep_learning", {}))
    cache_config["deep_learning"]["resnet1d"] = {
        "cache_dtype": "float16",
        "force_rebuild_cache": bool(strong_config.get("force_rebuild_cache", False)),
        "max_records": strong_config.get("max_records"),
    }
    return cache_config


def _sample_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=n_classes).astype("float64")
    class_weights = counts.sum() / np.maximum(counts, 1) / n_classes
    return class_weights[y.astype(int)].astype("float64")


def _fit_strong_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    checkpoint_path: Path,
    config: dict[str, Any],
    strong_config: dict[str, Any],
    use_amp: bool,
) -> tuple[list[dict[str, float]], int]:
    torch = _import_torch()
    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    patience = int(strong_config["patience"])

    for epoch in range(1, int(strong_config["epochs"]) + 1):
        train_result = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            gradient_clip=float(strong_config["gradient_clip"]),
        )
        val_result = _run_epoch(model, val_loader, criterion, device, use_amp=False)
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

        score = val_result.f1_macro
        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            ensure_dir(checkpoint_path.parent)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_f1_macro": best_score,
                    "best_epoch": best_epoch,
                    "class_names": config["labels"]["superclasses"],
                    "strong_config": strong_config,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    return history, best_epoch


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
    from sklearn.metrics import accuracy_score, f1_score, recall_score

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


def _predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    torch = _import_torch()
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch.to(device, non_blocking=True))
            y_true.append(y_batch.numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def _strong_checkpoint_path(config: dict[str, Any]) -> Path:
    models_dir = resolve_project_path(config["outputs"]["models_dir"], config.get("project_root"))
    return models_dir / "deep_learning_strong_best.pt"


def _save_strong_normalization(mean: np.ndarray, std: np.ndarray, config: dict[str, Any]) -> None:
    processed_dir = resolve_project_path(config["outputs"]["processed_dir"], config.get("project_root"))
    ensure_dir(processed_dir)
    np.savez(processed_dir / "deep_learning_strong_normalization_500hz.npz", mean=mean, std=std)


def _save_strong_outputs(
    history: list[dict[str, float]],
    metrics: dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strong_config: dict[str, Any],
    config: dict[str, Any],
) -> None:
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    figures_dir = resolve_project_path(config["outputs"]["figures_dir"], config.get("project_root"))
    models_dir = resolve_project_path(config["outputs"]["models_dir"], config.get("project_root"))
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)
    ensure_dir(models_dir)

    metrics_df = pd.DataFrame([metrics])
    save_table(metrics_df, tables_dir / "deep_learning_strong_metrics.csv", tables_dir / "deep_learning_strong_metrics.tex")
    report = classification_report_frame(y_true, y_pred, config["labels"]["superclasses"])
    save_table(
        report,
        tables_dir / "deep_learning_strong_classification_report.csv",
        tables_dir / "deep_learning_strong_classification_report.tex",
    )
    history_df = pd.DataFrame(history)
    save_table(history_df, tables_dir / "deep_learning_strong_history.csv")
    _plot_history(history_df, figures_dir / "fig_deep_learning_strong_training_curves.png")
    plot_confusion_matrix(
        y_true,
        y_pred,
        config["labels"]["superclasses"],
        figures_dir / "fig_deep_learning_strong_confusion_matrix.png",
        title="Matriz de confusao - InceptionTime 1D forte",
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        config["labels"]["superclasses"],
        figures_dir / "fig_deep_learning_strong_confusion_matrix_normalized.png",
        normalize="true",
        title="Matriz de confusao normalizada - InceptionTime 1D forte",
    )
    _update_final_comparison(metrics_df, config)
    _write_metadata(metrics, strong_config, config)
    _append_experiment_log(metrics, strong_config)


def _plot_history(history: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["epoch"], history["train_loss"], label="treino loss")
    ax.plot(history["epoch"], history["val_loss"], label="validacao loss")
    ax.plot(history["epoch"], history["train_f1_macro"], label="treino F1 macro")
    ax.plot(history["epoch"], history["val_f1_macro"], label="validacao F1 macro")
    ax.set_title("Curvas de treino e validacao - InceptionTime 1D forte")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Valor")
    ax.legend()
    save_figure(fig, output_path)


def _update_final_comparison(metrics_df: pd.DataFrame, config: dict[str, Any]) -> None:
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    figures_dir = resolve_project_path(config["outputs"]["figures_dir"], config.get("project_root"))
    frames = []
    for path in [
        tables_dir / "final_model_comparison.csv",
        tables_dir / "deep_learning_strong_metrics.csv",
    ]:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        frames.append(metrics_df)
    combined = pd.concat(frames, ignore_index=True)
    final = (
        combined.loc[combined["split"].eq("test")]
        .drop_duplicates(subset=["model", "signal_frequency"], keep="last")
        .sort_values("f1_macro", ascending=False)
    )
    save_table(final, tables_dir / "final_model_comparison.csv", tables_dir / "final_model_comparison.tex")
    plot_metrics_comparison(final, figures_dir / "fig_metrics_comparison.png")


def _write_metadata(metrics: dict[str, Any], strong_config: dict[str, Any], config: dict[str, Any]) -> None:
    models_dir = resolve_project_path(config["outputs"]["models_dir"], config.get("project_root"))
    metadata = {
        "checkpoint": "deep_learning_strong_best.pt",
        "architecture": "InceptionTime1D",
        "metrics": metrics,
        "config": strong_config,
        "methodology": "folds 1-8 treino, fold 9 validacao, fold 10 teste; teste usado apenas ao final.",
    }
    (models_dir / "deep_learning_strong_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _append_experiment_log(metrics: dict[str, Any], strong_config: dict[str, Any]) -> None:
    path = resolve_project_path("reports/experiment_log.md")
    ensure_dir(path.parent)
    text = path.read_text(encoding="utf-8") if path.exists() else "# Experiment Log\n\n"
    text += (
        "\n## Deep Learning Strong - InceptionTime1D\n\n"
        f"- Model: {strong_config['model_name']}\n"
        f"- Frequency: {strong_config['frequency']} Hz\n"
        f"- Loss: {strong_config['loss_type']}\n"
        f"- Accuracy: {metrics['accuracy']:.4f}\n"
        f"- F1 macro: {metrics['f1_macro']:.4f}\n"
        f"- Recall macro: {metrics['recall_macro']:.4f}\n"
        f"- Device: {metrics['device']}\n"
        f"- Epochs trained: {metrics['epochs_trained']}\n"
        f"- Best epoch: {metrics['best_epoch']}\n"
        f"- Training seconds: {metrics['training_seconds']:.2f}\n"
    )
    path.write_text(text, encoding="utf-8")
