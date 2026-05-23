"""Deep learning para sinais brutos do PTB-XL."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tcc_ecg.data import final_labeled_records
from tcc_ecg.evaluation import classification_report_frame, compute_classification_metrics
from tcc_ecg.features import load_signal, record_path_for_row
from tcc_ecg.models import split_by_folds
from tcc_ecg.paths import ensure_dir, resolve_project_path
from tcc_ecg.plots import plot_confusion_matrix, plot_metrics_comparison, save_figure
from tcc_ecg.utils import save_table


def _import_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depende do ambiente local
        raise ImportError(
            "TensorFlow nao esta instalado. Para executar o baseline de deep learning, "
            "rode: python -m pip install -e .[dl]"
        ) from exc
    return tf


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depende do ambiente local
        raise ImportError(
            "PyTorch nao esta instalado. Para executar a ResNet1D, rode: "
            "python -m pip install -e .[dl]"
        ) from exc
    return torch


def build_simple_1d_cnn(
    input_shape: tuple[int, int],
    n_classes: int,
    learning_rate: float = 0.001,
):
    """Cria CNN 1D pequena e justificavel para baseline adicional."""
    tf = _import_tensorflow()
    layers = tf.keras.layers
    model = tf.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(32, kernel_size=7, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            layers.Conv1D(64, kernel_size=5, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_signal_tensor(metadata: pd.DataFrame, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Carrega sinais brutos para o baseline CNN."""
    records = final_labeled_records(metadata)
    max_records = config.get("features", {}).get("max_records")
    if max_records:
        records = records.head(int(max_records))

    signals = []
    kept_index = []
    for ecg_id, row in records.iterrows():
        signal = load_signal(record_path_for_row(row, config)).astype("float32")
        signals.append(signal)
        kept_index.append(ecg_id)

    used = records.loc[kept_index].copy()
    X = np.stack(signals).astype("float32")
    y = used["target_id"].astype(int).to_numpy()
    return X, y, used


def train_deep_learning_baseline(metadata: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Treina CNN simples usando folds oficiais e salva metricas reais."""
    tf = _import_tensorflow()
    seed = int(config["project"]["seed"])
    tf.keras.utils.set_random_seed(seed)

    X, y, used_metadata = load_signal_tensor(metadata, config)
    tensor_df = used_metadata.reset_index().assign(row_idx=np.arange(len(used_metadata)), target_id=y)
    splits = split_by_folds(tensor_df, config)

    train_idx = splits["train"]["row_idx"].to_numpy()
    val_idx = splits["validation"]["row_idx"].to_numpy()
    test_idx = splits["test"]["row_idx"].to_numpy()

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    model = build_simple_1d_cnn(
        input_shape=X_train.shape[1:],
        n_classes=len(config["labels"]["superclasses"]),
        learning_rate=float(config["deep_learning"]["learning_rate"]),
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(config["deep_learning"]["patience"]),
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=int(config["deep_learning"]["epochs"]),
        batch_size=int(config["deep_learning"]["batch_size"]),
        callbacks=callbacks,
        verbose=1,
    )
    pred_proba = model.predict(X_test)
    y_pred = pred_proba.argmax(axis=1)
    metrics = compute_classification_metrics(
        y_test,
        y_pred,
        model_name="deep_learning_baseline",
        split="test",
    )
    metrics["signal_frequency"] = int(config["data"]["signal_frequency"])
    metrics["smote"] = False

    _save_deep_learning_outputs(model, history, metrics, y_test, y_pred, config)
    return {"model": model, "history": history.history, "metrics": metrics, "y_test": y_test, "y_pred": y_pred}


def _save_deep_learning_outputs(
    model,
    history,
    metrics: dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: dict[str, Any],
) -> None:
    models_dir = resolve_project_path(config["outputs"]["models_dir"], config.get("project_root"))
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    figures_dir = resolve_project_path(config["outputs"]["figures_dir"], config.get("project_root"))
    ensure_dir(models_dir)
    model.save(models_dir / "deep_learning_baseline.keras")

    metrics_df = pd.DataFrame([metrics])
    frequency = int(config["data"]["signal_frequency"])
    save_table(
        metrics_df,
        tables_dir / "deep_learning_baseline_metrics.csv",
        tables_dir / "deep_learning_baseline_metrics.tex",
    )
    save_table(
        metrics_df,
        tables_dir / f"deep_learning_baseline_metrics_{frequency}hz.csv",
        tables_dir / f"deep_learning_baseline_metrics_{frequency}hz.tex",
    )
    report = classification_report_frame(y_true, y_pred, config["labels"]["superclasses"])
    save_table(
        report,
        tables_dir / "deep_learning_baseline_classification_report.csv",
        tables_dir / "deep_learning_baseline_classification_report.tex",
    )

    # Nomes legados preservados para notebooks anteriores.
    save_table(metrics_df, tables_dir / "deep_learning_metrics.csv", tables_dir / "deep_learning_metrics.tex")
    save_table(
        metrics_df,
        tables_dir / f"deep_learning_metrics_{frequency}hz.csv",
        tables_dir / f"deep_learning_metrics_{frequency}hz.tex",
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history.history.get("loss", []), label="treino loss")
    ax.plot(history.history.get("val_loss", []), label="validacao loss")
    if "accuracy" in history.history:
        ax.plot(history.history["accuracy"], label="treino accuracy")
    if "val_accuracy" in history.history:
        ax.plot(history.history["val_accuracy"], label="validacao accuracy")
    ax.set_title("Curvas de treino e validacao - CNN 1D simples")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Valor")
    ax.legend()
    save_figure(fig, figures_dir / "fig_dl_training_curves.png")
    _plot_tensorflow_history(history, figures_dir / "fig_dl_baseline_training_curves.png")
    plot_confusion_matrix(
        y_true,
        y_pred,
        config["labels"]["superclasses"],
        figures_dir / "fig_dl_baseline_confusion_matrix.png",
        title="Matriz de confusao - CNN baseline simples",
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        config["labels"]["superclasses"],
        figures_dir / "fig_dl_baseline_confusion_matrix_normalized.png",
        normalize="true",
        title="Matriz de confusao normalizada - CNN baseline simples",
    )
    _update_final_model_comparison_with_deep_learning_baseline(metrics_df, config)


def _plot_tensorflow_history(history, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history.history.get("loss", []), label="treino loss")
    ax.plot(history.history.get("val_loss", []), label="validacao loss")
    if "accuracy" in history.history:
        ax.plot(history.history["accuracy"], label="treino accuracy")
    if "val_accuracy" in history.history:
        ax.plot(history.history["val_accuracy"], label="validacao accuracy")
    ax.set_title("Curvas de treino e validacao - CNN baseline simples")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Valor")
    ax.legend()
    save_figure(fig, output_path)


def _update_final_model_comparison_with_deep_learning_baseline(
    baseline_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    figures_dir = resolve_project_path(config["outputs"]["figures_dir"], config.get("project_root"))
    frequency = int(config["data"]["signal_frequency"])
    frames = []
    for path in [
        tables_dir / f"model_metrics_{frequency}hz.csv",
        tables_dir / f"deep_learning_baseline_metrics_{frequency}hz.csv",
        tables_dir / f"deep_learning_resnet1d_metrics_{frequency}hz.csv",
        tables_dir / "final_model_comparison.csv",
    ]:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        frames.append(baseline_metrics)
    combined = pd.concat(frames, ignore_index=True)
    if "signal_frequency" not in combined.columns:
        combined["signal_frequency"] = frequency
    final = (
        combined.loc[combined["split"].eq("test")]
        .drop_duplicates(subset=["model", "signal_frequency"], keep="last")
        .sort_values("f1_macro", ascending=False)
    )
    save_table(final, tables_dir / "final_model_comparison.csv", tables_dir / "final_model_comparison.tex")
    final_current_frequency = final.loc[final["signal_frequency"].eq(frequency)].copy()
    save_table(
        final_current_frequency,
        tables_dir / f"final_model_comparison_{frequency}hz.csv",
        tables_dir / f"final_model_comparison_{frequency}hz.tex",
    )
    plot_metrics_comparison(final, figures_dir / "fig_metrics_comparison.png")


class ECGMemmapDataset:
    """Dataset PyTorch baseado em cache `.npy`, sem carregar tudo na RAM."""

    def __init__(
        self,
        signals_path: str | Path,
        labels: np.ndarray,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.signals = np.load(signals_path, mmap_mode="r")
        self.labels = labels.astype("int64")
        self.indices = indices.astype("int64")
        self.mean = mean.astype("float32")
        self.std = std.astype("float32")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        torch = _import_torch()
        idx = int(self.indices[item])
        x = self.signals[idx].astype("float32")
        x = (x - self.mean[None, :]) / self.std[None, :]
        # PyTorch Conv1d espera (canais, tempo), enquanto WFDB retorna (tempo, canais).
        x_tensor = torch.from_numpy(np.ascontiguousarray(x.T))
        y_tensor = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x_tensor, y_tensor


def set_torch_seed(seed: int) -> None:
    """Define sementes para treino reprodutivel em PyTorch."""
    torch = _import_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_resnet1d_model(
    input_channels: int = 12,
    n_classes: int = 5,
    base_filters: int = 32,
    kernel_size: int = 7,
    dropout: float = 0.25,
):
    """Cria uma ResNet1D leve para ECG, sem arquiteturas profundas demais."""
    torch = _import_torch()
    nn = torch.nn

    class ResidualBlock1D(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
            super().__init__()
            padding = kernel_size // 2
            self.conv1 = nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            )
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(dropout)
            self.conv2 = nn.Conv1d(
                out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False
            )
            self.bn2 = nn.BatchNorm1d(out_channels)
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels),
                )
            else:
                self.shortcut = nn.Identity()

        def forward(self, x):
            residual = self.shortcut(x)
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out + residual)
            return out

    class ResNet1D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(
                    input_channels,
                    base_filters,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(base_filters),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            )
            self.blocks = nn.Sequential(
                ResidualBlock1D(base_filters, base_filters, stride=1),
                ResidualBlock1D(base_filters, base_filters * 2, stride=2),
                ResidualBlock1D(base_filters * 2, base_filters * 2, stride=1),
                ResidualBlock1D(base_filters * 2, base_filters * 4, stride=2),
                ResidualBlock1D(base_filters * 4, base_filters * 4, stride=1),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(base_filters * 4, n_classes),
            )

        def forward(self, x):
            return self.head(self.blocks(self.stem(x)))

    return ResNet1D()


def get_resnet1d_config(config: dict[str, Any]) -> dict[str, Any]:
    """Retorna parametros da ResNet1D com defaults conservadores."""
    defaults = {
        "enabled": True,
        "signal_frequency": 500,
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "patience": 8,
        "reduce_lr_patience": 3,
        "reduce_lr_factor": 0.5,
        "min_lr": 0.000001,
        "base_filters": 32,
        "kernel_size": 7,
        "dropout": 0.25,
        "num_workers": 0,
        "cache_dtype": "float16",
        "force_rebuild_cache": False,
        "max_records": None,
    }
    user_config = config.get("deep_learning", {}).get("resnet1d", {})
    return {**defaults, **user_config}


def build_resnet1d_cache(metadata: pd.DataFrame, config: dict[str, Any]) -> dict[str, Path]:
    """Cria cache dos sinais brutos para treinar sem reler WFDB a cada epoca."""
    resnet_config = get_resnet1d_config(config)
    frequency = int(config["data"]["signal_frequency"])
    processed_dir = resolve_project_path(config["outputs"]["processed_dir"], config.get("project_root"))
    ensure_dir(processed_dir)
    signals_path = processed_dir / f"signals_{frequency}hz_resnet1d.npy"
    labels_path = processed_dir / f"labels_{frequency}hz_resnet1d.npy"
    metadata_path = processed_dir / f"metadata_{frequency}hz_resnet1d.csv"

    if (
        signals_path.exists()
        and labels_path.exists()
        and metadata_path.exists()
        and not bool(resnet_config["force_rebuild_cache"])
    ):
        return {"signals": signals_path, "labels": labels_path, "metadata": metadata_path}

    records = final_labeled_records(metadata).copy()
    max_records = resnet_config.get("max_records")
    if max_records:
        records = records.head(int(max_records))
    records = records.sort_values("strat_fold")

    if records.empty:
        raise ValueError("Nenhum registro rotulado disponivel para criar cache ResNet1D.")

    first_signal = load_signal(record_path_for_row(records.iloc[0], config))
    dtype = np.dtype(resnet_config["cache_dtype"])
    signals = np.lib.format.open_memmap(
        signals_path,
        mode="w+",
        dtype=dtype,
        shape=(len(records), first_signal.shape[0], first_signal.shape[1]),
    )
    labels = np.empty(len(records), dtype="int64")

    for pos, (_, row) in enumerate(records.iterrows()):
        signal = load_signal(record_path_for_row(row, config))
        if signal.shape != first_signal.shape:
            raise ValueError(f"Sinal com shape inesperado: {signal.shape}; esperado {first_signal.shape}.")
        signals[pos] = signal.astype(dtype)
        labels[pos] = int(row["target_id"])

    np.save(labels_path, labels)
    cache_metadata = records.reset_index()[["ecg_id", "strat_fold", "target", "target_id"]].copy()
    cache_metadata.to_csv(metadata_path, index=False)
    return {"signals": signals_path, "labels": labels_path, "metadata": metadata_path}


def compute_train_channel_normalization(
    signals_path: str | Path,
    train_indices: np.ndarray,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula media/desvio por canal usando somente registros de treino."""
    signals = np.load(signals_path, mmap_mode="r")
    n_channels = signals.shape[2]
    channel_sum = np.zeros(n_channels, dtype="float64")
    channel_sumsq = np.zeros(n_channels, dtype="float64")
    count = 0

    for start in range(0, len(train_indices), batch_size):
        idx = train_indices[start : start + batch_size]
        batch = signals[idx].astype("float32")
        channel_sum += batch.sum(axis=(0, 1))
        channel_sumsq += np.square(batch, dtype="float32").sum(axis=(0, 1))
        count += batch.shape[0] * batch.shape[1]

    mean = channel_sum / count
    variance = np.maximum(channel_sumsq / count - mean**2, 1e-8)
    std = np.sqrt(variance)
    return mean.astype("float32"), std.astype("float32")


def train_resnet1d(metadata: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Treina ResNet1D leve com validacao no fold 9 e teste apenas ao final."""
    torch = _import_torch()
    from torch.utils.data import DataLoader

    resnet_config = get_resnet1d_config(config)
    set_torch_seed(int(config["project"]["seed"]))
    cache_paths = build_resnet1d_cache(metadata, config)
    labels = np.load(cache_paths["labels"])
    cache_metadata = pd.read_csv(cache_paths["metadata"]).assign(row_idx=np.arange(len(labels)))
    splits = split_by_folds(cache_metadata, config)

    train_idx = splits["train"]["row_idx"].to_numpy(dtype="int64")
    val_idx = splits["validation"]["row_idx"].to_numpy(dtype="int64")
    test_idx = splits["test"]["row_idx"].to_numpy(dtype="int64")

    mean, std = compute_train_channel_normalization(
        cache_paths["signals"], train_idx, batch_size=int(resnet_config["batch_size"])
    )
    _save_resnet_normalization(mean, std, config)

    train_dataset = ECGMemmapDataset(cache_paths["signals"], labels, train_idx, mean, std)
    val_dataset = ECGMemmapDataset(cache_paths["signals"], labels, val_idx, mean, std)
    test_dataset = ECGMemmapDataset(cache_paths["signals"], labels, test_idx, mean, std)

    loader_kwargs = {
        "batch_size": int(resnet_config["batch_size"]),
        "num_workers": int(resnet_config["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet1d_model(
        input_channels=12,
        n_classes=len(config["labels"]["superclasses"]),
        base_filters=int(resnet_config["base_filters"]),
        kernel_size=int(resnet_config["kernel_size"]),
        dropout=float(resnet_config["dropout"]),
    ).to(device)

    class_weights = _compute_class_weights(labels[train_idx], len(config["labels"]["superclasses"]))
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(resnet_config["learning_rate"]),
        weight_decay=float(resnet_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(resnet_config["reduce_lr_factor"]),
        patience=int(resnet_config["reduce_lr_patience"]),
        min_lr=float(resnet_config["min_lr"]),
    )
    checkpoint_path = _resnet_checkpoint_path(config)
    history = _fit_resnet1d(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        checkpoint_path,
        config,
        max_epochs=int(resnet_config["epochs"]),
        patience=int(resnet_config["patience"]),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_true, y_pred = _predict_resnet1d(model, test_loader, device)
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        model_name="resnet1d_light",
        split="test",
    )
    metrics["signal_frequency"] = int(config["data"]["signal_frequency"])
    metrics["smote"] = False
    _save_resnet1d_outputs(history, metrics, y_true, y_pred, config)
    return {
        "model": model,
        "history": history,
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "checkpoint": checkpoint_path,
        "device": str(device),
    }


def _compute_class_weights(y_train: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y_train.astype(int), minlength=n_classes).astype("float64")
    weights = counts.sum() / np.maximum(counts, 1) / n_classes
    weights[counts == 0] = 0.0
    return weights.astype("float32")


def _fit_resnet1d(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    checkpoint_path: Path,
    config: dict[str, Any],
    max_epochs: int,
    patience: int,
) -> list[dict[str, float]]:
    torch = _import_torch()
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        train_loss, train_accuracy = _run_resnet_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_accuracy = _run_resnet_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr,
        }
        history.append(row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            ensure_dir(checkpoint_path.parent)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                    "class_names": config["labels"]["superclasses"],
                    "signal_frequency": int(config["data"]["signal_frequency"]),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    return history


def _run_resnet_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    torch = _import_torch()
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * y_batch.size(0)
            total_correct += int((logits.argmax(dim=1) == y_batch).sum().item())
            total_examples += int(y_batch.size(0))

    return total_loss / total_examples, total_correct / total_examples


def _predict_resnet1d(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    torch = _import_torch()
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch.to(device))
            y_true.append(y_batch.numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def _resnet_checkpoint_path(config: dict[str, Any]) -> Path:
    models_dir = resolve_project_path(config["outputs"]["models_dir"], config.get("project_root"))
    return models_dir / "resnet1d_best.pt"


def _save_resnet_normalization(mean: np.ndarray, std: np.ndarray, config: dict[str, Any]) -> None:
    processed_dir = resolve_project_path(config["outputs"]["processed_dir"], config.get("project_root"))
    frequency = int(config["data"]["signal_frequency"])
    ensure_dir(processed_dir)
    np.savez(processed_dir / f"resnet1d_normalization_{frequency}hz.npz", mean=mean, std=std)


def _save_resnet1d_outputs(
    history: list[dict[str, float]],
    metrics: dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: dict[str, Any],
) -> None:
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    figures_dir = resolve_project_path(config["outputs"]["figures_dir"], config.get("project_root"))
    models_dir = resolve_project_path(config["outputs"]["models_dir"], config.get("project_root"))
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)
    ensure_dir(models_dir)

    frequency = int(config["data"]["signal_frequency"])
    metrics_df = pd.DataFrame([metrics])
    save_table(
        metrics_df,
        tables_dir / "deep_learning_resnet1d_metrics.csv",
        tables_dir / "deep_learning_resnet1d_metrics.tex",
    )
    save_table(
        metrics_df,
        tables_dir / f"deep_learning_resnet1d_metrics_{frequency}hz.csv",
        tables_dir / f"deep_learning_resnet1d_metrics_{frequency}hz.tex",
    )

    report = classification_report_frame(y_true, y_pred, config["labels"]["superclasses"])
    save_table(
        report,
        tables_dir / "deep_learning_resnet1d_classification_report.csv",
        tables_dir / "deep_learning_resnet1d_classification_report.tex",
    )

    history_df = pd.DataFrame(history)
    save_table(history_df, tables_dir / "deep_learning_resnet1d_history.csv")
    _plot_resnet_history(history_df, figures_dir / "fig_resnet1d_training_curves.png")
    plot_confusion_matrix(
        y_true,
        y_pred,
        config["labels"]["superclasses"],
        figures_dir / "fig_resnet1d_confusion_matrix.png",
        title="Matriz de confusao - ResNet1D leve",
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        config["labels"]["superclasses"],
        figures_dir / "fig_resnet1d_confusion_matrix_normalized.png",
        normalize="true",
        title="Matriz de confusao normalizada - ResNet1D leve",
    )
    _update_final_model_comparison_with_resnet(metrics_df, config)
    _write_resnet_metadata(config)


def _plot_resnet_history(history: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["train_loss"], label="treino loss")
    ax.plot(history["epoch"], history["val_loss"], label="validacao loss")
    ax.plot(history["epoch"], history["train_accuracy"], label="treino accuracy")
    ax.plot(history["epoch"], history["val_accuracy"], label="validacao accuracy")
    ax.set_title("Curvas de treino e validacao - ResNet1D leve")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Valor")
    ax.legend()
    save_figure(fig, output_path)


def _update_final_model_comparison_with_resnet(resnet_metrics: pd.DataFrame, config: dict[str, Any]) -> None:
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    frequency = int(config["data"]["signal_frequency"])
    frames = []
    for path in [
        tables_dir / f"model_metrics_{frequency}hz.csv",
        tables_dir / f"deep_learning_metrics_{frequency}hz.csv",
        tables_dir / f"deep_learning_resnet1d_metrics_{frequency}hz.csv",
    ]:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        frames.append(resnet_metrics)
    combined = pd.concat(frames, ignore_index=True)
    final = (
        combined.loc[combined["split"].eq("test")]
        .drop_duplicates(subset=["model", "signal_frequency"], keep="last")
        .sort_values("f1_macro", ascending=False)
    )
    save_table(final, tables_dir / "final_model_comparison.csv", tables_dir / "final_model_comparison.tex")
    final_current_frequency = final.loc[final["signal_frequency"].eq(frequency)].copy()
    save_table(
        final_current_frequency,
        tables_dir / f"final_model_comparison_{frequency}hz.csv",
        tables_dir / f"final_model_comparison_{frequency}hz.tex",
    )
    plot_metrics_comparison(final, resolve_project_path(config["outputs"]["figures_dir"], config.get("project_root")) / "fig_metrics_comparison.png")


def _write_resnet_metadata(config: dict[str, Any]) -> None:
    models_dir = resolve_project_path(config["outputs"]["models_dir"], config.get("project_root"))
    metadata = {
        "model": "resnet1d_light",
        "framework": "pytorch",
        "checkpoint": "resnet1d_best.pt",
        "signal_frequency": int(config["data"]["signal_frequency"]),
        "folds": config["folds"],
        "note": "Teste usado apenas apos selecao do checkpoint por val_loss no fold 9.",
    }
    (models_dir / "resnet1d_best_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# Interfaces do experimento pesado mantidas aqui como wrappers tardios para
# preservar um ponto publico unico sem criar import circular com o cache legado.
def get_torch_environment() -> dict[str, Any]:
    from tcc_ecg.dl_training import get_torch_environment as _get_torch_environment

    return _get_torch_environment()


def train_deep_learning_heavy(metadata: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    from tcc_ecg.dl_training import train_deep_learning_heavy as _train_deep_learning_heavy

    return _train_deep_learning_heavy(metadata, config)


def generate_balance_and_split_artifacts(metadata: pd.DataFrame, config: dict[str, Any]) -> dict[str, Path]:
    from tcc_ecg.dl_data import generate_balance_and_split_artifacts as _generate_balance_and_split_artifacts

    return _generate_balance_and_split_artifacts(metadata, config)
