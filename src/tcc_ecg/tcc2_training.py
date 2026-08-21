"""Interface de treino reproduzivel do TCC II."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from tcc_ecg.data import add_age_features, load_metadata
from tcc_ecg.external_repositories import load_repository_specs, repository_inventory
from tcc_ecg.features import get_feature_columns
from tcc_ecg.multilabel import build_multilabel_targets, label_column
from tcc_ecg.multilabel_data import (
    MultilabelMemmapDataset,
    compute_positive_weights,
    prepare_multilabel_signal_splits,
)
from tcc_ecg.multilabel_evaluation import (
    compute_multilabel_metrics,
    multilabel_report_frame,
    optimize_thresholds_on_validation,
)
from tcc_ecg.multilabel_models import (
    MODEL_SPECS,
    build_multilabel_classical_pipelines,
    build_tcn_model,
    get_model_spec,
)
from tcc_ecg.paths import ensure_dir, resolve_project_path
from tcc_ecg.protocol import (
    authorize_test_evaluation,
    config_hash,
    current_git_commit,
    mark_test_evaluated,
    split_by_official_folds,
    validate_tcc2_protocol,
    write_split_manifest,
)
from tcc_ecg.utils import save_table, set_random_seed


def prepare_multilabel_metadata(config: dict[str, Any]) -> pd.DataFrame:
    metadata, statements = load_metadata(config)
    metadata = add_age_features(metadata)
    return build_multilabel_targets(metadata, statements, config["labels"]["superclasses"])


def smoke_tcc2(config: dict[str, Any]) -> pd.DataFrame:
    """Executa verificacoes sinteticas sem produzir metricas cientificas."""
    validate_tcc2_protocol(config)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(config["project"]["seed"]))
    x = pd.DataFrame(rng.normal(size=(40, 8)), columns=[f"f{index}" for index in range(8)])
    y = np.zeros((40, 5), dtype="int8")
    for row in range(len(y)):
        y[row, row % 5] = 1
        if row % 7 == 0:
            y[row, (row + 1) % 5] = 1
    for name, pipeline in build_multilabel_classical_pipelines(config, list(x.columns)).items():
        pipeline.fit(x, y)
        prediction = pipeline.predict(x.iloc[:2])
        rows.append(
            {
                "model": name,
                "check": "synthetic_fit_predict",
                "status": "passed" if prediction.shape == (2, 5) else "failed",
                "scientific_result": False,
            }
        )

    try:
        import torch

        tcn_config = config["models"]["tcn"]
        model = build_tcn_model(
            input_channels=12,
            n_classes=5,
            channels=[int(value) for value in tcn_config["channels"]],
            kernel_size=int(tcn_config["kernel_size"]),
            dropout=float(tcn_config["dropout"]),
        )
        with torch.no_grad():
            output = model(torch.randn(2, 12, 256))
        rows.append(
            {
                "model": "tcn",
                "check": "synthetic_forward",
                "status": "passed" if tuple(output.shape) == (2, 5) else "failed",
                "scientific_result": False,
            }
        )
        synthetic_x = torch.randn(12, 12, 128)
        synthetic_y = torch.zeros(12, 5)
        for index in range(12):
            synthetic_y[index, index % 5] = 1.0
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(synthetic_x, synthetic_y), batch_size=4, shuffle=False
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        device = torch.device("cpu")
        train_metrics = _epoch(model, loader, criterion, device, optimizer, config)
        validation_metrics = _epoch(model, loader, criterion, device, None, config)
        rows.append(
            {
                "model": "tcn",
                "check": "synthetic_train_validation",
                "status": "passed"
                if math.isfinite(train_metrics["loss"]) and math.isfinite(validation_metrics["loss"])
                else "failed",
                "scientific_result": False,
            }
        )
    except ImportError:
        rows.append({"model": "tcn", "check": "synthetic_forward", "status": "dependency_missing", "scientific_result": False})

    repository_specs = load_repository_specs()
    inventory = repository_inventory(repository_specs)
    for spec in MODEL_SPECS:
        if spec.backend != "external":
            continue
        source_status = inventory.loc[inventory["name"].eq(spec.source), "status"]
        rows.append(
            {
                "model": spec.name,
                "check": "pinned_external_definition",
                "status": "passed" if spec.source in repository_specs else "failed",
                "checkout": source_status.iloc[0] if not source_status.empty else "not_configured",
                "scientific_result": False,
            }
        )
    frame = pd.DataFrame(rows)
    logs_dir = resolve_project_path(config["outputs"]["logs_dir"], config.get("project_root"))
    ensure_dir(logs_dir)
    frame.to_csv(logs_dir / "smoke_tcc2.csv", index=False)
    return frame


def train_tcn_validation(config: dict[str, Any]) -> dict[str, Any]:
    """Treina a TCN e salva somente resultados de validacao."""
    torch = _import_torch()

    validate_tcc2_protocol(config)
    set_random_seed(int(config["project"]["seed"]))
    torch.manual_seed(int(config["project"]["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["project"]["seed"]))

    metadata = prepare_multilabel_metadata(config)
    label_columns = [label_column(name) for name in config["labels"]["superclasses"]]
    write_split_manifest(metadata.loc[metadata["label_status_multilabel"].eq("kept")].reset_index(), config, label_columns)
    prepared = prepare_multilabel_signal_splits(metadata, config)
    train_loader, validation_loader = _build_train_validation_loaders(prepared, config)

    device = torch.device("cuda" if torch.cuda.is_available() and config["training"].get("device", "auto") != "cpu" else "cpu")
    tcn_config = config["models"]["tcn"]
    model = build_tcn_model(
        12,
        len(config["labels"]["superclasses"]),
        [int(value) for value in tcn_config["channels"]],
        int(tcn_config["kernel_size"]),
        float(tcn_config["dropout"]),
    ).to(device)
    pos_weight = compute_positive_weights(prepared["labels"][prepared["train_indices"]])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=max(2, int(config["training"]["patience"]) // 3),
        min_lr=float(config["training"]["min_learning_rate"]),
    )

    run_dir = _run_dir(config, "tcn")
    ensure_dir(run_dir)
    effective_config_path = run_dir / "effective_config.yaml"
    effective_config_path.write_text(yaml.safe_dump(_portable_config(config), sort_keys=False), encoding="utf-8")
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    history_path = run_dir / "history.csv"
    start_epoch, best_score, best_epoch = _resume_checkpoint(model, optimizer, last_path, config, device)
    history = pd.read_csv(history_path).to_dict("records") if start_epoch and history_path.exists() else []
    patience_counter = 0
    started = time.time()
    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        train_result = _epoch(model, train_loader, criterion, device, optimizer, config)
        validation_result = _epoch(model, validation_loader, criterion, device, None, config)
        score = float(validation_result["macro_auroc"])
        scheduler.step(score if math.isfinite(score) else -1.0)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_result["loss"],
            "validation_loss": validation_result["loss"],
            "train_macro_auroc": train_result["macro_auroc"],
            "validation_macro_auroc": validation_result["macro_auroc"],
            "validation_f1_macro_at_0_5": validation_result["f1_macro"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        if math.isfinite(score) and score > best_score:
            best_score = score
            best_epoch = epoch + 1
            patience_counter = 0
            save_training_checkpoint(best_path, model, optimizer, epoch + 1, best_score, config)
        else:
            patience_counter += 1
        save_training_checkpoint(last_path, model, optimizer, epoch + 1, best_score, config)
        if patience_counter >= int(config["training"]["patience"]):
            break

    if not best_path.exists():
        raise RuntimeError("Nenhum checkpoint de validacao foi produzido.")
    load_training_checkpoint(best_path, model, None, config, device)
    y_validation, score_validation = _predict(model, validation_loader, device)
    thresholds = optimize_thresholds_on_validation(y_validation, score_validation)
    validation_metrics = compute_multilabel_metrics(y_validation, score_validation, thresholds)
    report = multilabel_report_frame(y_validation, score_validation, config["labels"]["superclasses"], thresholds)
    validation_metrics.update(
        {
            "model": "tcn",
            "split": "validation",
            "best_epoch": best_epoch,
            "training_seconds": time.time() - started,
            "device": str(device),
            "parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        }
    )
    save_table(pd.DataFrame([validation_metrics]), run_dir / "validation_metrics.csv", run_dir / "validation_metrics.tex")
    save_table(report, run_dir / "validation_report.csv", run_dir / "validation_report.tex")
    candidate_manifest = _write_candidate_manifest(
        run_dir, "tcn", best_path, validation_metrics, thresholds.tolist(), config
    )
    return {
        "run_dir": run_dir,
        "checkpoint": best_path,
        "candidate_manifest": candidate_manifest,
        "validation_metrics": validation_metrics,
    }


def train_classical_validation(
    config: dict[str, Any],
    model_name: str,
    features_path: str | Path | None = None,
) -> dict[str, Any]:
    """Ajusta um modelo tabular no treino e mede apenas a validacao."""
    validate_tcc2_protocol(config)
    get_model_spec(model_name)
    feature_path = resolve_project_path(
        features_path or config["data"]["classical_features_path"], config.get("project_root")
    )
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Features multilabel ausentes: {feature_path}. Rode scripts/prepare_tcc2_features.py."
        )
    features = pd.read_parquet(feature_path)
    classes = list(config["labels"]["superclasses"])
    label_columns = [label_column(name) for name in classes]
    required = {"ecg_id", "strat_fold", *label_columns}
    missing = sorted(required - set(features.columns))
    if missing:
        raise KeyError(f"Features multilabel incompletas: {missing}")
    splits = split_by_official_folds(features, config)
    feature_columns = get_feature_columns(features)
    feature_columns = [column for column in feature_columns if column not in label_columns and column != "split"]
    pipeline = build_multilabel_classical_pipelines(config, feature_columns)[model_name]
    pipeline.fit(splits["train"][feature_columns], splits["train"][label_columns])
    score = _classical_scores(pipeline, splits["validation"][feature_columns])
    truth = splits["validation"][label_columns].to_numpy(dtype="int8")
    thresholds = optimize_thresholds_on_validation(truth, score)
    metrics = compute_multilabel_metrics(truth, score, thresholds)
    metrics.update({"model": model_name, "split": "validation"})
    run_dir = _run_dir(config, model_name)
    ensure_dir(run_dir)
    checkpoint = run_dir / "best.joblib"
    joblib.dump(pipeline, checkpoint)
    save_table(pd.DataFrame([metrics]), run_dir / "validation_metrics.csv", run_dir / "validation_metrics.tex")
    candidate_manifest = _write_candidate_manifest(run_dir, model_name, checkpoint, metrics, thresholds.tolist(), config)
    return {"run_dir": run_dir, "checkpoint": checkpoint, "candidate_manifest": candidate_manifest, "validation_metrics": metrics}


def evaluate_frozen_test(
    config: dict[str, Any],
    manifest_path: str | Path,
    features_path: str | Path | None = None,
) -> dict[str, Any]:
    """Avalia o fold 10 uma unica vez, depois da autorizacao do manifesto."""
    manifest_file = resolve_project_path(manifest_path, config.get("project_root"))
    manifest = authorize_test_evaluation(manifest_file, config, evaluate_test=True)
    model_name = str(manifest["model_name"])
    if model_name == "tcn":
        truth, score = _evaluate_tcn_test(config, manifest)
    else:
        truth, score = _evaluate_classical_test(config, manifest, features_path)
    thresholds = np.asarray(manifest["thresholds"], dtype="float32")
    metrics = compute_multilabel_metrics(truth, score, thresholds)
    metrics.update({"model": model_name, "split": "test", "frozen_manifest": str(manifest_file)})
    report = multilabel_report_frame(truth, score, config["labels"]["superclasses"], thresholds)
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    ensure_dir(tables_dir)
    metrics_path = tables_dir / f"{model_name}_selected_test_metrics.csv"
    save_table(pd.DataFrame([metrics]), metrics_path, tables_dir / f"{model_name}_selected_test_metrics.tex")
    save_table(report, tables_dir / f"{model_name}_selected_test_report.csv", tables_dir / f"{model_name}_selected_test_report.tex")
    mark_test_evaluated(manifest_file, metrics_path)
    return {"metrics": metrics, "metrics_path": metrics_path}


def _evaluate_tcn_test(config: dict[str, Any], manifest: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    torch = _import_torch()
    from torch.utils.data import DataLoader

    metadata = prepare_multilabel_metadata(config)
    prepared = prepare_multilabel_signal_splits(metadata, config)
    dataset = MultilabelMemmapDataset(
        prepared["cache_paths"]["signals"],
        prepared["labels"],
        prepared["test_indices"],
        prepared["mean"],
        prepared["std"],
        train=False,
    )
    workers = int(config["training"].get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and config["training"].get("device", "auto") != "cpu" else "cpu")
    tcn = config["models"]["tcn"]
    model = build_tcn_model(
        12,
        len(config["labels"]["superclasses"]),
        [int(value) for value in tcn["channels"]],
        int(tcn["kernel_size"]),
        float(tcn["dropout"]),
    ).to(device)
    load_training_checkpoint(manifest["checkpoint"], model, None, config, device)
    return _predict(model, loader, device)


def _evaluate_classical_test(
    config: dict[str, Any],
    manifest: dict[str, Any],
    features_path: str | Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    feature_path = resolve_project_path(
        features_path or config["data"]["classical_features_path"], config.get("project_root")
    )
    features = pd.read_parquet(feature_path)
    splits = split_by_official_folds(features, config)
    label_columns = [label_column(name) for name in config["labels"]["superclasses"]]
    feature_columns = [
        column
        for column in get_feature_columns(features)
        if column not in label_columns and column != "split"
    ]
    model = joblib.load(manifest["checkpoint"])
    truth = splits["test"][label_columns].to_numpy(dtype="int8")
    return truth, _classical_scores(model, splits["test"][feature_columns])


def save_training_checkpoint(path: str | Path, model, optimizer, epoch: int, best_score: float, config: dict[str, Any]) -> None:
    torch = _import_torch()
    output = Path(path)
    ensure_dir(output.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "epoch": int(epoch),
            "best_score": float(best_score),
            "config_hash": config_hash(config),
            "git_commit": current_git_commit(),
        },
        output,
    )


def load_training_checkpoint(path: str | Path, model, optimizer, config: dict[str, Any], device) -> dict[str, Any]:
    torch = _import_torch()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("config_hash") != config_hash(config):
        raise ValueError("Checkpoint criado com configuracao diferente.")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def _resume_checkpoint(model, optimizer, last_path: Path, config: dict[str, Any], device) -> tuple[int, float, int]:
    if not bool(config["training"].get("resume", True)) or not last_path.exists():
        return 0, float("-inf"), 0
    checkpoint = load_training_checkpoint(last_path, model, optimizer, config, device)
    return int(checkpoint["epoch"]), float(checkpoint["best_score"]), int(checkpoint["epoch"])


def _build_train_validation_loaders(prepared: dict[str, Any], config: dict[str, Any]):
    torch = _import_torch()
    from torch.utils.data import DataLoader

    training = config["training"]
    train_dataset = MultilabelMemmapDataset(
        prepared["cache_paths"]["signals"],
        prepared["labels"],
        prepared["train_indices"],
        prepared["mean"],
        prepared["std"],
        train=True,
        augmentations=training.get("augmentations"),
    )
    validation_dataset = MultilabelMemmapDataset(
        prepared["cache_paths"]["signals"],
        prepared["labels"],
        prepared["validation_indices"],
        prepared["mean"],
        prepared["std"],
        train=False,
    )
    workers = int(training.get("num_workers", 0))
    common = {
        "batch_size": int(training["batch_size"]),
        "num_workers": workers,
        "pin_memory": bool(torch.cuda.is_available()),
        "persistent_workers": workers > 0,
    }
    return DataLoader(train_dataset, shuffle=True, **common), DataLoader(validation_dataset, shuffle=False, **common)


def _epoch(model, loader, criterion, device, optimizer, config: dict[str, Any]) -> dict[str, float]:
    torch = _import_torch()
    training = optimizer is not None
    model.train(training)
    losses = []
    truths = []
    scores = []
    use_amp = bool(config["training"].get("use_amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
        losses.append(float(loss.detach().cpu()))
        truths.append(y.detach().cpu().numpy())
        scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    metrics = compute_multilabel_metrics(np.concatenate(truths), np.concatenate(scores), 0.5)
    return {"loss": float(np.mean(losses)), **metrics}


def _predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    torch = _import_torch()
    model.eval()
    truths, scores = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device, non_blocking=True))
            truths.append(y.numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(truths), np.concatenate(scores)


def _classical_scores(pipeline, features: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return np.asarray(pipeline.predict_proba(features), dtype="float64")
    decision = np.asarray(pipeline.decision_function(features), dtype="float64")
    return 1.0 / (1.0 + np.exp(-np.clip(decision, -40, 40)))


def _write_candidate_manifest(
    run_dir: Path,
    model_name: str,
    checkpoint: Path,
    validation_metrics: dict[str, Any],
    thresholds: list[float],
    config: dict[str, Any],
) -> Path:
    path = run_dir / "candidate_manifest.json"
    payload = {
        "state": "candidate",
        "model_name": model_name,
        "checkpoint": str(checkpoint.resolve()),
        "selection_split": "validation",
        "validation_metrics": validation_metrics,
        "thresholds": thresholds,
        "threshold_source": "validation",
        "config_hash": config_hash(config),
        "git_commit": current_git_commit(),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _run_dir(config: dict[str, Any], model_name: str) -> Path:
    root = resolve_project_path(config["outputs"]["models_dir"], config.get("project_root"))
    return root / model_name / f"seed_{int(config['project']['seed'])}"


def _portable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "project_root"}


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise ImportError("PyTorch nao instalado. Rode: python -m pip install -e .[tcc2]") from exc
    return torch
