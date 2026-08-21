"""Protocolo experimental e bloqueio metodologico do conjunto de teste."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tcc_ecg.paths import ensure_dir, resolve_project_path


class ProtocolError(RuntimeError):
    """Indica violacao de uma regra predefinida do protocolo experimental."""


def validate_tcc2_protocol(config: dict[str, Any]) -> None:
    """Valida as decisoes que nao podem mudar durante os experimentos."""
    if config.get("labels", {}).get("task") != "multilabel":
        raise ProtocolError("O protocolo do TCC II deve declarar labels.task=multilabel.")

    folds = config.get("folds", {})
    train = set(map(int, folds.get("train", [])))
    validation = set(map(int, folds.get("validation", [])))
    test = set(map(int, folds.get("test", [])))
    if train != set(range(1, 9)) or validation != {9} or test != {10}:
        raise ProtocolError("Use treino 1-8, validacao 9 e teste 10.")
    if train & validation or train & test or validation & test:
        raise ProtocolError("Os folds de treino, validacao e teste devem ser disjuntos.")

    protocol = config.get("protocol", {})
    if protocol.get("selection_split") != "validation":
        raise ProtocolError("A selecao de modelos deve usar somente a validacao.")
    if protocol.get("threshold_source") != "validation":
        raise ProtocolError("Thresholds multilabel devem ser determinados na validacao.")
    if not bool(protocol.get("lock_test", True)):
        raise ProtocolError("O bloqueio do fold 10 nao pode ser desativado no protocolo principal.")


def assign_official_split(folds: pd.Series, config: dict[str, Any]) -> pd.Series:
    """Mapeia ``strat_fold`` para treino, validacao ou teste."""
    mapping: dict[int, str] = {}
    for split in ("train", "validation", "test"):
        mapping.update({int(fold): split for fold in config["folds"][split]})
    return folds.map(mapping).astype("string")


def split_by_official_folds(
    records: pd.DataFrame,
    config: dict[str, Any],
    id_column: str = "ecg_id",
) -> dict[str, pd.DataFrame]:
    """Divide os registros e verifica sobreposicao entre identificadores."""
    validate_tcc2_protocol(config)
    if "strat_fold" not in records.columns:
        raise KeyError("A coluna strat_fold e obrigatoria.")
    frame = records.copy()
    frame["split"] = assign_official_split(frame["strat_fold"], config)
    if frame["split"].isna().any():
        unknown = sorted(frame.loc[frame["split"].isna(), "strat_fold"].unique().tolist())
        raise ProtocolError(f"Registros em folds nao configurados: {unknown}")
    splits = {name: frame.loc[frame["split"].eq(name)].copy() for name in ("train", "validation", "test")}
    _assert_disjoint_ids(splits, id_column)
    return splits


def _assert_disjoint_ids(splits: dict[str, pd.DataFrame], id_column: str) -> None:
    ids: dict[str, set[Any]] = {}
    for name, frame in splits.items():
        values = frame[id_column] if id_column in frame.columns else pd.Series(frame.index, index=frame.index)
        ids[name] = set(values.tolist())
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = ids[left] & ids[right]
        if overlap:
            raise ProtocolError(f"Sobreposicao entre {left} e {right}: {len(overlap)} registros.")


def write_split_manifest(
    records: pd.DataFrame,
    config: dict[str, Any],
    label_columns: list[str],
    stem: str = "ptbxl_multilabel_splits",
) -> dict[str, Path]:
    """Salva os identificadores e um resumo auditavel dos splits oficiais."""
    splits = split_by_official_folds(records, config)
    manifests_dir = resolve_project_path(config["outputs"]["manifests_dir"], config.get("project_root"))
    ensure_dir(manifests_dir)
    columns = [column for column in ["ecg_id", "strat_fold", "split", *label_columns] if column in records.columns or column == "split"]
    combined = pd.concat(splits.values(), ignore_index=True)[columns]
    csv_path = manifests_dir / f"{stem}.csv"
    summary_path = manifests_dir / f"{stem}.json"
    combined.to_csv(csv_path, index=False)
    summary = {
        "created_at": _utc_now(),
        "config_hash": config_hash(config),
        "git_commit": current_git_commit(),
        "counts": {name: int(len(frame)) for name, frame in splits.items()},
        "folds": config["folds"],
        "label_columns": label_columns,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return {"records": csv_path, "summary": summary_path}


def config_hash(config: dict[str, Any]) -> str:
    """Calcula hash estavel da configuracao sem o caminho local da raiz."""
    payload = {key: value for key, value in config.items() if key != "project_root"}
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def current_git_commit() -> str | None:
    """Retorna o commit atual quando executado dentro de um repositorio Git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def freeze_candidate(
    model_name: str,
    checkpoint_path: str | Path,
    validation_metrics: dict[str, Any],
    config: dict[str, Any],
    thresholds: list[float] | None = None,
    destination: str | Path | None = None,
) -> Path:
    """Congela um candidato escolhido exclusivamente pela validacao."""
    validate_tcc2_protocol(config)
    primary_metric = str(config["protocol"]["primary_metric"])
    if primary_metric not in validation_metrics:
        raise ProtocolError(f"Metrica primaria ausente na validacao: {primary_metric}")
    checkpoint = resolve_project_path(checkpoint_path, config.get("project_root"))
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {checkpoint}")
    manifests_dir = resolve_project_path(config["outputs"]["manifests_dir"], config.get("project_root"))
    ensure_dir(manifests_dir)
    output = resolve_project_path(destination, config.get("project_root")) if destination else manifests_dir / f"frozen_{model_name}.json"
    manifest = {
        "schema_version": 1,
        "state": "frozen",
        "model_name": model_name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "selection_split": "validation",
        "primary_metric": primary_metric,
        "validation_metrics": validation_metrics,
        "threshold_source": "validation",
        "thresholds": thresholds,
        "config_hash": config_hash(config),
        "git_commit": current_git_commit(),
        "frozen_at": _utc_now(),
        "test_evaluated": False,
    }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return output


def authorize_test_evaluation(
    manifest_path: str | Path,
    config: dict[str, Any],
    evaluate_test: bool,
) -> dict[str, Any]:
    """Libera o fold 10 apenas para um manifesto congelado e flag explicita."""
    if not evaluate_test:
        raise ProtocolError("A avaliacao do teste exige a flag explicita --evaluate-test.")
    validate_tcc2_protocol(config)
    path = resolve_project_path(manifest_path, config.get("project_root"))
    if not path.exists():
        raise FileNotFoundError(f"Manifesto congelado nao encontrado: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("state") != "frozen" or manifest.get("selection_split") != "validation":
        raise ProtocolError("O manifesto nao representa um modelo congelado pela validacao.")
    if manifest.get("config_hash") != config_hash(config):
        raise ProtocolError("A configuracao atual difere da configuracao congelada.")
    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.exists() or file_sha256(checkpoint) != manifest.get("checkpoint_sha256"):
        raise ProtocolError("O checkpoint congelado esta ausente ou foi modificado.")
    if manifest.get("test_evaluated") and not bool(config["protocol"].get("allow_repeat_test_evaluation", False)):
        raise ProtocolError("O teste ja foi avaliado para este manifesto.")
    return manifest


def mark_test_evaluated(manifest_path: str | Path, metrics_path: str | Path) -> None:
    """Registra a avaliacao depois que as metricas foram salvas com sucesso."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["test_evaluated"] = True
    manifest["test_evaluated_at"] = _utc_now()
    manifest["test_metrics_path"] = str(Path(metrics_path))
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()

