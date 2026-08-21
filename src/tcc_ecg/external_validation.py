"""Auditoria estrutural de candidatos a validacao externa."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from tcc_ecg.paths import ensure_dir, resolve_project_path
from tcc_ecg.utils import save_table


def load_external_candidates(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(resolve_project_path(path).read_text(encoding="utf-8")) or {}
    return dict(payload.get("candidates", {}))


def audit_external_candidates(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gera matriz de decisao e inventario de codigos sem executar inferencia."""
    candidates_path = config["external_validation"]["candidates_config"]
    candidates = load_external_candidates(candidates_path)
    rows: list[dict[str, Any]] = []
    diagnosis_rows: list[dict[str, Any]] = []
    for name, candidate in candidates.items():
        configured_path = os.getenv(candidate["path_env"])
        root = resolve_project_path(configured_path, config.get("project_root")) if configured_path else None
        audit = audit_wfdb_directory(root) if root and root.exists() else _empty_audit()
        status = "available" if root and root.exists() else "not_available"
        rows.append(
            {
                "candidate": name,
                "display_name": candidate["display_name"],
                "recommendation": candidate["recommendation"],
                "role": candidate["role"],
                "path_env": candidate["path_env"],
                "status": status,
                "records_found": audit["records_found"],
                "observed_leads": audit["observed_leads"],
                "observed_frequency": audit["observed_frequency"],
                "expected_leads": candidate.get("expected_leads"),
                "expected_frequency": candidate.get("expected_frequency"),
                "label_system": candidate.get("label_system"),
                "contamination_check": candidate.get("contamination_check"),
                "ready_for_external_test": bool(
                    status == "available"
                    and audit["records_found"] > 0
                    and audit["observed_leads"] == candidate.get("expected_leads")
                ),
            }
        )
        for code, count in audit["diagnosis_counts"].items():
            diagnosis_rows.append({"candidate": name, "diagnosis_code": code, "count": count})
    return pd.DataFrame(rows), pd.DataFrame(diagnosis_rows)


def audit_wfdb_directory(root: str | Path | None, max_headers: int | None = None) -> dict[str, Any]:
    """Le cabecalhos WFDB e resume frequencia, derivacoes e codigos diagnosticos."""
    if root is None:
        return _empty_audit()
    headers = sorted(Path(root).rglob("*.hea"))
    if max_headers is not None:
        headers = headers[: int(max_headers)]
    lead_counts: Counter[int] = Counter()
    frequencies: Counter[float] = Counter()
    diagnoses: Counter[str] = Counter()
    for header in headers:
        lines = header.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines:
            continue
        tokens = lines[0].split()
        if len(tokens) >= 3:
            try:
                lead_counts[int(tokens[1])] += 1
                frequencies[float(tokens[2].split("/")[0])] += 1
            except ValueError:
                pass
        for line in lines:
            if line.lower().startswith("#dx:"):
                diagnoses.update(code.strip() for code in line.split(":", 1)[1].split(",") if code.strip())
    return {
        "records_found": len(headers),
        "observed_leads": lead_counts.most_common(1)[0][0] if lead_counts else None,
        "observed_frequency": frequencies.most_common(1)[0][0] if frequencies else None,
        "diagnosis_counts": dict(diagnoses),
    }


def save_external_audit(
    candidate_frame: pd.DataFrame,
    diagnosis_frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Path]:
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    ensure_dir(tables_dir)
    candidates_csv = tables_dir / "external_dataset_audit.csv"
    diagnoses_csv = tables_dir / "external_diagnosis_codes.csv"
    save_table(candidate_frame, candidates_csv, tables_dir / "external_dataset_audit.tex")
    diagnosis_frame.to_csv(diagnoses_csv, index=False)
    return {"candidates": candidates_csv, "diagnoses": diagnoses_csv}


def _empty_audit() -> dict[str, Any]:
    return {
        "records_found": 0,
        "observed_leads": None,
        "observed_frequency": None,
        "diagnosis_counts": {},
    }

