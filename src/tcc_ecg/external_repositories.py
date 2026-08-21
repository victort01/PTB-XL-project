"""Obtencao reproduzivel de implementacoes externas em commits pinados."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from tcc_ecg.paths import ensure_dir, get_project_root, resolve_project_path


def load_repository_specs(path: str | Path = "configs/external_repositories.yaml") -> dict[str, dict[str, Any]]:
    config_path = resolve_project_path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return dict(payload.get("repositories", {}))


def external_repositories_root() -> Path:
    configured = os.getenv("TCC2_EXTERNAL_REPOS_DIR", "external")
    return resolve_project_path(configured)


def fetch_repository(name: str, spec: dict[str, Any], base_dir: str | Path | None = None) -> Path:
    """Clona ou atualiza um repositorio e posiciona exatamente no commit registrado."""
    root = Path(base_dir).resolve() if base_dir else external_repositories_root().resolve()
    ensure_dir(root)
    destination_name = Path(spec["destination"]).name
    destination = (root / destination_name).resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"Destino externo inseguro para {name}: {destination}")

    if not destination.exists():
        subprocess.run(["git", "clone", spec["url"], str(destination)], check=True)
    elif not (destination / ".git").exists():
        raise ValueError(f"Destino existe, mas nao e repositorio Git: {destination}")
    subprocess.run(["git", "-C", str(destination), "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", spec["commit"]], check=True)
    actual = repository_commit(destination)
    if actual != spec["commit"]:
        raise RuntimeError(f"Commit inesperado em {name}: {actual}")
    return destination


def repository_commit(path: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def repository_inventory(
    specs: dict[str, dict[str, Any]],
    base_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Compara os checkouts locais com os commits esperados."""
    root = Path(base_dir).resolve() if base_dir else external_repositories_root().resolve()
    rows = []
    for name, spec in specs.items():
        destination = root / Path(spec["destination"]).name
        actual = repository_commit(destination) if destination.exists() else None
        rows.append(
            {
                "name": name,
                "url": spec["url"],
                "expected_commit": spec["commit"],
                "actual_commit": actual,
                "license": spec.get("license"),
                "role": spec.get("role"),
                "frequency": spec.get("frequency"),
                "path": str(destination),
                "status": "ready" if actual == spec["commit"] else "not_fetched_or_mismatch",
            }
        )
    return pd.DataFrame(rows)


def write_repository_inventory(frame: pd.DataFrame) -> Path:
    output = get_project_root() / "reports" / "tables" / "tcc2" / "external_repository_inventory.csv"
    ensure_dir(output.parent)
    frame.to_csv(output, index=False)
    return output

