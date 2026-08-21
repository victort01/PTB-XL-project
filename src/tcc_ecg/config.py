"""Leitura de configuracao YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from tcc_ecg.paths import get_project_root, resolve_project_path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Carrega `configs/config.yaml` e aplica overrides simples por ambiente."""
    root = get_project_root()
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass
    config_path = resolve_project_path(path or "configs/config.yaml", root)
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuracao nao encontrado: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    config["project_root"] = str(root)

    env_data_dir = os.getenv("PTBXL_DATA_DIR")
    if env_data_dir:
        config.setdefault("data", {})["base_dir"] = env_data_dir

    return config
