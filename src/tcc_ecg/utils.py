"""Funcoes utilitarias compartilhadas."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcc_ecg.paths import ensure_dir


def setup_logging(level: int = logging.INFO) -> None:
    """Configura logs simples para notebooks e scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_random_seed(seed: int) -> None:
    """Define semente para bibliotecas usadas no projeto."""
    random.seed(seed)
    np.random.seed(seed)


def save_table(df: pd.DataFrame, csv_path: str | Path, tex_path: str | Path | None = None) -> None:
    """Salva uma tabela em CSV e, opcionalmente, em LaTeX."""
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=False)
    if tex_path is not None:
        tex_path = Path(tex_path)
        ensure_dir(tex_path.parent)
        tex_path.write_text(df.to_latex(index=False, escape=True), encoding="utf-8")


def as_list(value: Any) -> list[Any]:
    """Normaliza valores escalares ou nulos para lista."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
