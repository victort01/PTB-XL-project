"""Balanceamento de classes aplicado apenas ao conjunto de treino."""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE


def safe_smote_k_neighbors(y, requested_k: int = 5) -> int:
    """Ajusta k do SMOTE para a menor classe do treino."""
    _, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min())
    return max(1, min(requested_k, min_count - 1))


def make_smote(y_train, random_state: int, requested_k: int = 5) -> SMOTE | None:
    """Retorna SMOTE seguro ou None quando alguma classe tem menos de 2 exemplos."""
    _, counts = np.unique(y_train, return_counts=True)
    if int(counts.min()) < 2:
        return None
    return SMOTE(
        random_state=random_state,
        k_neighbors=safe_smote_k_neighbors(y_train, requested_k),
    )
