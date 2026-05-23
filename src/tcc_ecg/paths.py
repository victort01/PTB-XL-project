"""Utilitarios para caminhos relativos ao projeto."""

from __future__ import annotations

from pathlib import Path


def get_project_root(start: str | Path | None = None) -> Path:
    """Retorna a raiz do projeto procurando por `pyproject.toml`."""
    candidates = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.extend([Path.cwd().resolve(), Path(__file__).resolve()])

    for candidate in candidates:
        current = candidate if candidate.is_dir() else candidate.parent
        for parent in [current, *current.parents]:
            if (parent / "pyproject.toml").exists():
                return parent
    return Path.cwd().resolve()


def resolve_project_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Resolve caminho relativo a partir da raiz do projeto."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return (Path(root).resolve() if root else get_project_root()) / path_obj


def ensure_dir(path: str | Path) -> Path:
    """Cria um diretorio, se necessario, e retorna o Path."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj
