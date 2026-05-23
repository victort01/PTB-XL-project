"""Compara resultados salvos para execucoes em 100 Hz e 500 Hz.

Uso:
    python scripts/compare_frequencies.py

O script procura por arquivos `model_metrics_100hz.csv` e
`model_metrics_500hz.csv` em `reports/tables`. Esses arquivos são gerados
automaticamente pelo notebook de treinamento ao usar `signal_frequency`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcc_ecg.config import load_config
from tcc_ecg.paths import resolve_project_path
from tcc_ecg.utils import save_table


def build_frequency_comparison(tables_dir: Path) -> pd.DataFrame:
    rows = []
    for frequency in (100, 500):
        metrics_path = tables_dir / f"model_metrics_{frequency}hz.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        test_metrics = metrics.loc[metrics["split"].eq("test")].copy()
        if test_metrics.empty:
            continue
        best_f1 = test_metrics.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]
        best_acc = test_metrics.sort_values(["accuracy", "f1_macro"], ascending=False).iloc[0]
        rows.append(
            {
                "signal_frequency": frequency,
                "n_models_evaluated": int(test_metrics["model"].nunique()),
                "best_model_by_f1_macro": best_f1["model"],
                "best_f1_macro": best_f1["f1_macro"],
                "accuracy_of_best_f1_model": best_f1["accuracy"],
                "best_model_by_accuracy": best_acc["model"],
                "best_accuracy": best_acc["accuracy"],
                "f1_macro_of_best_accuracy_model": best_acc["f1_macro"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    tables_dir = resolve_project_path(config["outputs"]["tables_dir"], config.get("project_root"))
    comparison = build_frequency_comparison(tables_dir)
    if comparison.empty:
        raise SystemExit(
            "Nenhum arquivo model_metrics_100hz.csv ou model_metrics_500hz.csv encontrado. "
            "Execute o notebook 04 para cada frequencia antes de comparar."
        )
    save_table(
        comparison,
        tables_dir / "frequency_comparison.csv",
        tables_dir / "frequency_comparison.tex",
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
