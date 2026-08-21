"""Treina uma arquitetura original de Helme sem gerar predicoes do fold 10.

Execute este script dentro do ambiente Conda fornecido pelo repositorio de
Helme. A arquitetura e a classe de treino continuam sendo as originais; apenas
a orquestracao e separada para preservar o teste durante a selecao.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MODEL_CONFIG_NAMES = {
    "helme_inception1d": "conf_fastai_inception1d",
    "helme_xresnet1d101": "conf_fastai_xresnet1d101",
    "helme_resnet1d_wang": "conf_fastai_resnet1d_wang",
    "helme_fcn_wang": "conf_fastai_fcn_wang",
    "helme_lstm": "conf_fastai_lstm",
    "helme_lstm_bidir": "conf_fastai_lstm_bidir",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="external/ecg_ptbxl_benchmarking")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=sorted(MODEL_CONFIG_NAMES), required=True)
    parser.add_argument("--task", default="superdiagnostic")
    parser.add_argument("--frequency", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Path(args.repository).resolve()
    code_dir = repository / "code"
    if not code_dir.exists():
        raise FileNotFoundError(f"Checkout de Helme nao encontrado: {repository}")
    sys.path.insert(0, str(code_dir))

    from models.fastai_model import fastai_model  # noqa: PLC0415
    from utils import utils  # noqa: PLC0415

    from configs import fastai_configs  # noqa: PLC0415

    data_dir = str(Path(args.data_dir).resolve()) + "/"
    run_dir = Path(args.output_dir).resolve() / args.model
    model_dir = run_dir / "model"
    data_output = run_dir / "data"
    model_dir.mkdir(parents=True, exist_ok=True)
    data_output.mkdir(parents=True, exist_ok=True)

    data, raw_labels = utils.load_dataset(data_dir, args.frequency)
    labels = utils.compute_label_aggregations(raw_labels, data_dir, args.task)
    data, labels, targets, _ = utils.select_data(
        data, labels, args.task, 0, str(data_output) + "/"
    )
    train_mask = labels.strat_fold <= 8
    validation_mask = labels.strat_fold == 9
    x_train, y_train = data[train_mask], targets[train_mask]
    x_validation, y_validation = data[validation_mask], targets[validation_mask]
    x_train, x_validation, _ = utils.preprocess_signals(
        x_train, x_validation, [], str(data_output) + "/"
    )

    config = getattr(fastai_configs, MODEL_CONFIG_NAMES[args.model])
    original_name = config["modelname"]
    model = fastai_model(
        original_name,
        y_train.shape[1],
        args.frequency,
        str(model_dir),
        x_train[0].shape,
        **config["parameters"],
    )
    model.fit(x_train, y_train, x_validation, y_validation)
    validation_scores = model.predict(x_validation)
    np.save(run_dir / "y_validation.npy", y_validation)
    np.save(run_dir / "y_validation_score.npy", validation_scores)
    metrics = utils.evaluate_experiment(y_validation, validation_scores)
    validation_metrics = {
        "macro_auroc": float(metrics["macro_auc"]),
        "fmax": float(metrics["Fmax"]),
        "model": args.model,
        "split": "validation",
    }
    metrics_path = run_dir / "validation_metrics.json"
    metrics_path.write_text(json.dumps(validation_metrics, indent=2), encoding="utf-8")
    checkpoint = model_dir / "models" / f"{original_name}.pth"
    candidate = {
        "state": "candidate",
        "model_name": args.model,
        "checkpoint": str(checkpoint),
        "selection_split": "validation",
        "validation_metrics": validation_metrics,
        "thresholds": None,
        "threshold_source": "validation",
        "external_source": "helme_benchmark",
    }
    candidate_path = run_dir / "candidate_manifest.json"
    candidate_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    print(f"Validacao: {metrics_path}")
    print(f"Candidato: {candidate_path}")
    print("O fold 10 nao foi indexado nem predito por este script.")


if __name__ == "__main__":
    main()

