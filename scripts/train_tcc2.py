"""CLI unificada para preparar, treinar e congelar experimentos do TCC II."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcc_ecg.config import load_config
from tcc_ecg.multilabel_models import get_model_spec
from tcc_ecg.protocol import freeze_candidate
from tcc_ecg.tcc2_training import (
    evaluate_frozen_test,
    smoke_tcc2,
    train_classical_validation,
    train_tcn_validation,
)
from tcc_ecg.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Executa verificacoes sinteticas curtas.")
    smoke.add_argument("--config", default="configs/tcc2_multilabel.yaml")

    train = subparsers.add_parser("train", help="Treina somente com treino e validacao.")
    train.add_argument("--config", default="configs/tcc2_multilabel.yaml")
    train.add_argument("--model", required=True)
    train.add_argument("--features", default=None)

    freeze = subparsers.add_parser("freeze", help="Congela um candidato escolhido pela validacao.")
    freeze.add_argument("--config", default="configs/tcc2_multilabel.yaml")
    freeze.add_argument("--candidate-manifest", required=True)

    test = subparsers.add_parser("evaluate-test", help="Avalia um modelo previamente congelado.")
    test.add_argument("--config", default="configs/tcc2_multilabel.yaml")
    test.add_argument("--frozen-manifest", required=True)
    test.add_argument("--features", default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    if args.command == "smoke":
        frame = smoke_tcc2(config)
        print(frame.to_string(index=False))
        return
    if args.command == "train":
        spec = get_model_spec(args.model)
        if spec.backend == "external":
            raise SystemExit(
                f"{args.model} usa a implementacao externa pinada '{spec.source}'. "
                "Rode make fetch-external e consulte docs/tcc2/EXECUTION_GUIDE.md."
            )
        result = train_tcn_validation(config) if args.model == "tcn" else train_classical_validation(config, args.model, args.features)
        print(f"Candidato salvo em: {result['candidate_manifest']}")
        print("Nenhum dado do fold 10 foi avaliado.")
        return
    if args.command == "evaluate-test":
        result = evaluate_frozen_test(config, args.frozen_manifest, args.features)
        print(f"Metricas de teste salvas em: {result['metrics_path']}")
        return

    candidate_path = Path(args.candidate_manifest)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    frozen = freeze_candidate(
        model_name=candidate["model_name"],
        checkpoint_path=candidate["checkpoint"],
        validation_metrics=candidate["validation_metrics"],
        thresholds=candidate["thresholds"],
        config=config,
    )
    print(f"Manifesto congelado: {frozen}")


if __name__ == "__main__":
    main()
