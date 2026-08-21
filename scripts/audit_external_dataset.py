"""Audita candidatos externos sem treinar ou avaliar modelos."""

from __future__ import annotations

import argparse

from tcc_ecg.config import load_config
from tcc_ecg.external_validation import audit_external_candidates, save_external_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tcc2_multilabel.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    candidates, diagnoses = audit_external_candidates(config)
    outputs = save_external_audit(candidates, diagnoses, config)
    print(candidates.to_string(index=False))
    print(f"Tabela de candidatos: {outputs['candidates']}")
    print(f"Codigos diagnosticos: {outputs['diagnoses']}")


if __name__ == "__main__":
    main()

