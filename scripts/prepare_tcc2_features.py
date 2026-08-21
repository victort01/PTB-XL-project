"""Extrai os atributos tabulares multilabel em 500 Hz."""

from __future__ import annotations

import argparse

from tcc_ecg.config import load_config
from tcc_ecg.multilabel_data import build_multilabel_feature_table
from tcc_ecg.tcc2_training import prepare_multilabel_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tcc2_multilabel.yaml")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    config["data"]["signal_frequency"] = int(config["data"]["classical_frequency"])
    if args.max_records is not None:
        config["features"]["max_records"] = args.max_records
    metadata = prepare_multilabel_metadata(config)
    output = build_multilabel_feature_table(metadata, config)
    print(f"Features multilabel: {output}")


if __name__ == "__main__":
    main()

