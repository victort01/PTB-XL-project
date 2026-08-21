import json

import pandas as pd
import pytest

from tcc_ecg.protocol import (
    ProtocolError,
    authorize_test_evaluation,
    freeze_candidate,
    mark_test_evaluated,
    split_by_official_folds,
    validate_tcc2_protocol,
)


def protocol_config(tmp_path):
    return {
        "project": {"seed": 42},
        "project_root": str(tmp_path),
        "labels": {"task": "multilabel", "superclasses": ["NORM", "MI", "STTC", "CD", "HYP"]},
        "folds": {"train": list(range(1, 9)), "validation": [9], "test": [10]},
        "protocol": {
            "selection_split": "validation",
            "primary_metric": "macro_auroc",
            "threshold_source": "validation",
            "lock_test": True,
            "allow_repeat_test_evaluation": False,
        },
        "outputs": {"manifests_dir": "reports/manifests/tcc2"},
    }


def test_official_splits_are_disjoint(tmp_path):
    config = protocol_config(tmp_path)
    records = pd.DataFrame({"ecg_id": range(1, 11), "strat_fold": range(1, 11)})
    splits = split_by_official_folds(records, config)
    assert set(splits["train"]["ecg_id"]) == set(range(1, 9))
    assert splits["validation"]["ecg_id"].tolist() == [9]
    assert splits["test"]["ecg_id"].tolist() == [10]


def test_invalid_fold_protocol_is_rejected(tmp_path):
    config = protocol_config(tmp_path)
    config["folds"]["test"] = [9]
    with pytest.raises(ProtocolError):
        validate_tcc2_protocol(config)


def test_test_requires_frozen_manifest_and_is_single_use(tmp_path):
    config = protocol_config(tmp_path)
    checkpoint = tmp_path / "model.bin"
    checkpoint.write_bytes(b"checkpoint")
    manifest = freeze_candidate(
        "model",
        checkpoint,
        {"macro_auroc": 0.7},
        config,
        thresholds=[0.5] * 5,
    )
    with pytest.raises(ProtocolError):
        authorize_test_evaluation(manifest, config, evaluate_test=False)
    authorized = authorize_test_evaluation(manifest, config, evaluate_test=True)
    assert authorized["selection_split"] == "validation"
    mark_test_evaluated(manifest, tmp_path / "metrics.csv")
    assert json.loads(manifest.read_text())["test_evaluated"] is True
    with pytest.raises(ProtocolError):
        authorize_test_evaluation(manifest, config, evaluate_test=True)

