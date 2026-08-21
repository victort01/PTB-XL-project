import numpy as np

from tcc_ecg.multilabel_evaluation import (
    compute_multilabel_metrics,
    multilabel_report_frame,
    optimize_thresholds_on_validation,
)


def test_multilabel_metrics_and_thresholds_have_expected_shape():
    y_true = np.array(
        [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]],
        dtype="int8",
    )
    y_score = np.array(
        [[0.9, 0.2, 0.1], [0.2, 0.8, 0.2], [0.8, 0.7, 0.1], [0.1, 0.3, 0.9], [0.7, 0.2, 0.8], [0.2, 0.8, 0.7]],
        dtype="float32",
    )
    thresholds = optimize_thresholds_on_validation(y_true, y_score)
    metrics = compute_multilabel_metrics(y_true, y_score, thresholds)
    report = multilabel_report_frame(y_true, y_score, ["A", "B", "C"], thresholds)
    assert thresholds.shape == (3,)
    assert metrics["macro_auroc"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert report.shape[0] == 3

