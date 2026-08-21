import numpy as np
import pandas as pd

from tcc_ecg.multilabel import (
    build_multilabel_targets,
    final_multilabel_records,
    multilabel_target_matrix,
)


def test_multilabel_keeps_records_with_multiple_superclasses():
    metadata = pd.DataFrame(
        {
            "scp_codes": [
                "{'NORM': 100.0}",
                "{'NORM': 100.0, 'IMI': 80.0}",
                "{'UNKNOWN': 20.0}",
            ]
        },
        index=pd.Index([1, 2, 3], name="ecg_id"),
    )
    statements = pd.DataFrame(
        {"diagnostic": [1, 1], "diagnostic_class": ["NORM", "MI"]},
        index=["NORM", "IMI"],
    )

    labeled = build_multilabel_targets(metadata, statements, ["NORM", "MI", "STTC", "CD", "HYP"])
    kept = final_multilabel_records(labeled)
    matrix = multilabel_target_matrix(kept, ["NORM", "MI", "STTC", "CD", "HYP"])

    assert kept.index.tolist() == [1, 2]
    assert np.array_equal(matrix[0], [1, 0, 0, 0, 0])
    assert np.array_equal(matrix[1], [1, 1, 0, 0, 0])

