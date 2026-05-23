import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from tcc_ecg.features import extract_signal_features
from tcc_ecg.preprocessing import build_preprocessor


def test_synthetic_features_pass_basic_model_pipeline():
    rng = np.random.default_rng(42)
    rows = []
    y = []
    for idx in range(12):
        label = idx % 3
        signal = rng.normal(loc=label * 0.2, scale=1.0, size=(120, 12))
        row = extract_signal_features(signal)
        row["age_clean"] = 50 + idx
        row["age_is_anon_90_plus"] = 0
        row["sex"] = idx % 2
        rows.append(row)
        y.append(label)

    X = pd.DataFrame(rows)
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(list(X.columns), scale=True)),
            ("model", LogisticRegression(max_iter=300)),
        ]
    )

    pipeline.fit(X, y)
    pred = pipeline.predict(X)

    assert pred.shape == (12,)
