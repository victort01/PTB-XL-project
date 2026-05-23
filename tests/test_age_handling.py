import numpy as np
import pandas as pd

from tcc_ecg.data import add_age_features


def test_age_300_becomes_nan_and_flagged():
    df = pd.DataFrame({"age": [45, 300, np.nan]})

    result = add_age_features(df)

    assert result.loc[0, "age_clean"] == 45
    assert np.isnan(result.loc[1, "age_clean"])
    assert bool(result.loc[1, "age_is_anon_90_plus"]) is True
    assert bool(result.loc[0, "age_is_anon_90_plus"]) is False
