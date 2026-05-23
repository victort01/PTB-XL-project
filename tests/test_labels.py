import pandas as pd

from tcc_ecg.labels import build_multiclass_target, map_scp_to_superclasses, parse_scp_codes


def test_parse_scp_codes_from_string():
    parsed = parse_scp_codes("{'NORM': 100.0, 'IMI': 35.0}")

    assert parsed == {"NORM": 100.0, "IMI": 35.0}


def test_map_scp_to_superclasses_filters_diagnostic_codes():
    scp_statements = pd.DataFrame(
        {
            "diagnostic": [1, 1, 0],
            "diagnostic_class": ["NORM", "MI", "STTC"],
        },
        index=["NORM", "IMI", "NST_"],
    )

    mapped = map_scp_to_superclasses("{'NORM': 100.0, 'IMI': 35.0, 'NST_': 80.0}", scp_statements)

    assert mapped == {"NORM": 100.0, "MI": 35.0}


def test_strict_single_label_keeps_only_one_superclass():
    metadata = pd.DataFrame(
        {
            "scp_codes": [
                "{'NORM': 100.0}",
                "{'NORM': 100.0, 'IMI': 35.0}",
                "{'UNKNOWN': 20.0}",
            ]
        }
    )
    scp_statements = pd.DataFrame(
        {
            "diagnostic": [1, 1],
            "diagnostic_class": ["NORM", "MI"],
        },
        index=["NORM", "IMI"],
    )

    result = build_multiclass_target(metadata, scp_statements, strategy="strict_single_label")

    assert result.loc[0, "target"] == "NORM"
    assert pd.isna(result.loc[1, "target"])
    assert pd.isna(result.loc[2, "target"])
    assert result.loc[0, "target_id"] == 0
