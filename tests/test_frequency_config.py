import pytest

from tcc_ecg.data import get_records_dir_name, get_signal_filename_column, get_signal_frequency


def test_frequency_100_uses_low_resolution_records():
    config = {"data": {"signal_frequency": 100}}

    assert get_signal_frequency(config) == 100
    assert get_records_dir_name(config) == "records100"
    assert get_signal_filename_column(config) == "filename_lr"


def test_frequency_500_uses_high_resolution_records():
    config = {"data": {"signal_frequency": 500}}

    assert get_signal_frequency(config) == 500
    assert get_records_dir_name(config) == "records500"
    assert get_signal_filename_column(config) == "filename_hr"


def test_frequency_rejects_unsupported_values():
    config = {"data": {"signal_frequency": 250}}

    with pytest.raises(ValueError, match="100 ou 500"):
        get_signal_frequency(config)
