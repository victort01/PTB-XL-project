import pytest

from tcc_ecg.config import load_config
from tcc_ecg.multilabel_models import build_tcn_model, model_inventory_frame


def test_tcn_multilabel_forward_shape():
    torch = pytest.importorskip("torch")
    model = build_tcn_model(12, 5, [8, 16], kernel_size=5, dropout=0.1)
    with torch.no_grad():
        output = model(torch.randn(2, 12, 256))
    assert tuple(output.shape) == (2, 5)


def test_required_model_families_are_registered():
    inventory = model_inventory_frame()
    names = set(inventory["name"])
    assert {"helme_inception1d", "helme_xresnet1d101", "tcn", "s4", "ecg_jepa"} <= names


def test_tcc2_config_loads_and_declares_multilabel():
    config = load_config("configs/tcc2_multilabel.yaml")
    assert config["labels"]["task"] == "multilabel"
    assert config["protocol"]["selection_split"] == "validation"

