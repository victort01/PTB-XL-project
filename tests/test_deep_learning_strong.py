import numpy as np
import pytest

from tcc_ecg.deep_learning_strong import apply_ecg_augmentations, build_inceptiontime1d_model


def test_inceptiontime1d_forward_pass_shape():
    torch = pytest.importorskip("torch")
    model = build_inceptiontime1d_model(
        input_channels=12,
        n_classes=5,
        base_channels=16,
        block_channels=[16, 16, 32],
        kernel_sizes=[9, 19, 39],
        bottleneck_channels=8,
        dropout=0.1,
    )
    x = torch.randn(2, 12, 500)

    with torch.no_grad():
        y = model(x)

    assert tuple(y.shape) == (2, 5)


def test_ecg_augmentations_preserve_shape():
    signal = np.ones((100, 12), dtype="float32")
    augmented = apply_ecg_augmentations(
        signal,
        {
            "noise_std": 0.01,
            "scale_min": 0.95,
            "scale_max": 1.05,
            "time_shift": 5,
            "channel_dropout_prob": 0.1,
        },
    )

    assert augmented.shape == signal.shape
    assert augmented.dtype == np.float32
