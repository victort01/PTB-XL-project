import numpy as np
import pytest

from tcc_ecg.deep_learning import build_resnet1d_model, compute_train_channel_normalization


def test_resnet1d_forward_pass_shape():
    torch = pytest.importorskip("torch")
    model = build_resnet1d_model(
        input_channels=12,
        n_classes=5,
        base_filters=8,
        kernel_size=5,
        dropout=0.1,
    )
    x = torch.randn(2, 12, 500)

    with torch.no_grad():
        y = model(x)

    assert tuple(y.shape) == (2, 5)


def test_channel_normalization_uses_only_selected_indices(tmp_path):
    signals_path = tmp_path / "signals.npy"
    signals = np.lib.format.open_memmap(
        signals_path,
        mode="w+",
        dtype="float32",
        shape=(3, 4, 2),
    )
    signals[0] = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype="float32")
    signals[1] = np.array([[5, 50], [6, 60], [7, 70], [8, 80]], dtype="float32")
    signals[2] = np.array([[100, 1000], [100, 1000], [100, 1000], [100, 1000]], dtype="float32")
    del signals

    mean, std = compute_train_channel_normalization(signals_path, np.array([0, 1]), batch_size=1)

    train_values = np.load(signals_path, mmap_mode="r")[[0, 1]]
    np.testing.assert_allclose(mean, train_values.mean(axis=(0, 1)), rtol=1e-6)
    np.testing.assert_allclose(std, train_values.std(axis=(0, 1)), rtol=1e-6)
