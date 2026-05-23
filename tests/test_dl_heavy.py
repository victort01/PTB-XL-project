import numpy as np

from tcc_ecg.dl_data import compute_class_weights, sample_weights_for_labels
from tcc_ecg.dl_models import build_heavy_model, count_parameters
from tcc_ecg.dl_training import get_deep_learning_heavy_config


def test_heavy_models_forward_pass():
    import torch

    config = {
        "deep_learning_heavy": {
            "inceptiontime_deep": {
                "base_channels": 16,
                "block_channels": [16, 16],
                "kernel_sizes": [5, 9],
                "bottleneck_channels": 8,
                "dropout": 0.1,
            },
            "resnet1d_se": {
                "base_filters": 16,
                "stage_channels": [16, 32],
                "kernel_size": 5,
                "dropout": 0.1,
                "se_reduction": 4,
            },
        }
    }
    heavy_config = get_deep_learning_heavy_config(config)
    x = torch.randn(2, 12, 500)
    for architecture in ["inceptiontime_deep", "resnet1d_se"]:
        model = build_heavy_model(architecture, heavy_config, input_channels=12, n_classes=5)
        logits = model(x)
        assert logits.shape == (2, 5)
        assert count_parameters(model) > 0


def test_class_weights_use_training_labels_only():
    y_train = np.array([0, 0, 0, 1, 2], dtype=int)
    weights = compute_class_weights(y_train, n_classes=3)
    sample_weights = sample_weights_for_labels(y_train, n_classes=3)
    assert weights[0] < weights[1]
    assert weights[1] == weights[2]
    assert sample_weights.shape == y_train.shape
