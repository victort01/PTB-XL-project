"""Arquiteturas PyTorch 1D mais robustas para sinais ECG brutos."""

from __future__ import annotations

from typing import Any


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depende do ambiente local
        raise ImportError("PyTorch nao esta instalado. Rode: python -m pip install -e .[dl]") from exc
    return torch


def count_parameters(model) -> int:
    """Conta parametros treinaveis do modelo."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class HeavyInceptionTime1D:
    """Factory para InceptionTime 1D mais largo/profundo que a versao forte anterior."""

    def __new__(
        cls,
        input_channels: int,
        n_classes: int,
        base_channels: int,
        block_channels: list[int],
        kernel_sizes: list[int],
        bottleneck_channels: int,
        dropout: float,
    ):
        torch = _import_torch()
        nn = torch.nn

        class InceptionBlock1D(nn.Module):
            def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
                super().__init__()
                branch_channels = max(out_channels // (len(kernel_sizes) + 1), 8)
                bottleneck = min(int(bottleneck_channels), in_channels)
                self.reduce = nn.Conv1d(in_channels, bottleneck, kernel_size=1, bias=False)
                self.branches = nn.ModuleList(
                    [
                        nn.Conv1d(
                            bottleneck,
                            branch_channels,
                            kernel_size=int(kernel),
                            stride=stride,
                            padding=int(kernel) // 2,
                            bias=False,
                        )
                        for kernel in kernel_sizes
                    ]
                )
                self.pool_branch = nn.Sequential(
                    nn.MaxPool1d(kernel_size=3, stride=stride, padding=1),
                    nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
                )
                concat_channels = branch_channels * (len(kernel_sizes) + 1)
                self.proj = nn.Sequential(
                    nn.BatchNorm1d(concat_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(concat_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_channels),
                )
                self.shortcut = (
                    nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(out_channels),
                    )
                    if stride != 1 or in_channels != out_channels
                    else nn.Identity()
                )
                self.act = nn.GELU()

            def forward(self, x):
                reduced = self.reduce(x)
                out = torch.cat([branch(reduced) for branch in self.branches] + [self.pool_branch(x)], dim=1)
                return self.act(self.proj(out) + self.shortcut(x))

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv1d(input_channels, base_channels, kernel_size=9, stride=2, padding=4, bias=False),
                    nn.BatchNorm1d(base_channels),
                    nn.GELU(),
                    nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                )
                blocks = []
                in_channels = base_channels
                for idx, channels in enumerate(block_channels):
                    stride = 2 if idx in {2, 5} else 1
                    blocks.append(InceptionBlock1D(in_channels, int(channels), stride=stride))
                    in_channels = int(channels)
                self.blocks = nn.Sequential(*blocks)
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Dropout(dropout),
                    nn.Linear(in_channels, n_classes),
                )

            def forward(self, x):
                return self.head(self.blocks(self.stem(x)))

        return Model()


class ResNet1DSE:
    """Factory para ResNet1D com Squeeze-and-Excitation."""

    def __new__(
        cls,
        input_channels: int,
        n_classes: int,
        base_filters: int,
        stage_channels: list[int],
        kernel_size: int,
        dropout: float,
        se_reduction: int,
    ):
        torch = _import_torch()
        nn = torch.nn

        class SEBlock1D(nn.Module):
            def __init__(self, channels: int) -> None:
                super().__init__()
                hidden = max(channels // int(se_reduction), 4)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(channels, hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, channels),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                weights = self.fc(self.pool(x)).unsqueeze(-1)
                return x * weights

        class ResidualSEBlock1D(nn.Module):
            def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
                super().__init__()
                padding = kernel_size // 2
                self.main = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
                    nn.BatchNorm1d(out_channels),
                    SEBlock1D(out_channels),
                )
                self.shortcut = (
                    nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(out_channels),
                    )
                    if stride != 1 or in_channels != out_channels
                    else nn.Identity()
                )
                self.act = nn.GELU()

            def forward(self, x):
                return self.act(self.main(x) + self.shortcut(x))

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv1d(input_channels, base_filters, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, bias=False),
                    nn.BatchNorm1d(base_filters),
                    nn.GELU(),
                    nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                )
                blocks = []
                in_channels = base_filters
                for channels in stage_channels:
                    channels = int(channels)
                    stride = 2 if channels > in_channels else 1
                    blocks.append(ResidualSEBlock1D(in_channels, channels, stride=stride))
                    in_channels = channels
                self.blocks = nn.Sequential(*blocks)
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Dropout(dropout),
                    nn.Linear(in_channels, n_classes),
                )

            def forward(self, x):
                return self.head(self.blocks(self.stem(x)))

        return Model()


def build_heavy_model(
    architecture: str,
    heavy_config: dict[str, Any],
    input_channels: int = 12,
    n_classes: int = 5,
):
    """Constroi uma arquitetura pesada configuravel, sem usar Transformer."""
    if architecture == "inceptiontime_deep":
        cfg = heavy_config["inceptiontime_deep"]
        return HeavyInceptionTime1D(
            input_channels=input_channels,
            n_classes=n_classes,
            base_channels=int(cfg["base_channels"]),
            block_channels=[int(item) for item in cfg["block_channels"]],
            kernel_sizes=[int(item) for item in cfg["kernel_sizes"]],
            bottleneck_channels=int(cfg["bottleneck_channels"]),
            dropout=float(cfg["dropout"]),
        )
    if architecture == "resnet1d_se":
        cfg = heavy_config["resnet1d_se"]
        return ResNet1DSE(
            input_channels=input_channels,
            n_classes=n_classes,
            base_filters=int(cfg["base_filters"]),
            stage_channels=[int(item) for item in cfg["stage_channels"]],
            kernel_size=int(cfg["kernel_size"]),
            dropout=float(cfg["dropout"]),
            se_reduction=int(cfg["se_reduction"]),
        )
    raise ValueError(f"Arquitetura heavy desconhecida: {architecture}")
