"""Model definitions for EEG Task 1 classification."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """Simple MLP baseline over flattened EEG [B, 122, 500]."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 122,
        num_timesteps: int = 500,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        input_dim = num_channels * num_timesteps

        layers: list[nn.Module] = [nn.Flatten()]
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, int(hidden_dim)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            prev_dim = int(hidden_dim)

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalCNNBaseline(nn.Module):
    """Lightweight temporal CNN baseline over EEG [B, C, T].

    Shape flow:
    - input: [B, 122, 500]
    - conv1/relu/pool: [B, 128, 250]
    - conv2/relu/pool: [B, 256, 125]
    - global average pool over time: [B, 256]
    - classifier: [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 122,
        conv1_channels: int = 128,
        conv2_channels: int = 256,
        conv1_kernel_size: int = 9,
        conv2_kernel_size: int = 7,
        pool_kernel_size: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=conv1_channels,
                kernel_size=conv1_kernel_size,
                padding=conv1_kernel_size // 2,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size),
            nn.Conv1d(
                in_channels=conv1_channels,
                out_channels=conv2_channels,
                kernel_size=conv2_kernel_size,
                padding=conv2_kernel_size // 2,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(conv2_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape [B, C, T], got {tuple(x.shape)}")
        x = self.features(x)
        x = x.mean(dim=-1)  # Global average pooling over temporal axis.
        return self.classifier(x)


class EEGNetBaseline(nn.Module):
    """Compact EEGNet-style model.

    Input convention: x has shape [B, C, T] = [batch, 122, 500].
    Internal convention: reshape to [B, 1, C, T] for 2D convolutions.
    """

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 122,
        num_timesteps: int = 500,
        temporal_filters: int = 16,
        depth_multiplier: int = 2,
        separable_filters: int = 32,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(temporal_filters),
            nn.Conv2d(
                temporal_filters,
                temporal_filters * depth_multiplier,
                kernel_size=(num_channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters * depth_multiplier),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout),
        )

        in_filters = temporal_filters * depth_multiplier
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_filters,
                in_filters,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=in_filters,
                bias=False,
            ),
            nn.Conv2d(in_filters, separable_filters, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(separable_filters),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, num_timesteps)
            feat = self.block2(self.block1(dummy))
            flatten_dim = int(feat.numel())

        self.classifier = nn.Linear(flatten_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape [B, C, T], got {tuple(x.shape)}")
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


def build_model(model_name: str, num_classes: int, **kwargs: Any) -> nn.Module:
    """Build a classification model by name."""
    if model_name in {"cnn_baseline", "baseline_cnn"}:
        return TemporalCNNBaseline(
            num_classes=num_classes,
            num_channels=int(kwargs.get("num_channels", 122)),
            conv1_channels=int(kwargs.get("conv1_channels", 128)),
            conv2_channels=int(kwargs.get("conv2_channels", 256)),
            conv1_kernel_size=int(kwargs.get("conv1_kernel_size", 9)),
            conv2_kernel_size=int(kwargs.get("conv2_kernel_size", 7)),
            pool_kernel_size=int(kwargs.get("pool_kernel_size", 2)),
            dropout=float(kwargs.get("dropout", 0.3)),
        )

    if model_name == "mlp_baseline":
        return MLPBaseline(
            num_classes=num_classes,
            num_channels=int(kwargs.get("num_channels", 122)),
            num_timesteps=int(kwargs.get("num_timesteps", 500)),
            hidden_dims=tuple(kwargs.get("hidden_dims", (512, 256))),
            dropout=float(kwargs.get("dropout", 0.3)),
        )

    if model_name == "eegnet_baseline":
        return EEGNetBaseline(
            num_classes=num_classes,
            num_channels=int(kwargs.get("num_channels", 122)),
            num_timesteps=int(kwargs.get("num_timesteps", 500)),
            temporal_filters=int(kwargs.get("temporal_filters", 16)),
            depth_multiplier=int(kwargs.get("depth_multiplier", 2)),
            separable_filters=int(kwargs.get("separable_filters", 32)),
            dropout=float(kwargs.get("dropout", 0.5)),
        )

    raise ValueError(
        f"Unsupported model_name={model_name!r}. "
        "Expected one of: cnn_baseline, baseline_cnn, mlp_baseline, eegnet_baseline"
    )
