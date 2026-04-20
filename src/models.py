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


class SubjectAwareCNNTransformer(nn.Module):
    """Subject-aware EEG classifier with CNN token encoder and shared Transformer.

    Shape flow:
    - input: [B, 122, 500]
    - CNN encoder output: [B, 122, d_model]
    - Transformer output: [B, 122, d_model]
    - pooled shared feature: [B, d_model]
    - subject-specific logits: [B, num_classes]

    The CNN encoder processes each electrode's temporal signal independently with
    shared temporal convolutions, so each electrode becomes one token before the
    shared Transformer backbone models cross-electrode interactions.
    """

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 122,
        num_timesteps: int = 500,
        num_subjects: int = 13,
        cnn_out_channels: int = 64,
        d_model: int = 128,
        nhead: int = 8,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        head_hidden_dim: int | None = None,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.num_subjects = num_subjects
        self.d_model = d_model
        self.num_classes = num_classes

        # Each electrode is treated as its own temporal signal token.
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(cnn_out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_out_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(output_size=1),
        )
        self.token_projection = nn.Linear(cnn_out_channels, d_model)
        self.electrode_embedding = nn.Parameter(torch.randn(1, num_channels, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_transformer_layers,
        )
        if head_hidden_dim is None:
            self.heads = nn.ModuleList([nn.Linear(d_model, num_classes) for _ in range(num_subjects)])
        else:
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(d_model, head_hidden_dim),
                        nn.GELU(),
                        nn.Dropout(p=head_dropout),
                        nn.Linear(head_hidden_dim, num_classes),
                    )
                    for _ in range(num_subjects)
                ]
            )

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape [B, C, T], got {tuple(x.shape)}")
        if subject_ids.ndim != 1:
            raise ValueError(f"Expected subject_ids with shape [B], got {tuple(subject_ids.shape)}")
        if subject_ids.shape[0] != x.shape[0]:
            raise ValueError(
                "Batch size mismatch between x and subject_ids: "
                f"x.shape[0]={x.shape[0]}, subject_ids.shape[0]={subject_ids.shape[0]}"
            )
        if x.shape[1] != self.num_channels or x.shape[2] != self.num_timesteps:
            raise ValueError(
                "Unexpected EEG input shape. "
                f"Expected [B, {self.num_channels}, {self.num_timesteps}], got {tuple(x.shape)}"
            )

        batch_size, num_channels, num_timesteps = x.shape

        # [B, C, T] -> [B*C, 1, T] so one shared temporal CNN processes each electrode.
        x = x.reshape(batch_size * num_channels, 1, num_timesteps)
        x = self.temporal_encoder(x)  # [B*C, cnn_out_channels, 1]
        x = x.squeeze(-1)  # [B*C, cnn_out_channels]
        x = self.token_projection(x)  # [B*C, d_model]
        x = x.view(batch_size, num_channels, self.d_model)  # [B, C, d_model]

        x = x + self.electrode_embedding[:, :num_channels, :]
        x = self.transformer(x)  # [B, C, d_model]
        pooled = x.mean(dim=1)  # [B, d_model]

        logits = pooled.new_zeros((batch_size, self.num_classes))
        subject_ids = subject_ids.to(device=pooled.device, dtype=torch.long)
        for subject_idx in range(self.num_subjects):
            mask = subject_ids == subject_idx
            if mask.any():
                logits[mask] = self.heads[subject_idx](pooled[mask])

        invalid_mask = (subject_ids < 0) | (subject_ids >= self.num_subjects)
        if invalid_mask.any():
            invalid_ids = torch.unique(subject_ids[invalid_mask]).detach().cpu().tolist()
            raise ValueError(
                f"subject_ids contain values outside [0, {self.num_subjects - 1}]: {invalid_ids}"
            )

        return logits


class SharedHeadCNNTransformerClassifier(nn.Module):
    """CNN + shared Transformer + one shared classification head.

    Shape flow:
    - input: [B, 122, 500]
    - CNN encoder output: [B, 122, d_model]
    - Transformer output: [B, 122, d_model]
    - pooled feature: [B, d_model]
    - shared-head logits: [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 122,
        num_timesteps: int = 500,
        cnn_out_channels: int = 64,
        d_model: int = 128,
        nhead: int = 8,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.d_model = d_model

        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(cnn_out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_out_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(output_size=1),
        )
        self.token_projection = nn.Linear(cnn_out_channels, d_model)
        self.electrode_embedding = nn.Parameter(torch.randn(1, num_channels, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_transformer_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=min(dropout, 0.1)),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape [B, C, T], got {tuple(x.shape)}")
        if x.shape[1] != self.num_channels or x.shape[2] != self.num_timesteps:
            raise ValueError(
                "Unexpected EEG input shape. "
                f"Expected [B, {self.num_channels}, {self.num_timesteps}], got {tuple(x.shape)}"
            )

        batch_size, num_channels, num_timesteps = x.shape
        x = x.reshape(batch_size * num_channels, 1, num_timesteps)
        x = self.temporal_encoder(x)
        x = x.squeeze(-1)
        x = self.token_projection(x)
        x = x.view(batch_size, num_channels, self.d_model)
        x = x + self.electrode_embedding[:, :num_channels, :]
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


class SubjectEmbeddingCNNTransformerClassifier(nn.Module):
    """CNN + shared Transformer + subject embedding + shared classifier.

    Shape flow:
    - input: [B, 122, 500]
    - CNN encoder output: [B, 122, d_model]
    - Transformer output: [B, 122, d_model]
    - pooled EEG feature: [B, d_model]
    - subject embedding: [B, d_model]
    - fused feature: [B, 2 * d_model]
    - shared classifier logits: [B, num_classes]

    Subject information is injected only after the shared backbone has produced a
    sample-level EEG representation. This is intentionally less aggressive than
    fusing subject identity inside the Transformer token stream.
    """

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 122,
        num_timesteps: int = 500,
        num_subjects: int = 13,
        cnn_out_channels: int = 64,
        d_model: int = 128,
        nhead: int = 8,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        subject_embedding_dim: int | None = None,
        classifier_hidden_dim: int | None = None,
        fuse_mode: str = "concat",
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.num_subjects = num_subjects
        self.d_model = d_model
        self.fuse_mode = fuse_mode

        if subject_embedding_dim is None:
            subject_embedding_dim = d_model
        if classifier_hidden_dim is None:
            classifier_hidden_dim = d_model
        if fuse_mode not in {"concat", "add"}:
            raise ValueError(f"Unsupported fuse_mode={fuse_mode!r}. Expected 'concat' or 'add'.")
        if fuse_mode == "add" and subject_embedding_dim != d_model:
            raise ValueError(
                f"fuse_mode='add' requires subject_embedding_dim == d_model, got {subject_embedding_dim} and {d_model}"
            )

        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(cnn_out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_out_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(output_size=1),
        )
        self.token_projection = nn.Linear(cnn_out_channels, d_model)
        self.electrode_embedding = nn.Parameter(torch.randn(1, num_channels, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_transformer_layers,
        )

        self.subject_embedding = nn.Embedding(num_subjects, subject_embedding_dim)

        if fuse_mode == "concat":
            classifier_input_dim = d_model + subject_embedding_dim
        else:
            classifier_input_dim = d_model

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=min(dropout, 0.1)),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape [B, C, T], got {tuple(x.shape)}")
        if subject_ids.ndim != 1:
            raise ValueError(f"Expected subject_ids with shape [B], got {tuple(subject_ids.shape)}")
        if subject_ids.shape[0] != x.shape[0]:
            raise ValueError(
                "Batch size mismatch between x and subject_ids: "
                f"x.shape[0]={x.shape[0]}, subject_ids.shape[0]={subject_ids.shape[0]}"
            )
        if x.shape[1] != self.num_channels or x.shape[2] != self.num_timesteps:
            raise ValueError(
                "Unexpected EEG input shape. "
                f"Expected [B, {self.num_channels}, {self.num_timesteps}], got {tuple(x.shape)}"
            )

        invalid_mask = (subject_ids < 0) | (subject_ids >= self.num_subjects)
        if invalid_mask.any():
            invalid_ids = torch.unique(subject_ids[invalid_mask]).detach().cpu().tolist()
            raise ValueError(
                f"subject_ids contain values outside [0, {self.num_subjects - 1}]: {invalid_ids}"
            )

        batch_size, num_channels, num_timesteps = x.shape
        x = x.reshape(batch_size * num_channels, 1, num_timesteps)
        x = self.temporal_encoder(x)
        x = x.squeeze(-1)
        x = self.token_projection(x)
        x = x.view(batch_size, num_channels, self.d_model)
        x = x + self.electrode_embedding[:, :num_channels, :]
        x = self.transformer(x)
        pooled = x.mean(dim=1)

        subject_feature = self.subject_embedding(subject_ids.to(device=pooled.device, dtype=torch.long))
        if self.fuse_mode == "concat":
            fused = torch.cat([pooled, subject_feature], dim=-1)
        else:
            fused = pooled + subject_feature
        return self.classifier(fused)


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
    if model_name in {
        "subject_aware_cnn_transformer",
        "cnn_transformer_subject_head",
        "multihead_cnn_transformer",
    }:
        return SubjectAwareCNNTransformer(
            num_classes=num_classes,
            num_channels=int(kwargs.get("num_channels", 122)),
            num_timesteps=int(kwargs.get("num_timesteps", 500)),
            num_subjects=int(kwargs.get("num_subjects", 13)),
            cnn_out_channels=int(kwargs.get("cnn_out_channels", 64)),
            d_model=int(kwargs.get("d_model", 128)),
            nhead=int(kwargs.get("nhead", 8)),
            num_transformer_layers=int(kwargs.get("num_transformer_layers", 2)),
            dim_feedforward=int(kwargs.get("dim_feedforward", 256)),
            dropout=float(kwargs.get("dropout", 0.3)),
            head_hidden_dim=kwargs.get("head_hidden_dim"),
            head_dropout=float(kwargs.get("head_dropout", 0.1)),
        )

    if model_name in {
        "subject_embedding_cnn_transformer",
        "cnn_transformer_subject_embedding",
        "subject_conditioned_cnn_transformer",
    }:
        return SubjectEmbeddingCNNTransformerClassifier(
            num_classes=num_classes,
            num_channels=int(kwargs.get("num_channels", 122)),
            num_timesteps=int(kwargs.get("num_timesteps", 500)),
            num_subjects=int(kwargs.get("num_subjects", 13)),
            cnn_out_channels=int(kwargs.get("cnn_out_channels", 64)),
            d_model=int(kwargs.get("d_model", 128)),
            nhead=int(kwargs.get("nhead", 8)),
            num_transformer_layers=int(kwargs.get("num_transformer_layers", 2)),
            dim_feedforward=int(kwargs.get("dim_feedforward", 256)),
            dropout=float(kwargs.get("dropout", 0.3)),
            subject_embedding_dim=kwargs.get("subject_embedding_dim"),
            classifier_hidden_dim=kwargs.get("classifier_hidden_dim"),
            fuse_mode=str(kwargs.get("fuse_mode", "concat")),
        )

    if model_name in {"shared_head_cnn_transformer", "cnn_transformer_shared_head"}:
        return SharedHeadCNNTransformerClassifier(
            num_classes=num_classes,
            num_channels=int(kwargs.get("num_channels", 122)),
            num_timesteps=int(kwargs.get("num_timesteps", 500)),
            cnn_out_channels=int(kwargs.get("cnn_out_channels", 64)),
            d_model=int(kwargs.get("d_model", 128)),
            nhead=int(kwargs.get("nhead", 8)),
            num_transformer_layers=int(kwargs.get("num_transformer_layers", 2)),
            dim_feedforward=int(kwargs.get("dim_feedforward", 256)),
            dropout=float(kwargs.get("dropout", 0.3)),
        )

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
        "Expected one of: subject_aware_cnn_transformer, cnn_transformer_subject_head, "
        "multihead_cnn_transformer, subject_embedding_cnn_transformer, "
        "cnn_transformer_subject_embedding, subject_conditioned_cnn_transformer, "
        "shared_head_cnn_transformer, cnn_transformer_shared_head, "
        "cnn_baseline, baseline_cnn, mlp_baseline, eegnet_baseline"
    )
