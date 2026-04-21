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


class EEGNetEmbeddingEncoder(nn.Module):
    """EEGNet-style encoder that outputs a reusable EEG embedding.

    Shape flow:
    - input EEG: [B, C, T]
    - EEGNet feature map: [B, separable_filters, 1, T']
    - pooled feature: [B, separable_filters]
    - eeg embedding: [B, embedding_dim]
    - projected embedding (optional): [B, projection_dim]
    """

    def __init__(
        self,
        num_channels: int = 122,
        num_timesteps: int = 500,
        temporal_filters: int = 16,
        depth_multiplier: int = 2,
        separable_filters: int = 32,
        dropout: float = 0.5,
        embedding_dim: int = 256,
        projection_dim: int | None = None,
        projection_hidden_dim: int | None = None,
        projection_dropout: float = 0.0,
        normalize_projected_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.normalize_projected_embedding = normalize_projected_embedding

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
        self.embedding_head = nn.Sequential(
            nn.Linear(separable_filters, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        if projection_dim is None:
            self.projection_head: nn.Module | None = None
        else:
            hidden_dim = int(projection_hidden_dim or embedding_dim)
            self.projection_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=projection_dropout),
                nn.Linear(hidden_dim, projection_dim),
            )

    def extract_pooled_feature(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape [B, C, T], got {tuple(x.shape)}")
        if x.shape[1] != self.num_channels or x.shape[2] != self.num_timesteps:
            raise ValueError(
                "Unexpected EEG input shape. "
                f"Expected [B, {self.num_channels}, {self.num_timesteps}], got {tuple(x.shape)}"
            )
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        return x.mean(dim=(2, 3))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.extract_pooled_feature(x)
        return self.embedding_head(pooled)

    def project_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.projection_head is None:
            raise RuntimeError("Projection head is not configured for this encoder.")
        projected = self.projection_head(embedding)
        if self.normalize_projected_embedding:
            projected = torch.nn.functional.normalize(projected, dim=-1)
        return projected

    def encode_projected(self, x: torch.Tensor) -> torch.Tensor:
        return self.project_embedding(self.encode(x))


class GatedTemporalPooling(nn.Module):
    """Learned gated pooling over temporal feature maps [B, C, T]."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Conv1d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.gate(x), dim=-1)
        return (x * weights).sum(dim=-1)


class EEGNetResidualEmbeddingEncoder(nn.Module):
    """EEGNet-style encoder with temporal residual refinement and gated pooling.

    Shape flow:
    - input EEG: [B, C, T]
    - EEGNet stem: [B, separable_filters, 1, T']
    - temporal refinement: [B, separable_filters, T']
    - pooled feature: [B, separable_filters]
    - eeg embedding: [B, embedding_dim]
    - projected embedding (optional): [B, projection_dim]
    """

    def __init__(
        self,
        num_channels: int = 122,
        num_timesteps: int = 500,
        temporal_filters: int = 24,
        depth_multiplier: int = 2,
        separable_filters: int = 64,
        dropout: float = 0.15,
        embedding_dim: int = 256,
        temporal_kernel_size: int = 32,
        separable_kernel_size: int = 8,
        pool1_kernel_size: int = 2,
        pool2_kernel_size: int = 4,
        num_refinement_blocks: int = 1,
        refinement_kernel_size: int = 7,
        use_gated_pooling: bool = True,
        projection_dim: int | None = None,
        projection_hidden_dim: int | None = None,
        projection_dropout: float = 0.0,
        normalize_projected_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.normalize_projected_embedding = normalize_projected_embedding

        temporal_padding = temporal_kernel_size // 2
        separable_padding = separable_kernel_size // 2
        self.block1 = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, temporal_kernel_size), padding=(0, temporal_padding), bias=False),
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
            nn.AvgPool2d(kernel_size=(1, pool1_kernel_size)),
            nn.Dropout(p=dropout),
        )

        in_filters = temporal_filters * depth_multiplier
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_filters,
                in_filters,
                kernel_size=(1, separable_kernel_size),
                padding=(0, separable_padding),
                groups=in_filters,
                bias=False,
            ),
            nn.Conv2d(in_filters, separable_filters, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(separable_filters),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, pool2_kernel_size)),
            nn.Dropout(p=dropout),
        )
        self.refinement_blocks = nn.Sequential(
            *[
                ResidualTemporalRefinementBlock(
                    channels=separable_filters,
                    kernel_size=refinement_kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_refinement_blocks)
            ]
        )
        self.pool = GatedTemporalPooling(separable_filters) if use_gated_pooling else nn.AdaptiveAvgPool1d(1)
        self.embedding_head = nn.Sequential(
            nn.Linear(separable_filters, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(p=min(dropout, 0.1)),
        )
        if projection_dim is None:
            self.projection_head: nn.Module | None = None
        else:
            hidden_dim = int(projection_hidden_dim or embedding_dim)
            self.projection_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=projection_dropout),
                nn.Linear(hidden_dim, projection_dim),
            )

    def extract_pooled_feature(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape [B, C, T], got {tuple(x.shape)}")
        if x.shape[1] != self.num_channels or x.shape[2] != self.num_timesteps:
            raise ValueError(
                "Unexpected EEG input shape. "
                f"Expected [B, {self.num_channels}, {self.num_timesteps}], got {tuple(x.shape)}"
            )
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.squeeze(2)
        x = self.refinement_blocks(x)
        if isinstance(self.pool, GatedTemporalPooling):
            return self.pool(x)
        return self.pool(x).squeeze(-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.extract_pooled_feature(x)
        return self.embedding_head(pooled)

    def project_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.projection_head is None:
            raise RuntimeError("Projection head is not configured for this encoder.")
        projected = self.projection_head(embedding)
        if self.normalize_projected_embedding:
            projected = torch.nn.functional.normalize(projected, dim=-1)
        return projected

    def encode_projected(self, x: torch.Tensor) -> torch.Tensor:
        return self.project_embedding(self.encode(x))


class EEGNetResidualEncoderClassifier(nn.Module):
    """Residual EEGNet encoder with a thin linear classification head."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 122,
        num_timesteps: int = 500,
        temporal_filters: int = 24,
        depth_multiplier: int = 2,
        separable_filters: int = 64,
        dropout: float = 0.15,
        embedding_dim: int = 256,
        temporal_kernel_size: int = 32,
        separable_kernel_size: int = 8,
        pool1_kernel_size: int = 2,
        pool2_kernel_size: int = 4,
        num_refinement_blocks: int = 1,
        refinement_kernel_size: int = 7,
        use_gated_pooling: bool = True,
        projection_dim: int | None = None,
        projection_hidden_dim: int | None = None,
        projection_dropout: float = 0.0,
        normalize_projected_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = EEGNetResidualEmbeddingEncoder(
            num_channels=num_channels,
            num_timesteps=num_timesteps,
            temporal_filters=temporal_filters,
            depth_multiplier=depth_multiplier,
            separable_filters=separable_filters,
            dropout=dropout,
            embedding_dim=embedding_dim,
            temporal_kernel_size=temporal_kernel_size,
            separable_kernel_size=separable_kernel_size,
            pool1_kernel_size=pool1_kernel_size,
            pool2_kernel_size=pool2_kernel_size,
            num_refinement_blocks=num_refinement_blocks,
            refinement_kernel_size=refinement_kernel_size,
            use_gated_pooling=use_gated_pooling,
            projection_dim=projection_dim,
            projection_hidden_dim=projection_hidden_dim,
            projection_dropout=projection_dropout,
            normalize_projected_embedding=normalize_projected_embedding,
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(x)

    def project_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.encoder.project_embedding(embedding)

    def encode_projected(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_projected(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))


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


class EEGNetEmbeddingClassifier(nn.Module):
    """EEGNet embedding encoder with a thin linear classification head."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 122,
        num_timesteps: int = 500,
        temporal_filters: int = 16,
        depth_multiplier: int = 2,
        separable_filters: int = 32,
        dropout: float = 0.5,
        embedding_dim: int = 256,
        projection_dim: int | None = None,
        projection_hidden_dim: int | None = None,
        projection_dropout: float = 0.0,
        normalize_projected_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = EEGNetEmbeddingEncoder(
            num_channels=num_channels,
            num_timesteps=num_timesteps,
            temporal_filters=temporal_filters,
            depth_multiplier=depth_multiplier,
            separable_filters=separable_filters,
            dropout=dropout,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            projection_hidden_dim=projection_hidden_dim,
            projection_dropout=projection_dropout,
            normalize_projected_embedding=normalize_projected_embedding,
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(x)

    def project_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.encoder.project_embedding(embedding)

    def encode_projected(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_projected(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))


class EEGNetMLPBaseline(nn.Module):
    """EEGNet encoder followed by a small MLP classification head.

    Input convention: x has shape [B, C, T] = [batch, 122, 500].
    Internal convention: reshape to [B, 1, C, T] for 2D convolutions.

    Shape flow:
    - input: [B, 122, 500]
    - EEGNet feature map: [B, separable_filters, 1, T']
    - flattened feature: [B, flatten_dim]
    - MLP head hidden: [B, classifier_hidden_dim]
    - logits: [B, num_classes]
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
        classifier_hidden_dim: int = 256,
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

        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, classifier_hidden_dim),
            nn.ELU(inplace=True),
            nn.Dropout(p=min(dropout, 0.4)),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape [B, C, T], got {tuple(x.shape)}")
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


class EEGNetSubjectEmbeddingClassifier(nn.Module):
    """EEGNet encoder with late subject embedding fusion and a shared classifier.

    Shape flow:
    - input EEG: [B, 122, 500]
    - EEGNet feature map: [B, separable_filters, 1, T']
    - pooled EEG feature: [B, separable_filters]
    - subject embedding: [B, subject_embedding_dim]
    - fused feature: [B, separable_filters + subject_embedding_dim] or [B, separable_filters]
    - shared classifier logits: [B, num_classes]

    This keeps the EEGNet temporal/spatial inductive bias intact and injects subject
    information only after the shared encoder has produced a compact sample-level
    representation.
    """

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 122,
        num_timesteps: int = 500,
        num_subjects: int = 13,
        temporal_filters: int = 16,
        depth_multiplier: int = 2,
        separable_filters: int = 32,
        dropout: float = 0.5,
        subject_embedding_dim: int = 32,
        classifier_hidden_dim: int = 128,
        fuse_mode: str = "concat",
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.num_subjects = num_subjects
        self.separable_filters = separable_filters
        self.fuse_mode = fuse_mode

        if fuse_mode not in {"concat", "add"}:
            raise ValueError(f"Unsupported fuse_mode={fuse_mode!r}. Expected 'concat' or 'add'.")
        if fuse_mode == "add" and subject_embedding_dim != separable_filters:
            raise ValueError(
                "fuse_mode='add' requires subject_embedding_dim to match separable_filters. "
                f"Got subject_embedding_dim={subject_embedding_dim}, separable_filters={separable_filters}."
            )

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

        self.subject_embedding = nn.Embedding(num_subjects, subject_embedding_dim)
        classifier_input_dim = separable_filters + subject_embedding_dim if fuse_mode == "concat" else separable_filters
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_hidden_dim),
            nn.ELU(inplace=True),
            nn.Dropout(p=min(dropout, 0.25)),
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

        subject_ids = subject_ids.to(device=x.device, dtype=torch.long)
        invalid_mask = (subject_ids < 0) | (subject_ids >= self.num_subjects)
        if invalid_mask.any():
            invalid_ids = torch.unique(subject_ids[invalid_mask]).detach().cpu().tolist()
            raise ValueError(
                f"subject_ids contain values outside [0, {self.num_subjects - 1}]: {invalid_ids}"
            )

        # [B, C, T] -> [B, 1, C, T] for EEGNet-style temporal and spatial convolutions.
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        eeg_feature = x.mean(dim=(2, 3))  # [B, separable_filters]

        subject_feature = self.subject_embedding(subject_ids)
        if self.fuse_mode == "concat":
            fused = torch.cat([eeg_feature, subject_feature], dim=-1)
        else:
            fused = eeg_feature + subject_feature
        return self.classifier(fused)


class ResidualTemporalRefinementBlock(nn.Module):
    """Residual depthwise-separable temporal block over [B, channels, time]."""

    def __init__(self, channels: int, kernel_size: int = 15, dropout: float = 0.2) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=False),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ELU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=False),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class MultiScaleEEGNetBackbone(nn.Module):
    """Multi-scale EEG encoder that outputs a normalized embedding.

    Shape flow:
    - input EEG: [B, 122, 500]
    - temporal multi-scale stem: [B, temporal_filters * num_branches, 122, 500]
    - spatial EEGNet block: [B, stem_channels, 1, T']
    - residual temporal refinement: [B, separable_filters, T']
    - global pooled feature: [B, separable_filters]
    - embedding projection: [B, embedding_dim]

    This backbone is designed to be reused for both classification and later
    cross-modal alignment. The `encode(...)` method returns the normalized EEG
    embedding directly.
    """

    def __init__(
        self,
        num_channels: int = 122,
        num_timesteps: int = 500,
        temporal_filters: int = 8,
        branch_kernel_sizes: Sequence[int] = (15, 31, 63),
        depth_multiplier: int = 2,
        separable_filters: int = 64,
        num_refinement_blocks: int = 2,
        refinement_kernel_size: int = 15,
        stem_pool_kernel: int = 4,
        dropout: float = 0.3,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.embedding_dim = embedding_dim
        self.branch_kernel_sizes = tuple(int(k) for k in branch_kernel_sizes)
        if not self.branch_kernel_sizes:
            raise ValueError("branch_kernel_sizes must contain at least one kernel size.")

        self.temporal_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=temporal_filters,
                        kernel_size=(1, kernel_size),
                        padding=(0, kernel_size // 2),
                        bias=False,
                    ),
                    nn.BatchNorm2d(temporal_filters),
                    nn.ELU(inplace=True),
                )
                for kernel_size in self.branch_kernel_sizes
            ]
        )

        total_temporal_filters = temporal_filters * len(self.branch_kernel_sizes)
        stem_channels = total_temporal_filters * depth_multiplier
        self.spatial_block = nn.Sequential(
            nn.Conv2d(
                total_temporal_filters,
                stem_channels,
                kernel_size=(num_channels, 1),
                groups=total_temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, stem_pool_kernel)),
            nn.Dropout(p=dropout),
        )

        self.temporal_projection = nn.Sequential(
            nn.Conv1d(stem_channels, separable_filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(separable_filters),
            nn.ELU(inplace=True),
        )
        self.refinement_blocks = nn.Sequential(
            *[
                ResidualTemporalRefinementBlock(
                    channels=separable_filters,
                    kernel_size=refinement_kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_refinement_blocks)
            ]
        )
        self.embedding_projection = nn.Sequential(
            nn.Linear(separable_filters, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def extract_pooled_feature(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape [B, C, T], got {tuple(x.shape)}")
        if x.shape[1] != self.num_channels or x.shape[2] != self.num_timesteps:
            raise ValueError(
                "Unexpected EEG input shape. "
                f"Expected [B, {self.num_channels}, {self.num_timesteps}], got {tuple(x.shape)}"
            )

        x = x.unsqueeze(1)  # [B, 1, C, T]
        branch_features = [branch(x) for branch in self.temporal_branches]
        x = torch.cat(branch_features, dim=1)  # [B, total_temporal_filters, C, T]
        x = self.spatial_block(x)  # [B, stem_channels, 1, T']
        x = x.squeeze(2)  # [B, stem_channels, T']
        x = self.temporal_projection(x)  # [B, separable_filters, T']
        x = self.refinement_blocks(x)  # [B, separable_filters, T']
        return x.mean(dim=-1)  # Global temporal average pool.

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.extract_pooled_feature(x)
        return self.embedding_projection(pooled)


class MultiScaleEEGNetClassifier(nn.Module):
    """Multi-scale EEGNet-style encoder with a shared classification head.

    Shape flow:
    - input: [B, 122, 500]
    - encoder pooled feature: [B, separable_filters]
    - eeg embedding: [B, embedding_dim]
    - logits: [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 122,
        num_timesteps: int = 500,
        temporal_filters: int = 8,
        branch_kernel_sizes: Sequence[int] = (15, 31, 63),
        depth_multiplier: int = 2,
        separable_filters: int = 64,
        num_refinement_blocks: int = 2,
        refinement_kernel_size: int = 15,
        stem_pool_kernel: int = 4,
        dropout: float = 0.3,
        embedding_dim: int = 256,
        projection_dim: int | None = None,
        projection_hidden_dim: int | None = None,
        projection_dropout: float = 0.0,
        normalize_projected_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = MultiScaleEEGNetBackbone(
            num_channels=num_channels,
            num_timesteps=num_timesteps,
            temporal_filters=temporal_filters,
            branch_kernel_sizes=branch_kernel_sizes,
            depth_multiplier=depth_multiplier,
            separable_filters=separable_filters,
            num_refinement_blocks=num_refinement_blocks,
            refinement_kernel_size=refinement_kernel_size,
            stem_pool_kernel=stem_pool_kernel,
            dropout=dropout,
            embedding_dim=embedding_dim,
        )
        self.projection_dim = projection_dim
        self.normalize_projected_embedding = normalize_projected_embedding
        if projection_dim is None:
            self.projection_head: nn.Module | None = None
        else:
            hidden_dim = int(projection_hidden_dim or embedding_dim)
            self.projection_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=projection_dropout),
                nn.Linear(hidden_dim, projection_dim),
            )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(x)

    def project_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.projection_head is None:
            raise RuntimeError("Projection head is not configured for this model.")
        projected = self.projection_head(embedding)
        if self.normalize_projected_embedding:
            projected = torch.nn.functional.normalize(projected, dim=-1)
        return projected

    def encode_projected(self, x: torch.Tensor) -> torch.Tensor:
        return self.project_embedding(self.encode(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encode(x)
        return self.classifier(embedding)


class SubjectConditionedMultiScaleEEGNetClassifier(nn.Module):
    """Multi-scale EEG encoder with lightweight subject conditioning and shared classifier.

    Shape flow:
    - input EEG: [B, 122, 500]
    - shared pooled EEG feature: [B, separable_filters]
    - FiLM-conditioned pooled feature: [B, separable_filters]
    - eeg embedding: [B, embedding_dim]
    - shared classifier logits: [B, num_classes]

    Subject identity is injected with a small FiLM-style modulation on the pooled
    shared feature, rather than using separate classification heads.
    """

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 122,
        num_timesteps: int = 500,
        num_subjects: int = 13,
        temporal_filters: int = 8,
        branch_kernel_sizes: Sequence[int] = (15, 31, 63),
        depth_multiplier: int = 2,
        separable_filters: int = 64,
        num_refinement_blocks: int = 2,
        refinement_kernel_size: int = 15,
        stem_pool_kernel: int = 4,
        dropout: float = 0.3,
        embedding_dim: int = 256,
        subject_embedding_dim: int = 32,
        projection_dim: int | None = None,
        projection_hidden_dim: int | None = None,
        projection_dropout: float = 0.0,
        normalize_projected_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.num_subjects = num_subjects
        self.backbone = MultiScaleEEGNetBackbone(
            num_channels=num_channels,
            num_timesteps=num_timesteps,
            temporal_filters=temporal_filters,
            branch_kernel_sizes=branch_kernel_sizes,
            depth_multiplier=depth_multiplier,
            separable_filters=separable_filters,
            num_refinement_blocks=num_refinement_blocks,
            refinement_kernel_size=refinement_kernel_size,
            stem_pool_kernel=stem_pool_kernel,
            dropout=dropout,
            embedding_dim=embedding_dim,
        )
        self.projection_dim = projection_dim
        self.normalize_projected_embedding = normalize_projected_embedding
        self.subject_embedding = nn.Embedding(num_subjects, subject_embedding_dim)
        self.subject_to_scale = nn.Linear(subject_embedding_dim, separable_filters)
        self.subject_to_bias = nn.Linear(subject_embedding_dim, separable_filters)
        nn.init.zeros_(self.subject_to_scale.weight)
        nn.init.zeros_(self.subject_to_scale.bias)
        nn.init.zeros_(self.subject_to_bias.weight)
        nn.init.zeros_(self.subject_to_bias.bias)
        if projection_dim is None:
            self.projection_head: nn.Module | None = None
        else:
            hidden_dim = int(projection_hidden_dim or embedding_dim)
            self.projection_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=projection_dropout),
                nn.Linear(hidden_dim, projection_dim),
            )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        if subject_ids.ndim != 1:
            raise ValueError(f"Expected subject_ids with shape [B], got {tuple(subject_ids.shape)}")
        if subject_ids.shape[0] != x.shape[0]:
            raise ValueError(
                "Batch size mismatch between x and subject_ids: "
                f"x.shape[0]={x.shape[0]}, subject_ids.shape[0]={subject_ids.shape[0]}"
            )
        subject_ids = subject_ids.to(device=x.device, dtype=torch.long)
        invalid_mask = (subject_ids < 0) | (subject_ids >= self.num_subjects)
        if invalid_mask.any():
            invalid_ids = torch.unique(subject_ids[invalid_mask]).detach().cpu().tolist()
            raise ValueError(
                f"subject_ids contain values outside [0, {self.num_subjects - 1}]: {invalid_ids}"
            )

        pooled = self.backbone.extract_pooled_feature(x)
        subject_feature = self.subject_embedding(subject_ids)
        scale = torch.tanh(self.subject_to_scale(subject_feature))
        bias = self.subject_to_bias(subject_feature)
        conditioned = pooled * (1.0 + scale) + bias
        return self.backbone.embedding_projection(conditioned)

    def project_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.projection_head is None:
            raise RuntimeError("Projection head is not configured for this model.")
        projected = self.projection_head(embedding)
        if self.normalize_projected_embedding:
            projected = torch.nn.functional.normalize(projected, dim=-1)
        return projected

    def encode_projected(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        return self.project_embedding(self.encode(x, subject_ids))

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        embedding = self.encode(x, subject_ids)
        return self.classifier(embedding)


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

    if model_name in {
        "eegnet_embedding_classifier",
        "eegnet_embedding_baseline",
        "eegnet_encoder_classifier",
    }:
        return EEGNetEmbeddingClassifier(
            num_classes=num_classes,
            num_channels=int(kwargs.get("num_channels", 122)),
            num_timesteps=int(kwargs.get("num_timesteps", 500)),
            temporal_filters=int(kwargs.get("temporal_filters", 16)),
            depth_multiplier=int(kwargs.get("depth_multiplier", 2)),
            separable_filters=int(kwargs.get("separable_filters", 32)),
            dropout=float(kwargs.get("dropout", 0.5)),
            embedding_dim=int(kwargs.get("embedding_dim", 256)),
            projection_dim=kwargs.get("projection_dim"),
            projection_hidden_dim=kwargs.get("projection_hidden_dim"),
            projection_dropout=float(kwargs.get("projection_dropout", 0.0)),
            normalize_projected_embedding=bool(kwargs.get("normalize_projected_embedding", False)),
        )

    if model_name in {"eegnet_residual_encoder", "eegnet_residual_classifier", "eegnet_refined_encoder"}:
        return EEGNetResidualEncoderClassifier(
            num_classes=num_classes,
            num_channels=int(kwargs.get("num_channels", 122)),
            num_timesteps=int(kwargs.get("num_timesteps", 500)),
            temporal_filters=int(kwargs.get("temporal_filters", 24)),
            depth_multiplier=int(kwargs.get("depth_multiplier", 2)),
            separable_filters=int(kwargs.get("separable_filters", 64)),
            dropout=float(kwargs.get("dropout", 0.15)),
            embedding_dim=int(kwargs.get("embedding_dim", 256)),
            temporal_kernel_size=int(kwargs.get("temporal_kernel_size", 32)),
            separable_kernel_size=int(kwargs.get("separable_kernel_size", 8)),
            pool1_kernel_size=int(kwargs.get("pool1_kernel_size", 2)),
            pool2_kernel_size=int(kwargs.get("pool2_kernel_size", 4)),
            num_refinement_blocks=int(kwargs.get("num_refinement_blocks", 1)),
            refinement_kernel_size=int(kwargs.get("refinement_kernel_size", 7)),
            use_gated_pooling=bool(kwargs.get("use_gated_pooling", True)),
            projection_dim=kwargs.get("projection_dim"),
            projection_hidden_dim=kwargs.get("projection_hidden_dim"),
            projection_dropout=float(kwargs.get("projection_dropout", 0.0)),
            normalize_projected_embedding=bool(kwargs.get("normalize_projected_embedding", False)),
        )

    if model_name in {"eegnet_mlp_baseline", "eegnet_mlp_head", "eegnet_mlp_classifier"}:
        return EEGNetMLPBaseline(
            num_classes=num_classes,
            num_channels=int(kwargs.get("num_channels", 122)),
            num_timesteps=int(kwargs.get("num_timesteps", 500)),
            temporal_filters=int(kwargs.get("temporal_filters", 16)),
            depth_multiplier=int(kwargs.get("depth_multiplier", 2)),
            separable_filters=int(kwargs.get("separable_filters", 32)),
            dropout=float(kwargs.get("dropout", 0.5)),
            classifier_hidden_dim=int(kwargs.get("classifier_hidden_dim", 256)),
        )

    if model_name in {
        "multiscale_eegnet_classifier",
        "multiscale_eegnet",
        "multiscale_eegnet_encoder_classifier",
    }:
        return MultiScaleEEGNetClassifier(
            num_classes=num_classes,
            num_channels=int(kwargs.get("num_channels", 122)),
            num_timesteps=int(kwargs.get("num_timesteps", 500)),
            temporal_filters=int(kwargs.get("temporal_filters", 8)),
            branch_kernel_sizes=tuple(kwargs.get("branch_kernel_sizes", (15, 31, 63))),
            depth_multiplier=int(kwargs.get("depth_multiplier", 2)),
            separable_filters=int(kwargs.get("separable_filters", 64)),
            num_refinement_blocks=int(kwargs.get("num_refinement_blocks", 2)),
            refinement_kernel_size=int(kwargs.get("refinement_kernel_size", 15)),
            stem_pool_kernel=int(kwargs.get("stem_pool_kernel", 4)),
            dropout=float(kwargs.get("dropout", 0.3)),
            embedding_dim=int(kwargs.get("embedding_dim", 256)),
            projection_dim=kwargs.get("projection_dim"),
            projection_hidden_dim=kwargs.get("projection_hidden_dim"),
            projection_dropout=float(kwargs.get("projection_dropout", 0.0)),
            normalize_projected_embedding=bool(kwargs.get("normalize_projected_embedding", False)),
        )

    if model_name in {
        "subject_conditioned_multiscale_eegnet",
        "multiscale_eegnet_subject_conditioned",
        "subject_conditioned_multiscale_eegnet_classifier",
    }:
        return SubjectConditionedMultiScaleEEGNetClassifier(
            num_classes=num_classes,
            num_channels=int(kwargs.get("num_channels", 122)),
            num_timesteps=int(kwargs.get("num_timesteps", 500)),
            num_subjects=int(kwargs.get("num_subjects", 13)),
            temporal_filters=int(kwargs.get("temporal_filters", 8)),
            branch_kernel_sizes=tuple(kwargs.get("branch_kernel_sizes", (15, 31, 63))),
            depth_multiplier=int(kwargs.get("depth_multiplier", 2)),
            separable_filters=int(kwargs.get("separable_filters", 64)),
            num_refinement_blocks=int(kwargs.get("num_refinement_blocks", 2)),
            refinement_kernel_size=int(kwargs.get("refinement_kernel_size", 15)),
            stem_pool_kernel=int(kwargs.get("stem_pool_kernel", 4)),
            dropout=float(kwargs.get("dropout", 0.3)),
            embedding_dim=int(kwargs.get("embedding_dim", 256)),
            subject_embedding_dim=int(kwargs.get("subject_embedding_dim", 32)),
            projection_dim=kwargs.get("projection_dim"),
            projection_hidden_dim=kwargs.get("projection_hidden_dim"),
            projection_dropout=float(kwargs.get("projection_dropout", 0.0)),
            normalize_projected_embedding=bool(kwargs.get("normalize_projected_embedding", False)),
        )

    if model_name in {
        "eegnet_subject_embedding",
        "subject_embedding_eegnet",
        "subject_conditioned_eegnet",
    }:
        return EEGNetSubjectEmbeddingClassifier(
            num_classes=num_classes,
            num_channels=int(kwargs.get("num_channels", 122)),
            num_timesteps=int(kwargs.get("num_timesteps", 500)),
            num_subjects=int(kwargs.get("num_subjects", 13)),
            temporal_filters=int(kwargs.get("temporal_filters", 16)),
            depth_multiplier=int(kwargs.get("depth_multiplier", 2)),
            separable_filters=int(kwargs.get("separable_filters", 32)),
            dropout=float(kwargs.get("dropout", 0.5)),
            subject_embedding_dim=int(kwargs.get("subject_embedding_dim", 32)),
            classifier_hidden_dim=int(kwargs.get("classifier_hidden_dim", 128)),
            fuse_mode=str(kwargs.get("fuse_mode", "concat")),
        )

    raise ValueError(
        f"Unsupported model_name={model_name!r}. "
        "Expected one of: subject_aware_cnn_transformer, cnn_transformer_subject_head, "
        "multihead_cnn_transformer, subject_embedding_cnn_transformer, "
        "cnn_transformer_subject_embedding, subject_conditioned_cnn_transformer, "
        "shared_head_cnn_transformer, cnn_transformer_shared_head, "
        "cnn_baseline, baseline_cnn, mlp_baseline, eegnet_baseline, "
        "eegnet_embedding_classifier, eegnet_embedding_baseline, eegnet_encoder_classifier, "
        "eegnet_residual_encoder, eegnet_residual_classifier, eegnet_refined_encoder, "
        "eegnet_subject_embedding, subject_embedding_eegnet, subject_conditioned_eegnet, "
        "eegnet_mlp_baseline, eegnet_mlp_head, eegnet_mlp_classifier, "
        "multiscale_eegnet_classifier, multiscale_eegnet, multiscale_eegnet_encoder_classifier, "
        "subject_conditioned_multiscale_eegnet, multiscale_eegnet_subject_conditioned, "
        "subject_conditioned_multiscale_eegnet_classifier"
    )
