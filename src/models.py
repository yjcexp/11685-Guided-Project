"""Model definitions for EEG classification and retrieval"""
import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """Baseline MLP model for EEG classification"""

    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.mlp(x)


class EEGCNN(nn.Module):
    """CNN model for EEG classification"""

    def __init__(self, in_channels, conv_channels, kernel_size, pool_size, num_classes):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        prev_channels = in_channels

        for channels in conv_channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(prev_channels, channels, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(),
                    nn.MaxPool1d(pool_size)
                )
            )
            prev_channels = channels

        self.classifier = nn.Linear(prev_channels, num_classes)

    def forward(self, x):
        # x shape: (batch, channels, time)
        for conv in self.conv_layers:
            x = conv(x)

        x = x.mean(dim=-1)  # Global average pooling
        return self.classifier(x)


class RetrievalEncoder(nn.Module):
    """Encoder for EEG retrieval task"""

    def __init__(self, encoder_type, input_dim, embedding_dim, projection_dim):
        super().__init__()

        self.encoder_type = encoder_type

        if encoder_type == 'mlp':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(embedding_dim, embedding_dim)
            )

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        embedding = self.encoder(x)
        projection = self.projection(embedding)
        return embedding, projection
