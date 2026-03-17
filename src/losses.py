"""Loss functions for EEG classification and retrieval"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """Standard cross entropy loss for classification"""

    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.ce_loss(logits, targets)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for retrieval"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, eeg_embeddings, caption_embeddings):
        """Compute contrastive loss between EEG and caption embeddings"""
        # Normalize embeddings
        eeg_embeddings = F.normalize(eeg_embeddings, dim=-1)
        caption_embeddings = F.normalize(caption_embeddings, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(eeg_embeddings, caption_embeddings.T) / self.temperature

        # Create labels (positive pairs are on diagonal)
        batch_size = eeg_embeddings.size(0)
        labels = torch.arange(batch_size, device=eeg_embeddings.device)

        # Compute symmetric contrastive loss
        loss_eeg_to_caption = F.cross_entropy(logits, labels)
        loss_caption_to_eeg = F.cross_entropy(logits.T, labels)

        return (loss_eeg_to_caption + loss_caption_to_eeg) / 2
