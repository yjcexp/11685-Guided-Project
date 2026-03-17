"""Evaluation metrics for classification and retrieval"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
import matplotlib.pyplot as plt


def compute_accuracy(predictions, targets):
    """Compute classification accuracy"""
    preds = predictions.argmax(dim=-1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    return accuracy_score(targets_np, preds)


def compute_confusion_matrix(predictions, targets, num_classes):
    """Compute confusion matrix"""
    preds = predictions.argmax(dim=-1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    return confusion_matrix(targets_np, preds, labels=range(num_classes))


def compute_recall_at_k(query_embeddings, target_embeddings, k_values=[1, 5, 10]):
    """Compute Recall@K for retrieval"""
    # Normalize embeddings
    query_embeddings = query_embeddings / query_embeddings.norm(dim=-1, keepdim=True)
    target_embeddings = target_embeddings / target_embeddings.norm(dim=-1, keepdim=True)

    # Compute similarity scores
    similarity = torch.matmul(query_embeddings, target_embeddings.T)

    results = {}
    for k in k_values:
        # Get top-k predictions
        _, top_k_indices = similarity.topk(k, dim=-1)

        # Compute recall (assuming targets are on diagonal)
        batch_size = similarity.size(0)
        labels = torch.arange(batch_size, device=similarity.device).unsqueeze(-1)

        recall = (top_k_indices == labels).any(dim=-1).float().mean().item()
        results[f'Recall@{k}'] = recall

    return results


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig
