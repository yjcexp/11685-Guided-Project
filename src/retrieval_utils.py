"""Utilities for EEG retrieval task"""
import torch
import numpy as np


def extract_embeddings(model, dataloader, device):
    """Extract embeddings from a model"""
    model.eval()
    embeddings = []

    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)

            if hasattr(model, 'forward'):
                _, projection = model(data)
                embeddings.append(projection)
            else:
                embeddings.append(model(data))

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


def compute_similarity(query_embeddings, target_embeddings):
    """Compute cosine similarity between query and target embeddings"""
    # Normalize embeddings
    query_embeddings = query_embeddings / query_embeddings.norm(dim=-1, keepdim=True)
    target_embeddings = target_embeddings / target_embeddings.norm(dim=-1, keepdim=True)

    # Compute similarity
    similarity = torch.matmul(query_embeddings, target_embeddings.T)

    return similarity


def search_top_k(query_embeddings, target_embeddings, k):
    """Search for top-k most similar items"""
    similarity = compute_similarity(query_embeddings, target_embeddings)

    # Get top-k indices and scores
    scores, indices = similarity.topk(k, dim=-1)

    return scores, indices


def batch_retrieval(query_embeddings, target_embeddings, batch_size=256):
    """Perform retrieval in batches for efficiency"""
    num_queries = query_embeddings.size(0)

    all_scores = []
    all_indices = []

    for i in range(0, num_queries, batch_size):
        batch_query = query_embeddings[i:i+batch_size]
        scores, indices = search_top_k(batch_query, target_embeddings, k=10)

        all_scores.append(scores)
        all_indices.append(indices)

    all_scores = torch.cat(all_scores, dim=0)
    all_indices = torch.cat(all_indices, dim=0)

    return all_scores, all_indices


def save_embeddings(embeddings, save_path):
    """Save embeddings to disk"""
    torch.save(embeddings, save_path)


def load_embeddings(load_path, device='cpu'):
    """Load embeddings from disk"""
    embeddings = torch.load(load_path, map_location=device)
    return embeddings
