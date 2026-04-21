"""
Zero-shot Image ↔ Caption Retrieval with CLIP
=============================================

- Performs retrieval in both directions (Image→Caption, Caption→Image).
- Computes Recall@K for evaluation.
- Evaluates retrieved captions with BLEU, ROUGE, METEOR, BERTScore.
- Compares similarity distributions of true pairs vs random pairs.
"""

import os
os.chdir("/media/ahmed/Data/Deep_Learning/EEG-122-electrodes")

import random
import torch
import pandas as pd
#import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# HuggingFace Transformers
from transformers import CLIPProcessor, CLIPModel

# Text evaluation metrics
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score

#import matplotlib.pyplot as plt

# ---------------------------
# 1. CONFIGURATION
# ---------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

#### Choose ONE model from this list:
# "openai/clip-vit-base-patch16"
# "openai/clip-vit-base-patch32"
# "openai/clip-vit-large-patch14"
# "openai/clip-vit-large-patch14-336"
# "jinaai/jina-clip-v2"
# "zer0int/LongCLIP-GmP-ViT-L-14"
MODEL_NAME = "openai/clip-vit-base-patch32"

captions_file = "captions.txt"
img_folder = "All_images"

# Sample 10% of the dataset for faster evaluation
df = pd.read_csv(captions_file, sep="\t")
df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)


# ---------------------------
# 2. HELPER FUNCTIONS
# ---------------------------

def load_model(model_name):
    """Load CLIP model + processor from HuggingFace."""
    model = CLIPModel.from_pretrained(model_name).to(device) # Loads the pretrained CLIP
    processor = CLIPProcessor.from_pretrained(model_name) # Loads the matching processor that handles preprocessing both images and text
    return model, processor


def encode_texts(model, processor, texts, batch_size=64):
    """Encode captions into normalized embeddings."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device) # Converts raw text into tokenized tensors for CLIP�s text encoder.
        with torch.no_grad():
            embs = model.get_text_features(**inputs) # Gets the raw embedding vectors from CLIP�s text encoder.
            embs = embs / embs.norm(dim=-1, keepdim=True) # unit length (L2 norm = 1)
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0) # Combines all batches back into one big tensor of shape (num_texts, embedding_dim)


def encode_images(model, processor, image_names, batch_size=32):
    """Encode images into normalized embeddings (only if file exists)."""
    image_embeddings = []
    valid_indices = []
    for idx, name in tqdm(enumerate(image_names), total=len(image_names)):
        # try multiple extensions
        found = False
        for ext in [".jpg", ".jpeg", ".JPEG", ".JPG"]:
            path = os.path.join(img_folder, name + ext)
            if os.path.exists(path):
                found = True
                break
        if not found:
            continue

        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs) # Extracts the image embeddings
            emb = emb / emb.norm(dim=-1, keepdim=True) # embeddings are scaled to unit norm
        image_embeddings.append(emb.cpu())
        valid_indices.append(idx)

    if len(image_embeddings) == 0:
        raise RuntimeError("No valid images found!")
    return torch.cat(image_embeddings, dim=0).to(device), valid_indices


def recall_at_k(similarity, k=1):
    """Compute Recall@K (fraction of correct matches in top-k)."""
    n = similarity.shape[0] # Each row i represents similarities between query i (say a caption) and all candidates (say images).
    correct = 0
    for i in range(n):
        ranks = similarity[i].argsort(descending=True) # argsort(descending=True): Gets ranking of candidates by similarity (highest first)
        if i in ranks[:k]: # if correct item in the top k
            correct += 1
    return correct / n


def class_recall_at_k(similarity, classes, k=1):
    """
    Class-aware Recall@K.
    similarity: [N x N] similarity matrix (image vs caption).
    classes: list of class labels aligned with rows/cols.
    """
    n = similarity.shape[0]
    correct = 0
    for i in range(n):
        # get top-k retrieved captions for image i
        ranks = similarity[i].argsort(descending=True)[:k]
        
        # check if any retrieved caption has same class as the query
        if any(classes[j] == classes[i] for j in ranks):
            correct += 1
    return correct / n


def evaluate_caption(true_cap, retrieved_cap):
    """Evaluate retrieved caption against ground truth using multiple metrics."""
    metrics = {}

    # BLEU
    metrics["BLEU"] = sentence_bleu([true_cap.split()], retrieved_cap.split())

    # METEOR
    try:
        metrics["METEOR"] = meteor_score([true_cap], retrieved_cap)
    except:
        metrics["METEOR"] = 0.0

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = scorer.score(true_cap, retrieved_cap)
    metrics["ROUGE-L"] = rouge["rougeL"].fmeasure

    # BERTScore
    P, R, F1 = bert_score([retrieved_cap], [true_cap], lang="en", verbose=False)
    metrics["BERTScore_F1"] = F1.mean().item()

    return metrics


# pairing random image and text embeddings
def random_baseline(image_embs, text_embs, n=1000):
    """Generate random similarity scores for baseline distribution."""
    scores = []
    for _ in range(n):
        i = random.randint(0, len(image_embs)-1) # pick a random image embedding
        j = random.randint(0, len(text_embs)-1) # pick a random caption embedding
        sim = (image_embs[i] @ text_embs[j].T).item()
        scores.append(sim)
    return scores


# ---------------------------
# 3. MAIN WORKFLOW
# ---------------------------

print("\n=============================")
print(f"Evaluating model: {MODEL_NAME}")
print("=============================\n")

# ---- Load Model
model, processor = load_model(MODEL_NAME)

# ---- Encode text and images
text_embs = encode_texts(model, processor, df["abstracted"].tolist())
text_embs.shape
image_embs, valid_idx = encode_images(model, processor, df["image_name"].tolist())

# keep only valid subset
text_embs = text_embs[valid_idx]
df_valid = df.iloc[valid_idx].reset_index(drop=True)

# ---- Similarity matrix
similarity_matrix = image_embs @ text_embs.T
similarity_matrix.shape

# ---- Retrieval Metrics
print("Image → Caption Retrieval")
for k in [1, 5, 10]:
    r = recall_at_k(similarity_matrix, k)
    print(f"R@{k}: {r:.3f}")

print("\nCaption → Image Retrieval")
for k in [1, 5, 10]:
    r = recall_at_k(similarity_matrix.T, k)
    print(f"R@{k}: {r:.3f}")

# ---- Caption Evaluation
print("\nCaption Evaluation")

eval_results = []
true_caps = []
retrieved_caps = []

for i in range(len(df_valid)):
    # ground truth
    true_cap = df_valid.loc[i, "abstracted"]
    true_img = df_valid.loc[i, "image_name"]

    # retrieved best caption for this image
    best_caption_idx = similarity_matrix[i].argmax().item()
    retrieved_cap = df_valid.loc[best_caption_idx, "abstracted"]
    retrieved_img = df_valid.loc[best_caption_idx, "image_name"]

    # ---- compute metrics
    metrics = {}
    metrics["BLEU"] = sentence_bleu([true_cap.split()], retrieved_cap.split())

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = scorer.score(true_cap, retrieved_cap)
    metrics["ROUGE-L"] = rouge["rougeL"].fmeasure

    # keep track of captions and IDs for later BERTScore
    metrics["true_caption"] = true_cap
    metrics["retrieved_caption"] = retrieved_cap
    metrics["true_image"] = true_img
    metrics["retrieved_image"] = retrieved_img
    true_caps.append(true_cap)
    retrieved_caps.append(retrieved_cap)

    eval_results.append(metrics)

eval_df = pd.DataFrame(eval_results)

# ---- Compute overall statistics
# mean per metric
print("Mean scores:")
print(eval_df[["BLEU", "ROUGE-L"]].mean())

# std dev
print("\nStd deviation:")
print(eval_df[["BLEU", "ROUGE-L"]].std())

# ---- Add BERTScore
P, R, F1 = bert_score(retrieved_caps, true_caps, lang="en", verbose=False)
eval_df["BERTScore_F1"] = F1.cpu().numpy()


# Compute CLIPScore for true and retrieved captions
clip_score_true = []
clip_score_retrieved = []

for i in range(len(df_valid)):
    # ground-truth caption index = i (since text_embs aligned with df_valid)
    true_idx = i
    retrieved_idx = similarity_matrix[i].argmax().item()

    # CLIP similarity for true caption
    clip_score_true.append(similarity_matrix[i, true_idx].item())

    # CLIP similarity for retrieved caption
    clip_score_retrieved.append(similarity_matrix[i, retrieved_idx].item())

# add both to dataframe
eval_df["CLIPScore_true"] = clip_score_true
eval_df["CLIPScore_retrieved"] = clip_score_retrieved

# Compare their histograms in one plot

plt.figure(figsize=(8,5))
plt.hist(eval_df["CLIPScore_true"], bins=30, alpha=0.6, label="True captions", color="steelblue")
plt.hist(eval_df["CLIPScore_retrieved"], bins=30, alpha=0.6, label="Retrieved captions", color="orange")
plt.xlabel("CLIPScore")
plt.ylabel("Frequency")
plt.title("Distribution of CLIPScore (True vs Retrieved Captions)")
plt.legend()
plt.show()

# Compute CLIPScore for random captions
clip_score_random = []

for i in range(len(df_valid)):
    # pick a random caption index different from the true one
    rand_idx = random.choice([j for j in range(len(df_valid)) if j != i])
    clip_score_random.append(similarity_matrix[i, rand_idx].item())

eval_df["CLIPScore_random"] = clip_score_random

# Histogram comparison (True vs Retrieved vs Random)
plt.figure(figsize=(8,5))
plt.hist(eval_df["CLIPScore_true"], bins=30, alpha=0.6, label="True captions", color="steelblue")
plt.hist(eval_df["CLIPScore_retrieved"], bins=30, alpha=0.6, label="Retrieved captions", color="orange")
plt.hist(eval_df["CLIPScore_random"], bins=30, alpha=0.6, label="Random captions", color="green")
plt.xlabel("CLIPScore")
plt.ylabel("Frequency")
plt.title("Distribution of CLIPScore (True vs Retrieved vs Random)")
plt.legend()
plt.show()



classes = df_valid["category"].tolist()
print("Class-aware R@1:", class_recall_at_k(similarity_matrix, classes, k=1))
print("Class-aware R@5:", class_recall_at_k(similarity_matrix, classes, k=5))
print("Class-aware R@10:", class_recall_at_k(similarity_matrix, classes, k=10))


# ---- run BERTScore once in batch ----
P, R, F1 = bert_score(retrieved_caps, true_caps, lang="en", verbose=False)
bert_scores = F1.cpu().numpy()

# add to dataframe
eval_df = pd.DataFrame(eval_results)
eval_df["BERTScore_F1"] = bert_scores

print(eval_df.head())


eval_results = []

for i in range(len(df_valid)):
    true_cap = df_valid.loc[i, "abstracted"]
    true_img = df_valid.loc[i, "image_name"]

    # retrieve best caption
    best_caption_idx = similarity_matrix[i].argmax().item()
    retrieved_cap = df_valid.loc[best_caption_idx, "abstracted"]
    retrieved_img = df_valid.loc[best_caption_idx, "image_name"]

    # compute metrics
    metrics = evaluate_caption(true_cap, retrieved_cap)

    # add identifiers for interpretability
    metrics["true_image"] = true_img
    metrics["true_caption"] = true_cap
    metrics["retrieved_caption"] = retrieved_cap
    metrics["retrieved_image"] = retrieved_img

    eval_results.append(metrics)


# Show example of real and retrieved caption by image
def show_examples(similarity_matrix, df_valid, n=5):
    """
    Print qualitative retrieval examples for inspection.
    Shows true image + caption,
    then the top retrieved caption and image.
    """
    for i in range(min(n, len(df_valid))):
        true_img = df_valid.loc[i, "image_name"]
        true_cap = df_valid.loc[i, "abstracted"]

        # Best retrieved caption for this image
        best_caption_idx = similarity_matrix[i].argmax().item()
        retrieved_cap = df_valid.loc[best_caption_idx, "abstracted"]

        # Best retrieved image for this caption
        best_img_idx = similarity_matrix[:, i].argmax().item()
        retrieved_img = df_valid.loc[best_img_idx, "image_name"]

        print(f"\nExample {i+1}")
        print("-" * 40)
        print(f"True Image:      {true_img}")
        print(f"True Caption:    {true_cap}")
        print(f"Retrieved Caption: {retrieved_cap}")
        print(f"Retrieved Image:   {retrieved_img}")
        
show_examples(similarity_matrix, df_valid, n=5)

