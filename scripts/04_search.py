"""
Three types of search: semantic, with filters, and metric comparison
"""
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import torch
import torch.nn.functional as F

load_dotenv()

INDEX_NAME = "arxiv-papers"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base", use_safetensors=True)
model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True, use_safetensors=True)
model.eval()


def encode(text: str) -> list:
    with torch.no_grad():
        inputs = tokenizer(
            text, padding=True, truncation=True,
            return_tensors="pt", return_token_type_ids=False, max_length=512,
        )
        output = model(**inputs)
        emb = output.last_hidden_state[:, 0, :]
        return F.normalize(emb, p=2, dim=1).squeeze(0).tolist()


def print_matches(matches):
    for i, m in enumerate(matches, 1):
        meta = m["metadata"]
        print(f"  {i}. [{meta.get('category')} {meta.get('year')}] {meta.get('title')}")
        print(f"     score={m['score']:.4f} | {meta.get('abstract', '')[:100]}...")


# ── 1. Pure semantic search ───────────────────────────────────────────────────
print("=" * 60)
print("1. SEMANTIC SEARCH")
query = "teaching machines to recognize objects in pictures"
print(f"Query: '{query}'")
results = index.query(vector=encode(query), top_k=TOP_K, include_metadata=True)
print_matches(results["matches"])


# ── 2. Filtered search ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2a. FILTERED: category=cs.LG AND year >= 2021")
results_a = index.query(
    vector=encode("reinforcement learning"),
    top_k=TOP_K,
    include_metadata=True,
    filter={"$and": [{"category": {"$eq": "cs.LG"}}, {"year": {"$gte": 2021}}]},
)
print_matches(results_a["matches"])

print("\n" + "=" * 60)
print("2b. FILTERED: year < 2015")
results_b = index.query(
    vector=encode("reinforcement learning"),
    top_k=TOP_K,
    include_metadata=True,
    filter={"year": {"$lt": 2015}},
)
print_matches(results_b["matches"])


# ── 3. Local metric comparison ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. LOCAL METRIC COMPARISON")
query_m = "deep learning for natural language processing"
print(f"Query: '{query_m}'")
all_embeddings = np.load("embeddings/embeddings.npy")   # (N, 768), normalized
query_vec = np.array(encode(query_m))                   # (768,), normalized

cosine_scores = all_embeddings @ query_vec  # cosine == dot product for unit vectors
dot_scores    = all_embeddings @ query_vec
l2_distances  = np.linalg.norm(all_embeddings - query_vec, axis=1)

for label, scores, ascending in [
    ("Cosine similarity", cosine_scores, False),
    ("Dot product",       dot_scores,    False),
    ("L2 distance",       l2_distances,  True),
]:
    order = np.argsort(scores) if ascending else np.argsort(scores)[::-1]
    top = order[:TOP_K]
    print(f"\nTop-{TOP_K} by {label}:")
    for rank, idx in enumerate(top, 1):
        row = df.iloc[idx]
        print(f"  {rank}. [{row['category']} {row['year']}] {row['title']}  ({scores[idx]:.4f})")
