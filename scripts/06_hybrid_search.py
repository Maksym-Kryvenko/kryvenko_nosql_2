"""
Combining BM25 and vector search via RRF
"""
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import torch
import torch.nn.functional as F

load_dotenv()

INDEX_NAME = "arxiv-papers"
TOP_K = 10   # retrieve wider to give RRF more to re-rank
RRF_K = 60   # standard RRF constant

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


# ── Build BM25 index ──────────────────────────────────────────────────────────
corpus_texts = (df["title"] + " " + df["abstract"]).tolist()
tokenized    = [doc.lower().split() for doc in corpus_texts]
bm25         = BM25Okapi(tokenized)
print(f"BM25 index built over {len(corpus_texts)} documents.")


# ── Search functions ──────────────────────────────────────────────────────────

def bm25_search(query: str, top_k: int = TOP_K) -> list:
    """Returns list of (df_index, score)."""
    scores  = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_idx]


def vector_search(query: str, top_k: int = TOP_K) -> list:
    """Returns list of (pinecone_id, score, metadata)."""
    results = index.query(vector=encode(query), top_k=top_k, include_metadata=True)
    return [(m["id"], m["score"], m["metadata"]) for m in results["matches"]]


def rrf_fusion(bm25_results: list, vector_results: list, k: int = RRF_K) -> list:
    """
    Reciprocal Rank Fusion.
    bm25_results:   [(df_index, score), ...]
    vector_results: [(pinecone_id, score, metadata), ...]
    Returns: [(pinecone_id, rrf_score), ...] sorted descending.
    """
    rrf_scores: dict = {}

    for rank, (df_idx, _) in enumerate(bm25_results):
        pid = f"paper_{df.iloc[df_idx]['id']}"
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank + 1)

    for rank, (pid, _, _) in enumerate(vector_results):
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank + 1)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ── Display helpers ───────────────────────────────────────────────────────────

def show_bm25(results):
    print("  BM25 top-5:")
    for rank, (idx, score) in enumerate(results[:5], 1):
        row = df.iloc[idx]
        print(f"    {rank}. [{row['category']} {row['year']}] {row['title']}  (BM25={score:.4f})")


def show_vector(results):
    print("  Vector top-5:")
    for rank, (pid, score, meta) in enumerate(results[:5], 1):
        print(f"    {rank}. [{meta.get('category')} {meta.get('year')}] {meta.get('title')}  (cos={score:.4f})")


def show_hybrid(rrf_results, meta_map):
    print("  Hybrid (RRF) top-5:")
    for rank, (pid, rrf_score) in enumerate(rrf_results[:5], 1):
        if pid in meta_map:
            meta = meta_map[pid]
            title    = meta.get("title", pid)
            category = meta.get("category", "?")
            year     = meta.get("year", "?")
        else:
            arxiv_id = pid.replace("paper_", "")
            row      = df[df["id"] == arxiv_id]
            if not row.empty:
                r        = row.iloc[0]
                title    = r["title"]
                category = r["category"]
                year     = r["year"]
            else:
                title, category, year = pid, "?", "?"
        print(f"    {rank}. [{category} {year}] {title}  (RRF={rrf_score:.5f})")


# ── Demo queries ──────────────────────────────────────────────────────────────

queries = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text",
]

for query in queries:
    print(f"\n{'=' * 60}")
    print(f"Query: '{query}'")

    bm25_res   = bm25_search(query)
    vec_res    = vector_search(query)
    rrf_res    = rrf_fusion(bm25_res, vec_res)
    meta_map   = {pid: meta for pid, _, meta in vec_res}

    print()
    show_bm25(bm25_res)
    print()
    show_vector(vec_res)
    print()
    show_hybrid(rrf_res, meta_map)
