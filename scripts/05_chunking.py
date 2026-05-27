"""
We break down long texts, compare chunking strategies
"""
import os
import re
import time
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import torch
import torch.nn.functional as F

load_dotenv()

FIXED_INDEX    = "arxiv-chunks-fixed"
SEMANTIC_INDEX = "arxiv-chunks-semantic"
VECTOR_DIM     = 768
BATCH_SIZE     = 32      # smaller batches — chunks are shorter, padding less wasteful
CHUNK_SIZE     = 100     # words per fixed chunk
OVERLAP        = 20      # word overlap between fixed chunks
MAX_WORDS      = 100     # max words per semantic chunk

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base", use_safetensors=True)
model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True, use_safetensors=True)
model.eval()


def encode_batch(texts: list) -> np.ndarray:
    with torch.no_grad():
        inputs = tokenizer(
            texts, padding=True, truncation=True,
            return_tensors="pt", return_token_type_ids=False, max_length=512,
        )
        output = model(**inputs)
        emb = output.last_hidden_state[:, 0, :]
        return F.normalize(emb, p=2, dim=1).cpu().numpy()


# ── Chunking strategies ───────────────────────────────────────────────────────

def fixed_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list:
    words = text.split()
    step = size - overlap
    result = []
    for i in range(0, max(1, len(words) - overlap), step):
        chunk = " ".join(words[i : i + size])
        if chunk:
            result.append(chunk)
    return result


def semantic_chunks(text: str, max_words: int = MAX_WORDS) -> list:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    result, current, current_wc = [], [], 0
    for sent in sentences:
        wc = len(sent.split())
        if current_wc + wc > max_words and current:
            result.append(" ".join(current))
            current, current_wc = [], 0
        current.append(sent)
        current_wc += wc
    if current:
        result.append(" ".join(current))
    return result


# ── Pinecone helpers ──────────────────────────────────────────────────────────

def ensure_index(name: str) -> object:
    if not pc.has_index(name):
        pc.create_index(
            name=name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)
    return pc.Index(name)


def batch_gen(iterable, size):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, size))


# ── Data ──────────────────────────────────────────────────────────────────────

df = pd.read_parquet("data/arxiv_subset.parquet")
df["abstract_len"] = df["abstract"].str.split().str.len()
top30 = df.nlargest(30, "abstract_len").reset_index(drop=True)

print(f"Selected 30 articles. Longest abstract: {top30['abstract_len'].max()} words")

fixed_idx    = ensure_index(FIXED_INDEX)
semantic_idx = ensure_index(SEMANTIC_INDEX)


# ── Build + upsert ────────────────────────────────────────────────────────────

def process_and_upsert(pinecone_index, chunk_fn, strategy_name):
    records = []
    for _, row in top30.iterrows():
        for i, chunk_text in enumerate(chunk_fn(row["abstract"])):
            records.append({
                "chunk_id":  f"{row['id']}_chunk_{i}",
                "chunk_text": chunk_text,
                "arxiv_id":  row["id"],
                "title":     row["title"],
                "chunk_i":   i,
                "year":      int(row["year"]),
                "category":  row["category"],
            })

    print(f"\n{strategy_name}: {len(records)} chunks from 30 articles")

    for batch in tqdm(batch_gen(records, BATCH_SIZE), desc=f"Upserting {strategy_name}"):
        embeddings = encode_batch([r["chunk_text"] for r in batch])
        vectors = [
            {
                "id": r["chunk_id"],
                "values": emb.tolist(),
                "metadata": {
                    "arxiv_id":   r["arxiv_id"],
                    "title":      r["title"],
                    "chunk_text": r["chunk_text"][:500],
                    "chunk_i":    r["chunk_i"],
                    "year":       r["year"],
                    "category":   r["category"],
                },
            }
            for r, emb in zip(batch, embeddings)
        ]
        pinecone_index.upsert(vectors=vectors)


process_and_upsert(fixed_idx,    fixed_chunks,    "Fixed-size")
process_and_upsert(semantic_idx, semantic_chunks, "Semantic")


# ── Search comparison ─────────────────────────────────────────────────────────

def search(pinecone_index, query: str, top_k: int = 5) -> list:
    vec = encode_batch([query])[0].tolist()
    return pinecone_index.query(vector=vec, top_k=top_k, include_metadata=True)["matches"]


test_queries = [
    "neural network image classification",
    "natural language processing transformers",
    "reinforcement learning reward optimization",
]

for query in test_queries:
    print(f"\n{'=' * 60}")
    print(f"Query: '{query}'")
    for strategy, idx in [("Fixed-size", fixed_idx), ("Semantic", semantic_idx)]:
        print(f"\n  [{strategy}]")
        for m in search(idx, query):
            meta = m["metadata"]
            print(f"    score={m['score']:.4f} | chunk {meta['chunk_i']} | {meta['title']}")
            print(f"      '{meta['chunk_text'][:120]}...'")
