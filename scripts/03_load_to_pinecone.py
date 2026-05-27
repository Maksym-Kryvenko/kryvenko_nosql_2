"""
Create an index in Pinecone, load vectors with metadata
"""

import os
import time
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200 

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(INDEX_NAME)

data = pd.read_parquet(INPUT_PARQUET)
embed = np.load(INPUT_EMBEDDINGS)

pinecone_records = []
for _e, _m in zip(embed, data.to_dict("records")):
    pinecone_records.append({
        "id": f"paper_{_m['id']}",
        "values": _e.tolist(),
        "metadata": {
            "arxiv_id": _m["id"],
            "title": _m["title"],
            "abstract": _m["abstract"][:500],
            "authors": _m["authors"][:200],
            "year": int(_m["year"]),
            "category": _m["category"],
        }
    })

def chunks(iterable, batch_size=200):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

for ids_vectors_chunk in tqdm(chunks(pinecone_records, batch_size=BATCH_SIZE), desc="Upserting"):
    index.upsert(vectors=list(ids_vectors_chunk))

stats = index.describe_index_stats()
print(f"Total vectors in index: {stats.total_vector_count}")
