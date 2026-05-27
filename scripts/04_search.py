"""
Three types of search: semantic, with filters, and metric comparison
"""

import pandas as pd
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from tqdm import tqdm
import torch.nn.functional as F
import torch
import numpy as np
import os

INDEX_NAME = "arxiv-papers"

tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
model = AutoAdapterModel.from_pretrained('allenai/specter2_base', use_safetensors=True)
model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True, use_safetensors=True)

input_text = input("Input your query: ")

model.eval()
with torch.no_grad():
    inputs = tokenizer(input_text, padding=True, truncation=True,
                            return_tensors="pt", return_token_type_ids=False, max_length=512)
    output = model(**inputs)
    embedding = output.last_hidden_state[:, 0, :]
    normalized_embedding = F.normalize(embedding, p=2, dim=1)

pc = Pinecone(api_key="YOUR_API_KEY")
index = pc.Index(host=INDEX_NAME)

results = index.search(
    query={
        "top_k": t,
        "inputs": {
            'vector': normalized_embedding
        }
    }
)

for res in results['result']:
    print(res)
