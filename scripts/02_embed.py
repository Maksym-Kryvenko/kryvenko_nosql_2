"""
We encode texts with the specter2 model, store vectors
"""
import pandas as pd
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from tqdm import tqdm
import torch.nn.functional as F
import torch
import numpy as np
import os

data = pd.read_parquet("data/arxiv_subset.parquet")
tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
model = AutoAdapterModel.from_pretrained('allenai/specter2_base', use_safetensors=True)
model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True, use_safetensors=True)

data['to_embed'] = data.apply(lambda x: x["title"] + tokenizer.sep_token + x.get('abstract', ""), axis=1)

all_embeddings = []
model.eval()
with torch.no_grad():
    for i in tqdm(range(0, len(data), 100)):
        text_batch = data.iloc[i:i+10]["to_embed"].values.tolist()
        inputs = tokenizer(text_batch, padding=True, truncation=True,
                                return_tensors="pt", return_token_type_ids=False, max_length=512)
        output = model(**inputs)
        embeddings = output.last_hidden_state[:, 0, :]
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.extend(
            normalized_embeddings.cpu().numpy()
        )

print(f"Processed {len(all_embeddings)} records")
print(f"Embed size: {len(all_embeddings[0])}")
print(f"Vector norm: {np.linalg.norm(all_embeddings[0], ord=2)}")

try:
    os.makedirs("embeddings", exist_ok=True)
    np.save("embeddings/embeddings.npy", all_embeddings)
    print(f"Embeddings successfully saved to embeddings/embeddings.npy")
except:
    print(f"Failed to save the file {e}")