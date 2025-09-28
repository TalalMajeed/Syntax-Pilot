import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Load your dataset
with open("commands.json") as f:
    data = json.load(f)

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode queries into embeddings
embeddings = np.array([model.encode(d["query"]) for d in data], dtype="float32")

# Build FAISS index
dimension = embeddings.shape[1]   # should be 384 for all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save the index
faiss.write_index(index, "commands.index")
print("Index built and saved as commands.index")