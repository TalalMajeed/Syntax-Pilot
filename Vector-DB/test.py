import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Load index + metadata
index = faiss.read_index("commands.index")
with open("commands.json") as f:
    data = json.load(f)

# Load the same embedding model used for building
model = SentenceTransformer("all-MiniLM-L6-v2")

# Search function
def search_command(user_query, k=1):
    vec = np.array([model.encode(user_query)], dtype="float32")
    distances, indices = index.search(vec, k)
    return [(data[i]["command"], float(distances[0][j])) for j, i in enumerate(indices[0])]

# Example
print(search_command("I need a new nextjs project"))
print(search_command("Please give me a nextjs boilerplate"))