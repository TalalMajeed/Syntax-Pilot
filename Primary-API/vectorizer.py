import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Load environment variables (expects PINECONE_API_KEY in .env)
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define index name
index_name = "syntax-pilot-commands"

# Create index if it doesn’t exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,             # SentenceTransformer "all-MiniLM-L6-v2"
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",           # or "gcp"
            region="us-east-1"     # check region in your Pinecone console
        )
    )

index = pc.Index(index_name)

with open("commands.json") as f:
    data = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

vectors = []
for i, d in enumerate(data):
    emb = model.encode(d["query"]).tolist()
    vectors.append({
        "id": f"cmd-{i}",
        "values": emb,
        "metadata": {
            "command": d["command"],
            "query": d["query"]
        }
    })

index.upsert(vectors=vectors)

print(f"Upserted {len(vectors)} commands into Pinecone index '{index_name}'")