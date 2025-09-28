import pinecone
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-west1-gcp"
)

index = pinecone.Index("cli-commands")

model = SentenceTransformer("all-MiniLM-L6-v2")

def search_command(user_query, k=1):
    query_emb = model.encode(user_query).tolist()
    results = index.query(vector=query_emb, top_k=k, include_metadata=True)
    return [(m["metadata"]["command"], m["score"]) for m in results["matches"]]

# Example
print(search_command("I need a new nextjs project"))