import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to your index
index_name = "syntax-pilot-commands"
index = pc.Index(index_name)

# Load the same embedding model used in vectorizer.py
model = SentenceTransformer("all-MiniLM-L6-v2")

# Search function
def search_command(user_query, k=1):
    # Turn query into embedding
    query_emb = model.encode(user_query).tolist()
    
    # Query Pinecone
    results = index.query(
        vector=query_emb,
        top_k=k,
        include_metadata=True
    )
    
    # Extract command(s)
    return [(m["metadata"]["command"], m["score"]) for m in results["matches"]]

# 🔎 Example queries
print(search_command("I need a new nextjs project"))
print(search_command("please give me a nextjs boilerplate"))