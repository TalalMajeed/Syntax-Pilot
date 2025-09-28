import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Define collection name
collection_name = "syntax-pilot-commands"

# Wrap into a Chroma embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Get or create collection
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"}  # cosine similarity
)

# Seed commands (only run once; IDs must be unique)
commands = [
    {"id": "1", "command": "npx create-next-app", "text": "I need a new nextjs project"},
    {"id": "2", "command": "git clone https://github.com/vercel/nextjs-boilerplate.git", "text": "please give me a nextjs boilerplate"},
]

try:
    collection.add(
        ids=[cmd["id"] for cmd in commands],
        documents=[cmd["text"] for cmd in commands],
        metadatas=[{"command": cmd["command"]} for cmd in commands],
    )
    print("Commands added successfully!")
except Exception as e:
    print("Skipping add (maybe already exists):", e)