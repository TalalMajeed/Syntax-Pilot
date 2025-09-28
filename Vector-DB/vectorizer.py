import openai
import faiss
import numpy as np
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
# Initialize OpenAI client
client = openai.OpenAI()

# Load dataset
with open("commands.json") as f:
    data = json.load(f)

queries = [d["query"] for d in data]

# Get embeddings
def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

embeddings = [get_embedding(q) for q in queries]
embeddings = np.array(embeddings, dtype="float32")

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "commands.index")