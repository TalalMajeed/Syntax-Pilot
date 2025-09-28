import chromadb
from chromadb.utils import embedding_functions

# Initialize Chroma client (same persistence path)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Define collection name (same as vectorizer.py)
collection_name = "syntax-pilot-commands"

# Wrap into the same embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Load the collection
collection = chroma_client.get_collection(
    name=collection_name,
    embedding_function=sentence_transformer_ef,
)

# Search function
def search_command(user_query):
    results = collection.query(
        query_texts=[user_query],
        n_results=1,
        include=["metadatas", "distances"]
    )

    best_match = results["metadatas"][0][0]["command"]
    score = results["distances"][0][0]
    return best_match, score

if __name__ == "__main__":
    print("🔎 Syntax Pilot - Command Search")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter your query: ")
        if query.lower() in ["exit", "quit"]:
            break

        result = search_command(query)

        if result:
            command, score = result
            print(f"Best Match: {command} (Score: {score:.4f})\n")
        else:
            print("No matching command found.\n")
