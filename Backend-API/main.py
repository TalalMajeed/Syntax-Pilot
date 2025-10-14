from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from typing import List
import sqlite3
import chromadb
from chromadb.utils import embedding_functions
import os

DATABASE = "commands.db"
CHROMA_PATH = "chroma"

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS commands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            command TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

if not os.path.exists(CHROMA_PATH):
    os.makedirs(CHROMA_PATH)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = chroma_client.get_or_create_collection(
    name="commands_collection",
    embedding_function=embedding_func
)

def rebuild_chroma():
    """Rebuild the entire Chroma DB from SQLite."""
    print("Rebuilding Chroma database...")
    collection.delete(where={})  # Clear collection

    conn = get_db()
    rows = conn.execute("SELECT id, query, command FROM commands").fetchall()
    conn.close()

    if not rows:
        print("ℹNo commands found in SQLite.")
        return

    ids = [str(row["id"]) for row in rows]
    docs = [row["query"] for row in rows]
    metas = [{"command": row["command"]} for row in rows]

    collection.add(documents=docs, metadatas=metas, ids=ids)
    print(f"Indexed {len(rows)} commands into Chroma.")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

class CommandItem(BaseModel):
    query: str
    command: str

class CommandList(BaseModel):
    commands: List[CommandItem]

app = FastAPI(
    title="Ecello Pilot API",
    description="AI-powered RAG Command Server for Ecello Pilot",
    version="2.0.0"
)

@app.get("/", response_class=HTMLResponse)
def read_root():
    html = """
    <div style="font-family: monospace; margin: 20px;">
        <h1>Ecello Pilot API</h1>
        <p>Version 2.0.0 - Ecello Labs</p>
    </div>
    """
    return html

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/commands", response_model=CommandList)
def get_all_commands():
    conn = get_db()
    rows = conn.execute("SELECT query, command FROM commands").fetchall()
    conn.close()
    cmds = [CommandItem(query=row["query"], command=row["command"]) for row in rows]
    return CommandList(commands=cmds)

@app.post("/commands")
def add_command(cmd: CommandItem):
    conn = get_db()
    conn.execute("INSERT INTO commands (query, command) VALUES (?, ?)", (cmd.query, cmd.command))
    conn.commit()
    conn.close()

    rebuild_chroma()
    return {"message": "Command added and Chroma rebuilt successfully."}


@app.post("/query", response_model=QueryResponse)
def query_command(request: QueryRequest):
    print(f"Received query: {request.query}")

    results = collection.query(query_texts=[request.query], n_results=1)
    if results["documents"] and len(results["documents"][0]) > 0:
        best_match = results["metadatas"][0][0]["command"]
        print(f"Matched command: {best_match}")
        return QueryResponse(response=best_match)

    print("No match found, returning default response.")
    return QueryResponse(response="echo hello world")


@app.on_event("startup")
def on_startup():
    init_db()
    rebuild_chroma()
    print("Ecello Pilot RAG API ready!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)