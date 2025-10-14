from fastapi import FastAPI
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

def seed_chroma_if_empty():
    """Seed Chroma from SQLite only if empty (non-destructive)."""
    try:
        count = collection.count()
    except Exception:
        # Fallback in case count is unavailable in this version
        try:
            peek = collection.peek()
            count = len(peek.get("ids", []))
        except Exception:
            count = 0

    if count and count > 0:
        print(f"Chroma already has {count} items. Skipping initial seed.")
        return

    print("Seeding Chroma from SQLite (initial only)...")
    conn = get_db()
    rows = conn.execute("SELECT id, query, command FROM commands").fetchall()
    conn.close()

    if not rows:
        print("ℹ No commands found in SQLite to seed.")
        return

    ids = [str(row["id"]) for row in rows]
    docs = [row["query"] for row in rows]
    metas = [{"command": row["command"]} for row in rows]

    collection.add(documents=docs, metadatas=metas, ids=ids)
    print(f"Seeded {len(rows)} commands into Chroma.")

def rebuild_chroma():
    """Legacy rebuild: non-destructive. Seeds only if empty."""
    print("Rebuilding Chroma database (non-destructive seed if empty)...")
    seed_chroma_if_empty()

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
    cur = conn.execute(
        "INSERT INTO commands (query, command) VALUES (?, ?)", (cmd.query, cmd.command)
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()

    # Append the new item to Chroma without rebuilding
    try:
        collection.add(
            documents=[cmd.query],
            metadatas=[{"command": cmd.command}],
            ids=[str(new_id)],
        )
    except Exception as e:
        print(f"Warning: Failed to index new command in Chroma: {e}")

    return {"message": "Command added and indexed successfully.", "id": new_id}


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
    seed_chroma_if_empty()
    print("Ecello Pilot RAG API ready!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
