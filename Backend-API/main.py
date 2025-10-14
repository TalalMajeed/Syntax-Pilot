from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Ecello Pilot API",
    description="API for Ecello Pilot Application",
    version="1.0.0"
)

@app.get("/")
def read_root():
    html_content = """
    <div style="font-family: Consolas, monospace; margin: 20px;">
        <h1>Ecello Pilot API</h1>
        <p>Version 1.0.0 - Ecello Labs</p>
    </div>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/health")
def health_check():
    return {"status": "ok"}

class QueryRequest(BaseModel):
    query: str 

class QueryResponse(BaseModel):
    response: str

@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    print(f"Received query: {request.query}")
    response_text = "echo hello world"
    return QueryResponse(response=response_text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)