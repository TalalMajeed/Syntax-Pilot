# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Ecello Pilot API",
    description="API for Ecello Pilot Application",
    version="1.0.0"
)

# Example data model
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    in_stock: bool = True


# Root route
@app.get("/")
def read_root():
    html_content = """
    <div style="font-family: Consolas, monospace; margin: 20px;">
        <h1>Ecello Pilot API</h1>
        <p>Version 1.0.0 - Ecello Labs</p>
    </div>
    """
    return HTMLResponse(content=html_content, status_code=200)

# Example health check route
@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)