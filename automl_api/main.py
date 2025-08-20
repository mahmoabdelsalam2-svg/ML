# automl_api/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from automl_api import settings
# Correctly import all routers at once
from automl_api.routers import health, tabular_router, image_router, gemini_router

app = FastAPI(title="Unified AutoML API", version="3.0.0")

# --- Middleware ---
# This allows your frontend to communicate with the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Routers ---
# Register all the different parts of your API.
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(tabular_router.router, prefix="/api/v1/tabular", tags=["Tabular AutoML"])
app.include_router(image_router.router, prefix="/api/v1/image", tags=["Image AutoML (CNN)"])
app.include_router(gemini_router.router, prefix="/api/v1/gemini", tags=["Generative AI"])

# --- Serve Frontend ---
# This section makes your FastAPI server also serve the index.html file.
frontend_path = Path(__file__).parent.parent / "frontend"

app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/", response_class=FileResponse, tags=["Root"])
async def read_root():
    """Serves the main index.html file as the root page."""
    return str(frontend_path / "index.html")
