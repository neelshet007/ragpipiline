"""
FastAPI Server Entrypoint
Mounts REST routes, WebSocket routes, CORS middleware, and static files.
"""

import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app.api.routes import router as api_router
from backend.app.api.websockets import ws_router

app = FastAPI(
    title="HH Goa 2026 — Voice-Enabled RAG System",
    description="Production-quality multilingual sub-200ms Voice-Enabled RAG System built on ai4bharat/MSMARCO-XI dataset.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration for Frontend Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(api_router)
app.include_router(ws_router)

# Mount Frontend Static Web Interface
frontend_path = os.path.abspath("frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {
            "title": "HH Goa 2026 — Voice-Enabled RAG System",
            "docs": "/docs",
            "health": "/health",
            "api_v1_query": "/api/v1/query",
            "api_v1_voice": "/api/v1/voice",
            "websocket": "/ws/rag"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
