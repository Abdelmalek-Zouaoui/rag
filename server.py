"""
FastAPI server for the RAG web UI.

Serves the static frontend (index.html / static/style.css / static/app.js),
accepts document uploads into data/, re-triggers ingestion, and answers
questions through the existing rag.chain.ask() pipeline.

Assumes the project layout described in README.md:

    server.py          <- this file, at repo root
    ui/
        index.html
        style.css
        app.js
    data/               <- uploaded documents live here (already gitignored)
    src/rag/
        config.py
        ingest.py       <- exposes build_index()
        chain.py        <- exposes ask(question)
        cli.py

Run with:
    uvicorn server:app --reload
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Same public API your cli.py already relies on.
from rag.chain import ask, ask_stream
from rag.ingest import build_index

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
STATIC_DIR = ROOT_DIR / "ui"
INDEX_HTML = ROOT_DIR / "ui" / "index.html"

DATA_DIR.mkdir(exist_ok=True)

# Keep this in sync with the frontend's <input accept="..."> and with
# whatever loaders ingest.py actually supports.
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".csv", ".html"}

# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before deploying anywhere public
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


class UploadResponse(BaseModel):
    saved: List[str]
    skipped: List[str]


# ------------------------------------------------------------------
# Frontend
# ------------------------------------------------------------------
@app.get("/")
def serve_index() -> FileResponse:
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=500, detail="index.html not found next to server.py")
    return FileResponse(INDEX_HTML)


# ------------------------------------------------------------------
# Files
# ------------------------------------------------------------------
@app.get("/api/files")
def list_files() -> dict:
    files = sorted(
        p.name for p in DATA_DIR.iterdir() if p.is_file() and not p.name.startswith(".")
    )
    return {"files": files}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)) -> UploadResponse:
    saved: List[str] = []
    skipped: List[str] = []

    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            skipped.append(upload.filename)
            continue

        dest = DATA_DIR / upload.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved.append(upload.filename)

    if not saved:
        raise HTTPException(
            status_code=400,
            detail=f"No supported files were uploaded. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # MVP approach: re-embed everything in data/ on every upload.
    # PLAN.md already tracks "incremental re-indexing" as a follow-up —
    # swap this call once build_index() supports it, no other change needed here.
    try:
        build_index()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    return UploadResponse(saved=saved, skipped=skipped)


# ------------------------------------------------------------------
# Chat (streaming — tokens sent word-by-word like ChatGPT)
# ------------------------------------------------------------------
@app.post("/api/chat")
def chat(payload: ChatRequest):
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    def token_generator():
        try:
            for chunk in ask_stream(question):
                yield chunk
        except Exception as exc:
            yield f"\n\n⚠️ Error: {exc}"

    return StreamingResponse(token_generator(), media_type="text/plain")