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
import threading
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Same public API your cli.py already relies on.
from rag.chain import ask_stream, invalidate_chain_cache
from rag.config import CHROMA_PERSIST_DIR
from rag.ingest import build_index

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
STATIC_DIR = ROOT_DIR / "ui"
INDEX_HTML = ROOT_DIR / "ui" / "index.html"

DATA_DIR.mkdir(exist_ok=True)

# Each browser gets its own uploaded documents via a session cookie, so one
# visitor's PDF is never searchable in another visitor's chat. The session
# ID doubles as both the per-session data subfolder name and the Chroma
# collection name - only these cheap, per-session pieces are isolated; the
# embedding/reranker models stay shared (see rag.models) since duplicating
# those per session would blow the memory budget almost immediately.
SESSION_COOKIE = "session_id"
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


def get_session_id(request: Request) -> str:
    return request.cookies.get(SESSION_COOKIE) or uuid.uuid4().hex


def session_data_dir(session_id: str) -> Path:
    path = DATA_DIR / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path

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

# The embedding/reranker models are large enough relative to a 512MB instance
# that running two requests through them at once can exceed the memory limit
# on its own. Serializing upload/chat work trades a bit of queuing latency
# for not crashing - reasonable for a low-traffic demo.
GENERATION_LOCK = threading.Lock()


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
def list_files(request: Request, response: Response) -> dict:
    session_id = get_session_id(request)
    response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax", max_age=SESSION_COOKIE_MAX_AGE)
    data_dir = session_data_dir(session_id)
    files = sorted(
        p.name for p in data_dir.iterdir() if p.is_file() and not p.name.startswith(".")
    )
    return {"files": files}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_files(request: Request, response: Response, files: List[UploadFile] = File(...)) -> UploadResponse:
    session_id = get_session_id(request)
    response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax", max_age=SESSION_COOKIE_MAX_AGE)
    data_dir = session_data_dir(session_id)

    saved: List[str] = []
    skipped: List[str] = []

    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            skipped.append(upload.filename)
            continue

        dest = data_dir / upload.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved.append(upload.filename)

    if not saved:
        raise HTTPException(
            status_code=400,
            detail=f"No supported files were uploaded. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # MVP approach: re-embed everything in this session's folder on every upload.
    # PLAN.md already tracks "incremental re-indexing" as a follow-up —
    # swap this call once build_index() supports it, no other change needed here.
    if not GENERATION_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Server is busy processing another request. Please try again in a few seconds.",
        )
    try:
        build_index(data_dir=str(data_dir), collection_name=session_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc
    finally:
        GENERATION_LOCK.release()
    invalidate_chain_cache(data_dir=str(data_dir), persist_dir=CHROMA_PERSIST_DIR, collection_name=session_id)

    return UploadResponse(saved=saved, skipped=skipped)


# ------------------------------------------------------------------
# Chat (streaming — tokens sent word-by-word like ChatGPT)
# ------------------------------------------------------------------
@app.post("/api/chat")
def chat(payload: ChatRequest, request: Request):
    session_id = get_session_id(request)
    data_dir = session_data_dir(session_id)
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    def token_generator():
        if not GENERATION_LOCK.acquire(blocking=False):
            yield "⚠️ Server is busy processing another request. Please try again in a few seconds."
            return
        try:
            for chunk in ask_stream(question, data_dir=str(data_dir), collection_name=session_id):
                yield chunk
        except Exception as exc:
            yield f"\n\n⚠️ Error: {exc}"
        finally:
            GENERATION_LOCK.release()

    resp = StreamingResponse(token_generator(), media_type="text/plain")
    resp.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax", max_age=SESSION_COOKIE_MAX_AGE)
    return resp