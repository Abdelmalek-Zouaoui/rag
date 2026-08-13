# rag

A retrieval-augmented generation (RAG) pipeline built with LangChain: generation via the
[Groq API](https://console.groq.com) (fast hosted inference of open models like Llama 3), and
embeddings via a local HuggingFace `sentence-transformers` model (runs on CPU, no API key needed).

A FastAPI server + web UI (`server.py` / `ui/`) sits on top of the same pipeline, so you can
upload documents and chat from the browser instead of the CLI.

## Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/keys)

## Setup (local Python)

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install -r requirements.txt
copy .env.example .env
```

Set `GROQ_API_KEY` in `.env`.

## Usage (local Python, CLI)

1. Drop your documents (`.txt`, `.md`, `.pdf`) into `data/`.
2. Build the index:
   ```
   python -m rag.cli ingest
   ```
3. Ask questions:
   ```
   python -m rag.cli query "What is this document about?"
   ```

## Usage (web server)

The web UI wraps the same `ingest`/`query` pipeline behind a browser chat interface, with
drag-and-drop-free file upload straight into `data/`.

1. Finish the local Python setup above (venv, `pip install -e .`, `pip install -r requirements.txt`,
   `.env` with `GROQ_API_KEY` set).
2. Start the server:
   ```
   uvicorn server:app --reload
   ```
3. Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
4. Upload a document with the attach button — it's saved into `data/` and the index is rebuilt
   automatically. Then ask questions in the chat box.

Notes:
- Every upload currently triggers a full re-index of everything in `data/` (see `PLAN.md`'s
  "incremental re-indexing" item for the planned improvement).
- `--reload` is for local development only; drop it for anything closer to production.

## Usage (Docker)

1. Set `GROQ_API_KEY` in your shell or a `.env` file next to `docker-compose.yml`.
2. Build the image:
   ```
   docker compose build
   ```
3. Drop your documents (`.txt`, `.md`, `.pdf`) into `data/` (mounted into the app container).
4. Build the index:
   ```
   docker compose run --rm app ingest
   ```
5. Ask questions:
   ```
   docker compose run --rm app query "What is this document about?"
   ```

Indexed vectors persist in the `chroma_db` named volume. The first run downloads the embedding
model (~90 MB) into the container's HuggingFace cache.

## Project layout

```
src/rag/
  config.py   environment-driven settings
  ingest.py   document loading, chunking, and embedding into Chroma
  chain.py    retrieval + generation chain
  cli.py      command-line entrypoint
data/         source documents (gitignored contents, kept via .gitkeep)
ui/           web frontend (index.html, style.css, app.js)
server.py     FastAPI server: serves ui/, handles uploads, answers chat questions
```

## Swapping models or providers

Model names and the Groq API key are read from `.env` (see `.env.example`). To use a different
LLM or embeddings provider, swap the `ChatGroq`/`HuggingFaceEmbeddings` instances in
`chain.py`/`ingest.py` for the equivalent LangChain integration.

## Project history and roadmap

- [PROJECT_STEPS.md](PROJECT_STEPS.md) — log of what's been set up so far
- [PLAN.md](PLAN.md) — remaining work, split between contributors