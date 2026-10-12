# Plan

Current state: a working MVP — ingest documents into Chroma with local HuggingFace embeddings,
answer questions with a Groq-hosted LLM. This splits the next round of work by pipeline stage.

## Abdelmalek — Ingestion (`ingest.py` and around it)

- [x] Support more file types (`.docx`, `.html`, `.csv`)
- [x] Smarter chunking — section/semantic-aware splitting instead of fixed-size
      `RecursiveCharacterTextSplitter`
- [x] Attach richer metadata to chunks (source file, page number) so answers can cite sources
- [x] Incremental re-indexing — skip re-embedding documents that haven't changed
- [x] Evaluate embedding models — compare `all-MiniLM-L6-v2` against larger local models on
      quality vs. speed (kept `all-MiniLM-L6-v2`; see `scripts/evaluate_embeddings.py`)

## Amine — Retrieval & Generation (`chain.py` and around it)

- [x] Hybrid retrieval — combine BM25 keyword search with vector search
- [x] Add a reranking step (e.g. a cross-encoder) over retrieved chunks
- [x] Prompt improvements — have the model cite which chunk/source it used
- [x] Stream responses from Groq instead of waiting for the full answer
- [x] Add a simple chat UI (FastAPI + HTML/JS/CSS) on top of `build_chain()`
- [x] Put together a small Q&A test set to sanity-check answer quality as the chain changes
- [x] Automated Ragas evaluation benchmark (`scripts/evaluate_rag.py`)

## Shared / infra

- [ ] Keep `Dockerfile` / `docker-compose.yml` in sync as dependencies change
- [ ] Add basic automated tests (at least smoke tests for `ingest` and `query`)
- [ ] Keep `README.md` and `.env.example` up to date with new config options
- [ ] Review each other's PRs before merging to `main`

## Workflow

- One branch per feature, PR into `main`, one review before merge
- Log notable milestones in [PROJECT_STEPS.md](PROJECT_STEPS.md)
