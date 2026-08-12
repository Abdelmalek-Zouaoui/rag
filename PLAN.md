# Plan

Current state: a working MVP — ingest documents into Chroma with local HuggingFace embeddings,
answer questions with a Groq-hosted LLM. This splits the next round of work by pipeline stage.

## Abdelmalek — Ingestion (`ingest.py` and around it)

- [ ] Support more file types (`.docx`, `.html`, `.csv`)
- [ ] Smarter chunking — section/semantic-aware splitting instead of fixed-size
      `RecursiveCharacterTextSplitter`
- [ ] Attach richer metadata to chunks (source file, page number) so answers can cite sources
- [ ] Incremental re-indexing — skip re-embedding documents that haven't changed
- [ ] Evaluate embedding models — compare `all-MiniLM-L6-v2` against larger local models on
      quality vs. speed

## Amine — Retrieval & Generation (`chain.py` and around it)

- [ ] Hybrid retrieval — combine BM25 keyword search with vector search
- [ ] Add a reranking step (e.g. a cross-encoder) over retrieved chunks
- [ ] Prompt improvements — have the model cite which chunk/source it used
- [ ] Stream responses from Groq instead of waiting for the full answer
- [ ] Add a simple chat UI (e.g. Streamlit) on top of `build_chain()`
- [ ] Put together a small Q&A test set to sanity-check answer quality as the chain changes

## Shared / infra

- [ ] Keep `Dockerfile` / `docker-compose.yml` in sync as dependencies change
- [ ] Add basic automated tests (at least smoke tests for `ingest` and `query`)
- [ ] Keep `README.md` and `.env.example` up to date with new config options
- [ ] Review each other's PRs before merging to `main`

## Workflow

- One branch per feature, PR into `main`, one review before merge
- Log notable milestones in [PROJECT_STEPS.md](PROJECT_STEPS.md)
