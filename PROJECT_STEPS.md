# Project Steps

A log of what's been done to set up the `rag` project, in order.

1. **Scaffolded the project** — Python + LangChain RAG pipeline (`src/rag/`: `config.py`,
   `ingest.py`, `chain.py`, `cli.py`), with a Chroma vector store for retrieval.
2. **Initialized git and created the GitHub repository** — private repo at
   [Abdelmalek-Zouaoui/rag](https://github.com/Abdelmalek-Zouaoui/rag), initial scaffold pushed
   to `main`.
3. **Added a collaborator** — invited `AmineOuatt` with write access.
4. **Containerized the app** — added `Dockerfile`, `docker-compose.yml`, and `.dockerignore` so
   the pipeline can run without a local Python install.
5. **Switched model providers** — moved generation from local Ollama to the
   [Groq API](https://console.groq.com) (`llama-3.3-70b-versatile`) for fast hosted inference,
   and moved embeddings to a local HuggingFace `sentence-transformers` model
   (`all-MiniLM-L6-v2`, CPU, no API key) since Groq doesn't serve embedding models.

See [PLAN.md](PLAN.md) for what's next and how the remaining work is split.
