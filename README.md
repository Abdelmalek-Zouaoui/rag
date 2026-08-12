# rag

A local retrieval-augmented generation (RAG) pipeline built with LangChain, running entirely on
open-source models via [Ollama](https://ollama.com) — no external API keys required.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Pull the models used by default:
  ```
  ollama pull llama3.1
  ollama pull nomic-embed-text
  ```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install -r requirements.txt
copy .env.example .env
```

## Usage

1. Drop your documents (`.txt`, `.md`, `.pdf`) into `data/`.
2. Build the index:
   ```
   python -m rag.cli ingest
   ```
3. Ask questions:
   ```
   python -m rag.cli query "What is this document about?"
   ```

## Project layout

```
src/rag/
  config.py   environment-driven settings
  ingest.py   document loading, chunking, and embedding into Chroma
  chain.py    retrieval + generation chain
  cli.py      command-line entrypoint
data/         source documents (gitignored contents, kept via .gitkeep)
```

## Swapping models or providers

Model names and the Ollama endpoint are read from `.env` (see `.env.example`). To use a different
provider (OpenAI, Anthropic, etc.) instead of local models, swap the `OllamaEmbeddings`/`ChatOllama`
instances in `ingest.py`/`chain.py` for the equivalent LangChain integration.
