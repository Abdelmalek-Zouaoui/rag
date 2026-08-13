"""Compare local embedding models on speed and retrieval quality for real project data.

Usage: python scripts/evaluate_embeddings.py
Free/local models only (see PLAN.md) - no paid embedding APIs are considered.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag.config import DATA_DIR
from rag.ingest import enrich_metadata, load_document, scan_data_dir, split_documents

CANDIDATE_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",  # current default
    "BAAI/bge-small-en-v1.5",
]

TEST_QUESTIONS = [
    "How does someone become a member of MicroClub?",
    "What happens if a member is inactive on Discord?",
    "What are the rules around department recruitment?",
]


def load_chunks(data_dir: str):
    documents = []
    for filename in scan_data_dir(data_dir):
        documents += load_document(next(Path(data_dir).rglob(filename)))
    return enrich_metadata(split_documents(documents))


def evaluate(model_name: str, chunks):
    print(f"\n{'=' * 70}\n{model_name}\n{'=' * 70}")

    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    start = time.time()
    store = Chroma.from_documents(chunks, embedding=embeddings)
    elapsed = time.time() - start
    print(f"Embedded {len(chunks)} chunks in {elapsed:.2f}s ({elapsed / len(chunks) * 1000:.1f} ms/chunk)")

    for question in TEST_QUESTIONS:
        print(f"\nQ: {question}")
        for rank, doc in enumerate(store.similarity_search(question, k=2), start=1):
            snippet = doc.page_content[:160].replace("\n", " ")
            print(f"  [{rank}] {snippet}...")

    return elapsed


def main():
    chunks = load_chunks(DATA_DIR)
    print(f"Loaded {len(chunks)} chunks from '{DATA_DIR}' for evaluation.")

    timings = {model: evaluate(model, chunks) for model in CANDIDATE_MODELS}

    print(f"\n{'=' * 70}\nSpeed summary\n{'=' * 70}")
    for model, elapsed in timings.items():
        print(f"{model}: {elapsed:.2f}s total")


if __name__ == "__main__":
    main()
