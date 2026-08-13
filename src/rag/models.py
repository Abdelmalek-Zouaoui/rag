"""Shared, lazily-loaded model singletons.

The embedding and reranker models are the single biggest memory cost in this
app (verified: ~150-200MB each). Per-session isolation must not create a
separate copy of either per session, or a handful of concurrent visitors
would exceed a 512MB instance on model weights alone. Everything that is
actually session-specific (uploaded files, the Chroma collection, the BM25
index) stays cheap and per-session; these two models are the one thing kept
global.
"""

import threading

from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_community.embeddings import FastEmbedEmbeddings

from rag.config import EMBEDDING_MODEL, ONNX_PROVIDERS

RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

_lock = threading.Lock()
_embeddings = None
_cross_encoder_model = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        with _lock:
            if _embeddings is None:
                _embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL, providers=ONNX_PROVIDERS)
    return _embeddings


def get_cross_encoder_model():
    global _cross_encoder_model
    if _cross_encoder_model is None:
        with _lock:
            if _cross_encoder_model is None:
                _cross_encoder_model = TextCrossEncoder(RERANKER_MODEL, providers=ONNX_PROVIDERS)
    return _cross_encoder_model
