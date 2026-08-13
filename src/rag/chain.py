import threading
from collections import defaultdict

from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders.base import BaseCrossEncoder
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq

from rag.config import CHROMA_PERSIST_DIR, DATA_DIR, GROQ_API_KEY, LLM_MODEL
from rag.ingest import enrich_metadata, load_documents, split_documents
from rag.models import get_cross_encoder_model, get_embeddings

# Broad "what's in here" questions have no single chunk that represents the
# whole document, so similarity search tends to surface an unrelated chunk
# that merely shares vocabulary with the question. Route these to the
# opening chunks of each file instead, where titles/intros actually live.
SUMMARY_PHRASES = (
    "what is this document about",
    "what is this about",
    "what are these documents about",
    "summarize",
    "summary",
    "overview",
    "what does this cover",
    "main topic",
    "main points",
    "key points",
)


def is_summary_question(question: str) -> bool:
    q = question.lower()
    return any(phrase in q for phrase in SUMMARY_PHRASES)


def leading_chunks_per_file(chunks, n: int = 3):
    by_source = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.metadata.get("source", "")].append(chunk)
    return [chunk for file_chunks in by_source.values() for chunk in file_chunks[:n]]


class FastEmbedCrossEncoder(BaseCrossEncoder):
    """Adapts fastembed's shared ONNX cross-encoder to LangChain's reranker interface."""

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        return list(get_cross_encoder_model().rerank_pairs(text_pairs))

PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful document assistant. Answer the question using ONLY the context below.

Formatting rules:
- Use clear paragraphs separated by blank lines.
- Use bullet points or numbered lists when listing multiple items.
- Use **bold** for key terms or important phrases.
- Keep your answer well-structured and easy to read.
- At the end, list the sources you used (e.g. [Source: filename, Page X]).
- If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}
"""
)


def format_docs(docs):
    formatted_chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", None)
        page_str = f", Page {page + 1}" if page is not None else ""
        header_str = f"[Source {i}: {source}{page_str}]\n"
        formatted_chunks.append(header_str + doc.page_content)
    return "\n\n".join(formatted_chunks)


def build_chain(data_dir: str = DATA_DIR, persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = "default"):
    # 1. Vector Retriever (Semantic search via ChromaDB)
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=get_embeddings(), collection_name=collection_name)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. Keyword Retriever (Exact text search via BM25)
    raw_docs = load_documents(data_dir)
    chunks = enrich_metadata(split_documents(raw_docs))
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3

    # 3. Hybrid Retriever (Combine 50% BM25 + 50% Chroma) → returns 6 candidates
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )

    # 4. Reranker (Cross-Encoder scores each candidate and keeps top 3 best)
    reranker = CrossEncoderReranker(model=FastEmbedCrossEncoder(), top_n=3)
    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble_retriever
    )

    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY)

    overview_docs = leading_chunks_per_file(chunks)

    def get_context(question: str) -> str:
        docs = overview_docs if is_summary_question(question) else retriever.invoke(question)
        return format_docs(docs)

    return (
        {"context": RunnableLambda(get_context), "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


_chain_cache: dict = {}
_chain_cache_lock = threading.Lock()


def get_chain(data_dir: str = DATA_DIR, persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = "default"):
    """Reuse a built chain per session instead of reloading it on every message.

    Without this, concurrent chat requests each build their own copy of the BM25
    index of the whole corpus at the same time. The embedding/reranker models
    themselves are separately shared across ALL sessions (see rag.models) -
    only the cheap, session-specific pieces (BM25, the Chroma collection
    reference) get built per collection_name.
    """
    key = (data_dir, persist_dir, collection_name)
    if key not in _chain_cache:
        with _chain_cache_lock:
            if key not in _chain_cache:
                _chain_cache[key] = build_chain(data_dir=data_dir, persist_dir=persist_dir, collection_name=collection_name)
    return _chain_cache[key]


def invalidate_chain_cache(data_dir: str = None, persist_dir: str = None, collection_name: str = None):
    """Call after build_index() adds/removes documents so BM25 picks up the change.

    With no arguments, clears every session's cached chain. Pass all three to
    drop just one session's entry instead of forcing every other session to
    rebuild too.
    """
    if data_dir is None:
        _chain_cache.clear()
        return
    _chain_cache.pop((data_dir, persist_dir, collection_name), None)


def ask(question: str, data_dir: str = DATA_DIR, persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = "default") -> str:
    return get_chain(data_dir=data_dir, persist_dir=persist_dir, collection_name=collection_name).invoke(question)


def ask_stream(question: str, data_dir: str = DATA_DIR, persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = "default"):
    return get_chain(data_dir=data_dir, persist_dir=persist_dir, collection_name=collection_name).stream(question)