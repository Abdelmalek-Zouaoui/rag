from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from rag.config import CHROMA_PERSIST_DIR, DATA_DIR, EMBEDDING_MODEL, GROQ_API_KEY, LLM_MODEL
from rag.ingest import load_documents, split_documents

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


def build_chain(data_dir: str = DATA_DIR, persist_dir: str = CHROMA_PERSIST_DIR):
    # 1. Vector Retriever (Semantic search via ChromaDB)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # 2. Keyword Retriever (Exact text search via BM25)
    raw_docs = load_documents(data_dir)
    chunks = split_documents(raw_docs)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 6

    # 3. Hybrid Retriever (Combine 50% BM25 + 50% Chroma) → returns 12 candidates
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )

    # 4. Reranker (Cross-Encoder scores each candidate and keeps top 4 best)
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=4)
    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble_retriever
    )

    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


def ask(question: str, data_dir: str = DATA_DIR, persist_dir: str = CHROMA_PERSIST_DIR) -> str:
    return build_chain(data_dir=data_dir, persist_dir=persist_dir).invoke(question)


def ask_stream(question: str, data_dir: str = DATA_DIR, persist_dir: str = CHROMA_PERSIST_DIR):
    return build_chain(data_dir=data_dir, persist_dir=persist_dir).stream(question)