from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

from rag.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, LLM_MODEL, OLLAMA_BASE_URL

PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the context below. If the context doesn't
contain the answer, say you don't know.

Context:
{context}

Question: {question}
"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(persist_dir: str = CHROMA_PERSIST_DIR):
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


def ask(question: str, persist_dir: str = CHROMA_PERSIST_DIR) -> str:
    return build_chain(persist_dir).invoke(question)
