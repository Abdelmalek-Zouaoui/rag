from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from rag.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, GROQ_API_KEY, LLM_MODEL

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
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


def ask(question: str, persist_dir: str = CHROMA_PERSIST_DIR) -> str:
    return build_chain(persist_dir).invoke(question)
