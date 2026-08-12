from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import CHROMA_PERSIST_DIR, DATA_DIR, EMBEDDING_MODEL

LOADERS_BY_EXTENSION = {
    "txt": TextLoader,
    "md": TextLoader,
    "pdf": PyPDFLoader,
    "docx": Docx2txtLoader,
    "html": UnstructuredHTMLLoader,
    "csv": CSVLoader,
}


def load_documents(data_dir: str = DATA_DIR):
    docs = []
    for extension, loader_cls in LOADERS_BY_EXTENSION.items():
        docs += DirectoryLoader(data_dir, glob=f"**/*.{extension}", loader_cls=loader_cls).load()
    return docs


def build_index(data_dir: str = DATA_DIR, persist_dir: str = CHROMA_PERSIST_DIR):
    documents = load_documents(data_dir)
    if not documents:
        raise ValueError(f"No documents found in '{data_dir}'. Add .txt, .md, or .pdf files first.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents into '{persist_dir}'.")


if __name__ == "__main__":
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    build_index()
