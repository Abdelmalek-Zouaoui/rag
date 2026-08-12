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
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rag.config import CHROMA_PERSIST_DIR, DATA_DIR, EMBEDDING_MODEL

LOADERS_BY_EXTENSION = {
    "txt": TextLoader,
    "md": TextLoader,
    "pdf": PyPDFLoader,
    "docx": Docx2txtLoader,
    "html": UnstructuredHTMLLoader,
    "csv": CSVLoader,
}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MARKDOWN_HEADERS = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]


def load_documents(data_dir: str = DATA_DIR):
    docs = []
    for extension, loader_cls in LOADERS_BY_EXTENSION.items():
        docs += DirectoryLoader(data_dir, glob=f"**/*.{extension}", loader_cls=loader_cls).load()
    return docs


def split_documents(documents):
    """Split markdown docs by heading/section first, everything else by size alone."""
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    header_splitter = MarkdownHeaderTextSplitter(MARKDOWN_HEADERS, strip_headers=False)

    def is_markdown(doc):
        return Path(doc.metadata.get("source", "")).suffix.lower() == ".md"

    markdown_docs = [d for d in documents if is_markdown(d)]
    other_docs = [d for d in documents if not is_markdown(d)]

    chunks = char_splitter.split_documents(other_docs)
    for doc in markdown_docs:
        for section in header_splitter.split_text(doc.page_content):
            section.metadata.update(doc.metadata)
            chunks += char_splitter.split_documents([section])

    return chunks


def build_index(data_dir: str = DATA_DIR, persist_dir: str = CHROMA_PERSIST_DIR):
    documents = load_documents(data_dir)
    if not documents:
        raise ValueError(f"No documents found in '{data_dir}'. Add .txt, .md, or .pdf files first.")

    chunks = split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents into '{persist_dir}'.")


if __name__ == "__main__":
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    build_index()
