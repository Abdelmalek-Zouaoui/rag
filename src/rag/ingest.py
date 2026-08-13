import hashlib
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rag.config import CHROMA_PERSIST_DIR, DATA_DIR, EMBEDDING_MODEL, ONNX_PROVIDERS

LOADERS_BY_EXTENSION = {
    "txt": TextLoader,
    "md": TextLoader,
    "pdf": PyPDFLoader,
    "docx": Docx2txtLoader,
    "html": BSHTMLLoader,
    "csv": CSVLoader,
}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MARKDOWN_HEADERS = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
MANIFEST_FILENAME = "manifest.json"


def load_documents(data_dir: str = DATA_DIR):
    docs = []
    for extension, loader_cls in LOADERS_BY_EXTENSION.items():
        docs += DirectoryLoader(data_dir, glob=f"**/*.{extension}", loader_cls=loader_cls).load()
    return docs


def load_document(path: Path):
    """Load a single file with the loader matching its extension."""
    loader_cls = LOADERS_BY_EXTENSION[path.suffix.lstrip(".").lower()]
    return loader_cls(str(path)).load()


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan_data_dir(data_dir: str) -> dict:
    """Return {filename: content_hash} for every supported file currently in data_dir."""
    return {
        path.name: hash_file(path)
        for path in Path(data_dir).rglob("*")
        if path.is_file() and path.suffix.lstrip(".").lower() in LOADERS_BY_EXTENSION
    }


def load_manifest(persist_dir: str) -> dict:
    manifest_path = Path(persist_dir) / MANIFEST_FILENAME
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {}


def save_manifest(manifest: dict, persist_dir: str):
    (Path(persist_dir) / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))


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


def enrich_metadata(chunks):
    """Trim source to a bare filename; add a chunk_index for file types with no natural location marker."""
    chunk_counts = {}
    for chunk in chunks:
        filename = Path(chunk.metadata.get("source", "")).name
        chunk.metadata["source"] = filename

        has_location = "page" in chunk.metadata or "row" in chunk.metadata or "Header 1" in chunk.metadata
        if not has_location:
            chunk_counts[filename] = chunk_counts.get(filename, 0) + 1
            chunk.metadata["chunk_index"] = chunk_counts[filename] - 1

    return chunks


def build_index(data_dir: str = DATA_DIR, persist_dir: str = CHROMA_PERSIST_DIR):
    manifest = load_manifest(persist_dir)
    current_hashes = scan_data_dir(data_dir)

    if not current_hashes and not manifest:
        raise ValueError(f"No documents found in '{data_dir}'. Add .txt, .md, or .pdf files first.")

    changed_or_new = [name for name, digest in current_hashes.items() if manifest.get(name) != digest]
    deleted = [name for name in manifest if name not in current_hashes]

    if not changed_or_new and not deleted:
        print("No changes detected. Index is already up to date.")
        return

    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL, providers=ONNX_PROVIDERS)
    store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    for filename in deleted + changed_or_new:
        store.delete(where={"source": filename})

    if changed_or_new:
        documents = []
        for filename in changed_or_new:
            match = next(Path(data_dir).rglob(filename))
            documents += load_document(match)
        chunks = enrich_metadata(split_documents(documents))
        store.add_documents(chunks)

    for filename in deleted:
        manifest.pop(filename)
    manifest.update({name: current_hashes[name] for name in changed_or_new})
    save_manifest(manifest, persist_dir)

    print(
        f"Updated {len(changed_or_new)} file(s), removed {len(deleted)} file(s) "
        f"from '{persist_dir}'."
    )


if __name__ == "__main__":
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    build_index()
