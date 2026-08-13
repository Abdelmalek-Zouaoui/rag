import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "data")

# onnxruntime's default CPU arena strategy doubles its allocation each time it
# grows and never shrinks back - on a 512MB instance this alone can eat 100MB+
# per model. Requesting exactly what's needed keeps fastembed's models lean.
ONNX_PROVIDERS = [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]
