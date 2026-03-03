# rag-service/app/rag/core/vectorstore.py

from pathlib import Path
import shutil
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ===============================
# Configuration
# ===============================

BASE_VECTOR_DIR = Path("vectorstores")
BASE_VECTOR_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ===============================
# Embeddings
# ===============================

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


# ===============================
# Internal Helpers
# ===============================

def _store_path(store_type: str, name: str) -> Path:
    """
    store_type: 'repo' or 'pdf'
    name: repo_name or pdf_name
    """
    path = BASE_VECTOR_DIR / store_type / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _meta_path(store_path: Path) -> Path:
    return store_path / "meta.json"


def _write_meta(store_path: Path, chunk_count: int):
    meta = {
        "embedding_model": EMBED_MODEL,
        "chunk_count": chunk_count,
    }
    with open(_meta_path(store_path), "w") as f:
        json.dump(meta, f)


def _read_meta(store_path: Path):
    meta_file = _meta_path(store_path)
    if not meta_file.exists():
        return None
    with open(meta_file, "r") as f:
        return json.load(f)


# ===============================
# Build / Load Vectorstore
# ===============================

def build_vectorstore(documents, store_type: str, name: str, force: bool = False):
    embeddings = get_embeddings()
    store_path = BASE_VECTOR_DIR / store_type / name

    # If store exists
    if store_path.exists():

        # If not forcing rebuild → just load
        if not force and (store_path / "index.faiss").exists():
            return FAISS.load_local(
                str(store_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )

        # If forcing rebuild → try delete safely
        if force:
            try:
                shutil.rmtree(store_path)
            except PermissionError:
                raise RuntimeError(
                    "Vectorstore is currently in use. Restart server before rebuilding."
                )

    # Create fresh store
    store_path.mkdir(parents=True, exist_ok=True)

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(str(store_path))
    _write_meta(store_path, len(documents))

    return vectorstore

def load_vectorstore(store_type: str, name: str):
    embeddings = get_embeddings()
    store_path = _store_path(store_type, name)

    return FAISS.load_local(
        str(store_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )