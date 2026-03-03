from pathlib import Path
from langchain_core.documents import Document
from rag.code.chunker import chunk_directory
from rag.core.vectorstore import build_vectorstore


def ingest_code(repo_path: str):
    repo_name = Path(repo_path).name

    chunks = chunk_directory(repo_path)
    documents = []

    for chunk in chunks:
        documents.append(
            Document(
                page_content=(
                    f"Repository: {repo_name}\n"
                    f"File: {chunk.file}\n"
                    f"Language: {chunk.language}\n"
                    f"Type: {chunk.type}\n"
                    f"Lines: {chunk.start_line}-{chunk.end_line}\n\n"
                    f"{chunk.text}"
                ),
                metadata={
                    "id": chunk.id,
                    "language": chunk.language,
                    "file": chunk.file,
                    "type": chunk.type,
                    "name": chunk.name,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                },
            )
        )

    print(f"Total chunks: {len(chunks)}")
    print(f"Total documents: {len(documents)}")

    return documents