from pathlib import Path
from langchain_core.documents import Document

from rag.pdf.loader_pdf import extract_pdf_text
from rag.pdf.chunker_pdf import clean_text, split_by_paragraphs, smart_chunk


def ingest_pdf(path: str):
    """
    Full PDF ingestion pipeline:
    Extract → Clean → Split → Chunk → Convert to Documents
    """

    source_name = Path(path).name

    # 1️⃣ Extract
    raw_text = extract_pdf_text(path)

    if not raw_text.strip():
        return []

    # 2️⃣ Clean
    cleaned = clean_text(raw_text)

    # 3️⃣ Split
    paragraphs = split_by_paragraphs(cleaned)

    # 4️⃣ Chunk
    chunks = smart_chunk(paragraphs)

    # 5️⃣ Convert to Documents
    documents = []

    for i, chunk in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source_type": "pdf",
                    "source_name": source_name,
                    "chunk_id": i,
                },
            )
        )

    return documents