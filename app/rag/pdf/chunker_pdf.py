import re


# -------------------------
# Cleaning
# -------------------------

def clean_text(text: str) -> str:
    """
    Normalize PDF extracted text.
    """
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# -------------------------
# Paragraph Splitting
# -------------------------

def split_by_paragraphs(text: str):
    """
    Split into meaningful paragraphs.
    """
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if len(p.strip()) > 50]


# -------------------------
# Smart Chunking
# -------------------------

def smart_chunk(paragraphs, chunk_size=800, overlap=150):
    """
    Create size-controlled chunks with overlap.
    """

    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) <= chunk_size:
            current += para + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            current = para + "\n\n"

    if current:
        chunks.append(current.strip())

    # Sliding window for large chunks
    final_chunks = []

    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            start = 0
            while start < len(chunk):
                end = start + chunk_size
                final_chunks.append(chunk[start:end])
                start = end - overlap

    return final_chunks