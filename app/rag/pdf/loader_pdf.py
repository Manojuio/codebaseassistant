import PyPDF2


def extract_pdf_text(path: str) -> str:
    """
    Extract raw text from PDF.
    No cleaning.
    No chunking.
    """

    text = ""

    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n\n"

    return text