from io import BytesIO

from pypdf import PdfReader


def extract_pdf_text(uploaded_file, max_pages=20, max_chars=20000):
    """
    Extract text from an uploaded PDF file.
    Limits pages/chars so very large PDFs do not overwhelm search.
    """
    if uploaded_file is None:
        return ""

    filename = (uploaded_file.filename or "").lower()
    content_type = (uploaded_file.mimetype or "").lower()

    if not filename.endswith(".pdf") and content_type != "application/pdf":
        return ""

    try:
        file_bytes = uploaded_file.read()
        if not file_bytes:
            return ""

        reader = PdfReader(BytesIO(file_bytes))
        chunks = []

        for page in reader.pages[:max_pages]:
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if page_text:
                chunks.append(page_text)
            if sum(len(chunk) for chunk in chunks) >= max_chars:
                break

        text = "\n\n".join(chunks).strip()
        return text[:max_chars]
    except Exception:
        return ""
    finally:
        try:
            uploaded_file.stream.seek(0)
        except Exception:
            pass
