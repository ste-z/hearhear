import re


def normalize_paragraph_text(text):
    value = "" if text is None else str(text)
    return re.sub(r"\s+", " ", value).strip()


def _split_long_paragraph(paragraph, max_chars):
    if not max_chars or max_chars <= 0 or len(paragraph) <= max_chars:
        return [paragraph]

    chunks = []
    remaining = paragraph
    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars)
        if split_at < int(max_chars * 0.5):
            split_at = max_chars
        chunk = remaining[:split_at].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


def split_into_paragraphs(text, min_chars=20, max_chars=None):
    value = "" if text is None else str(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not value:
        return []

    raw_paragraphs = re.split(r"\n\s*\n+", value)
    if len(raw_paragraphs) <= 1:
        raw_paragraphs = value.split("\n")

    paragraphs = []
    for raw_paragraph in raw_paragraphs:
        paragraph = normalize_paragraph_text(raw_paragraph)
        if len(paragraph) < min_chars:
            continue
        paragraphs.extend(_split_long_paragraph(paragraph, max_chars=max_chars))

    return paragraphs


def paragraph_rows_from_text(
    article_text,
    article_id=None,
    prefix="p",
    min_chars=20,
    max_chars=None,
):
    paragraphs = split_into_paragraphs(
        article_text,
        min_chars=min_chars,
        max_chars=max_chars,
    )
    rows = []
    for idx, paragraph in enumerate(paragraphs):
        rows.append(
            {
                "paragraph_id": f"{prefix}{idx}",
                "paragraph_index": idx,
                "article_id": article_id,
                "paragraph": paragraph,
            }
        )
    return rows
