import math

from backend.text_processing.sentence_splitter import sentence_rows_from_text


DEFAULT_SEMANTIC_BREAK_SIMILARITY_THRESHOLD = 0.2
DEFAULT_SEMANTIC_MAX_CHARS = 15000


def _clean_text(value):
    return " ".join(str(value or "").split()).strip()


def _cosine_similarity(left, right):
    if left is None or right is None:
        return None
    try:
        left_values = [float(value) for value in left]
        right_values = [float(value) for value in right]
    except (TypeError, ValueError):
        return None

    if not left_values or not right_values or len(left_values) != len(right_values):
        return None

    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left_values, right_values)
    )
    return float(dot_product / (left_norm * right_norm))


def _project_sentence(processor, sentence):
    if processor is None or not hasattr(processor, "project_query"):
        return None
    try:
        return processor.project_query(sentence, normalize=True)
    except Exception:
        return None


def semantic_chunk_rows_from_text(
    article_text,
    article_id=None,
    prefix="sc",
    svd_processor=None,
    similarity_threshold=DEFAULT_SEMANTIC_BREAK_SIMILARITY_THRESHOLD,
    max_chars=DEFAULT_SEMANTIC_MAX_CHARS,
):
    sentence_rows = sentence_rows_from_text(article_text, prefix=f"{prefix}_s")
    if not sentence_rows:
        return []

    resolved_threshold = float(similarity_threshold)
    resolved_max_chars = int(max_chars or DEFAULT_SEMANTIC_MAX_CHARS)

    chunks = []
    current_sentences = []
    current_start = 0
    previous_vector = None

    def flush():
        if not current_sentences:
            return
        text = _clean_text(" ".join(current_sentences))
        if not text:
            return
        chunk_index = len(chunks)
        chunks.append(
            {
                "paragraph_id": f"{prefix}{chunk_index}",
                "paragraph_index": chunk_index,
                "article_id": article_id,
                "paragraph": text,
                "sentence_start_index": current_start,
                "sentence_end_index": current_start + len(current_sentences) - 1,
            }
        )

    for sentence_index, sentence_row in enumerate(sentence_rows):
        sentence = _clean_text(sentence_row.get("sentence"))
        if not sentence:
            continue

        sentence_vector = _project_sentence(svd_processor, sentence)
        current_text = _clean_text(" ".join([*current_sentences, sentence]))
        should_break = False

        if current_sentences:
            similarity = _cosine_similarity(previous_vector, sentence_vector)
            if similarity is not None and similarity < resolved_threshold:
                should_break = True
            if resolved_max_chars > 0 and len(current_text) > resolved_max_chars:
                should_break = True

        if should_break:
            flush()
            current_sentences = [sentence]
            current_start = sentence_index
        elif not current_sentences:
            current_sentences = [sentence]
            current_start = sentence_index
        else:
            current_sentences.append(sentence)

        if sentence_vector is not None:
            previous_vector = sentence_vector

    flush()
    return chunks
