import pandas as pd


def _normalize_terms(terms):
    if terms is None:
        raise ValueError("terms cannot be None.")

    normalized = []
    seen = set()
    for term in list(terms):
        value = str(term).strip()
        if not value:
            raise ValueError("terms contains empty values.")
        if value in seen:
            raise ValueError(f"Duplicate term found: '{value}'.")
        normalized.append(value)
        seen.add(value)
    return normalized


def _normalize_id_series(id_values, field_name):
    if id_values is None:
        raise ValueError(f"{field_name} cannot be None.")

    normalized = pd.Series(id_values, dtype="string").fillna("").str.strip()
    if (normalized == "").any():
        raise ValueError(f"{field_name} contains blank ids.")

    duplicates = normalized[normalized.duplicated()].unique().tolist()
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"{field_name} has duplicate ids (sample: {preview}).")

    return normalized


def _normalize_doc_ids(doc_ids):
    if doc_ids is None:
        raise ValueError("doc_ids cannot be None.")
    return _normalize_id_series(doc_ids, "doc_ids").tolist()


def _normalize_articles_for_doc_ids(articles, doc_ids, id_column):
    if not isinstance(articles, pd.DataFrame):
        raise TypeError("articles must be a pandas DataFrame.")
    if id_column not in articles.columns:
        raise ValueError(f"articles must contain id column '{id_column}'.")

    normalized = articles.reset_index(drop=True).copy()
    normalized[id_column] = _normalize_id_series(normalized[id_column], id_column)

    indexed = normalized.set_index(id_column, drop=False)
    missing_ids = [doc_id for doc_id in doc_ids if doc_id not in indexed.index]
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(
            f"articles is missing ids from doc_ids (sample: {preview})."
        )

    return indexed.loc[doc_ids].reset_index(drop=True)
