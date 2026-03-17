import json
import pandas as pd
from pathlib import Path


def parse_json_list_cell(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return []
    try:
        val = json.loads(s)
        return val if isinstance(val, list) else []
    except Exception:
        return []


def _is_blank_string(series):
    series = series.astype("string")
    return series.isna() | (series.str.strip() == "")


def _normalize_name_list(value):
    if not isinstance(value, list):
        return []

    cleaned = []
    seen = set()
    for item in value:
        if item is None:
            continue
        name = str(item).strip()
        if not name:
            continue
        if name not in seen:
            cleaned.append(name)
            seen.add(name)
    return cleaned


def _authors_from_row(row):
    contributors = row.get("contributors", [])
    if isinstance(contributors, list) and contributors:
        return contributors

    author_raw = row.get("author_raw", "")
    if pd.isna(author_raw):
        return []
    author_name = str(author_raw).strip()
    return [author_name] if author_name else []


def load_guardian_year(year, folder):
    path = Path(folder) / f"guardian_opinion_{year}.csv"
    df = pd.read_csv(path, keep_default_na=False)

    string_cols = [
        "id", "title", "summary", "url", "author_raw",
        "section_id", "section_name", "body_text"
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    for col in ["keywords", "contributors"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_json_list_cell)

    if "n_contributors" in df.columns:
        df["n_contributors"] = pd.to_numeric(df["n_contributors"], errors="coerce").fillna(0).astype(int)

    df["year"] = int(year)
    return df


def load_guardian_years(years, folder, drop_duplicates=True):
    dfs = []

    for year in years:
        df_year = load_guardian_year(year, folder=folder)
        dfs.append(df_year)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    if drop_duplicates and "id" in df.columns:
        df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    return df


def clean_guardian_articles(df, min_body_text_chars=1000):
    """
    Replicates cleaning logic from guardian_data_exploration.ipynb
    (excluding exploration/plotting cells):
      1) Remove rows with missing author info:
         n_contributors == 0 and author_raw is missing/blank
      2) Remove rows with missing/blank body_text
      3) Remove rows with missing/blank summary
      4) Keep rows where body_text_length >= min_body_text_chars
    Also normalizes contributor lists and builds a robust authors list.
    """
    if df.empty:
        return df.copy()

    cleaned = df.copy()

    # Ensure consistent list payloads for contributor/keyword columns.
    if "contributors" in cleaned.columns:
        cleaned["contributors"] = cleaned["contributors"].apply(_normalize_name_list)
    else:
        cleaned["contributors"] = [[] for _ in range(len(cleaned))]

    if "keywords" in cleaned.columns:
        cleaned["keywords"] = cleaned["keywords"].apply(_normalize_name_list)
    else:
        cleaned["keywords"] = [[] for _ in range(len(cleaned))]

    # Notebook logic: invalid author only when both n_contributors==0 and author_raw is blank.
    n_contrib = pd.to_numeric(cleaned.get("n_contributors", 0), errors="coerce").fillna(0).astype(int)
    author_raw = cleaned.get("author_raw", pd.Series([""] * len(cleaned), index=cleaned.index))
    body_text = cleaned.get("body_text", pd.Series([""] * len(cleaned), index=cleaned.index))
    summary = cleaned.get("summary", pd.Series([""] * len(cleaned), index=cleaned.index))

    missing_author = (n_contrib == 0) & _is_blank_string(author_raw)
    missing_body_text = _is_blank_string(body_text)
    missing_summary = _is_blank_string(summary)

    cleaned = cleaned.loc[~(missing_author | missing_body_text | missing_summary)].copy()

    cleaned["body_text"] = cleaned["body_text"].astype("string")
    cleaned["body_text_length"] = cleaned["body_text"].str.len()
    cleaned = cleaned.loc[cleaned["body_text_length"] >= int(min_body_text_chars)].copy()

    # Handle multiple authors robustly for app usage.
    cleaned["authors"] = cleaned.apply(_authors_from_row, axis=1)
    cleaned["n_contributors"] = cleaned["authors"].apply(len).astype(int)
    cleaned["author_display"] = cleaned["authors"].apply(lambda names: ", ".join(names))

    # section_id/section_name are constant in this dataset and not needed downstream.
    cleaned = cleaned.drop(columns=["section_id", "section_name"], errors="ignore")

    return cleaned.reset_index(drop=True)


def load_and_clean_guardian_years(years, folder, drop_duplicates=True, min_body_text_chars=1000):
    df = load_guardian_years(years=years, folder=folder, drop_duplicates=drop_duplicates)
    return clean_guardian_articles(df, min_body_text_chars=min_body_text_chars)
