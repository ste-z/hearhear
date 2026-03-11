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