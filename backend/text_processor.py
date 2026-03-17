from abc import ABC
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VectorTextProcessor(ABC):
    def __init__(
        self,
        articles,
        text_column="body_text",
        id_column="id",
    ):
        if not isinstance(articles, pd.DataFrame):
            raise TypeError("articles must be a pandas DataFrame.")
        if text_column not in articles.columns:
            raise ValueError(f"Column '{text_column}' not found in articles.")
        if id_column not in articles.columns:
            raise ValueError(f"Column '{id_column}' not found in articles.")

        self.articles = articles.reset_index(drop=True).copy()
        self.text_column = text_column
        self.id_column = id_column

        text_series = self.articles[text_column].astype("string").fillna("").str.strip()
        if (text_series == "").all():
            raise ValueError(f"Column '{text_column}' contains no non-empty text.")
        self.corpus = text_series.tolist()

        id_series = self.articles[id_column].astype("string").fillna("").str.strip()
        if (id_series == "").any():
            raise ValueError(f"Column '{id_column}' contains blank document ids.")
        duplicate_ids = id_series[id_series.duplicated()].unique()
        if len(duplicate_ids) > 0:
            sample_dupes = ", ".join(duplicate_ids[:5].tolist())
            raise ValueError(
                f"Duplicate document ids found in '{id_column}' (sample: {sample_dupes})."
            )

        self.doc_ids = id_series.tolist()
        self.doc_to_idx = dict(zip(self.doc_ids, range(len(self.doc_ids))))
        self.n_docs = len(self.corpus)

        self.token_pattern = r"(?u)\b\w\w+\b"
        self.lowercase = True
        self.strip_accents = "unicode"
        self.stop_words = "english"

        self.fitted = False

        # Subclasses should set self.vectorizer to an instance with appropriate parameters.
        self.vectorizer = None

    def fit_transform(self):
        if self.vectorizer is None:
            raise RuntimeError("vectorizer is not configured.")
        self.term_doc_matrix = self.vectorizer.fit_transform(self.corpus)
        self.fitted = True
        self.terms = self.vectorizer.get_feature_names_out()
        self.term_to_idx = self.vectorizer.vocabulary_
        self.n_terms = len(self.terms)
        return self.term_doc_matrix

    def get_term_doc_matrix(self):
        if not self.fitted:
            raise RuntimeError("Must call fit_transform() before get_term_doc_matrix().")
        return self.term_doc_matrix

    def get_doc_idx_by_id(self, doc_id):
        idx = self.doc_to_idx.get(doc_id)
        if idx is None:
            raise ValueError(f"Document ID {doc_id} not found in corpus.")
        return idx

    def get_doc_vector(self, doc_id):
        if not self.fitted:
            raise RuntimeError("Must call fit_transform() before get_doc_vector().")
        idx = self.get_doc_idx_by_id(doc_id)
        return self.term_doc_matrix[idx]

    def get_term_vector(self, term):
        if not self.fitted:
            raise RuntimeError("Must call fit_transform() before get_term_vector().")
        if term is None:
            raise ValueError("Term must be a non-empty string.")
        term_key = str(term).strip()
        if self.lowercase:
            term_key = term_key.lower()
        term_idx = self.term_to_idx.get(term_key)
        if term_idx is None:
            raise ValueError(f"Term '{term}' not found in vocabulary.")
        return self.term_doc_matrix[:, term_idx]

    def search(self, query, top_n=100):
        if not self.fitted:
            raise RuntimeError("Must call fit_transform() before search().")
        
        if top_n is None:
            top_n = self.n_docs
        
        if top_n <= 0:
            return []

        query_vec = self.vectorizer.transform([str(query)])
        cosine_similarities = cosine_similarity(query_vec, self.term_doc_matrix).ravel()
        top_indices = np.argsort(cosine_similarities)[::-1][:top_n]
        return [
            (self.articles.iloc[i].to_dict(), float(cosine_similarities[i]))
            for i in top_indices
            if cosine_similarities[i] > 0
        ]


class TfProcessor(VectorTextProcessor):
    def __init__(
        self,
        articles,
        text_column="body_text",
        id_column="id",
    ):
        super().__init__(
            articles,
            text_column=text_column,
            id_column=id_column,
        )
        self.vectorizer = CountVectorizer(
            analyzer="word",
            token_pattern=self.token_pattern,
            lowercase=self.lowercase,
            strip_accents=self.strip_accents,
            stop_words=self.stop_words,
        )

class TfidfProcessor(VectorTextProcessor):
    def __init__(
        self,
        articles,
        text_column="body_text",
        id_column="id",
    ):
        super().__init__(
            articles,
            text_column=text_column,
            id_column=id_column,
        )
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=self.token_pattern,
            lowercase=self.lowercase,
            strip_accents=self.strip_accents,
            stop_words=self.stop_words,
        )
