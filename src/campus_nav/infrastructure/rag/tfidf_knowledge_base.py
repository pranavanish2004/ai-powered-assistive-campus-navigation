"""
TF-IDF knowledge base.

Design rationale
-----------------
A genuine improvement over `KeywordKnowledgeBase`'s plain term-overlap
scoring, while staying in the same "zero new ML dependencies" tier: TF-IDF
weights rare, distinctive terms (e.g. "Biomedical") far more heavily than
common ones (e.g. "the", "building"), which directly fixes the failure
mode plain overlap has -- a query like "library" matching a document that
merely *mentions* "Library" in passing just as strongly as the document
*about* the Library. Cosine similarity over TF-IDF vectors is a standard,
well-understood retrieval baseline, implemented here with only `numpy`
(already a core dependency since Phase 1.3's vision pipeline) -- no
`scikit-learn`, no embedding model, no vector database.

The index is rebuilt lazily on the next `search()` call after any
`add_document()` -- acceptable for a knowledge base sized for a campus
(tens to low-hundreds of documents), where rebuilding a term-frequency
matrix is effectively instantaneous, and far simpler than maintaining an
incrementally-updated TF-IDF matrix.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from campus_nav.interfaces.rag_interface import (
    KnowledgeDocument,
    RAGRepositoryInterface,
    RetrievalResult,
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# A minimal, standard English stopword list. Without this, common words
# (e.g. "for", "the", "is") that happen to appear in one document more
# than another can dominate the cosine similarity score for a query whose
# actual distinguishing term (e.g. "lunch") appears in no document at all
# -- surfacing an unrelated document with high confidence instead of
# correctly finding nothing relevant. This is not exhaustive linguistic
# stopword removal (no stemming, no domain-specific list); it targets the
# small set of extremely high-frequency function words most likely to
# create exactly this false-positive-match failure mode.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "with",
        "by",
        "from",
        "as",
        "and",
        "or",
        "but",
        "if",
        "so",
        "than",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "i",
        "me",
        "my",
        "you",
        "your",
        "we",
        "our",
        "they",
        "their",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "what",
        "where",
        "when",
        "how",
        "who",
        "which",
        "whats",
    }
)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_PATTERN.findall(text.lower()) if t not in _STOPWORDS]


class TFIDFKnowledgeBase(RAGRepositoryInterface):
    """In-memory TF-IDF retrieval with cosine similarity ranking."""

    def __init__(self) -> None:
        self._documents: dict[str, KnowledgeDocument] = {}
        self._dirty = True
        self._doc_ids: list[str] = []
        self._vocabulary: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._doc_vectors: np.ndarray | None = None

    def add_document(self, document: KnowledgeDocument) -> None:
        self._documents[document.id] = document
        self._dirty = True

    def document_count(self) -> int:
        return len(self._documents)

    def _rebuild_index(self) -> None:
        self._doc_ids = list(self._documents.keys())
        doc_token_lists = [
            _tokenize(f"{self._documents[doc_id].title} {self._documents[doc_id].content}")
            for doc_id in self._doc_ids
        ]

        vocabulary: dict[str, int] = {}
        for tokens in doc_token_lists:
            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = len(vocabulary)
        self._vocabulary = vocabulary

        num_docs = len(self._doc_ids)
        num_terms = len(vocabulary)

        if num_docs == 0 or num_terms == 0:
            self._idf = np.zeros(num_terms)
            self._doc_vectors = np.zeros((num_docs, num_terms))
            self._dirty = False
            return

        term_frequency_matrix = np.zeros((num_docs, num_terms))
        document_frequency = np.zeros(num_terms)

        for row, tokens in enumerate(doc_token_lists):
            counts = Counter(tokens)
            total = len(tokens) or 1
            for token, count in counts.items():
                col = vocabulary[token]
                term_frequency_matrix[row, col] = count / total
                document_frequency[col] += 1

        # Standard smoothed IDF: log((1 + N) / (1 + df)) + 1, avoids a
        # zero or negative weight for terms appearing in every document.
        idf = np.log((1.0 + num_docs) / (1.0 + document_frequency)) + 1.0
        self._idf = idf
        self._doc_vectors = term_frequency_matrix * idf
        self._dirty = False

    def _vectorize_query(self, query: str) -> np.ndarray:
        assert self._idf is not None
        vector = np.zeros(len(self._vocabulary))
        tokens = _tokenize(query)
        if not tokens:
            return vector

        counts = Counter(tokens)
        total = len(tokens)
        for token, count in counts.items():
            col = self._vocabulary.get(token)
            if col is not None:
                vector[col] = (count / total) * self._idf[col]
        return vector

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        if self._dirty:
            self._rebuild_index()

        if not self._doc_ids or self._doc_vectors is None:
            return []

        query_vector = self._vectorize_query(query)
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []

        doc_norms = np.linalg.norm(self._doc_vectors, axis=1)
        # Avoid division by zero for documents with no indexable tokens
        # (e.g. an empty content string) by treating their similarity as 0
        # rather than NaN.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarities = (self._doc_vectors @ query_vector) / (doc_norms * query_norm)
        similarities = np.nan_to_num(similarities, nan=0.0)

        ranked_indices = np.argsort(-similarities)
        results: list[RetrievalResult] = []
        for idx in ranked_indices[:top_k]:
            score = float(similarities[idx])
            if score <= 0:
                continue
            document = self._documents[self._doc_ids[idx]]
            results.append(
                RetrievalResult(document=document, score=score, retrieval_method="tfidf")
            )
        return results
