"""
Keyword-based knowledge base -- the default, zero-ML-dependency RAG
backend.

Design rationale
-----------------
For a pilot with a few dozen to a few hundred campus facts, exact/partial
term overlap is a perfectly serviceable retrieval strategy, and it has a
decisive advantage over an embedding-based approach for this project's
Phase 1.4 stage: it's exact, debuggable, and requires no model download,
no GPU, and no vector database. `ChromaKnowledgeBase` (optional `[rag]`
extra) exists for when genuine semantic search becomes worth its
complexity — e.g. matching "where can I get food" to a document titled
"Student Canteen" with no literal word overlap.
"""

from __future__ import annotations

import re

from campus_nav.interfaces.rag_interface import (
    KnowledgeDocument,
    RAGRepositoryInterface,
    RetrievalResult,
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Same rationale as tfidf_knowledge_base.py's _STOPWORDS: without this, a
# query like "what's for lunch" can match on the common word "for" alone
# and surface an unrelated document with a nonzero score.
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


def _tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_PATTERN.findall(text.lower()) if t not in _STOPWORDS}


_TITLE_SUBSTRING_BONUS = 0.75


class KeywordKnowledgeBase(RAGRepositoryInterface):
    """In-memory, pure-Python term-overlap retriever."""

    def __init__(self) -> None:
        self._documents: dict[str, KnowledgeDocument] = {}
        self._token_index: dict[str, set[str]] = {}  # doc id -> content+title tokens

    def add_document(self, document: KnowledgeDocument) -> None:
        self._documents[document.id] = document
        self._token_index[document.id] = _tokenize(f"{document.title} {document.content}")

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        query_tokens = _tokenize(query)
        query_lower = query.lower().strip()

        scored: list[RetrievalResult] = []
        for doc_id, doc_tokens in self._token_index.items():
            document = self._documents[doc_id]

            overlap_score = 0.0
            if query_tokens:
                overlap_score = len(query_tokens & doc_tokens) / len(query_tokens)

            title_bonus = (
                _TITLE_SUBSTRING_BONUS
                if query_lower and query_lower in document.title.lower()
                else 0.0
            )

            score = overlap_score + title_bonus
            if score > 0:
                scored.append(
                    RetrievalResult(document=document, score=score, retrieval_method="keyword")
                )

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def document_count(self) -> int:
        return len(self._documents)
