"""
FAISS + sentence-transformers knowledge base -- optional semantic search
backend using a local FAISS vector index.

IMPORTANT -- optional dependency
--------------------------------
Requires the `[rag]` extra (`faiss-cpu`, `sentence-transformers`, which
pulls in `torch`). Install with:

    pip install -e ".[rag]"

Distinct from `ChromaKnowledgeBase`: FAISS is a pure similarity-search
library with no built-in persistence or document store of its own, so
this class owns document storage/embedding directly and uses FAISS only
for the nearest-neighbor index -- a different architectural trade-off
(more control, no external service dependency) than Chroma's all-in-one
approach. Both implement the same `RAGRepositoryInterface`, so which one
is "better" for a given deployment is a configuration choice, not a code
choice -- exactly the point of keeping the retrieval backend pluggable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from campus_nav.core.exceptions import ModelLoadError, RAGError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.rag_interface import (
    KnowledgeDocument,
    RAGRepositoryInterface,
    RetrievalResult,
)

logger = get_logger(__name__)

_INSTALL_HINT = (
    "faiss-cpu/sentence-transformers are not installed. Install the optional "
    'rag extra with: pip install -e ".[rag]"'
)
_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class FAISSKnowledgeBase(RAGRepositoryInterface):
    """Semantic search using a local FAISS flat index and a
    sentence-transformers embedding model. Rebuilds its index lazily
    (like TFIDFKnowledgeBase) on the next search after any add_document
    call -- appropriate for a campus-sized knowledge base, and far
    simpler than maintaining an incrementally-updated FAISS index."""

    def __init__(self, embedding_model: str = _DEFAULT_EMBEDDING_MODEL) -> None:
        self._embedding_model_name = embedding_model
        self._embedder: Any | None = None
        self._documents: dict[str, KnowledgeDocument] = {}
        self._dirty = True
        self._doc_ids: list[str] = []
        self._index: Any | None = None

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ModelLoadError(_INSTALL_HINT) from exc

            try:
                self._embedder = SentenceTransformer(self._embedding_model_name)
            except Exception as exc:
                raise ModelLoadError(f"Failed to load embedding model: {exc}") from exc
        return self._embedder

    def add_document(self, document: KnowledgeDocument) -> None:
        self._documents[document.id] = document
        self._dirty = True

    def document_count(self) -> int:
        return len(self._documents)

    def _rebuild_index(self) -> None:
        self._doc_ids = list(self._documents.keys())
        if not self._doc_ids:
            self._index = None
            self._dirty = False
            return

        try:
            import faiss
        except ImportError as exc:
            raise ModelLoadError(_INSTALL_HINT) from exc

        embedder = self._get_embedder()

        texts = [
            f"{self._documents[doc_id].title}\n{self._documents[doc_id].content}"
            for doc_id in self._doc_ids
        ]
        try:
            embeddings = np.asarray(embedder.encode(texts), dtype="float32")
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(
                embeddings.shape[1]
            )  # inner product on normalized vectors = cosine similarity
            index.add(embeddings)
        except Exception as exc:
            raise RAGError(f"Failed to build FAISS index: {exc}") from exc

        self._index = index
        self._dirty = False

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        if self._dirty:
            self._rebuild_index()

        if self._index is None or not self._doc_ids:
            return []

        embedder = self._get_embedder()
        try:
            import faiss

            query_vector = np.asarray(embedder.encode([query]), dtype="float32")
            faiss.normalize_L2(query_vector)
            scores, indices = self._index.search(query_vector, min(top_k, len(self._doc_ids)))
        except Exception as exc:
            raise RAGError(f"FAISS query failed: {exc}") from exc

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            document = self._documents[self._doc_ids[idx]]
            results.append(
                RetrievalResult(document=document, score=float(score), retrieval_method="faiss")
            )
        return results
