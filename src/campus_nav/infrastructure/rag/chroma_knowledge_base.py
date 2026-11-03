"""
ChromaDB + sentence-transformers knowledge base -- optional semantic
search backend.

IMPORTANT -- optional dependency
--------------------------------
Requires the `[rag]` extra (`chromadb`, `sentence-transformers`, which
pulls in `torch`). Install with:

    pip install -e ".[rag]"

Lazy-imported on first use, matching the WhisperSTT/YOLOv8Detector pattern
established in Phases 1.1/1.3.
"""

from __future__ import annotations

from typing import Any

from campus_nav.core.exceptions import ModelLoadError, RAGError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.rag_interface import (
    KnowledgeDocument,
    RAGRepositoryInterface,
    RetrievalResult,
)

logger = get_logger(__name__)

_INSTALL_HINT = (
    "chromadb/sentence-transformers are not installed. Install the optional "
    'rag extra with: pip install -e ".[rag]"'
)


class ChromaKnowledgeBase(RAGRepositoryInterface):
    """Semantic search over campus knowledge using a local (in-memory or
    on-disk) ChromaDB collection and a sentence-transformers embedding
    model."""

    def __init__(
        self, collection_name: str = "campus_knowledge", persist_directory: str | None = None
    ) -> None:
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._collection: Any | None = None
        self._documents: dict[str, KnowledgeDocument] = {}

    def _get_collection(self) -> Any:
        if self._collection is None:
            try:
                import chromadb
                from chromadb.utils import embedding_functions
            except ImportError as exc:
                raise ModelLoadError(_INSTALL_HINT) from exc

            try:
                client = (
                    chromadb.PersistentClient(path=self._persist_directory)
                    if self._persist_directory
                    else chromadb.Client()
                )
                embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
                self._collection = client.get_or_create_collection(
                    name=self._collection_name, embedding_function=embedding_fn
                )
            except Exception as exc:
                raise ModelLoadError(f"Failed to initialize ChromaDB collection: {exc}") from exc
        return self._collection

    def add_document(self, document: KnowledgeDocument) -> None:
        collection = self._get_collection()
        try:
            collection.upsert(
                ids=[document.id],
                documents=[f"{document.title}\n{document.content}"],
                metadatas=[{"category": document.category.value, **document.metadata}],
            )
            self._documents[document.id] = document
        except Exception as exc:
            raise RAGError(f"Failed to index document {document.id!r}: {exc}") from exc

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        collection = self._get_collection()
        try:
            results = collection.query(query_texts=[query], n_results=top_k)
        except Exception as exc:
            raise RAGError(f"ChromaDB query failed: {exc}") from exc

        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: list[RetrievalResult] = []
        for doc_id, distance in zip(ids, distances, strict=False):
            document = self._documents.get(doc_id)
            if document is None:
                continue
            # Chroma reports distance (lower = more similar); convert to a
            # score where higher = more relevant, consistent with
            # RAGRepositoryInterface's documented convention.
            score = 1.0 / (1.0 + distance)
            retrieved.append(
                RetrievalResult(document=document, score=score, retrieval_method="chroma")
            )

        return retrieved

    def document_count(self) -> int:
        return len(self._documents)
