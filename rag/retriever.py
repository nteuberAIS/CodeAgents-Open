"""RAG retriever — query interface for the Notion content vector database.

Queries ChromaDB for relevant content chunks based on semantic similarity,
with optional metadata filtering (entity type, status, sprint). Returns
ranked results that can be formatted as prompt-ready context strings.
"""

from __future__ import annotations

import logging
from typing import Any

from rag.ingest import COLLECTION_NAME

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Query interface for the Notion content vector database."""

    def __init__(self, settings: Any) -> None:
        """Initialize ChromaDB client and load the collection.

        Args:
            settings: Settings instance with chroma_db_path, embedding_model,
                      ollama_base_url, rag_top_k, rag_score_threshold.
        """
        try:
            import chromadb
            from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
        except ImportError as e:
            raise ImportError(
                "chromadb is required for retrieval. Install with: pip install chromadb"
            ) from e

        self._default_top_k: int = getattr(settings, "rag_top_k", 5)
        self._default_score_threshold: float | None = getattr(
            settings, "rag_score_threshold", None
        )

        client = chromadb.PersistentClient(path=str(settings.chroma_db_path))
        ef = OllamaEmbeddingFunction(
            model_name=settings.embedding_model,
            url=settings.ollama_base_url,
        )

        try:
            self._collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
            )
        except ValueError:
            logger.warning(
                "Collection '%s' not found. Run 'python main.py ingest' first.",
                COLLECTION_NAME,
            )
            self._collection = None

    def query(
        self,
        text: str,
        top_k: int | None = None,
        entity_types: list[str] | None = None,
        status: list[str] | None = None,
        sprint_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """Query the vector DB for relevant content.

        Args:
            text: The query text (task description, sprint goal, etc.)
            top_k: Maximum results to return. None uses the settings default.
            entity_types: Filter to specific types (e.g., ["work_item", "doc"]).
            status: Filter to specific statuses (e.g., ["Ready", "In Progress"]).
            sprint_id: Filter to a specific sprint's content.
            score_threshold: Minimum similarity score (0.0-1.0). None uses the
                settings default; pass explicitly to override.

        Returns:
            List of result dicts ordered by relevance (best first).
        """
        if self._collection is None:
            return []

        top_k = top_k if top_k is not None else self._default_top_k
        threshold = (
            score_threshold
            if score_threshold is not None
            else self._default_score_threshold
        )

        # Build where filter from non-None parameters
        where = self._build_where(entity_types, status, sprint_id)

        query_kwargs: dict[str, Any] = {
            "query_texts": [text],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        raw = self._collection.query(**query_kwargs)

        # Unpack ChromaDB's nested list structure (one query → index 0)
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        results: list[dict] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            score = 1.0 - dist  # cosine distance → similarity
            if threshold is not None and score < threshold:
                continue
            result = {
                "content": doc,
                "score": score,
            }
            result.update(meta)
            results.append(result)

        return results

    def format_results(self, results: list[dict], max_chars: int = 4000) -> str:
        """Format retrieval results as a prompt-ready string.

        Each result is formatted as a section with metadata header and content.
        Truncates to max_chars budget.

        Args:
            results: Output from query().
            max_chars: Character budget for the formatted output.

        Returns:
            Formatted string ready for prompt injection, or empty string.
        """
        if not results:
            return ""

        parts: list[str] = []
        remaining = max_chars

        for r in results:
            entity_type = r.get("entity_type", "unknown")
            name = r.get("name", "Untitled")
            status = r.get("status", "")
            content = r.get("content", "")

            header = f"[{entity_type}] {name}"
            if status:
                header += f" \u2014 {status}"

            section = f"{header}\n{content}"

            if len(section) > remaining:
                # If nothing added yet, truncate this result to fit
                if not parts:
                    parts.append(f"{header}\n{content[:remaining - len(header) - 1]}")
                # Otherwise, try to fit a truncated version if there's room for header + some content
                elif remaining > len(header) + 50:
                    content_budget = remaining - len(header) - 1
                    parts.append(f"{header}\n{content[:content_budget]}")
                # Not enough room — stop
                break

            parts.append(section)
            remaining -= len(section) + 2  # account for separator

        return "\n\n".join(parts)

    @staticmethod
    def _build_where(
        entity_types: list[str] | None,
        status: list[str] | None,
        sprint_id: str | None,
    ) -> dict | None:
        """Build a ChromaDB where-filter dict from optional parameters."""
        filters: list[dict] = []

        if entity_types:
            if len(entity_types) == 1:
                filters.append({"entity_type": entity_types[0]})
            else:
                filters.append({"entity_type": {"$in": entity_types}})

        if status:
            if len(status) == 1:
                filters.append({"status": status[0]})
            else:
                filters.append({"status": {"$in": status}})

        if sprint_id:
            filters.append({"sprint_id": sprint_id})

        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}
