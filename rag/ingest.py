"""Notion content ingestion into ChromaDB.

Loads Notion page content and entity metadata from local JSON snapshots,
chunks the content using a hybrid strategy (document-level for short pages,
section-level for long pages), and stores embeddings in ChromaDB via
Ollama's nomic-embed-text model.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

COLLECTION_NAME = "notion_content"

# Notion renderer markup tags that inflate token count without semantic value
_MARKUP_TAG_RE = re.compile(
    r"</?(?:table|tr|td|th|callout|toggle|column_list|column)[^>]*>",
    re.IGNORECASE,
)


def _strip_markup(text: str) -> str:
    """Strip Notion-specific markup tags to reduce token count.

    Removes HTML-like tags used by the Notion block renderer (table, tr, td,
    callout, etc.) that inflate token count without adding semantic value.
    Collapses resulting multiple blank lines.
    """
    stripped = _MARKUP_TAG_RE.sub("", text)
    # Collapse runs of blank lines
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()

# JSON snapshot filename → (entity_type, name_field)
_SNAPSHOT_REGISTRY: list[tuple[str, str, str]] = [
    ("work_items.json", "work_item", "name"),
    ("sprints.json", "sprint", "name"),
    ("docs_specs.json", "doc", "name"),
    ("decisions.json", "decision", "title"),
    ("risks_issues.json", "risk", "name"),
]


def ingest_notion_content(
    settings: Any,
    content_dir: Path,
    snapshot_dir: Path,
    force: bool = False,
) -> dict:
    """Ingest Notion content into ChromaDB.

    Args:
        settings: Settings instance with chroma_db_path, embedding_model, etc.
        content_dir: Path to data/notion/content/ with .md files.
        snapshot_dir: Path to data/notion/ with JSON snapshot files.
        force: If True, delete and recreate the collection.

    Returns:
        {"documents_ingested": int, "chunks_created": int, "collection_name": str}
    """
    # Lazy imports — ChromaDB may not be installed during testing
    try:
        import chromadb
        from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
    except ImportError as e:
        raise ImportError(
            "chromadb is required for ingestion. Install with: pip install chromadb"
        ) from e

    # Build entity metadata lookup
    entity_lookup = _build_entity_lookup(snapshot_dir)

    # Collect and chunk all content files
    all_chunks: list[dict] = []
    docs_ingested = 0

    md_files = sorted(content_dir.glob("*.md"))
    for md_path in md_files:
        notion_id = md_path.stem
        content = md_path.read_text(encoding="utf-8")
        if not content.strip():
            continue

        chunks = _chunk_content(
            content=content,
            notion_id=notion_id,
            chunk_size=settings.rag_chunk_size,
            max_chunk_size=settings.rag_max_chunk_size,
            overlap=settings.rag_chunk_overlap,
        )

        # Attach metadata from entity lookup
        metadata = entity_lookup.get(notion_id, {"entity_type": "unknown", "name": ""})
        for chunk in chunks:
            chunk["metadata"] = {
                "notion_id": notion_id,
                "entity_type": metadata.get("entity_type", "unknown"),
                "name": metadata.get("name", ""),
                "status": metadata.get("status", "") or "",
            }
            # Optional fields — only include if present and non-empty
            for field in ("priority", "sprint_id", "type", "severity"):
                value = metadata.get(field)
                if value:
                    chunk["metadata"][field] = value

        all_chunks.extend(chunks)
        docs_ingested += 1

    # Initialize ChromaDB
    chroma_path = Path(settings.chroma_db_path)
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))

    ef = OllamaEmbeddingFunction(
        model_name=settings.embedding_model,
        url=settings.ollama_base_url,
    )

    # Handle force re-ingestion
    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Deleted existing collection '%s'", COLLECTION_NAME)
        except ValueError:
            pass  # Collection doesn't exist

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Add chunks in batches.
    # Strip Notion markup tags before embedding to reduce token count,
    # but store the original text as the document for retrieval.
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts_for_embedding = [_strip_markup(c["text"]) for c in batch]
        embeddings = ef(texts_for_embedding)
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embeddings,
            metadatas=[c["metadata"] for c in batch],
        )
        logger.info("Embedded %d/%d chunks", min(i + batch_size, len(all_chunks)), len(all_chunks))

    return {
        "documents_ingested": docs_ingested,
        "chunks_created": len(all_chunks),
        "collection_name": COLLECTION_NAME,
    }


def _build_entity_lookup(snapshot_dir: Path) -> dict[str, dict]:
    """Load all JSON snapshots and build notion_id → metadata mapping.

    Returns:
        Dict mapping notion_id to metadata dict with keys like
        entity_type, name, status, priority, sprint_id, type, severity.
    """
    lookup: dict[str, dict] = {}

    for filename, entity_type, name_field in _SNAPSHOT_REGISTRY:
        path = snapshot_dir / filename
        if not path.exists():
            continue

        entities = json.loads(path.read_text(encoding="utf-8"))
        for entity in entities:
            notion_id = entity.get("notion_id", "")
            if not notion_id:
                continue

            meta: dict[str, str] = {
                "entity_type": entity_type,
                "name": entity.get(name_field, ""),
                "status": entity.get("status") or "",
            }

            # Entity-specific optional fields
            if entity_type == "work_item":
                for field in ("priority", "sprint_id", "type"):
                    if entity.get(field):
                        meta[field] = entity[field]
            elif entity_type == "sprint":
                if entity.get("type"):
                    meta["type"] = entity["type"]
            elif entity_type == "risk":
                for field in ("severity", "sprint_id"):
                    if entity.get(field):
                        meta[field] = entity[field]

            lookup[notion_id] = meta

    return lookup


def _chunk_content(
    content: str,
    notion_id: str,
    chunk_size: int = 4000,
    max_chunk_size: int = 8000,
    overlap: int = 200,
) -> list[dict]:
    """Chunk content using hybrid strategy.

    - Files <= chunk_size: single document chunk
    - Files > chunk_size: split on ## headers
    - Sections < 200 chars: merge with adjacent
    - Sections > max_chunk_size: sub-split at paragraph boundaries

    Args:
        content: The markdown content to chunk.
        notion_id: Notion page UUID (used for chunk IDs).
        chunk_size: Threshold for section splitting.
        max_chunk_size: Hard cap per chunk.
        overlap: Overlap for secondary splits.

    Returns:
        List of dicts with "text" and "chunk_id" keys.
    """
    if len(content) <= chunk_size:
        return [{"text": content, "chunk_id": f"{notion_id}_0"}]

    # Split on ## headers (keep the header with its section)
    sections = re.split(r"^(?=## )", content, flags=re.MULTILINE)
    sections = [s for s in sections if s.strip()]

    if not sections:
        return [{"text": content, "chunk_id": f"{notion_id}_0"}]

    # Merge tiny sections (<200 chars) with adjacent
    merged: list[str] = []
    for section in sections:
        if merged and len(merged[-1]) < 200:
            merged[-1] += "\n" + section
        else:
            merged.append(section)

    # If the last section is tiny, merge it with the previous
    if len(merged) > 1 and len(merged[-1]) < 200:
        merged[-2] += "\n" + merged[-1]
        merged.pop()

    # Sub-split oversized sections
    final_sections: list[str] = []
    for section in merged:
        if len(section) <= max_chunk_size:
            final_sections.append(section)
        else:
            final_sections.extend(
                _split_by_paragraphs(section, max_chunk_size, overlap)
            )

    return [
        {"text": s, "chunk_id": f"{notion_id}_{i}"}
        for i, s in enumerate(final_sections)
    ]


def _split_by_paragraphs(
    text: str,
    max_size: int,
    overlap: int,
) -> list[str]:
    """Split text at paragraph boundaries (\\n\\n) respecting max_size.

    Every returned chunk is guaranteed to be <= max_size characters.

    Args:
        text: Text to split.
        max_size: Maximum chunk size in characters.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunks, each <= max_size.
    """
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # If a single paragraph exceeds max_size, force-split it first
        if len(para) > max_size:
            # Flush current buffer
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(para), max_size - overlap):
                chunks.append(para[i : i + max_size])
            continue

        if current and len(current) + len(para) + 2 > max_size:
            chunks.append(current)
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current:
        chunks.append(current)

    # Final safety pass — force-split anything still over the limit
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_size:
            final.append(chunk)
        else:
            for i in range(0, len(chunk), max_size - overlap):
                final.append(chunk[i : i + max_size])

    return final
