"""Relational query interface over Notion JSON snapshot data.

Indexes all entities by notion_id for O(1) lookup, enabling
relation traversal without semantic search. Operates on the dict
already loaded by _snapshot_to_context() — no file I/O.
"""

from __future__ import annotations

from typing import Any

# Mapping from snapshot context dict keys to entity_type labels
_ENTITY_TYPE_MAP = {
    "work_items": "work_item",
    "sprints": "sprint",
    "docs": "doc",
    "decisions": "decision",
    "risks": "risk",
}


class SnapshotLookup:
    """Relational query interface over Notion JSON snapshot data.

    Indexes all entities by notion_id for O(1) lookup, enabling
    relation traversal without semantic search.
    """

    def __init__(self, snapshot_context: dict) -> None:
        """Build index from the snapshot context dict.

        Args:
            snapshot_context: Dict with keys: work_items, sprints, docs,
                              decisions, risks — each a list of entity dicts
                              (from _snapshot_to_context()).
        """
        self._index: dict[str, dict] = {}

        for context_key, entity_type in _ENTITY_TYPE_MAP.items():
            for entity in snapshot_context.get(context_key, []):
                notion_id = entity.get("notion_id")
                if notion_id:
                    entry = dict(entity)
                    entry["entity_type"] = entity_type
                    self._index[notion_id] = entry

    def get_entity(self, notion_id: str) -> dict | None:
        """Look up any entity by notion_id.

        Returns the full entity dict (with entity_type added) or None.
        """
        return self._index.get(notion_id)

    def get_related(self, notion_id: str, relation_field: str) -> list[dict]:
        """Follow a relation field and return the linked entity dicts.

        Handles both scalar relations (str | None) and list relations
        (list[str]).

        Example:
            get_related("task-uuid", "doc_ids")
            → looks up the task, reads its doc_ids list, returns each
              linked DocSpec dict

        Returns:
            List of resolved entity dicts. Returns [] if entity not found,
            field doesn't exist, or field is empty.
        """
        entity = self._index.get(notion_id)
        if entity is None:
            return []

        value = entity.get(relation_field)
        if value is None:
            return []

        # Scalar relation (e.g., parent_epic_id: str, sprint_id: str)
        if isinstance(value, str):
            related = self._index.get(value)
            return [related] if related is not None else []

        # List relation (e.g., doc_ids: list[str])
        if isinstance(value, list):
            results = []
            for related_id in value:
                if isinstance(related_id, str):
                    related = self._index.get(related_id)
                    if related is not None:
                        results.append(related)
            return results

        return []

    def get_related_ids(self, notion_id: str, relation_field: str) -> list[str]:
        """Return just the linked notion_ids for a relation field.

        Useful for passing to RAGRetriever as a filter:
            ids = snapshot.get_related_ids(task_id, "doc_ids")
            results = retriever.query("security", notion_ids=ids)

        Returns:
            List of notion_id strings. Returns [] if entity not found,
            field doesn't exist, or field is empty.
        """
        entity = self._index.get(notion_id)
        if entity is None:
            return []

        value = entity.get(relation_field)
        if value is None:
            return []

        # Scalar relation
        if isinstance(value, str):
            return [value] if value else []

        # List relation
        if isinstance(value, list):
            return [rid for rid in value if isinstance(rid, str) and rid]

        return []
