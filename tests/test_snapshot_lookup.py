"""Tests for rag/snapshot_lookup.py.

No external services required — operates on in-memory dicts.
"""

from __future__ import annotations

import pytest

from rag.snapshot_lookup import SnapshotLookup


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_snapshot_context() -> dict:
    """Build a minimal snapshot context with relations across entity types."""
    return {
        "work_items": [
            {
                "notion_id": "wi-001",
                "name": "Deploy Firewall",
                "type": "Task",
                "status": "Ready",
                "priority": "P1",
                "sprint_id": "sp-001",
                "parent_epic_id": "wi-epic-001",
                "child_item_ids": [],
                "dependency_ids": ["wi-002"],
                "decision_ids": ["dec-001"],
                "doc_ids": ["doc-001", "doc-002"],
                "risk_ids": ["risk-001"],
            },
            {
                "notion_id": "wi-002",
                "name": "Configure NSG Rules",
                "type": "Task",
                "status": "In Progress",
                "priority": "P2",
                "sprint_id": "sp-001",
                "parent_epic_id": None,
                "child_item_ids": [],
                "dependency_ids": [],
                "decision_ids": [],
                "doc_ids": [],
                "risk_ids": [],
            },
            {
                "notion_id": "wi-epic-001",
                "name": "Infrastructure Epic",
                "type": "Epic",
                "status": "In Progress",
                "priority": "P0",
                "sprint_id": None,
                "parent_epic_id": None,
                "child_item_ids": ["wi-001"],
                "dependency_ids": [],
                "decision_ids": [],
                "doc_ids": [],
                "risk_ids": [],
            },
        ],
        "sprints": [
            {
                "notion_id": "sp-001",
                "name": "Sprint 1.4",
                "status": "Active",
                "goal": "Deploy infrastructure",
                "work_item_ids": ["wi-001", "wi-002"],
                "risk_ids": ["risk-001"],
            },
        ],
        "docs": [
            {
                "notion_id": "doc-001",
                "name": "Firewall Architecture",
                "doc_type": "Architecture",
                "status": "Final",
                "work_item_ids": ["wi-001"],
                "decision_ids": ["dec-001"],
                "sprint_ids": ["sp-001"],
                "supersedes_ids": [],
            },
            {
                "notion_id": "doc-002",
                "name": "NSG Security Spec",
                "doc_type": "Security",
                "status": "Draft",
                "work_item_ids": [],
                "decision_ids": [],
                "sprint_ids": [],
                "supersedes_ids": [],
            },
        ],
        "decisions": [
            {
                "notion_id": "dec-001",
                "title": "ADR-001: Use Azure Firewall",
                "adr_id": "ADR-001",
                "status": "Accepted",
                "work_item_ids": ["wi-001"],
                "doc_ids": ["doc-001"],
                "supersedes_ids": [],
                "superseded_by_ids": [],
            },
        ],
        "risks": [
            {
                "notion_id": "risk-001",
                "name": "Firewall latency risk",
                "type": "Risk",
                "status": "Open",
                "severity": "High",
                "mitigation_plan": "Use premium tier",
                "work_item_ids": ["wi-001"],
                "sprint_id": "sp-001",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Tests — get_entity
# ---------------------------------------------------------------------------

class TestGetEntity:
    def test_finds_work_item(self):
        lookup = SnapshotLookup(_make_snapshot_context())
        entity = lookup.get_entity("wi-001")

        assert entity is not None
        assert entity["name"] == "Deploy Firewall"
        assert entity["entity_type"] == "work_item"

    def test_finds_sprint(self):
        lookup = SnapshotLookup(_make_snapshot_context())
        entity = lookup.get_entity("sp-001")

        assert entity is not None
        assert entity["name"] == "Sprint 1.4"
        assert entity["entity_type"] == "sprint"

    def test_finds_doc(self):
        lookup = SnapshotLookup(_make_snapshot_context())
        entity = lookup.get_entity("doc-001")

        assert entity is not None
        assert entity["name"] == "Firewall Architecture"
        assert entity["entity_type"] == "doc"

    def test_finds_decision(self):
        lookup = SnapshotLookup(_make_snapshot_context())
        entity = lookup.get_entity("dec-001")

        assert entity is not None
        assert entity["title"] == "ADR-001: Use Azure Firewall"
        assert entity["entity_type"] == "decision"

    def test_finds_risk(self):
        lookup = SnapshotLookup(_make_snapshot_context())
        entity = lookup.get_entity("risk-001")

        assert entity is not None
        assert entity["name"] == "Firewall latency risk"
        assert entity["entity_type"] == "risk"

    def test_returns_none_for_missing(self):
        lookup = SnapshotLookup(_make_snapshot_context())
        assert lookup.get_entity("nonexistent") is None

    def test_empty_snapshot(self):
        lookup = SnapshotLookup({})
        assert lookup.get_entity("wi-001") is None


# ---------------------------------------------------------------------------
# Tests — get_related
# ---------------------------------------------------------------------------

class TestGetRelated:
    def test_follows_list_relation(self):
        """doc_ids is a list relation — should return resolved entities."""
        lookup = SnapshotLookup(_make_snapshot_context())
        related = lookup.get_related("wi-001", "doc_ids")

        assert len(related) == 2
        names = {r["name"] for r in related}
        assert "Firewall Architecture" in names
        assert "NSG Security Spec" in names

    def test_follows_scalar_relation(self):
        """parent_epic_id is a scalar relation — should return single entity."""
        lookup = SnapshotLookup(_make_snapshot_context())
        related = lookup.get_related("wi-001", "parent_epic_id")

        assert len(related) == 1
        assert related[0]["name"] == "Infrastructure Epic"

    def test_scalar_none_returns_empty(self):
        """parent_epic_id is None — should return []."""
        lookup = SnapshotLookup(_make_snapshot_context())
        related = lookup.get_related("wi-002", "parent_epic_id")
        assert related == []

    def test_empty_list_returns_empty(self):
        """dependency_ids is [] — should return []."""
        lookup = SnapshotLookup(_make_snapshot_context())
        related = lookup.get_related("wi-002", "dependency_ids")
        assert related == []

    def test_missing_entity_returns_empty(self):
        lookup = SnapshotLookup(_make_snapshot_context())
        related = lookup.get_related("nonexistent", "doc_ids")
        assert related == []

    def test_missing_field_returns_empty(self):
        lookup = SnapshotLookup(_make_snapshot_context())
        related = lookup.get_related("wi-001", "nonexistent_field")
        assert related == []

    def test_skips_unresolvable_ids(self):
        """If a relation ID doesn't exist in the snapshot, skip it."""
        ctx = _make_snapshot_context()
        # Add a fake doc ID that doesn't exist
        ctx["work_items"][0]["doc_ids"].append("doc-nonexistent")
        lookup = SnapshotLookup(ctx)
        related = lookup.get_related("wi-001", "doc_ids")

        # Should still have 2 (doc-001, doc-002), not 3
        assert len(related) == 2

    def test_cross_entity_type_relation(self):
        """Follow sprint_id from work_item to sprint."""
        lookup = SnapshotLookup(_make_snapshot_context())
        related = lookup.get_related("wi-001", "sprint_id")

        assert len(related) == 1
        assert related[0]["name"] == "Sprint 1.4"
        assert related[0]["entity_type"] == "sprint"

    def test_risk_relation(self):
        """Follow risk_ids from work_item to risks."""
        lookup = SnapshotLookup(_make_snapshot_context())
        related = lookup.get_related("wi-001", "risk_ids")

        assert len(related) == 1
        assert related[0]["name"] == "Firewall latency risk"


# ---------------------------------------------------------------------------
# Tests — get_related_ids
# ---------------------------------------------------------------------------

class TestGetRelatedIds:
    def test_list_relation_ids(self):
        lookup = SnapshotLookup(_make_snapshot_context())
        ids = lookup.get_related_ids("wi-001", "doc_ids")
        assert ids == ["doc-001", "doc-002"]

    def test_scalar_relation_id(self):
        lookup = SnapshotLookup(_make_snapshot_context())
        ids = lookup.get_related_ids("wi-001", "parent_epic_id")
        assert ids == ["wi-epic-001"]

    def test_scalar_none_returns_empty(self):
        lookup = SnapshotLookup(_make_snapshot_context())
        ids = lookup.get_related_ids("wi-002", "parent_epic_id")
        assert ids == []

    def test_empty_list_returns_empty(self):
        lookup = SnapshotLookup(_make_snapshot_context())
        ids = lookup.get_related_ids("wi-002", "dependency_ids")
        assert ids == []

    def test_missing_entity_returns_empty(self):
        lookup = SnapshotLookup(_make_snapshot_context())
        ids = lookup.get_related_ids("nonexistent", "doc_ids")
        assert ids == []

    def test_missing_field_returns_empty(self):
        lookup = SnapshotLookup(_make_snapshot_context())
        ids = lookup.get_related_ids("wi-001", "nonexistent_field")
        assert ids == []


# ---------------------------------------------------------------------------
# Tests — BaseAgent.lookup_relations integration
# ---------------------------------------------------------------------------

class TestBaseAgentLookupRelations:
    def test_without_snapshot_returns_empty(self):
        from unittest.mock import MagicMock
        from agents.base import BaseAgent

        class DummyAgent(BaseAgent):
            name = "dummy"
            def run(self, user_input):
                return {}

        agent = DummyAgent(llm=MagicMock(), context=None)
        assert agent.lookup_relations("wi-001", "doc_ids") == []

    def test_with_snapshot_delegates(self):
        from unittest.mock import MagicMock
        from agents.base import BaseAgent

        class DummyAgent(BaseAgent):
            name = "dummy"
            def run(self, user_input):
                return {}

        snapshot = SnapshotLookup(_make_snapshot_context())
        agent = DummyAgent(llm=MagicMock(), context=None, snapshot=snapshot)
        related = agent.lookup_relations("wi-001", "doc_ids")

        assert len(related) == 2

    def test_snapshot_keyword_only(self):
        """Verify snapshot cannot be passed as a positional argument."""
        from unittest.mock import MagicMock
        from agents.base import BaseAgent

        class DummyAgent(BaseAgent):
            name = "dummy"
            def run(self, user_input):
                return {}

        with pytest.raises(TypeError):
            DummyAgent(MagicMock(), None, MagicMock(), MagicMock())  # snapshot as positional
