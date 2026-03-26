"""Tests for the local Notion write-back tool."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from schemas.notion_models import (
    ChangeLog,
    Decision,
    DocSpec,
    NotionSnapshot,
    PendingChange,
    RiskIssue,
    Sprint,
    WorkItem,
)
from tools.notion_write_tool import NotionWriteError, NotionWriteTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(data_dir: Path) -> MagicMock:
    """Create a minimal Settings-like object for testing."""
    settings = MagicMock()
    settings.data_dir = data_dir
    return settings


def _seed_cloud_snapshot(data_dir: Path) -> None:
    """Write minimal cloud snapshot files for testing."""
    notion_dir = data_dir / "notion"
    notion_dir.mkdir(parents=True, exist_ok=True)

    work_items = [
        WorkItem(
            notion_id="wi-001", name="Task 1", status="Ready",
            priority="P1", sprint_id="sp-001",
        ).model_dump(),
        WorkItem(
            notion_id="wi-002", name="Task 2", status="Backlog",
        ).model_dump(),
    ]
    sprints = [
        Sprint(
            notion_id="sp-001", name="Sprint 8", status="Active",
            sprint_number=8,
        ).model_dump(),
    ]
    docs = [
        DocSpec(
            notion_id="doc-001", name="Arch Doc", doc_type="Architecture",
            status="Draft", active=True, owner="Alice", tags=["infra"],
        ).model_dump(),
    ]
    decisions = [
        Decision(
            notion_id="dec-001", title="Use Pydantic", adr_id="ADR-001",
            status="Accepted", date="2026-01-15",
        ).model_dump(),
    ]
    risks = [
        RiskIssue(
            notion_id="rsk-001", name="API Rate Limits", type="Risk",
            status="Open", severity="High", owner="Bob",
        ).model_dump(),
    ]

    (notion_dir / "work_items.json").write_text(json.dumps(work_items))
    (notion_dir / "sprints.json").write_text(json.dumps(sprints))
    (notion_dir / "docs_specs.json").write_text(json.dumps(docs))
    (notion_dir / "decisions.json").write_text(json.dumps(decisions))
    (notion_dir / "risks_issues.json").write_text(json.dumps(risks))
    (notion_dir / "sync_meta.json").write_text(json.dumps({
        "synced_at": "2026-03-26T00:00:00Z",
        "databases": {}, "counts": {},
    }))


# ---------------------------------------------------------------------------
# Tests — Init
# ---------------------------------------------------------------------------

class TestNotionWriteToolInit:
    def test_creates_with_valid_settings(self, tmp_path):
        settings = _make_settings(tmp_path)
        tool = NotionWriteTool(settings)
        assert tool.settings is settings

    def test_data_dir_set_correctly(self, tmp_path):
        settings = _make_settings(tmp_path)
        tool = NotionWriteTool(settings)
        assert tool.data_dir == tmp_path / "notion"


# ---------------------------------------------------------------------------
# Tests — Load cloud snapshot
# ---------------------------------------------------------------------------

class TestLoadCloudSnapshot:
    def test_loads_cloud_snapshot(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        snapshot = tool._load_cloud_snapshot()
        assert len(snapshot.work_items) == 2
        assert len(snapshot.sprints) == 1
        assert len(snapshot.docs) == 1
        assert len(snapshot.decisions) == 1
        assert len(snapshot.risks) == 1

    def test_raises_when_no_sync_done(self, tmp_path):
        tool = NotionWriteTool(_make_settings(tmp_path))
        with pytest.raises(NotionWriteError, match="No cloud snapshot found"):
            tool._load_cloud_snapshot()


# ---------------------------------------------------------------------------
# Tests — Load local snapshot
# ---------------------------------------------------------------------------

class TestLoadLocalSnapshot:
    def test_returns_local_snapshot_when_exists(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))

        # Create a local snapshot via an update
        tool.update_work_item("wi-001", status="In Progress")

        snapshot = tool.load_local_snapshot()
        assert snapshot is not None
        wi = next(w for w in snapshot.work_items if w.notion_id == "wi-001")
        assert wi.status == "In Progress"

    def test_falls_back_to_cloud_snapshot(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        snapshot = tool.load_local_snapshot()
        assert snapshot is not None
        assert len(snapshot.work_items) == 2

    def test_returns_none_when_no_snapshot(self, tmp_path):
        tool = NotionWriteTool(_make_settings(tmp_path))
        assert tool.load_local_snapshot() is None


# ---------------------------------------------------------------------------
# Tests — Update work item
# ---------------------------------------------------------------------------

class TestUpdateWorkItem:
    def test_updates_status(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.update_work_item("wi-001", status="In Progress")
        assert result.status == "In Progress"
        assert result.notion_id == "wi-001"

    def test_updates_multiple_fields(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.update_work_item(
            "wi-001", status="In Progress", priority="P0",
        )
        assert result.status == "In Progress"
        assert result.priority == "P0"

    def test_records_pending_change_with_old_new(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        tool.update_work_item("wi-001", status="In Progress")

        changelog = tool.load_pending_changes()
        assert len(changelog.changes) == 1
        change = changelog.changes[0]
        assert change.action == "update"
        assert change.entity_type == "work_item"
        assert change.entity_id == "wi-001"
        assert change.field == "status"
        assert change.old_value == "Ready"
        assert change.new_value == "In Progress"

    def test_saves_local_snapshot(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        tool.update_work_item("wi-001", status="Done")

        local_path = tmp_path / "notion" / "local_snapshot.json"
        assert local_path.exists()
        data = json.loads(local_path.read_text())
        snapshot = NotionSnapshot.model_validate(data)
        wi = next(w for w in snapshot.work_items if w.notion_id == "wi-001")
        assert wi.status == "Done"

    def test_raises_for_nonexistent_entity(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        with pytest.raises(NotionWriteError, match="not found"):
            tool.update_work_item("wi-999", status="Done")

    def test_raises_for_disallowed_field(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        with pytest.raises(NotionWriteError, match="not allowed"):
            tool.update_work_item("wi-001", name="Renamed")

    def test_sequential_updates_accumulate(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        tool.update_work_item("wi-001", status="In Progress")
        tool.update_work_item("wi-002", status="Ready")

        changelog = tool.load_pending_changes()
        assert len(changelog.changes) == 2
        assert changelog.changes[0].entity_id == "wi-001"
        assert changelog.changes[1].entity_id == "wi-002"

    def test_reads_from_local_snapshot_not_cloud(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))

        # First update
        tool.update_work_item("wi-001", status="In Progress")
        # Second update should see "In Progress" as old value
        tool.update_work_item("wi-001", status="Done")

        changelog = tool.load_pending_changes()
        assert len(changelog.changes) == 2
        assert changelog.changes[1].old_value == "In Progress"
        assert changelog.changes[1].new_value == "Done"


# ---------------------------------------------------------------------------
# Tests — Update sprint
# ---------------------------------------------------------------------------

class TestUpdateSprint:
    def test_updates_status(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.update_sprint("sp-001", status="Done")
        assert result.status == "Done"

    def test_updates_goal(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.update_sprint("sp-001", goal="Ship Phase 2c")
        assert result.goal == "Ship Phase 2c"

    def test_raises_for_nonexistent_sprint(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        with pytest.raises(NotionWriteError, match="not found"):
            tool.update_sprint("sp-999", status="Done")


# ---------------------------------------------------------------------------
# Tests — Update doc
# ---------------------------------------------------------------------------

class TestUpdateDoc:
    def test_updates_status(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.update_doc("doc-001", status="Final")
        assert result.status == "Final"

    def test_updates_tags(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.update_doc("doc-001", tags=["infra", "security"])
        assert result.tags == ["infra", "security"]

    def test_updates_active(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.update_doc("doc-001", active=False)
        assert result.active is False


# ---------------------------------------------------------------------------
# Tests — Update decision
# ---------------------------------------------------------------------------

class TestUpdateDecision:
    def test_updates_status(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.update_decision("dec-001", status="Superseded")
        assert result.status == "Superseded"

    def test_raises_for_disallowed_field(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        with pytest.raises(NotionWriteError, match="not allowed"):
            tool.update_decision("dec-001", title="New Title")


# ---------------------------------------------------------------------------
# Tests — Update risk
# ---------------------------------------------------------------------------

class TestUpdateRisk:
    def test_updates_severity(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.update_risk("rsk-001", severity="Low")
        assert result.severity == "Low"

    def test_updates_mitigation_plan(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.update_risk("rsk-001", mitigation_plan="Add caching")
        assert result.mitigation_plan == "Add caching"


# ---------------------------------------------------------------------------
# Tests — Create work item
# ---------------------------------------------------------------------------

class TestCreateWorkItem:
    def test_creates_with_required_fields_only(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.create_work_item("New Task")
        assert result.name == "New Task"
        assert result.notion_id.startswith("local-")

    def test_creates_with_all_optional_fields(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.create_work_item(
            "New Task", type="Task", status="Ready",
            priority="P2", sprint_id="sp-001",
            estimate_hrs=4.0, owner="Alice",
        )
        assert result.type == "Task"
        assert result.status == "Ready"
        assert result.priority == "P2"
        assert result.estimate_hrs == 4.0

    def test_generated_id_has_local_prefix(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.create_work_item("Task")
        assert result.notion_id.startswith("local-")

    def test_records_create_change(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.create_work_item("Task")

        changelog = tool.load_pending_changes()
        assert len(changelog.changes) == 1
        change = changelog.changes[0]
        assert change.action == "create"
        assert change.entity_type == "work_item"
        assert change.entity_id == result.notion_id
        assert change.entity_snapshot is not None
        assert change.entity_snapshot["name"] == "Task"

    def test_new_item_in_local_snapshot(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.create_work_item("Task 3")

        snapshot = tool.load_local_snapshot()
        assert len(snapshot.work_items) == 3
        ids = [w.notion_id for w in snapshot.work_items]
        assert result.notion_id in ids

    def test_multiple_creates_accumulate(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        tool.create_work_item("Task A")
        tool.create_work_item("Task B")

        changelog = tool.load_pending_changes()
        assert len(changelog.changes) == 2
        snapshot = tool.load_local_snapshot()
        assert len(snapshot.work_items) == 4


# ---------------------------------------------------------------------------
# Tests — Create sprint
# ---------------------------------------------------------------------------

class TestCreateSprint:
    def test_creates_sprint(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.create_sprint("Sprint 9", sprint_number=9)
        assert result.name == "Sprint 9"
        assert result.sprint_number == 9

    def test_appears_in_local_snapshot(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.create_sprint("Sprint 9")
        snapshot = tool.load_local_snapshot()
        assert len(snapshot.sprints) == 2


# ---------------------------------------------------------------------------
# Tests — Create doc
# ---------------------------------------------------------------------------

class TestCreateDoc:
    def test_creates_doc(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.create_doc("Security Doc", doc_type="Security")
        assert result.name == "Security Doc"
        assert result.doc_type == "Security"


# ---------------------------------------------------------------------------
# Tests — Create decision
# ---------------------------------------------------------------------------

class TestCreateDecision:
    def test_creates_decision(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.create_decision("Use SQLite")
        assert result.title == "Use SQLite"
        assert result.notion_id.startswith("local-")


# ---------------------------------------------------------------------------
# Tests — Create risk
# ---------------------------------------------------------------------------

class TestCreateRisk:
    def test_creates_risk(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        result = tool.create_risk("Data Loss", type="Risk")
        assert result.name == "Data Loss"
        assert result.type == "Risk"


# ---------------------------------------------------------------------------
# Tests — Pending changes
# ---------------------------------------------------------------------------

class TestPendingChanges:
    def test_has_pending_changes_false_initially(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        assert tool.has_pending_changes() is False

    def test_has_pending_changes_after_update(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        tool.update_work_item("wi-001", status="Done")
        assert tool.has_pending_changes() is True

    def test_has_pending_changes_after_create(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        tool.create_work_item("New")
        assert tool.has_pending_changes() is True

    def test_load_pending_changes_empty(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        changelog = tool.load_pending_changes()
        assert len(changelog.changes) == 0

    def test_load_pending_changes_accumulated(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        tool.update_work_item("wi-001", status="Done")
        tool.create_work_item("New")
        changelog = tool.load_pending_changes()
        assert len(changelog.changes) == 2

    def test_changes_have_timestamps(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        tool.update_work_item("wi-001", status="Done")
        changelog = tool.load_pending_changes()
        assert changelog.changes[0].timestamp is not None
        assert changelog.created_at is not None
        assert changelog.last_modified is not None


# ---------------------------------------------------------------------------
# Tests — Discard pending changes
# ---------------------------------------------------------------------------

class TestDiscardPendingChanges:
    def test_removes_pending_changes_file(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        tool.update_work_item("wi-001", status="Done")
        tool.discard_pending_changes()
        assert not (tmp_path / "notion" / "pending_changes.json").exists()

    def test_removes_local_snapshot_file(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        tool.update_work_item("wi-001", status="Done")
        tool.discard_pending_changes()
        assert not (tmp_path / "notion" / "local_snapshot.json").exists()

    def test_falls_back_to_cloud_after_discard(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        tool.update_work_item("wi-001", status="Done")
        tool.discard_pending_changes()

        snapshot = tool.load_local_snapshot()
        wi = next(w for w in snapshot.work_items if w.notion_id == "wi-001")
        assert wi.status == "Ready"  # Back to original cloud value

    def test_has_pending_changes_false_after_discard(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        tool.update_work_item("wi-001", status="Done")
        tool.discard_pending_changes()
        assert tool.has_pending_changes() is False

    def test_discard_is_noop_when_no_changes(self, tmp_path):
        _seed_cloud_snapshot(tmp_path)
        tool = NotionWriteTool(_make_settings(tmp_path))
        tool.discard_pending_changes()  # Should not raise


# ---------------------------------------------------------------------------
# Tests — Field validation
# ---------------------------------------------------------------------------

class TestFieldValidation:
    def test_passes_for_allowed_fields(self, tmp_path):
        tool = NotionWriteTool(_make_settings(tmp_path))
        tool._validate_update_fields("work_item", {"status": "Done"})

    def test_raises_for_notion_id(self, tmp_path):
        tool = NotionWriteTool(_make_settings(tmp_path))
        with pytest.raises(NotionWriteError, match="not allowed"):
            tool._validate_update_fields("work_item", {"notion_id": "x"})

    def test_raises_for_name(self, tmp_path):
        tool = NotionWriteTool(_make_settings(tmp_path))
        with pytest.raises(NotionWriteError, match="not allowed"):
            tool._validate_update_fields("work_item", {"name": "x"})

    def test_raises_for_relation_fields(self, tmp_path):
        tool = NotionWriteTool(_make_settings(tmp_path))
        with pytest.raises(NotionWriteError, match="not allowed"):
            tool._validate_update_fields("work_item", {"dependency_ids": []})

    def test_raises_for_unknown_fields(self, tmp_path):
        tool = NotionWriteTool(_make_settings(tmp_path))
        with pytest.raises(NotionWriteError, match="not allowed"):
            tool._validate_update_fields("work_item", {"foo_bar": "x"})


# ---------------------------------------------------------------------------
# Tests — Local ID generation
# ---------------------------------------------------------------------------

class TestLocalIdGeneration:
    def test_starts_with_local_prefix(self, tmp_path):
        tool = NotionWriteTool(_make_settings(tmp_path))
        local_id = tool._generate_local_id()
        assert local_id.startswith("local-")

    def test_ids_are_unique(self, tmp_path):
        tool = NotionWriteTool(_make_settings(tmp_path))
        ids = {tool._generate_local_id() for _ in range(100)}
        assert len(ids) == 100
