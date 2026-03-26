"""Tests for Pydantic models in schemas/notion_models.py."""

from schemas.notion_models import (
    Decision,
    DocSpec,
    NotionSnapshot,
    RiskIssue,
    Sprint,
    SyncMeta,
    WorkItem,
)


class TestWorkItem:
    def test_minimal(self):
        wi = WorkItem(notion_id="abc-123", name="Deploy VM")
        assert wi.notion_id == "abc-123"
        assert wi.name == "Deploy VM"
        assert wi.type is None
        assert wi.dependency_ids == []

    def test_full(self):
        wi = WorkItem(
            notion_id="abc-123",
            name="Deploy VM",
            type="Task",
            status="In Progress",
            priority="P1",
            estimate_hrs=4.0,
            owner="Nathan Teuber",
            due_date="2026-03-28",
            sprint_id="spr-001",
            dependency_ids=["dep-001", "dep-002"],
            parent_epic_id="epc-001",
            child_item_ids=[],
            decision_ids=["dec-001"],
            doc_ids=["doc-001"],
            risk_ids=["rsk-001"],
            definition_of_done="VM deployed and SHIR registered.",
            links="https://example.com",
            notion_url="https://www.notion.so/abc123",
        )
        assert wi.estimate_hrs == 4.0
        assert len(wi.dependency_ids) == 2
        assert wi.sprint_id == "spr-001"

    def test_round_trip(self):
        wi = WorkItem(
            notion_id="abc-123",
            name="Test task",
            status="Done",
            dependency_ids=["x", "y"],
        )
        data = wi.model_dump()
        restored = WorkItem.model_validate(data)
        assert restored == wi


class TestSprint:
    def test_minimal(self):
        s = Sprint(notion_id="spr-001", name="Sprint 1.3")
        assert s.sprint_number is None
        assert s.work_item_ids == []

    def test_full(self):
        s = Sprint(
            notion_id="spr-001",
            name="Sprint 1.3",
            type="Sprint",
            status="Active",
            sprint_number=13,
            goal="Deploy SHIR",
            start_date="2026-03-10",
            end_date="2026-03-28",
            work_item_ids=["wi-1", "wi-2"],
            risk_ids=["rsk-1"],
        )
        assert s.sprint_number == 13
        assert s.status == "Active"

    def test_round_trip(self):
        s = Sprint(notion_id="spr-001", name="Sprint 1.3", sprint_number=13)
        assert Sprint.model_validate(s.model_dump()) == s


class TestDocSpec:
    def test_defaults(self):
        d = DocSpec(notion_id="doc-001", name="Architecture doc")
        assert d.active is True
        assert d.tags == []

    def test_full(self):
        d = DocSpec(
            notion_id="doc-001",
            name="Network Egress Rules",
            doc_type="Architecture",
            status="Reviewed",
            active=True,
            owner="Nathan Teuber",
            tags=["Networking", "Security"],
            last_reviewed="2026-03-15",
            work_item_ids=["wi-1"],
            decision_ids=["dec-1"],
            sprint_ids=["spr-1"],
            supersedes_ids=[],
        )
        assert len(d.tags) == 2
        assert d.doc_type == "Architecture"


class TestDecision:
    def test_minimal(self):
        d = Decision(notion_id="dec-001", title="ADR-004: Auth via Entra")
        assert d.adr_id is None
        assert d.supersedes_ids == []

    def test_full(self):
        d = Decision(
            notion_id="dec-001",
            title="ADR-004: Auth via Entra",
            adr_id="ADR-004",
            status="Accepted",
            date="2026-02-15",
            work_item_ids=["wi-1"],
            doc_ids=["doc-1"],
            supersedes_ids=[],
            superseded_by_ids=[],
        )
        assert d.status == "Accepted"


class TestRiskIssue:
    def test_minimal(self):
        r = RiskIssue(notion_id="rsk-001", name="SSRS KCD dependency")
        assert r.severity is None
        assert r.sprint_id is None

    def test_full(self):
        r = RiskIssue(
            notion_id="rsk-001",
            name="SSRS KCD dependency",
            type="Risk",
            status="Open",
            severity="High",
            owner="Nathan Teuber",
            mitigation_plan="Validate via spike.",
            next_review="2026-04-01",
            work_item_ids=["wi-1"],
            sprint_id="spr-1",
        )
        assert r.severity == "High"
        assert r.type == "Risk"


class TestSyncMeta:
    def test_creation(self):
        m = SyncMeta(
            synced_at="2026-03-26T10:00:00Z",
            databases={"work_items": "abc", "sprints": "def"},
            counts={"work_items": 50, "sprints": 5},
        )
        assert m.counts["work_items"] == 50


class TestNotionSnapshot:
    def test_empty_snapshot(self):
        snap = NotionSnapshot(
            work_items=[],
            sprints=[],
            docs=[],
            decisions=[],
            risks=[],
            meta=SyncMeta(
                synced_at="2026-03-26T10:00:00Z",
                databases={},
                counts={},
            ),
        )
        assert len(snap.work_items) == 0

    def test_round_trip(self):
        snap = NotionSnapshot(
            work_items=[WorkItem(notion_id="wi-1", name="Task 1")],
            sprints=[Sprint(notion_id="s-1", name="Sprint 1")],
            docs=[DocSpec(notion_id="d-1", name="Doc 1")],
            decisions=[Decision(notion_id="dec-1", title="ADR-001")],
            risks=[RiskIssue(notion_id="r-1", name="Risk 1")],
            meta=SyncMeta(
                synced_at="2026-03-26T10:00:00Z",
                databases={"work_items": "abc"},
                counts={"work_items": 1},
            ),
        )
        data = snap.model_dump()
        restored = NotionSnapshot.model_validate(data)
        assert restored.work_items[0].name == "Task 1"
        assert restored.sprints[0].name == "Sprint 1"
