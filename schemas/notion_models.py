"""Pydantic models for Notion database entities.

Maps the 5 hub-and-spoke databases in the Synergy Data Platform teamspace
to clean domain objects. All cross-database relations are stored as page
UUID strings; resolution (ID → name) is handled by the sync tool.
"""

from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Work Items (hub)
# ---------------------------------------------------------------------------

class WorkItem(BaseModel):
    """A work item from the Work Items database (Epic, Task, Bug, or Spike)."""

    notion_id: str
    name: str
    type: str | None = None                  # Epic, Task, Bug, Spike
    status: str | None = None                # Backlog, Blocked, Ready, In Progress, In Review, Done
    priority: str | None = None              # P0, P1, P2, P3
    estimate_hrs: float | None = None
    owner: str | None = None                 # Resolved display name
    due_date: str | None = None              # ISO date string
    sprint_id: str | None = None             # → Phases & Sprints UUID
    dependency_ids: list[str] = []           # → Work Items UUIDs (self)
    parent_epic_id: str | None = None        # → Work Items UUID (self, limit 1)
    child_item_ids: list[str] = []           # → Work Items UUIDs (self)
    decision_ids: list[str] = []             # → Decisions (ADRs) UUIDs
    doc_ids: list[str] = []                  # → Docs & Specs UUIDs
    risk_ids: list[str] = []                 # → Risks & Issues UUIDs
    definition_of_done: str | None = None
    links: str | None = None                 # URL
    notion_url: str | None = None            # Notion page URL


# ---------------------------------------------------------------------------
# Phases & Sprints
# ---------------------------------------------------------------------------

class Sprint(BaseModel):
    """A phase or sprint from the Phases & Sprints database."""

    notion_id: str
    name: str
    type: str | None = None                  # Phase, Sprint
    status: str | None = None                # Not started, Active, Done
    sprint_number: int | None = None
    goal: str | None = None
    start_date: str | None = None            # ISO date string
    end_date: str | None = None              # ISO date string
    work_item_ids: list[str] = []            # → Work Items UUIDs
    risk_ids: list[str] = []                 # → Risks & Issues UUIDs


# ---------------------------------------------------------------------------
# Docs & Specs
# ---------------------------------------------------------------------------

class DocSpec(BaseModel):
    """A document from the Docs & Specs database."""

    notion_id: str
    name: str
    doc_type: str | None = None              # Architecture, Security, Networking, etc.
    status: str | None = None                # Not started, Draft, Reviewed, Final, Superseded
    active: bool = True
    owner: str | None = None
    tags: list[str] = []                     # Multi-select values
    last_reviewed: str | None = None         # ISO date string
    deprecated: str | None = None            # ISO date string
    work_item_ids: list[str] = []            # → Work Items UUIDs
    decision_ids: list[str] = []             # → Decisions (ADRs) UUIDs
    sprint_ids: list[str] = []               # → Phases & Sprints UUIDs
    supersedes_ids: list[str] = []           # → Docs & Specs UUIDs (self)
    notion_url: str | None = None


# ---------------------------------------------------------------------------
# Decisions (ADRs)
# ---------------------------------------------------------------------------

class Decision(BaseModel):
    """An architecture decision record from the Decisions (ADRs) database."""

    notion_id: str
    title: str
    adr_id: str | None = None               # e.g. "ADR-001"
    status: str | None = None                # Proposed, Accepted, Superseded
    date: str | None = None                  # ISO date string
    work_item_ids: list[str] = []            # → Work Items UUIDs
    doc_ids: list[str] = []                  # → Docs & Specs UUIDs
    supersedes_ids: list[str] = []           # → Decisions UUIDs (self)
    superseded_by_ids: list[str] = []        # → Decisions UUIDs (self, backlink)
    notion_url: str | None = None


# ---------------------------------------------------------------------------
# Risks & Issues
# ---------------------------------------------------------------------------

class RiskIssue(BaseModel):
    """A risk or issue from the Risks & Issues database."""

    notion_id: str
    name: str
    type: str | None = None                  # Risk, Issue
    status: str | None = None                # Open, Monitoring, Mitigated, Closed
    severity: str | None = None              # High, Med, Low
    owner: str | None = None
    mitigation_plan: str | None = None
    next_review: str | None = None           # ISO date string
    work_item_ids: list[str] = []            # → Work Items UUIDs
    sprint_id: str | None = None             # → Phases & Sprints UUID (limit 1)
    notion_url: str | None = None


# ---------------------------------------------------------------------------
# Sync metadata & snapshot
# ---------------------------------------------------------------------------

class SyncMeta(BaseModel):
    """Metadata about the last sync operation."""

    synced_at: str                           # ISO timestamp
    databases: dict[str, str]                # name → collection ID
    counts: dict[str, int]                   # name → page count


class NotionSnapshot(BaseModel):
    """The complete local snapshot of all Notion databases."""

    work_items: list[WorkItem]
    sprints: list[Sprint]
    docs: list[DocSpec]
    decisions: list[Decision]
    risks: list[RiskIssue]
    meta: SyncMeta
