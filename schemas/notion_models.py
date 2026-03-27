"""Pydantic models for Notion database entities.

Maps the 5 hub-and-spoke databases in the Synergy Data Platform teamspace
to clean domain objects. All cross-database relations are stored as page
UUID strings; resolution (ID → name) is handled by the sync tool.
"""

from __future__ import annotations

from typing import Any

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
    notion_url: str | None = None
    has_content: bool = False                # True if content/*.md exists            # Notion page URL


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
    has_content: bool = False                # True if content/*.md exists


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
    has_content: bool = False                # True if content/*.md exists


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
    has_content: bool = False                # True if content/*.md exists


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
    has_content: bool = False                # True if content/*.md exists


# ---------------------------------------------------------------------------
# Template metadata
# ---------------------------------------------------------------------------

class TemplateInfo(BaseModel):
    """Metadata about a database template."""

    notion_id: str
    title: str
    db_name: str                             # Which database this belongs to
    filename: str                            # e.g. "epic_template.md"


# ---------------------------------------------------------------------------
# Sync metadata & snapshot
# ---------------------------------------------------------------------------

class SyncMeta(BaseModel):
    """Metadata about the last sync operation."""

    synced_at: str                           # ISO timestamp
    databases: dict[str, str]                # name → collection ID
    counts: dict[str, int]                   # name → page count
    content_counts: dict[str, int] = {}      # name → pages with content synced
    template_counts: dict[str, int] = {}     # name → template count per db
    templates: list[TemplateInfo] = []       # All template metadata


class NotionSnapshot(BaseModel):
    """The complete local snapshot of all Notion databases."""

    work_items: list[WorkItem]
    sprints: list[Sprint]
    docs: list[DocSpec]
    decisions: list[Decision]
    risks: list[RiskIssue]
    meta: SyncMeta


# ---------------------------------------------------------------------------
# Local write-back (Phase 2c)
# ---------------------------------------------------------------------------

class PendingChange(BaseModel):
    """A single tracked mutation to a local Notion entity."""

    timestamp: str                        # ISO timestamp of when change was made
    action: str                           # "update" or "create"
    entity_type: str                      # "work_item", "sprint", "doc", "decision", "risk"
    entity_id: str                        # notion_id (existing) or generated UUID (new)
    field: str | None = None              # Field that changed (None for "create")
    old_value: Any = None                 # Previous value (None for "create")
    new_value: Any = None                 # New value (None for "create" — full entity in entity_snapshot)
    entity_snapshot: dict | None = None   # Full entity state after change (for creates)


class ChangeLog(BaseModel):
    """Collection of pending changes not yet pushed to cloud Notion."""

    changes: list[PendingChange] = []
    created_at: str | None = None         # When changelog was started
    last_modified: str | None = None      # When last change was recorded
