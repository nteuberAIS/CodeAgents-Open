"""Local Notion write-back tool.

Provides create/update operations on the local Notion snapshot.
Cloud snapshot files from sync() are never modified — all mutations
are tracked in pending_changes.json and applied to local_snapshot.json.

Cloud push is NOT implemented here. That requires human approval and
will be a separate phase.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from schemas.notion_models import (
    ChangeLog,
    Decision,
    DocSpec,
    NotionSnapshot,
    PendingChange,
    RiskIssue,
    Sprint,
    SyncMeta,
    WorkItem,
)


class NotionWriteError(Exception):
    """Raised when a local write operation fails."""


# Entity type → (snapshot attribute, cloud filename, model class)
_ENTITY_REGISTRY: dict[str, tuple[str, str, type[BaseModel]]] = {
    "work_item": ("work_items", "work_items.json", WorkItem),
    "sprint": ("sprints", "sprints.json", Sprint),
    "doc": ("docs", "docs_specs.json", DocSpec),
    "decision": ("decisions", "decisions.json", Decision),
    "risk": ("risks", "risks_issues.json", RiskIssue),
}

_ALLOWED_UPDATE_FIELDS: dict[str, set[str]] = {
    "work_item": {
        "status", "priority", "sprint_id", "estimate_hrs",
        "owner", "due_date", "definition_of_done",
    },
    "sprint": {"status", "goal", "start_date", "end_date"},
    "doc": {"status", "active", "owner", "tags", "last_reviewed"},
    "decision": {"status", "date"},
    "risk": {"status", "severity", "owner", "mitigation_plan", "next_review"},
}


class NotionWriteTool:
    """Local write-back for Notion entities.

    Reads the cloud snapshot, applies local mutations, writes results to:
    - data/notion/pending_changes.json (append-only changelog)
    - data/notion/local_snapshot.json (full merged state)

    Cloud snapshot files are NEVER modified.
    """

    def __init__(self, settings: Any) -> None:
        self.data_dir: Path = Path(settings.data_dir) / "notion"
        self.settings = settings

    # ------------------------------------------------------------------ #
    #  Public API — Updates                                                #
    # ------------------------------------------------------------------ #

    def update_work_item(self, notion_id: str, **fields: Any) -> WorkItem:
        """Update fields on an existing work item.

        Allowed fields: status, priority, sprint_id, estimate_hrs,
                        owner, due_date, definition_of_done

        Returns the updated WorkItem.
        Raises NotionWriteError if entity not found or invalid field.
        """
        return self._update_entity("work_item", notion_id, fields)

    def update_sprint(self, notion_id: str, **fields: Any) -> Sprint:
        """Update fields on an existing sprint.

        Allowed fields: status, goal, start_date, end_date
        """
        return self._update_entity("sprint", notion_id, fields)

    def update_doc(self, notion_id: str, **fields: Any) -> DocSpec:
        """Update fields on an existing doc.

        Allowed fields: status, active, owner, tags, last_reviewed
        """
        return self._update_entity("doc", notion_id, fields)

    def update_decision(self, notion_id: str, **fields: Any) -> Decision:
        """Update fields on an existing decision.

        Allowed fields: status, date
        """
        return self._update_entity("decision", notion_id, fields)

    def update_risk(self, notion_id: str, **fields: Any) -> RiskIssue:
        """Update fields on an existing risk/issue.

        Allowed fields: status, severity, owner, mitigation_plan, next_review
        """
        return self._update_entity("risk", notion_id, fields)

    # ------------------------------------------------------------------ #
    #  Public API — Creates                                                #
    # ------------------------------------------------------------------ #

    def create_work_item(self, name: str, **fields: Any) -> WorkItem:
        """Create a new work item in the local snapshot.

        Required: name
        Optional: type, status, priority, sprint_id, estimate_hrs, owner

        Generates a local UUID (prefixed 'local-' to distinguish from
        real Notion IDs). Returns the created WorkItem.
        """
        return self._create_entity("work_item", WorkItem, {"name": name, **fields})

    def create_sprint(self, name: str, **fields: Any) -> Sprint:
        """Create a new sprint in the local snapshot.

        Required: name
        Optional: type, sprint_number, goal, start_date, end_date
        """
        return self._create_entity("sprint", Sprint, {"name": name, **fields})

    def create_doc(self, name: str, **fields: Any) -> DocSpec:
        """Create a new doc in the local snapshot.

        Required: name
        Optional: doc_type, status, owner, tags
        """
        return self._create_entity("doc", DocSpec, {"name": name, **fields})

    def create_decision(self, title: str, **fields: Any) -> Decision:
        """Create a new decision in the local snapshot.

        Required: title
        Optional: adr_id, status, date
        """
        return self._create_entity("decision", Decision, {"title": title, **fields})

    def create_risk(self, name: str, **fields: Any) -> RiskIssue:
        """Create a new risk/issue in the local snapshot.

        Required: name
        Optional: type, status, severity, owner, mitigation_plan
        """
        return self._create_entity("risk", RiskIssue, {"name": name, **fields})

    # ------------------------------------------------------------------ #
    #  Public API — Read local state                                       #
    # ------------------------------------------------------------------ #

    def load_local_snapshot(self) -> NotionSnapshot | None:
        """Load the local snapshot (cloud + pending changes applied).

        Returns local_snapshot.json if it exists, otherwise falls back
        to the cloud snapshot (same as NotionTool.load_snapshot()).
        Returns None if no snapshot exists at all.
        """
        local_path = self.data_dir / "local_snapshot.json"
        if local_path.exists():
            data = json.loads(local_path.read_text(encoding="utf-8"))
            return NotionSnapshot.model_validate(data)

        # Fall back to cloud snapshot
        meta_path = self.data_dir / "sync_meta.json"
        if not meta_path.exists():
            return None
        return self._load_cloud_snapshot()

    def load_pending_changes(self) -> ChangeLog:
        """Load the current pending changes log.

        Returns empty ChangeLog if no changes have been made.
        """
        path = self.data_dir / "pending_changes.json"
        if not path.exists():
            return ChangeLog()
        data = json.loads(path.read_text(encoding="utf-8"))
        return ChangeLog.model_validate(data)

    def has_pending_changes(self) -> bool:
        """Check if there are any uncommitted local changes."""
        path = self.data_dir / "pending_changes.json"
        if not path.exists():
            return False
        changelog = self.load_pending_changes()
        return len(changelog.changes) > 0

    def discard_pending_changes(self) -> None:
        """Discard all pending changes and remove local_snapshot.json.

        This resets the local state back to the cloud snapshot.
        Deletes pending_changes.json and local_snapshot.json.
        """
        pending_path = self.data_dir / "pending_changes.json"
        local_path = self.data_dir / "local_snapshot.json"
        if pending_path.exists():
            pending_path.unlink()
        if local_path.exists():
            local_path.unlink()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _load_cloud_snapshot(self) -> NotionSnapshot:
        """Load the original cloud snapshot from sync files.

        Raises NotionWriteError if no sync has been done yet.
        """
        meta_path = self.data_dir / "sync_meta.json"
        if not meta_path.exists():
            raise NotionWriteError(
                "No cloud snapshot found. Run sync() first."
            )

        meta = SyncMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))

        def _load_list(filename: str, model_cls: type) -> list:
            path = self.data_dir / filename
            if not path.exists():
                return []
            raw = json.loads(path.read_text(encoding="utf-8"))
            return [model_cls.model_validate(item) for item in raw]

        return NotionSnapshot(
            work_items=_load_list("work_items.json", WorkItem),
            sprints=_load_list("sprints.json", Sprint),
            docs=_load_list("docs_specs.json", DocSpec),
            decisions=_load_list("decisions.json", Decision),
            risks=_load_list("risks_issues.json", RiskIssue),
            meta=meta,
        )

    def _get_current_snapshot(self) -> NotionSnapshot:
        """Get the current effective snapshot.

        Returns local_snapshot.json if it exists, otherwise cloud snapshot.
        This is what agents should read from to get the latest state.
        """
        local_path = self.data_dir / "local_snapshot.json"
        if local_path.exists():
            data = json.loads(local_path.read_text(encoding="utf-8"))
            return NotionSnapshot.model_validate(data)
        return self._load_cloud_snapshot()

    def _find_entity(
        self, snapshot: NotionSnapshot, entity_type: str, notion_id: str,
    ) -> tuple[BaseModel, list]:
        """Find an entity by type and ID in a snapshot.

        Returns (entity, entity_list) tuple.
        Raises NotionWriteError if not found.
        """
        attr_name = _ENTITY_REGISTRY[entity_type][0]
        entity_list = getattr(snapshot, attr_name)

        id_field = "title" if entity_type == "decision" else "name"
        _ = id_field  # unused, we search by notion_id

        for entity in entity_list:
            if entity.notion_id == notion_id:
                return entity, entity_list

        raise NotionWriteError(
            f"{entity_type} with notion_id '{notion_id}' not found."
        )

    def _record_change(self, change: PendingChange) -> None:
        """Append a change to pending_changes.json."""
        changelog = self.load_pending_changes()
        now = datetime.now(timezone.utc).isoformat()

        if not changelog.created_at:
            changelog.created_at = now
        changelog.last_modified = now
        changelog.changes.append(change)

        path = self.data_dir / "pending_changes.json"
        path.write_text(
            json.dumps(changelog.model_dump(), default=str, indent=2),
            encoding="utf-8",
        )

    def _save_local_snapshot(self, snapshot: NotionSnapshot) -> None:
        """Write the merged snapshot to local_snapshot.json."""
        path = self.data_dir / "local_snapshot.json"
        path.write_text(
            json.dumps(snapshot.model_dump(), default=str, indent=2),
            encoding="utf-8",
        )

    def _validate_update_fields(
        self, entity_type: str, fields: dict,
    ) -> None:
        """Validate that only allowed fields are being updated.

        Raises NotionWriteError if a disallowed field is passed.
        """
        allowed = _ALLOWED_UPDATE_FIELDS.get(entity_type)
        if allowed is None:
            raise NotionWriteError(f"Unknown entity type: {entity_type}")

        for field_name in fields:
            if field_name not in allowed:
                raise NotionWriteError(
                    f"Field '{field_name}' is not allowed for "
                    f"{entity_type} updates. Allowed: {sorted(allowed)}"
                )

    def _generate_local_id(self) -> str:
        """Generate a UUID prefixed with 'local-' for new entities."""
        return f"local-{uuid.uuid4()}"

    # ------------------------------------------------------------------ #
    #  Generic update/create (reduces boilerplate)                         #
    # ------------------------------------------------------------------ #

    def _update_entity(
        self, entity_type: str, notion_id: str, fields: dict,
    ) -> BaseModel:
        """Generic update logic shared by all update_* methods."""
        self._validate_update_fields(entity_type, fields)
        snapshot = self._get_current_snapshot()
        entity, entity_list = self._find_entity(snapshot, entity_type, notion_id)

        # Record each field change and build the updated entity
        for field_name, new_value in fields.items():
            old_value = getattr(entity, field_name)
            change = PendingChange(
                timestamp=datetime.now(timezone.utc).isoformat(),
                action="update",
                entity_type=entity_type,
                entity_id=notion_id,
                field=field_name,
                old_value=old_value,
                new_value=new_value,
            )
            self._record_change(change)

        # Create immutable copy with updates applied
        updated = entity.model_copy(update=fields)

        # Replace in snapshot list
        idx = entity_list.index(entity)
        entity_list[idx] = updated

        self._save_local_snapshot(snapshot)
        return updated

    def _create_entity(
        self, entity_type: str, model_cls: type[BaseModel], fields: dict,
    ) -> BaseModel:
        """Generic create logic shared by all create_* methods."""
        snapshot = self._get_current_snapshot()
        local_id = self._generate_local_id()

        entity = model_cls(notion_id=local_id, **fields)

        change = PendingChange(
            timestamp=datetime.now(timezone.utc).isoformat(),
            action="create",
            entity_type=entity_type,
            entity_id=local_id,
            entity_snapshot=entity.model_dump(),
        )
        self._record_change(change)

        attr_name = _ENTITY_REGISTRY[entity_type][0]
        getattr(snapshot, attr_name).append(entity)

        self._save_local_snapshot(snapshot)
        return entity
