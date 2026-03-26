"""Read-only Notion sync tool.

Pulls all 5 hub-and-spoke databases from the Synergy Data Platform
teamspace into local JSON files. Agents read from the local snapshot;
no writes to cloud Notion occur in this phase.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from notion_client import Client
from notion_client.errors import APIResponseError

from schemas.notion_models import (
    Decision,
    DocSpec,
    NotionSnapshot,
    RiskIssue,
    Sprint,
    SyncMeta,
    WorkItem,
)


class NotionSyncError(Exception):
    """Raised when a Notion sync operation fails."""


class NotionTool:
    """Read-only Notion sync. Pulls all 5 databases to local JSON."""

    # (settings_attr, model_class, mapper_method, filename)
    _DB_REGISTRY: list[tuple[str, str, type, str, str]] = [
        ("work_items",  "notion_work_items_db",  WorkItem,   "_map_work_item",  "work_items.json"),
        ("sprints",     "notion_sprints_db",      Sprint,     "_map_sprint",     "sprints.json"),
        ("docs",        "notion_docs_db",         DocSpec,    "_map_doc",        "docs_specs.json"),
        ("decisions",   "notion_decisions_db",    Decision,   "_map_decision",   "decisions.json"),
        ("risks",       "notion_risks_db",        RiskIssue,  "_map_risk",       "risks_issues.json"),
    ]

    def __init__(self, settings: Any) -> None:
        if not settings.notion_api_key:
            raise NotionSyncError(
                "NOTION_API_KEY is not set. Add it to your .env file."
            )
        self.client = Client(auth=settings.notion_api_key)
        self.data_dir: Path = Path(settings.data_dir) / "notion"
        self.settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync(self, dry_run: bool = False) -> SyncMeta:
        """Full pull of all 5 databases → local JSON files.

        Args:
            dry_run: If True, query Notion for counts but don't write files.

        Returns:
            SyncMeta with timestamp, database IDs, and page counts.
        """
        results: dict[str, list] = {}
        counts: dict[str, int] = {}
        db_ids: dict[str, str] = {}

        for name, settings_attr, model_cls, mapper_name, filename in self._DB_REGISTRY:
            db_id = getattr(self.settings, settings_attr)
            db_ids[name] = db_id
            mapper = getattr(self, mapper_name)

            try:
                raw_pages = self._query_all_pages(db_id)
            except APIResponseError as e:
                raise NotionSyncError(
                    f"Failed to query '{name}' database ({db_id}): {e.message}"
                ) from e

            items = [mapper(page) for page in raw_pages]
            results[name] = items
            counts[name] = len(items)

        meta = SyncMeta(
            synced_at=datetime.now(timezone.utc).isoformat(),
            databases=db_ids,
            counts=counts,
        )

        if dry_run:
            return meta

        # Write JSON files
        self.data_dir.mkdir(parents=True, exist_ok=True)

        for name, _, model_cls, _, filename in self._DB_REGISTRY:
            items = results[name]
            path = self.data_dir / filename
            path.write_text(
                json.dumps([item.model_dump() for item in items], indent=2),
                encoding="utf-8",
            )

        meta_path = self.data_dir / "sync_meta.json"
        meta_path.write_text(
            json.dumps(meta.model_dump(), indent=2),
            encoding="utf-8",
        )

        return meta

    def load_snapshot(self) -> NotionSnapshot | None:
        """Load the most recent local snapshot from disk.

        Returns None if no sync has been done yet.
        """
        meta_path = self.data_dir / "sync_meta.json"
        if not meta_path.exists():
            return None

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

    def get_work_items(
        self,
        sprint_id: str | None = None,
        status: str | None = None,
        item_type: str | None = None,
    ) -> list[WorkItem]:
        """Filter local work items by sprint, status, and/or type."""
        snapshot = self.load_snapshot()
        if not snapshot:
            return []
        items = snapshot.work_items
        if sprint_id:
            items = [i for i in items if i.sprint_id == sprint_id]
        if status:
            items = [i for i in items if i.status == status]
        if item_type:
            items = [i for i in items if i.type == item_type]
        return items

    def get_active_sprint(self) -> Sprint | None:
        """Return the sprint with status='Active', or None."""
        snapshot = self.load_snapshot()
        if not snapshot:
            return None
        for s in snapshot.sprints:
            if s.status == "Active":
                return s
        return None

    # ------------------------------------------------------------------
    # Notion API helpers
    # ------------------------------------------------------------------

    def _query_all_pages(self, data_source_id: str) -> list[dict]:
        """Paginated data_sources.query() — fetches all pages."""
        pages: list[dict] = []
        cursor: str | None = None

        while True:
            kwargs: dict[str, Any] = {"data_source_id": data_source_id, "page_size": 100}
            if cursor:
                kwargs["start_cursor"] = cursor

            response = self.client.data_sources.query(**kwargs)
            pages.extend(response["results"])

            if not response.get("has_more"):
                break
            cursor = response.get("next_cursor")

        return pages

    # ------------------------------------------------------------------
    # Property extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_title(props: dict, name: str) -> str:
        """Extract a title property value."""
        prop = props.get(name)
        if not prop or not prop.get("title"):
            return ""
        parts = prop["title"]
        return "".join(p.get("plain_text", "") for p in parts)

    @staticmethod
    def _get_rich_text(props: dict, name: str) -> str | None:
        """Extract a rich_text property as plain text."""
        prop = props.get(name)
        if not prop or not prop.get("rich_text"):
            return None
        parts = prop["rich_text"]
        text = "".join(p.get("plain_text", "") for p in parts)
        return text or None

    @staticmethod
    def _get_select(props: dict, name: str) -> str | None:
        """Extract a select property value."""
        prop = props.get(name)
        if not prop or not prop.get("select"):
            return None
        return prop["select"].get("name")

    @staticmethod
    def _get_multi_select(props: dict, name: str) -> list[str]:
        """Extract a multi_select property as a list of names."""
        prop = props.get(name)
        if not prop or not prop.get("multi_select"):
            return []
        return [opt["name"] for opt in prop["multi_select"]]

    @staticmethod
    def _get_status(props: dict, name: str) -> str | None:
        """Extract a status property value."""
        prop = props.get(name)
        if not prop or not prop.get("status"):
            return None
        return prop["status"].get("name")

    @staticmethod
    def _get_number(props: dict, name: str) -> float | None:
        """Extract a number property value."""
        prop = props.get(name)
        if not prop:
            return None
        return prop.get("number")

    @staticmethod
    def _get_checkbox(props: dict, name: str) -> bool:
        """Extract a checkbox property value."""
        prop = props.get(name)
        if not prop:
            return False
        return prop.get("checkbox", False)

    @staticmethod
    def _get_date(props: dict, name: str) -> str | None:
        """Extract the start date from a date property."""
        prop = props.get(name)
        if not prop or not prop.get("date"):
            return None
        return prop["date"].get("start")

    @staticmethod
    def _get_url(props: dict, name: str) -> str | None:
        """Extract a URL property value."""
        prop = props.get(name)
        if not prop:
            return None
        return prop.get("url")

    @staticmethod
    def _get_relation_ids(props: dict, name: str) -> list[str]:
        """Extract page UUIDs from a relation property."""
        prop = props.get(name)
        if not prop or not prop.get("relation"):
            return []
        return [rel["id"] for rel in prop["relation"]]

    @staticmethod
    def _get_person_name(props: dict, name: str) -> str | None:
        """Extract the first person's name from a people property."""
        prop = props.get(name)
        if not prop or not prop.get("people"):
            return None
        people = prop["people"]
        if not people:
            return None
        person = people[0]
        return person.get("name") or person.get("id")

    @staticmethod
    def _page_url(page: dict) -> str:
        """Build the Notion page URL from a page dict."""
        return page.get("url", f"https://www.notion.so/{page['id'].replace('-', '')}")

    # ------------------------------------------------------------------
    # Mappers: Notion API page → Pydantic model
    # ------------------------------------------------------------------

    def _map_work_item(self, page: dict) -> WorkItem:
        props = page["properties"]
        return WorkItem(
            notion_id=page["id"],
            name=self._get_title(props, "Name"),
            type=self._get_select(props, "Type"),
            status=self._get_status(props, "Status"),
            priority=self._get_select(props, "Priority"),
            estimate_hrs=self._get_number(props, "Estimate (hrs)"),
            owner=self._get_person_name(props, "Owner"),
            due_date=self._get_date(props, "Due date"),
            sprint_id=self._get_relation_ids(props, "Phase/Sprint")[0]
                if self._get_relation_ids(props, "Phase/Sprint") else None,
            dependency_ids=self._get_relation_ids(props, "Dependencies"),
            parent_epic_id=self._get_relation_ids(props, "Parent Epic")[0]
                if self._get_relation_ids(props, "Parent Epic") else None,
            child_item_ids=self._get_relation_ids(props, "Child Work Items"),
            decision_ids=self._get_relation_ids(props, "Decisions (ADRs)"),
            doc_ids=self._get_relation_ids(props, "Docs & Specs"),
            risk_ids=self._get_relation_ids(props, "Risks & Issues"),
            definition_of_done=self._get_rich_text(props, "Definition of Done"),
            links=self._get_url(props, "Links"),
            notion_url=self._page_url(page),
        )

    def _map_sprint(self, page: dict) -> Sprint:
        props = page["properties"]
        num = self._get_number(props, "Sprint #")
        return Sprint(
            notion_id=page["id"],
            name=self._get_title(props, "Name"),
            type=self._get_select(props, "Type"),
            status=self._get_status(props, "Status"),
            sprint_number=int(num) if num is not None else None,
            goal=self._get_rich_text(props, "Goal"),
            start_date=self._get_date(props, "Start date"),
            end_date=self._get_date(props, "End date"),
            work_item_ids=self._get_relation_ids(props, "Work Items"),
            risk_ids=self._get_relation_ids(props, "Risks & Issues"),
        )

    def _map_doc(self, page: dict) -> DocSpec:
        props = page["properties"]
        return DocSpec(
            notion_id=page["id"],
            name=self._get_title(props, "Name"),
            doc_type=self._get_select(props, "Doc Type"),
            status=self._get_status(props, "Status"),
            active=self._get_checkbox(props, "Active"),
            owner=self._get_person_name(props, "Owner"),
            tags=self._get_multi_select(props, "Tags"),
            last_reviewed=self._get_date(props, "Last Reviewed"),
            deprecated=self._get_date(props, "Deprecated"),
            work_item_ids=self._get_relation_ids(props, "Related Work Items"),
            decision_ids=self._get_relation_ids(props, "Decisions (ADRs)"),
            sprint_ids=self._get_relation_ids(props, "Related Phase/Sprint"),
            supersedes_ids=self._get_relation_ids(props, "Supersedes"),
            notion_url=self._page_url(page),
        )

    def _map_decision(self, page: dict) -> Decision:
        props = page["properties"]
        return Decision(
            notion_id=page["id"],
            title=self._get_title(props, "Title"),
            adr_id=self._get_rich_text(props, "ADR ID"),
            status=self._get_select(props, "Status"),
            date=self._get_date(props, "Date"),
            work_item_ids=self._get_relation_ids(props, "Related Work Items"),
            doc_ids=self._get_relation_ids(props, "Related Docs"),
            supersedes_ids=self._get_relation_ids(props, "Supersedes / Superseded by"),
            superseded_by_ids=self._get_relation_ids(
                props, "Supersedes / Superseded by (backlink)"
            ),
            notion_url=self._page_url(page),
        )

    def _map_risk(self, page: dict) -> RiskIssue:
        props = page["properties"]
        return RiskIssue(
            notion_id=page["id"],
            name=self._get_title(props, "Name"),
            type=self._get_select(props, "Type"),
            status=self._get_select(props, "Status"),
            severity=self._get_select(props, "Severity"),
            owner=self._get_person_name(props, "Owner"),
            mitigation_plan=self._get_rich_text(props, "Mitigation Plan"),
            next_review=self._get_date(props, "Next Review"),
            work_item_ids=self._get_relation_ids(props, "Related Work Items"),
            sprint_id=self._get_relation_ids(props, "Phase/Sprint")[0]
                if self._get_relation_ids(props, "Phase/Sprint") else None,
            notion_url=self._page_url(page),
        )
