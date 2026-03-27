"""Tests for tools/notion_tool.py.

All tests mock the Notion client — no API calls are made.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from schemas.notion_models import NotionSnapshot, WorkItem
from tools.notion_tool import NotionSyncError, NotionTool

FIXTURES = Path(__file__).parent / "fixtures"

# Empty blocks response for mocking blocks.children.list
_EMPTY_BLOCKS = {"results": [], "has_more": False, "next_cursor": None}


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _make_settings(**overrides):
    """Create a minimal Settings-like object for testing."""
    defaults = {
        "notion_api_key": "test-api-key",
        "notion_work_items_db": "wi-db-id",
        "notion_sprints_db": "sp-db-id",
        "notion_docs_db": "doc-db-id",
        "notion_decisions_db": "dec-db-id",
        "notion_risks_db": "rsk-db-id",
        "data_dir": Path("/tmp/test-data"),
    }
    defaults.update(overrides)
    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


def _setup_sync_mocks(mock_client, fixtures=None):
    """Configure a mock client for sync tests (properties + empty blocks)."""
    if fixtures is None:
        fixtures = {
            "wi-db-id": _load_fixture("work_items_raw.json"),
            "sp-db-id": _load_fixture("sprints_raw.json"),
            "doc-db-id": _load_fixture("docs_specs_raw.json"),
            "dec-db-id": _load_fixture("decisions_raw.json"),
            "rsk-db-id": _load_fixture("risks_issues_raw.json"),
        }
    mock_client.data_sources.query.side_effect = lambda **kw: fixtures[kw["data_source_id"]]
    mock_client.blocks.children.list.return_value = _EMPTY_BLOCKS


class TestNotionToolInit:
    def test_raises_without_api_key(self):
        settings = _make_settings(notion_api_key=None)
        with pytest.raises(NotionSyncError, match="NOTION_API_KEY"):
            NotionTool(settings)

    @patch("tools.notion_tool.Client")
    def test_creates_client(self, mock_client_cls):
        settings = _make_settings()
        tool = NotionTool(settings)
        mock_client_cls.assert_called_once_with(auth="test-api-key")
        assert tool.data_dir == Path("/tmp/test-data/notion")


class TestPropertyExtractors:
    """Test individual _get_* static methods."""

    @patch("tools.notion_tool.Client")
    def setup_method(self, method, mock_client_cls):
        self.tool = NotionTool(_make_settings())

    def test_get_title(self):
        props = {"Name": {"type": "title", "title": [{"plain_text": "Hello"}]}}
        assert self.tool._get_title(props, "Name") == "Hello"

    def test_get_title_empty(self):
        props = {"Name": {"type": "title", "title": []}}
        assert self.tool._get_title(props, "Name") == ""

    def test_get_title_missing(self):
        assert self.tool._get_title({}, "Name") == ""

    def test_get_select(self):
        props = {"Type": {"type": "select", "select": {"name": "Task"}}}
        assert self.tool._get_select(props, "Type") == "Task"

    def test_get_select_none(self):
        props = {"Type": {"type": "select", "select": None}}
        assert self.tool._get_select(props, "Type") is None

    def test_get_status(self):
        props = {"Status": {"type": "status", "status": {"name": "In Progress"}}}
        assert self.tool._get_status(props, "Status") == "In Progress"

    def test_get_number(self):
        props = {"Estimate (hrs)": {"type": "number", "number": 4.5}}
        assert self.tool._get_number(props, "Estimate (hrs)") == 4.5

    def test_get_number_null(self):
        props = {"Estimate (hrs)": {"type": "number", "number": None}}
        assert self.tool._get_number(props, "Estimate (hrs)") is None

    def test_get_date(self):
        props = {"Due date": {"type": "date", "date": {"start": "2026-03-28"}}}
        assert self.tool._get_date(props, "Due date") == "2026-03-28"

    def test_get_date_null(self):
        props = {"Due date": {"type": "date", "date": None}}
        assert self.tool._get_date(props, "Due date") is None

    def test_get_url(self):
        props = {"Links": {"type": "url", "url": "https://example.com"}}
        assert self.tool._get_url(props, "Links") == "https://example.com"

    def test_get_rich_text(self):
        props = {"DoD": {"type": "rich_text", "rich_text": [{"plain_text": "Deployed."}]}}
        assert self.tool._get_rich_text(props, "DoD") == "Deployed."

    def test_get_rich_text_empty(self):
        props = {"DoD": {"type": "rich_text", "rich_text": []}}
        assert self.tool._get_rich_text(props, "DoD") is None

    def test_get_checkbox(self):
        props = {"Active": {"type": "checkbox", "checkbox": True}}
        assert self.tool._get_checkbox(props, "Active") is True

    def test_get_checkbox_false(self):
        props = {"Active": {"type": "checkbox", "checkbox": False}}
        assert self.tool._get_checkbox(props, "Active") is False

    def test_get_multi_select(self):
        props = {"Tags": {"type": "multi_select", "multi_select": [
            {"name": "Networking"}, {"name": "Security"}
        ]}}
        assert self.tool._get_multi_select(props, "Tags") == ["Networking", "Security"]

    def test_get_multi_select_empty(self):
        props = {"Tags": {"type": "multi_select", "multi_select": []}}
        assert self.tool._get_multi_select(props, "Tags") == []

    def test_get_relation_ids(self):
        props = {"Deps": {"type": "relation", "relation": [
            {"id": "aaa"}, {"id": "bbb"}
        ]}}
        assert self.tool._get_relation_ids(props, "Deps") == ["aaa", "bbb"]

    def test_get_relation_ids_empty(self):
        props = {"Deps": {"type": "relation", "relation": []}}
        assert self.tool._get_relation_ids(props, "Deps") == []

    def test_get_person_name(self):
        props = {"Owner": {"type": "people", "people": [
            {"id": "u1", "name": "Nathan Teuber"}
        ]}}
        assert self.tool._get_person_name(props, "Owner") == "Nathan Teuber"

    def test_get_person_no_name_falls_back_to_id(self):
        props = {"Owner": {"type": "people", "people": [{"id": "u1"}]}}
        assert self.tool._get_person_name(props, "Owner") == "u1"

    def test_get_person_empty(self):
        props = {"Owner": {"type": "people", "people": []}}
        assert self.tool._get_person_name(props, "Owner") is None


class TestMappers:
    """Test _map_* methods with fixture data."""

    @patch("tools.notion_tool.Client")
    def setup_method(self, method, mock_client_cls):
        self.tool = NotionTool(_make_settings())

    def test_map_work_item(self):
        fixture = _load_fixture("work_items_raw.json")
        page = fixture["results"][0]
        wi = self.tool._map_work_item(page)

        assert wi.notion_id == "aaa11111-1111-1111-1111-111111111111"
        assert wi.name == "Deploy SHIR VM to Azure"
        assert wi.type == "Task"
        assert wi.status == "In Progress"
        assert wi.priority == "P1"
        assert wi.estimate_hrs == 4.0
        assert wi.owner == "Nathan Teuber"
        assert wi.due_date == "2026-03-28"
        assert wi.sprint_id == "spr11111-1111-1111-1111-111111111111"
        assert wi.dependency_ids == ["bbb22222-2222-2222-2222-222222222222"]
        assert wi.parent_epic_id == "epc11111-1111-1111-1111-111111111111"
        assert wi.decision_ids == ["dec11111-1111-1111-1111-111111111111"]
        assert wi.doc_ids == ["doc11111-1111-1111-1111-111111111111"]
        assert wi.risk_ids == ["rsk11111-1111-1111-1111-111111111111"]
        assert wi.definition_of_done == "VM deployed and SHIR registered in ADF."
        assert wi.links == "https://dev.azure.com/project/wiki"

    def test_map_work_item_with_nulls(self):
        fixture = _load_fixture("work_items_raw.json")
        page = fixture["results"][1]  # Item with many null fields
        wi = self.tool._map_work_item(page)

        assert wi.name == "Configure VNet and subnets"
        assert wi.status == "Done"
        assert wi.due_date is None
        assert wi.parent_epic_id is None
        assert wi.dependency_ids == []
        assert wi.definition_of_done is None
        assert wi.links is None

    def test_map_sprint(self):
        fixture = _load_fixture("sprints_raw.json")
        page = fixture["results"][0]
        s = self.tool._map_sprint(page)

        assert s.notion_id == "spr11111-1111-1111-1111-111111111111"
        assert s.name == "Sprint 1.3"
        assert s.type == "Sprint"
        assert s.status == "Active"
        assert s.sprint_number == 1  # int(1.3) = 1
        assert s.goal == "Deploy SHIR and first ADF pipeline"
        assert s.start_date == "2026-03-10"
        assert s.end_date == "2026-03-28"
        assert len(s.work_item_ids) == 2
        assert len(s.risk_ids) == 1

    def test_map_doc(self):
        fixture = _load_fixture("docs_specs_raw.json")
        page = fixture["results"][0]
        d = self.tool._map_doc(page)

        assert d.name == "Network Egress & PE Rules"
        assert d.doc_type == "Architecture"
        assert d.active is True
        assert d.tags == ["Networking", "Security"]
        assert d.last_reviewed == "2026-03-15"
        assert len(d.work_item_ids) == 1
        assert len(d.decision_ids) == 1
        assert len(d.sprint_ids) == 1

    def test_map_decision(self):
        fixture = _load_fixture("decisions_raw.json")
        page = fixture["results"][0]
        d = self.tool._map_decision(page)

        assert d.title == "ADR-004: SSRS VM Authentication via Entra + KCD"
        assert d.adr_id == "ADR-004"
        assert d.status == "Accepted"
        assert d.date == "2026-02-15"
        assert len(d.work_item_ids) == 1
        assert len(d.doc_ids) == 1

    def test_map_risk(self):
        fixture = _load_fixture("risks_issues_raw.json")
        page = fixture["results"][0]
        r = self.tool._map_risk(page)

        assert r.name == "SSRS KCD may require on-prem AD dependency"
        assert r.type == "Risk"
        assert r.status == "Open"
        assert r.severity == "High"
        assert r.owner == "Nathan Teuber"
        assert r.mitigation_plan == "Validate via spike; fallback to Power BI paginated reports."
        assert r.next_review == "2026-04-01"
        assert r.sprint_id == "spr11111-1111-1111-1111-111111111111"


class TestSync:
    """Test sync() and load_snapshot() with mocked Notion client."""

    @patch("tools.notion_tool.Client")
    def test_sync_writes_files(self, mock_client_cls, tmp_path):
        settings = _make_settings(data_dir=tmp_path)
        mock_client = mock_client_cls.return_value
        _setup_sync_mocks(mock_client)

        tool = NotionTool(settings)
        tool._REQUEST_INTERVAL = 0  # Disable rate limiting in tests
        meta = tool.sync()

        # Check files were created
        notion_dir = tmp_path / "notion"
        assert (notion_dir / "work_items.json").exists()
        assert (notion_dir / "sprints.json").exists()
        assert (notion_dir / "docs_specs.json").exists()
        assert (notion_dir / "decisions.json").exists()
        assert (notion_dir / "risks_issues.json").exists()
        assert (notion_dir / "sync_meta.json").exists()

        # Check counts
        assert meta.counts["work_items"] == 2
        assert meta.counts["sprints"] == 1
        assert meta.counts["docs"] == 1
        assert meta.counts["decisions"] == 1
        assert meta.counts["risks"] == 1

        # Verify JSON content
        items = json.loads((notion_dir / "work_items.json").read_text())
        assert len(items) == 2
        assert items[0]["name"] == "Deploy SHIR VM to Azure"

        # Verify content directory was created
        assert (notion_dir / "content").exists()

    @patch("tools.notion_tool.Client")
    def test_sync_dry_run_no_files(self, mock_client_cls, tmp_path):
        settings = _make_settings(data_dir=tmp_path)
        mock_client = mock_client_cls.return_value
        _setup_sync_mocks(mock_client)

        tool = NotionTool(settings)
        tool._REQUEST_INTERVAL = 0
        meta = tool.sync(dry_run=True)

        assert meta.counts["work_items"] == 2
        assert not (tmp_path / "notion").exists()

    @patch("tools.notion_tool.Client")
    def test_load_snapshot_round_trip(self, mock_client_cls, tmp_path):
        settings = _make_settings(data_dir=tmp_path)
        mock_client = mock_client_cls.return_value
        _setup_sync_mocks(mock_client)

        tool = NotionTool(settings)
        tool._REQUEST_INTERVAL = 0
        tool.sync()

        snapshot = tool.load_snapshot()
        assert snapshot is not None
        assert len(snapshot.work_items) == 2
        assert len(snapshot.sprints) == 1
        assert len(snapshot.docs) == 1
        assert len(snapshot.decisions) == 1
        assert len(snapshot.risks) == 1
        assert snapshot.work_items[0].name == "Deploy SHIR VM to Azure"

    @patch("tools.notion_tool.Client")
    def test_load_snapshot_no_data(self, mock_client_cls, tmp_path):
        settings = _make_settings(data_dir=tmp_path)
        tool = NotionTool(settings)
        assert tool.load_snapshot() is None


class TestQueryHelpers:
    @patch("tools.notion_tool.Client")
    def test_get_work_items_filter_by_status(self, mock_client_cls, tmp_path):
        settings = _make_settings(data_dir=tmp_path)
        mock_client = mock_client_cls.return_value
        _setup_sync_mocks(mock_client)

        tool = NotionTool(settings)
        tool._REQUEST_INTERVAL = 0
        tool.sync()

        done_items = tool.get_work_items(status="Done")
        assert len(done_items) == 1
        assert done_items[0].name == "Configure VNet and subnets"

        in_progress = tool.get_work_items(status="In Progress")
        assert len(in_progress) == 1

    @patch("tools.notion_tool.Client")
    def test_get_active_sprint(self, mock_client_cls, tmp_path):
        settings = _make_settings(data_dir=tmp_path)
        mock_client = mock_client_cls.return_value
        _setup_sync_mocks(mock_client)

        tool = NotionTool(settings)
        tool._REQUEST_INTERVAL = 0
        tool.sync()

        sprint = tool.get_active_sprint()
        assert sprint is not None
        assert sprint.name == "Sprint 1.3"
        assert sprint.status == "Active"


class TestPagination:
    @patch("tools.notion_tool.Client")
    def test_handles_pagination(self, mock_client_cls):
        settings = _make_settings()
        mock_client = mock_client_cls.return_value

        page1 = {
            "results": [{"id": "p1", "properties": {
                "Name": {"type": "title", "title": [{"plain_text": "Item 1"}]},
                "Type": {"type": "select", "select": None},
                "Status": {"type": "status", "status": None},
                "Priority": {"type": "select", "select": None},
                "Estimate (hrs)": {"type": "number", "number": None},
                "Owner": {"type": "people", "people": []},
                "Due date": {"type": "date", "date": None},
                "Phase/Sprint": {"type": "relation", "relation": []},
                "Dependencies": {"type": "relation", "relation": []},
                "Parent Epic": {"type": "relation", "relation": []},
                "Child Work Items": {"type": "relation", "relation": []},
                "Decisions (ADRs)": {"type": "relation", "relation": []},
                "Docs & Specs": {"type": "relation", "relation": []},
                "Risks & Issues": {"type": "relation", "relation": []},
                "Definition of Done": {"type": "rich_text", "rich_text": []},
                "Links": {"type": "url", "url": None},
            }}],
            "has_more": True,
            "next_cursor": "cursor-abc",
        }
        page2 = {
            "results": [{"id": "p2", "properties": {
                "Name": {"type": "title", "title": [{"plain_text": "Item 2"}]},
                "Type": {"type": "select", "select": None},
                "Status": {"type": "status", "status": None},
                "Priority": {"type": "select", "select": None},
                "Estimate (hrs)": {"type": "number", "number": None},
                "Owner": {"type": "people", "people": []},
                "Due date": {"type": "date", "date": None},
                "Phase/Sprint": {"type": "relation", "relation": []},
                "Dependencies": {"type": "relation", "relation": []},
                "Parent Epic": {"type": "relation", "relation": []},
                "Child Work Items": {"type": "relation", "relation": []},
                "Decisions (ADRs)": {"type": "relation", "relation": []},
                "Docs & Specs": {"type": "relation", "relation": []},
                "Risks & Issues": {"type": "relation", "relation": []},
                "Definition of Done": {"type": "rich_text", "rich_text": []},
                "Links": {"type": "url", "url": None},
            }}],
            "has_more": False,
            "next_cursor": None,
        }

        mock_client.data_sources.query.side_effect = [page1, page2]

        tool = NotionTool(settings)
        tool._REQUEST_INTERVAL = 0
        pages = tool._query_all_pages("test-db")
        assert len(pages) == 2
        assert mock_client.data_sources.query.call_count == 2
        # Second call should include start_cursor
        second_call_kwargs = mock_client.data_sources.query.call_args_list[1][1]
        assert second_call_kwargs["start_cursor"] == "cursor-abc"


class TestFetchPageBlocks:
    """Test _fetch_page_blocks with mocked blocks API."""

    @patch("tools.notion_tool.Client")
    def test_fetches_single_page_blocks(self, mock_client_cls):
        settings = _make_settings()
        mock_client = mock_client_cls.return_value
        mock_client.blocks.children.list.return_value = {
            "results": [
                {"id": "b1", "type": "paragraph", "has_children": False,
                 "paragraph": {"rich_text": [{"type": "text", "text": {"content": "Hello"}}], "color": "default"}},
            ],
            "has_more": False,
            "next_cursor": None,
        }

        tool = NotionTool(settings)
        tool._REQUEST_INTERVAL = 0
        blocks = tool._fetch_page_blocks("page-id-1")

        assert len(blocks) == 1
        assert blocks[0]["type"] == "paragraph"
        mock_client.blocks.children.list.assert_called_once()

    @patch("tools.notion_tool.Client")
    def test_handles_pagination(self, mock_client_cls):
        settings = _make_settings()
        mock_client = mock_client_cls.return_value

        mock_client.blocks.children.list.side_effect = [
            {"results": [{"id": "b1", "type": "paragraph", "has_children": False}],
             "has_more": True, "next_cursor": "cur-1"},
            {"results": [{"id": "b2", "type": "paragraph", "has_children": False}],
             "has_more": False, "next_cursor": None},
        ]

        tool = NotionTool(settings)
        tool._REQUEST_INTERVAL = 0
        blocks = tool._fetch_page_blocks("page-id-1")

        assert len(blocks) == 2
        assert mock_client.blocks.children.list.call_count == 2

    @patch("tools.notion_tool.Client")
    def test_recursive_children(self, mock_client_cls):
        settings = _make_settings()
        mock_client = mock_client_cls.return_value

        def mock_list(**kwargs):
            block_id = kwargs["block_id"]
            if block_id == "page-id-1":
                return {
                    "results": [
                        {"id": "b1", "type": "callout", "has_children": True,
                         "callout": {"rich_text": [], "color": "blue_bg"}},
                    ],
                    "has_more": False, "next_cursor": None,
                }
            if block_id == "b1":
                return {
                    "results": [
                        {"id": "b1-child", "type": "paragraph", "has_children": False,
                         "paragraph": {"rich_text": [], "color": "default"}},
                    ],
                    "has_more": False, "next_cursor": None,
                }
            return _EMPTY_BLOCKS

        mock_client.blocks.children.list.side_effect = mock_list

        tool = NotionTool(settings)
        tool._REQUEST_INTERVAL = 0
        blocks = tool._fetch_page_blocks("page-id-1")

        assert len(blocks) == 1
        assert "children" in blocks[0]
        assert len(blocks[0]["children"]) == 1
        assert blocks[0]["children"][0]["id"] == "b1-child"

    @patch("tools.notion_tool.Client")
    def test_depth_limit(self, mock_client_cls):
        settings = _make_settings()
        mock_client = mock_client_cls.return_value

        # Should return empty at max_depth
        tool = NotionTool(settings)
        tool._REQUEST_INTERVAL = 0
        blocks = tool._fetch_page_blocks("page-id", depth=4, max_depth=4)
        assert blocks == []
        mock_client.blocks.children.list.assert_not_called()

    @patch("tools.notion_tool.Client")
    def test_dedup_prevents_cycles(self, mock_client_cls):
        settings = _make_settings()
        mock_client = mock_client_cls.return_value

        tool = NotionTool(settings)
        tool._REQUEST_INTERVAL = 0
        seen = {"already-seen-id"}
        blocks = tool._fetch_page_blocks("already-seen-id", seen=seen)
        assert blocks == []
        mock_client.blocks.children.list.assert_not_called()

    @patch("tools.notion_tool.Client")
    def test_child_page_not_recursed(self, mock_client_cls):
        settings = _make_settings()
        mock_client = mock_client_cls.return_value

        mock_client.blocks.children.list.return_value = {
            "results": [
                {"id": "cp1", "type": "child_page", "has_children": True,
                 "child_page": {"title": "Sub Page"}},
            ],
            "has_more": False, "next_cursor": None,
        }

        tool = NotionTool(settings)
        tool._REQUEST_INTERVAL = 0
        blocks = tool._fetch_page_blocks("page-id")

        assert len(blocks) == 1
        assert "children" not in blocks[0]
        # Only one call — child_page was not recursed
        assert mock_client.blocks.children.list.call_count == 1

    @patch("tools.notion_tool.Client")
    def test_empty_page(self, mock_client_cls):
        settings = _make_settings()
        mock_client = mock_client_cls.return_value
        mock_client.blocks.children.list.return_value = _EMPTY_BLOCKS

        tool = NotionTool(settings)
        tool._REQUEST_INTERVAL = 0
        blocks = tool._fetch_page_blocks("page-id")
        assert blocks == []


class TestSyncWithContent:
    """Test sync() with page content fetching."""

    @patch("tools.notion_tool.Client")
    def test_sync_creates_content_directory(self, mock_client_cls, tmp_path):
        settings = _make_settings(data_dir=tmp_path)
        mock_client = mock_client_cls.return_value
        _setup_sync_mocks(mock_client)

        tool = NotionTool(settings)
        tool._REQUEST_INTERVAL = 0
        tool.sync()

        assert (tmp_path / "notion" / "content").exists()

    @patch("tools.notion_tool.Client")
    def test_sync_writes_markdown_files(self, mock_client_cls, tmp_path):
        settings = _make_settings(data_dir=tmp_path)
        mock_client = mock_client_cls.return_value
        _setup_sync_mocks(mock_client)

        # Override blocks for one specific page to return content
        original_blocks_list = mock_client.blocks.children.list.return_value
        def mock_blocks(**kwargs):
            if kwargs["block_id"] == "aaa11111-1111-1111-1111-111111111111":
                return {
                    "results": [
                        {"id": "b1", "type": "heading_1", "has_children": False,
                         "heading_1": {"rich_text": [{"type": "text", "text": {"content": "Goal"}, "annotations": {"bold": False, "italic": False, "strikethrough": False, "code": False, "underline": False, "color": "default"}}], "color": "default"}},
                        {"id": "b2", "type": "paragraph", "has_children": False,
                         "paragraph": {"rich_text": [{"type": "text", "text": {"content": "Deploy SHIR"}, "annotations": {"bold": False, "italic": False, "strikethrough": False, "code": False, "underline": False, "color": "default"}}], "color": "default"}},
                    ],
                    "has_more": False, "next_cursor": None,
                }
            return _EMPTY_BLOCKS

        mock_client.blocks.children.list.side_effect = mock_blocks

        tool = NotionTool(settings)
        tool._REQUEST_INTERVAL = 0
        meta = tool.sync()

        # Check markdown file was written
        md_path = tmp_path / "notion" / "content" / "aaa11111-1111-1111-1111-111111111111.md"
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "# Goal" in content
        assert "Deploy SHIR" in content

        # Check content_counts
        assert meta.content_counts["work_items"] >= 1

    @patch("tools.notion_tool.Client")
    def test_sync_meta_includes_content_counts(self, mock_client_cls, tmp_path):
        settings = _make_settings(data_dir=tmp_path)
        mock_client = mock_client_cls.return_value
        _setup_sync_mocks(mock_client)

        tool = NotionTool(settings)
        tool._REQUEST_INTERVAL = 0
        meta = tool.sync()

        assert "work_items" in meta.content_counts
        assert "sprints" in meta.content_counts

    @patch("tools.notion_tool.Client")
    def test_load_page_content(self, mock_client_cls, tmp_path):
        settings = _make_settings(data_dir=tmp_path)
        mock_client = mock_client_cls.return_value

        tool = NotionTool(settings)
        # Write a test content file
        content_dir = tmp_path / "notion" / "content"
        content_dir.mkdir(parents=True)
        (content_dir / "test-id.md").write_text("# Test\nContent here", encoding="utf-8")

        result = tool.load_page_content("test-id")
        assert result == "# Test\nContent here"

    @patch("tools.notion_tool.Client")
    def test_load_page_content_nonexistent(self, mock_client_cls, tmp_path):
        settings = _make_settings(data_dir=tmp_path)
        tool = NotionTool(settings)
        assert tool.load_page_content("nonexistent-id") is None


class TestTemplateSync:
    """Test template identification and storage."""

    @patch("tools.notion_tool.Client")
    def test_templates_filtered_from_entity_lists(self, mock_client_cls, tmp_path):
        settings = _make_settings(data_dir=tmp_path)
        mock_client = mock_client_cls.return_value

        # Create fixture with a template item
        wi_fixture = _load_fixture("work_items_raw.json")
        template_page = {
            "id": "tmpl-1111-1111-1111-111111111111",
            "url": "https://www.notion.so/tmpl1111",
            "properties": {
                "Name": {"type": "title", "title": [{"plain_text": "Epic template"}]},
                "Type": {"type": "select", "select": {"name": "Epic"}},
                "Status": {"type": "status", "status": {"name": "Backlog"}},
                "Priority": {"type": "select", "select": None},
                "Estimate (hrs)": {"type": "number", "number": None},
                "Owner": {"type": "people", "people": []},
                "Due date": {"type": "date", "date": None},
                "Phase/Sprint": {"type": "relation", "relation": []},
                "Dependencies": {"type": "relation", "relation": []},
                "Parent Epic": {"type": "relation", "relation": []},
                "Child Work Items": {"type": "relation", "relation": []},
                "Decisions (ADRs)": {"type": "relation", "relation": []},
                "Docs & Specs": {"type": "relation", "relation": []},
                "Risks & Issues": {"type": "relation", "relation": []},
                "Definition of Done": {"type": "rich_text", "rich_text": []},
                "Links": {"type": "url", "url": None},
            },
        }
        wi_fixture["results"].append(template_page)

        fixtures = {
            "wi-db-id": wi_fixture,
            "sp-db-id": _load_fixture("sprints_raw.json"),
            "doc-db-id": _load_fixture("docs_specs_raw.json"),
            "dec-db-id": _load_fixture("decisions_raw.json"),
            "rsk-db-id": _load_fixture("risks_issues_raw.json"),
        }
        mock_client.data_sources.query.side_effect = lambda **kw: fixtures[kw["data_source_id"]]
        mock_client.blocks.children.list.return_value = _EMPTY_BLOCKS

        tool = NotionTool(settings)
        tool._REQUEST_INTERVAL = 0
        meta = tool.sync()

        # Template should be filtered out — only 2 work items, not 3
        assert meta.counts["work_items"] == 2

        # Template should be recorded in metadata
        assert meta.template_counts["work_items"] == 1
        assert len(meta.templates) == 1
        assert meta.templates[0].title == "Epic template"

        # Template directory should be created
        assert (tmp_path / "notion" / "templates" / "work_items").exists()

    @patch("tools.notion_tool.Client")
    def test_template_content_written(self, mock_client_cls, tmp_path):
        settings = _make_settings(data_dir=tmp_path)
        mock_client = mock_client_cls.return_value

        wi_fixture = _load_fixture("work_items_raw.json")
        template_page = {
            "id": "tmpl-2222-2222-2222-222222222222",
            "url": "https://www.notion.so/tmpl2222",
            "properties": {
                "Name": {"type": "title", "title": [{"plain_text": "Bug template"}]},
                "Type": {"type": "select", "select": {"name": "Bug"}},
                "Status": {"type": "status", "status": {"name": "Backlog"}},
                "Priority": {"type": "select", "select": None},
                "Estimate (hrs)": {"type": "number", "number": None},
                "Owner": {"type": "people", "people": []},
                "Due date": {"type": "date", "date": None},
                "Phase/Sprint": {"type": "relation", "relation": []},
                "Dependencies": {"type": "relation", "relation": []},
                "Parent Epic": {"type": "relation", "relation": []},
                "Child Work Items": {"type": "relation", "relation": []},
                "Decisions (ADRs)": {"type": "relation", "relation": []},
                "Docs & Specs": {"type": "relation", "relation": []},
                "Risks & Issues": {"type": "relation", "relation": []},
                "Definition of Done": {"type": "rich_text", "rich_text": []},
                "Links": {"type": "url", "url": None},
            },
        }
        wi_fixture["results"].append(template_page)

        fixtures = {
            "wi-db-id": wi_fixture,
            "sp-db-id": _load_fixture("sprints_raw.json"),
            "doc-db-id": _load_fixture("docs_specs_raw.json"),
            "dec-db-id": _load_fixture("decisions_raw.json"),
            "rsk-db-id": _load_fixture("risks_issues_raw.json"),
        }
        mock_client.data_sources.query.side_effect = lambda **kw: fixtures[kw["data_source_id"]]

        def mock_blocks(**kwargs):
            if kwargs["block_id"] == "tmpl-2222-2222-2222-222222222222":
                return {
                    "results": [
                        {"id": "tb1", "type": "heading_1", "has_children": False,
                         "heading_1": {"rich_text": [{"type": "text", "text": {"content": "Bug Report"}, "annotations": {"bold": False, "italic": False, "strikethrough": False, "code": False, "underline": False, "color": "default"}}], "color": "default"}},
                    ],
                    "has_more": False, "next_cursor": None,
                }
            return _EMPTY_BLOCKS

        mock_client.blocks.children.list.side_effect = mock_blocks

        tool = NotionTool(settings)
        tool._REQUEST_INTERVAL = 0
        tool.sync()

        # Check template file was written
        tmpl_files = list((tmp_path / "notion" / "templates" / "work_items").glob("*.md"))
        assert len(tmpl_files) == 1
        content = tmpl_files[0].read_text(encoding="utf-8")
        assert "# Bug Report" in content
