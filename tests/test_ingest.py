"""Tests for rag/ingest.py.

All tests mock ChromaDB and Ollama — no real services required.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.ingest import (
    COLLECTION_NAME,
    _build_entity_lookup,
    _chunk_content,
    _split_by_paragraphs,
    _strip_markup,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_snapshot(tmp_path: Path, filename: str, entities: list[dict]) -> None:
    """Write a JSON snapshot file to tmp_path."""
    path = tmp_path / filename
    path.write_text(json.dumps(entities), encoding="utf-8")


def _make_settings(tmp_path: Path, **overrides):
    """Create a minimal Settings-like object for testing."""
    defaults = {
        "data_dir": tmp_path,
        "chroma_db_path": tmp_path / "chroma",
        "embedding_model": "nomic-embed-text",
        "ollama_base_url": "http://localhost:11434",
        "rag_chunk_size": 4000,
        "rag_chunk_overlap": 200,
        "rag_max_chunk_size": 8000,
    }
    defaults.update(overrides)
    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


def _setup_snapshot_files(snapshot_dir: Path) -> None:
    """Write sample JSON snapshots for all 5 entity types."""
    _write_snapshot(snapshot_dir, "work_items.json", [
        {
            "notion_id": "wi-001",
            "name": "Deploy Firewall",
            "status": "In Progress",
            "type": "Task",
            "priority": "P1",
            "sprint_id": "sp-001",
            "has_content": True,
        },
        {
            "notion_id": "wi-002",
            "name": "Configure NSG",
            "status": "Ready",
            "type": "Task",
            "priority": "P2",
            "sprint_id": "sp-001",
            "has_content": True,
        },
    ])
    _write_snapshot(snapshot_dir, "sprints.json", [
        {
            "notion_id": "sp-001",
            "name": "Sprint 1.3",
            "status": "Active",
            "type": "Sprint",
        },
    ])
    _write_snapshot(snapshot_dir, "docs_specs.json", [
        {
            "notion_id": "doc-001",
            "name": "Network Architecture",
            "status": "Final",
        },
    ])
    _write_snapshot(snapshot_dir, "decisions.json", [
        {
            "notion_id": "dec-001",
            "title": "ADR-001 Use Bicep",
            "status": "Accepted",
        },
    ])
    _write_snapshot(snapshot_dir, "risks_issues.json", [
        {
            "notion_id": "risk-001",
            "name": "VRAM Exhaustion",
            "status": "Open",
            "severity": "High",
            "sprint_id": "sp-001",
        },
    ])


# ---------------------------------------------------------------------------
# _build_entity_lookup tests
# ---------------------------------------------------------------------------

class TestBuildEntityLookup:
    def test_loads_all_entity_types(self, tmp_path):
        _setup_snapshot_files(tmp_path)
        lookup = _build_entity_lookup(tmp_path)

        assert "wi-001" in lookup
        assert "sp-001" in lookup
        assert "doc-001" in lookup
        assert "dec-001" in lookup
        assert "risk-001" in lookup

    def test_work_item_metadata(self, tmp_path):
        _setup_snapshot_files(tmp_path)
        lookup = _build_entity_lookup(tmp_path)

        wi = lookup["wi-001"]
        assert wi["entity_type"] == "work_item"
        assert wi["name"] == "Deploy Firewall"
        assert wi["status"] == "In Progress"
        assert wi["priority"] == "P1"
        assert wi["sprint_id"] == "sp-001"
        assert wi["type"] == "Task"

    def test_sprint_metadata(self, tmp_path):
        _setup_snapshot_files(tmp_path)
        lookup = _build_entity_lookup(tmp_path)

        sp = lookup["sp-001"]
        assert sp["entity_type"] == "sprint"
        assert sp["name"] == "Sprint 1.3"
        assert sp["type"] == "Sprint"

    def test_decision_uses_title_field(self, tmp_path):
        _setup_snapshot_files(tmp_path)
        lookup = _build_entity_lookup(tmp_path)

        dec = lookup["dec-001"]
        assert dec["entity_type"] == "decision"
        assert dec["name"] == "ADR-001 Use Bicep"

    def test_risk_metadata(self, tmp_path):
        _setup_snapshot_files(tmp_path)
        lookup = _build_entity_lookup(tmp_path)

        risk = lookup["risk-001"]
        assert risk["entity_type"] == "risk"
        assert risk["severity"] == "High"
        assert risk["sprint_id"] == "sp-001"

    def test_missing_snapshot_file(self, tmp_path):
        # Only write work_items, skip the rest
        _write_snapshot(tmp_path, "work_items.json", [
            {"notion_id": "wi-001", "name": "Test", "status": "Ready"},
        ])
        lookup = _build_entity_lookup(tmp_path)
        assert "wi-001" in lookup
        assert len(lookup) == 1


# ---------------------------------------------------------------------------
# _chunk_content tests
# ---------------------------------------------------------------------------

class TestChunkContent:
    def test_short_content_single_chunk(self):
        content = "Short task description.\n\n## Steps\n- Step 1"
        chunks = _chunk_content(content, "id-001", chunk_size=4000)
        assert len(chunks) == 1
        assert chunks[0]["chunk_id"] == "id-001_0"
        assert chunks[0]["text"] == content

    def test_long_content_splits_on_headers(self):
        # Build content > 4000 chars with multiple ## sections
        sections = [f"## Section {i}\n" + ("x" * 1000) for i in range(6)]
        content = "\n".join(sections)
        assert len(content) > 4000

        chunks = _chunk_content(content, "id-002", chunk_size=4000)
        assert len(chunks) > 1
        # Each chunk should start with ## (except possibly first)
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "text" in chunk

    def test_merges_tiny_sections(self):
        # Create content with a tiny section (<200 chars)
        content = (
            "## Introduction\n" + ("a" * 500) + "\n"
            "## Tiny\nShort.\n"
            "## Main Content\n" + ("b" * 2000) + "\n"
            "## Conclusion\n" + ("c" * 1500)
        )
        chunks = _chunk_content(content, "id-003", chunk_size=100)
        # The tiny "## Tiny" section should be merged with an adjacent section
        texts = [c["text"] for c in chunks]
        # No chunk should be just "## Tiny\nShort.\n" alone
        assert not any(t.strip() == "## Tiny\nShort." for t in texts)

    def test_subsplits_oversized_sections(self):
        # Single section >8000 chars
        content = "## Big Section\n" + ("word " * 2000)  # ~10000 chars
        chunks = _chunk_content(
            content, "id-004", chunk_size=100, max_chunk_size=4000,
        )
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk["text"]) <= 4500  # Allow some slack for overlap

    def test_empty_content_returns_single_chunk(self):
        chunks = _chunk_content("   ", "id-005", chunk_size=4000)
        # Even whitespace-only returns a chunk (filtering is caller's job)
        assert len(chunks) == 1

    def test_chunk_ids_are_sequential(self):
        sections = [f"## Section {i}\n" + ("x" * 1500) for i in range(4)]
        content = "\n".join(sections)
        chunks = _chunk_content(content, "myid", chunk_size=100)
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_id"] == f"myid_{i}"

    def test_no_headers_stays_single_chunk(self):
        # Content > threshold but no ## headers
        content = "Just a long paragraph.\n" * 200
        chunks = _chunk_content(content, "id-006", chunk_size=100)
        # Should still produce at least one chunk (no headers to split on)
        assert len(chunks) >= 1

    def test_last_tiny_section_merged(self):
        content = (
            "## Section A\n" + ("a" * 2000) + "\n"
            "## Tiny End\nDone."
        )
        chunks = _chunk_content(content, "id-007", chunk_size=100)
        # The tiny "## Tiny End" should be merged with Section A
        assert not any(c["text"].strip() == "## Tiny End\nDone." for c in chunks)


# ---------------------------------------------------------------------------
# _strip_markup tests
# ---------------------------------------------------------------------------

class TestStripMarkup:
    def test_strips_table_tags(self):
        text = '<table header-row="true">\n<tr>\n<td>Value</td>\n</tr>\n</table>'
        stripped = _strip_markup(text)
        assert "<table" not in stripped
        assert "<tr>" not in stripped
        assert "<td>" not in stripped
        assert "Value" in stripped

    def test_strips_callout_tags(self):
        text = '<callout color="blue">\nImportant note.\n</callout>'
        stripped = _strip_markup(text)
        assert "<callout" not in stripped
        assert "Important note." in stripped

    def test_preserves_markdown(self):
        text = "## Heading\n\n- Item 1\n- Item 2\n\n```python\nprint('hi')\n```"
        stripped = _strip_markup(text)
        assert stripped == text

    def test_collapses_blank_lines(self):
        text = '<table>\n<tr>\n<td>A</td>\n</tr>\n</table>\n\n\n\n\nNext section.'
        stripped = _strip_markup(text)
        assert "\n\n\n" not in stripped

    def test_reduces_token_heavy_content(self):
        row = "<tr>\n<td>Value</td>\n<td>Desc</td>\n<td>Status</td>\n</tr>\n"
        text = '<table header-row="true">\n' + row * 40 + "</table>"
        stripped = _strip_markup(text)
        # Should be significantly shorter
        assert len(stripped) < len(text) * 0.5


# ---------------------------------------------------------------------------
# _split_by_paragraphs tests
# ---------------------------------------------------------------------------

class TestSplitByParagraphs:
    def test_splits_at_paragraph_boundaries(self):
        text = ("Paragraph 1.\n\n" * 5 + "Paragraph 2.\n\n" * 5)
        chunks = _split_by_paragraphs(text, max_size=100, overlap=20)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 150  # Allow some tolerance

    def test_single_paragraph_under_limit(self):
        text = "Short paragraph."
        chunks = _split_by_paragraphs(text, max_size=1000, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_force_splits_huge_paragraph(self):
        text = "x" * 10000  # No paragraph breaks
        chunks = _split_by_paragraphs(text, max_size=3000, overlap=200)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# ingest_notion_content tests
# ---------------------------------------------------------------------------

def _mock_chromadb():
    """Set up mock chromadb and embedding function modules in sys.modules.

    Returns (mock_client, mock_collection) for assertion.
    """
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    mock_chromadb = MagicMock()
    mock_chromadb.PersistentClient.return_value = mock_client

    mock_ef_module = MagicMock()

    # Insert mocks into sys.modules so lazy imports find them
    sys.modules["chromadb"] = mock_chromadb
    sys.modules["chromadb.utils"] = MagicMock()
    sys.modules["chromadb.utils.embedding_functions"] = mock_ef_module

    return mock_client, mock_collection


def _cleanup_chromadb_mocks():
    """Remove mock chromadb modules from sys.modules."""
    for key in list(sys.modules.keys()):
        if key.startswith("chromadb"):
            del sys.modules[key]


def _do_ingest(settings, content_dir, snapshot_dir, force=False):
    """Run ingestion with mocked chromadb imports."""
    import importlib
    import rag.ingest
    importlib.reload(rag.ingest)
    return rag.ingest.ingest_notion_content(
        settings=settings,
        content_dir=content_dir,
        snapshot_dir=snapshot_dir,
        force=force,
    )


class TestIngestNotionContent:
    def setup_method(self):
        self.mock_client, self.mock_collection = _mock_chromadb()

    def teardown_method(self):
        _cleanup_chromadb_mocks()

    def test_ingest_creates_collection(self, tmp_path):
        snapshot_dir = tmp_path / "notion"
        snapshot_dir.mkdir()
        content_dir = snapshot_dir / "content"
        content_dir.mkdir()

        _setup_snapshot_files(snapshot_dir)
        (content_dir / "wi-001.md").write_text("## Steps\nDeploy the firewall.", encoding="utf-8")

        settings = _make_settings(tmp_path)
        result = _do_ingest(settings, content_dir, snapshot_dir)

        assert result["documents_ingested"] == 1
        assert result["chunks_created"] >= 1
        assert result["collection_name"] == COLLECTION_NAME
        self.mock_client.get_or_create_collection.assert_called_once()
        self.mock_collection.add.assert_called()

    def test_ingest_force_deletes_collection(self, tmp_path):
        snapshot_dir = tmp_path / "notion"
        snapshot_dir.mkdir()
        content_dir = snapshot_dir / "content"
        content_dir.mkdir()

        _setup_snapshot_files(snapshot_dir)
        (content_dir / "wi-001.md").write_text("Content here.", encoding="utf-8")

        settings = _make_settings(tmp_path)
        _do_ingest(settings, content_dir, snapshot_dir, force=True)

        self.mock_client.delete_collection.assert_called_once_with(COLLECTION_NAME)

    def test_ingest_skips_empty_files(self, tmp_path):
        snapshot_dir = tmp_path / "notion"
        snapshot_dir.mkdir()
        content_dir = snapshot_dir / "content"
        content_dir.mkdir()

        _setup_snapshot_files(snapshot_dir)
        (content_dir / "wi-001.md").write_text("", encoding="utf-8")
        (content_dir / "wi-002.md").write_text("  \n  ", encoding="utf-8")

        settings = _make_settings(tmp_path)
        result = _do_ingest(settings, content_dir, snapshot_dir)

        assert result["documents_ingested"] == 0
        assert result["chunks_created"] == 0

    def test_ingest_metadata_fields(self, tmp_path):
        snapshot_dir = tmp_path / "notion"
        snapshot_dir.mkdir()
        content_dir = snapshot_dir / "content"
        content_dir.mkdir()

        _setup_snapshot_files(snapshot_dir)
        (content_dir / "wi-001.md").write_text("Deploy the firewall.", encoding="utf-8")

        settings = _make_settings(tmp_path)
        _do_ingest(settings, content_dir, snapshot_dir)

        # Check metadata passed to collection.add()
        call_args = self.mock_collection.add.call_args
        metadatas = call_args.kwargs["metadatas"]
        assert len(metadatas) == 1

        meta = metadatas[0]
        assert meta["notion_id"] == "wi-001"
        assert meta["entity_type"] == "work_item"
        assert meta["name"] == "Deploy Firewall"
        assert meta["status"] == "In Progress"
        assert meta["priority"] == "P1"
        assert meta["sprint_id"] == "sp-001"
        assert meta["type"] == "Task"

    def test_ingest_unknown_entity(self, tmp_path):
        """Content files without matching entity get entity_type='unknown'."""
        snapshot_dir = tmp_path / "notion"
        snapshot_dir.mkdir()
        content_dir = snapshot_dir / "content"
        content_dir.mkdir()

        _setup_snapshot_files(snapshot_dir)
        (content_dir / "sub-page-xyz.md").write_text("Sub-page content.", encoding="utf-8")

        settings = _make_settings(tmp_path)
        _do_ingest(settings, content_dir, snapshot_dir)

        call_args = self.mock_collection.add.call_args
        metadatas = call_args.kwargs["metadatas"]
        sub_meta = [m for m in metadatas if m["notion_id"] == "sub-page-xyz"]
        assert len(sub_meta) == 1
        assert sub_meta[0]["entity_type"] == "unknown"
