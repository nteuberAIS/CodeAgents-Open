"""Tests for rag/retriever.py.

All tests mock ChromaDB and Ollama — no real services required.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_settings(tmp_path, **overrides):
    """Create a minimal Settings-like object for retriever tests."""
    defaults = {
        "chroma_db_path": tmp_path / "chroma",
        "embedding_model": "nomic-embed-text",
        "ollama_base_url": "http://localhost:11434",
        "rag_top_k": 5,
        "rag_score_threshold": None,
    }
    defaults.update(overrides)
    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


def _mock_chromadb(collection=None, collection_exists=True):
    """Set up mock chromadb modules in sys.modules.

    Args:
        collection: Mock collection to return. If None, creates one.
        collection_exists: If False, get_collection raises ValueError.

    Returns:
        (mock_client, mock_collection)
    """
    mock_collection = collection or MagicMock()
    mock_client = MagicMock()

    if collection_exists:
        mock_client.get_collection.return_value = mock_collection
    else:
        mock_client.get_collection.side_effect = ValueError("Collection not found")

    mock_chromadb_mod = MagicMock()
    mock_chromadb_mod.PersistentClient.return_value = mock_client

    mock_ef_module = MagicMock()

    sys.modules["chromadb"] = mock_chromadb_mod
    sys.modules["chromadb.utils"] = MagicMock()
    sys.modules["chromadb.utils.embedding_functions"] = mock_ef_module

    return mock_client, mock_collection


def _cleanup_chromadb_mocks():
    """Remove mock chromadb modules from sys.modules."""
    for key in list(sys.modules.keys()):
        if key.startswith("chromadb"):
            del sys.modules[key]


def _make_query_result(documents, metadatas, distances):
    """Build a ChromaDB-style query result (nested lists)."""
    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }


def _reload_retriever():
    """Reload the retriever module to pick up mocked imports."""
    import importlib
    import rag.retriever
    importlib.reload(rag.retriever)
    return rag.retriever.RAGRetriever


# ---------------------------------------------------------------------------
# RAGRetriever tests
# ---------------------------------------------------------------------------

class TestRAGRetrieverQuery:
    def setup_method(self):
        self.mock_client, self.mock_collection = _mock_chromadb()

    def teardown_method(self):
        _cleanup_chromadb_mocks()

    def test_basic_query(self, tmp_path):
        self.mock_collection.query.return_value = _make_query_result(
            documents=["Deploy the firewall.", "Configure NSG rules."],
            metadatas=[
                {"notion_id": "wi-001", "entity_type": "work_item", "name": "Deploy Firewall", "status": "Ready"},
                {"notion_id": "wi-002", "entity_type": "work_item", "name": "Configure NSG", "status": "In Progress"},
            ],
            distances=[0.2, 0.4],
        )

        RAGRetriever = _reload_retriever()
        retriever = RAGRetriever(_make_settings(tmp_path))
        results = retriever.query("firewall rules", top_k=2)

        assert len(results) == 2
        assert results[0]["content"] == "Deploy the firewall."
        assert results[0]["score"] == pytest.approx(0.8)
        assert results[0]["notion_id"] == "wi-001"
        assert results[0]["entity_type"] == "work_item"
        assert results[0]["name"] == "Deploy Firewall"
        assert results[1]["score"] == pytest.approx(0.6)

    def test_uses_settings_defaults(self, tmp_path):
        self.mock_collection.query.return_value = _make_query_result([], [], [])

        RAGRetriever = _reload_retriever()
        retriever = RAGRetriever(_make_settings(tmp_path, rag_top_k=10))
        retriever.query("test")

        call_kwargs = self.mock_collection.query.call_args.kwargs
        assert call_kwargs["n_results"] == 10

    def test_per_call_top_k_overrides_default(self, tmp_path):
        self.mock_collection.query.return_value = _make_query_result([], [], [])

        RAGRetriever = _reload_retriever()
        retriever = RAGRetriever(_make_settings(tmp_path, rag_top_k=10))
        retriever.query("test", top_k=3)

        call_kwargs = self.mock_collection.query.call_args.kwargs
        assert call_kwargs["n_results"] == 3

    def test_filter_entity_types(self, tmp_path):
        self.mock_collection.query.return_value = _make_query_result([], [], [])

        RAGRetriever = _reload_retriever()
        retriever = RAGRetriever(_make_settings(tmp_path))
        retriever.query("test", entity_types=["work_item", "doc"])

        call_kwargs = self.mock_collection.query.call_args.kwargs
        assert call_kwargs["where"] == {"entity_type": {"$in": ["work_item", "doc"]}}

    def test_filter_single_entity_type(self, tmp_path):
        self.mock_collection.query.return_value = _make_query_result([], [], [])

        RAGRetriever = _reload_retriever()
        retriever = RAGRetriever(_make_settings(tmp_path))
        retriever.query("test", entity_types=["work_item"])

        call_kwargs = self.mock_collection.query.call_args.kwargs
        assert call_kwargs["where"] == {"entity_type": "work_item"}

    def test_filter_status(self, tmp_path):
        self.mock_collection.query.return_value = _make_query_result([], [], [])

        RAGRetriever = _reload_retriever()
        retriever = RAGRetriever(_make_settings(tmp_path))
        retriever.query("test", status=["Ready", "In Progress"])

        call_kwargs = self.mock_collection.query.call_args.kwargs
        assert call_kwargs["where"] == {"status": {"$in": ["Ready", "In Progress"]}}

    def test_filter_sprint_id(self, tmp_path):
        self.mock_collection.query.return_value = _make_query_result([], [], [])

        RAGRetriever = _reload_retriever()
        retriever = RAGRetriever(_make_settings(tmp_path))
        retriever.query("test", sprint_id="sp-001")

        call_kwargs = self.mock_collection.query.call_args.kwargs
        assert call_kwargs["where"] == {"sprint_id": "sp-001"}

    def test_combined_filters(self, tmp_path):
        self.mock_collection.query.return_value = _make_query_result([], [], [])

        RAGRetriever = _reload_retriever()
        retriever = RAGRetriever(_make_settings(tmp_path))
        retriever.query("test", entity_types=["work_item"], status=["Ready"], sprint_id="sp-001")

        call_kwargs = self.mock_collection.query.call_args.kwargs
        where = call_kwargs["where"]
        assert "$and" in where
        assert len(where["$and"]) == 3
        assert {"entity_type": "work_item"} in where["$and"]
        assert {"status": "Ready"} in where["$and"]
        assert {"sprint_id": "sp-001"} in where["$and"]

    def test_filter_single_notion_id(self, tmp_path):
        self.mock_collection.query.return_value = _make_query_result([], [], [])

        RAGRetriever = _reload_retriever()
        retriever = RAGRetriever(_make_settings(tmp_path))
        retriever.query("test", notion_ids=["wi-001"])

        call_kwargs = self.mock_collection.query.call_args.kwargs
        assert call_kwargs["where"] == {"notion_id": "wi-001"}

    def test_filter_multiple_notion_ids(self, tmp_path):
        self.mock_collection.query.return_value = _make_query_result([], [], [])

        RAGRetriever = _reload_retriever()
        retriever = RAGRetriever(_make_settings(tmp_path))
        retriever.query("test", notion_ids=["wi-001", "wi-002", "doc-001"])

        call_kwargs = self.mock_collection.query.call_args.kwargs
        assert call_kwargs["where"] == {"notion_id": {"$in": ["wi-001", "wi-002", "doc-001"]}}

    def test_notion_ids_combined_with_entity_types(self, tmp_path):
        self.mock_collection.query.return_value = _make_query_result([], [], [])

        RAGRetriever = _reload_retriever()
        retriever = RAGRetriever(_make_settings(tmp_path))
        retriever.query("test", entity_types=["doc"], notion_ids=["doc-001", "doc-002"])

        call_kwargs = self.mock_collection.query.call_args.kwargs
        where = call_kwargs["where"]
        assert "$and" in where
        assert {"entity_type": "doc"} in where["$and"]
        assert {"notion_id": {"$in": ["doc-001", "doc-002"]}} in where["$and"]

    def test_no_filter_omits_where(self, tmp_path):
        self.mock_collection.query.return_value = _make_query_result([], [], [])

        RAGRetriever = _reload_retriever()
        retriever = RAGRetriever(_make_settings(tmp_path))
        retriever.query("test")

        call_kwargs = self.mock_collection.query.call_args.kwargs
        assert "where" not in call_kwargs

    def test_score_threshold_filters_results(self, tmp_path):
        self.mock_collection.query.return_value = _make_query_result(
            documents=["Good match", "Bad match"],
            metadatas=[
                {"notion_id": "a", "entity_type": "doc", "name": "Good", "status": "Ready"},
                {"notion_id": "b", "entity_type": "doc", "name": "Bad", "status": "Ready"},
            ],
            distances=[0.1, 0.8],  # scores: 0.9, 0.2
        )

        RAGRetriever = _reload_retriever()
        retriever = RAGRetriever(_make_settings(tmp_path))
        results = retriever.query("test", score_threshold=0.5)

        assert len(results) == 1
        assert results[0]["name"] == "Good"

    def test_score_threshold_from_settings(self, tmp_path):
        self.mock_collection.query.return_value = _make_query_result(
            documents=["Good match", "Bad match"],
            metadatas=[
                {"notion_id": "a", "entity_type": "doc", "name": "Good", "status": "Ready"},
                {"notion_id": "b", "entity_type": "doc", "name": "Bad", "status": "Ready"},
            ],
            distances=[0.1, 0.8],
        )

        RAGRetriever = _reload_retriever()
        retriever = RAGRetriever(_make_settings(tmp_path, rag_score_threshold=0.5))
        results = retriever.query("test")

        assert len(results) == 1

    def test_empty_results(self, tmp_path):
        self.mock_collection.query.return_value = _make_query_result([], [], [])

        RAGRetriever = _reload_retriever()
        retriever = RAGRetriever(_make_settings(tmp_path))
        results = retriever.query("nonexistent topic")

        assert results == []


class TestRAGRetrieverNoCollection:
    def setup_method(self):
        self.mock_client, _ = _mock_chromadb(collection_exists=False)

    def teardown_method(self):
        _cleanup_chromadb_mocks()

    def test_no_collection_returns_empty(self, tmp_path):
        RAGRetriever = _reload_retriever()
        retriever = RAGRetriever(_make_settings(tmp_path))
        results = retriever.query("anything")

        assert results == []


# ---------------------------------------------------------------------------
# format_results tests
# ---------------------------------------------------------------------------

class TestFormatResults:
    def setup_method(self):
        self.mock_client, self.mock_collection = _mock_chromadb()

    def teardown_method(self):
        _cleanup_chromadb_mocks()

    def _make_retriever(self, tmp_path):
        RAGRetriever = _reload_retriever()
        return RAGRetriever(_make_settings(tmp_path))

    def test_basic_format(self, tmp_path):
        retriever = self._make_retriever(tmp_path)
        results = [
            {"content": "Deploy config.", "score": 0.9, "entity_type": "work_item", "name": "Deploy Firewall", "status": "Ready"},
        ]
        formatted = retriever.format_results(results)

        assert "[work_item] Deploy Firewall \u2014 Ready" in formatted
        assert "Deploy config." in formatted

    def test_empty_results(self, tmp_path):
        retriever = self._make_retriever(tmp_path)
        assert retriever.format_results([]) == ""

    def test_truncation_at_max_chars(self, tmp_path):
        retriever = self._make_retriever(tmp_path)
        results = [
            {"content": "A" * 500, "score": 0.9, "entity_type": "doc", "name": "Doc1", "status": "Final"},
            {"content": "B" * 500, "score": 0.8, "entity_type": "doc", "name": "Doc2", "status": "Final"},
            {"content": "C" * 500, "score": 0.7, "entity_type": "doc", "name": "Doc3", "status": "Final"},
        ]
        formatted = retriever.format_results(results, max_chars=600)

        assert len(formatted) <= 600
        # First result should be present
        assert "Doc1" in formatted

    def test_no_status_omits_dash(self, tmp_path):
        retriever = self._make_retriever(tmp_path)
        results = [
            {"content": "Content here.", "score": 0.9, "entity_type": "doc", "name": "MyDoc", "status": ""},
        ]
        formatted = retriever.format_results(results)

        assert "[doc] MyDoc" in formatted
        assert "\u2014" not in formatted

    def test_multiple_results_separated(self, tmp_path):
        retriever = self._make_retriever(tmp_path)
        results = [
            {"content": "First.", "score": 0.9, "entity_type": "work_item", "name": "Task1", "status": "Ready"},
            {"content": "Second.", "score": 0.8, "entity_type": "doc", "name": "Doc1", "status": "Final"},
        ]
        formatted = retriever.format_results(results)

        assert "Task1" in formatted
        assert "Doc1" in formatted
        assert "\n\n" in formatted  # separated by blank line


# ---------------------------------------------------------------------------
# BaseAgent.retrieve() tests
# ---------------------------------------------------------------------------

class TestBaseAgentRetrieve:
    def test_retrieve_without_rag_returns_empty(self):
        from agents.base import BaseAgent

        class DummyAgent(BaseAgent):
            name = "dummy"
            def run(self, user_input):
                return {}

        agent = DummyAgent(llm=MagicMock(), context=None)
        assert agent.retrieve("test query") == []
        assert agent.rag is None

    def test_retrieve_delegates_to_rag(self):
        from agents.base import BaseAgent

        class DummyAgent(BaseAgent):
            name = "dummy"
            def run(self, user_input):
                return {}

        mock_rag = MagicMock()
        mock_rag.query.return_value = [{"content": "result", "score": 0.9}]

        agent = DummyAgent(llm=MagicMock(), context=None, rag=mock_rag)
        results = agent.retrieve("test query", top_k=3)

        assert len(results) == 1
        mock_rag.query.assert_called_once_with("test query", top_k=3)

    def test_rag_keyword_only(self):
        """Verify rag cannot be passed as a positional argument."""
        from agents.base import BaseAgent

        class DummyAgent(BaseAgent):
            name = "dummy"
            def run(self, user_input):
                return {}

        with pytest.raises(TypeError):
            DummyAgent(MagicMock(), None, MagicMock())  # rag as positional should fail
