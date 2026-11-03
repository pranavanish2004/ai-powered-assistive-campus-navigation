"""Tests for merging campus graph landmark descriptions with the static
supplementary knowledge file into a populated knowledge base."""

import json
import tempfile
from pathlib import Path

import pytest

from campus_nav.infrastructure.rag.campus_knowledge_loader import (
    build_campus_knowledge_base,
    documents_from_graph,
    documents_from_static_file,
)
from campus_nav.infrastructure.rag.keyword_knowledge_base import KeywordKnowledgeBase
from campus_nav.interfaces.rag_interface import KnowledgeCategory

pytestmark = pytest.mark.unit


def test_documents_from_graph_only_includes_nodes_with_landmark_description(simple_graph):
    # simple_graph fixture nodes have no landmark_description set.
    documents = documents_from_graph(simple_graph)
    assert documents == []


def test_documents_from_graph_extracts_pilot_graph_landmarks(pilot_graph):
    documents = documents_from_graph(pilot_graph)
    assert len(documents) == pilot_graph.node_count()  # all pilot nodes have descriptions
    library_docs = [d for d in documents if d.metadata.get("node_id") == "library"]
    assert len(library_docs) == 1
    assert "glass-fronted" in library_docs[0].content.lower()


def test_documents_from_graph_sets_metadata_and_category(pilot_graph):
    documents = documents_from_graph(pilot_graph)
    doc = next(d for d in documents if d.metadata.get("node_id") == "library")
    assert doc.category == KnowledgeCategory.BUILDING
    assert doc.metadata["is_accessible"] is True


def test_documents_from_static_file_loads_valid_entries():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "knowledge.json"
        path.write_text(
            json.dumps(
                [{"id": "e1", "title": "Test", "content": "Test content.", "category": "emergency"}]
            )
        )
        documents = documents_from_static_file(path)
        assert len(documents) == 1
        assert documents[0].category == KnowledgeCategory.EMERGENCY


def test_documents_from_static_file_missing_file_returns_empty_list():
    documents = documents_from_static_file("/nonexistent/path/knowledge.json")
    assert documents == []


def test_documents_from_static_file_malformed_json_returns_empty_list():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "bad.json"
        path.write_text("{not valid json")
        assert documents_from_static_file(path) == []


def test_documents_from_static_file_skips_malformed_entries():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "knowledge.json"
        path.write_text(
            json.dumps(
                [
                    {"id": "good", "title": "Good", "content": "Fine."},
                    {"id": "bad_missing_title"},  # missing required "title"/"content"
                    {
                        "id": "bad_category",
                        "title": "X",
                        "content": "Y",
                        "category": "not-a-real-category",
                    },
                ]
            )
        )
        documents = documents_from_static_file(path)
        assert len(documents) == 1
        assert documents[0].id == "good"


def test_build_campus_knowledge_base_merges_both_sources(pilot_graph):
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "knowledge.json"
        path.write_text(
            json.dumps([{"id": "static1", "title": "Static Fact", "content": "A fact."}])
        )

        kb = build_campus_knowledge_base(KeywordKnowledgeBase(), pilot_graph, path)

        assert kb.document_count() == pilot_graph.node_count() + 1


def test_build_campus_knowledge_base_with_missing_static_file_still_loads_graph(pilot_graph):
    kb = build_campus_knowledge_base(KeywordKnowledgeBase(), pilot_graph, "/nonexistent.json")
    assert kb.document_count() == pilot_graph.node_count()


def test_real_campus_knowledge_fixture_loads_and_is_searchable(pilot_graph):
    kb = build_campus_knowledge_base(
        KeywordKnowledgeBase(), pilot_graph, "tests/fixtures/campus_knowledge.json"
    )
    results = kb.search("emergency")
    assert any(r.document.id == "emergency_contacts" for r in results)
