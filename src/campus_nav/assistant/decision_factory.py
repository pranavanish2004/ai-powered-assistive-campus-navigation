"""DecisionEngine factory -- wires the RAG and MLLM backends selected by
Settings into a fully-configured DecisionEngine. Dashboard/navigation/
vision data is intentionally NOT wired here: those are per-request,
already-live app.state objects (see api/dependencies.py) gathered fresh
for each call, not something to bake into a process-wide singleton."""

from __future__ import annotations

from campus_nav.assistant.decision_engine import DecisionEngine
from campus_nav.config.settings import Settings
from campus_nav.infrastructure.mllm.mllm_factory import create_mllm_provider
from campus_nav.infrastructure.rag.rag_factory import create_campus_knowledge_base
from campus_nav.navigation.graph.models import CampusGraph


def create_decision_engine(settings: Settings, graph: CampusGraph) -> DecisionEngine:
    knowledge_base = create_campus_knowledge_base(settings, graph)
    mllm_provider = create_mllm_provider(settings)
    return DecisionEngine(
        rag_repository=knowledge_base,
        mllm_provider=mllm_provider,
        rag_top_k=settings.rag_top_k,
        mllm_max_tokens=settings.mllm_max_tokens,
    )
