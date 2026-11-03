"""
DecisionEngine -- the Phase 1.4d fusion orchestrator.

Design rationale
-----------------
Mirrors the role `NavigationEngine` (Phase 1.2) and `VisionPipeline`
(Phase 1.3) play for their respective domains: this class contains no
retrieval math, no prompt-formatting logic, and no provider-specific code
of its own -- it only sequences calls to `RAGRepositoryInterface`,
`prompt_builder`, and `MLLMProviderInterface`, which is what keeps each of
those independently unit-testable.

Two hard boundaries, enforced here in code (not just documented):

1. **This class only ever produces CONVERSATIONAL-lane speech.** It has no
   access to `NavigationStateMachine.transition()`, `SafetyEngine`, or
   `VoiceGuidance.announce_safety_alert()` -- it cannot trigger a safety
   announcement or a state change even if asked to, because it was never
   given those objects. Obstacle warnings and route-adherence alerts
   continue to flow through `SafetyEngine` exactly as in Phase 1.2/1.3,
   completely independently of whether the Decision Engine or any MLLM
   provider is healthy.
2. **MLLM failure degrades to an honest, static fallback, never a crash
   or a fabricated answer.** `MLLMError` is caught here and turned into a
   plain "I'm having trouble answering right now" response -- consistent
   with treating the MLLM (like the Dashboard) as an enhancement, not a
   dependency the rest of the assistant relies on to function.
"""

from __future__ import annotations

from campus_nav.assistant.context import AssistantContext, AssistantResponse
from campus_nav.assistant.prompt_builder import build_messages, is_dashboard_relevant
from campus_nav.core.exceptions import MLLMError, RAGError
from campus_nav.core.logging_config import get_logger
from campus_nav.interfaces.mllm_interface import MLLMProviderInterface
from campus_nav.interfaces.rag_interface import RAGRepositoryInterface

logger = get_logger(__name__)

_FALLBACK_RESPONSE_TEXT = (
    "I'm having trouble answering that right now, but navigation and safety "
    "alerts are unaffected. Please try asking again in a moment."
)


class DecisionEngine:
    """Fuses retrieved campus knowledge with pre-gathered navigation/
    vision/dashboard context into one natural-language response via the
    configured MLLM provider."""

    def __init__(
        self,
        rag_repository: RAGRepositoryInterface,
        mllm_provider: MLLMProviderInterface,
        rag_top_k: int = 3,
        mllm_max_tokens: int = 512,
    ) -> None:
        self._rag = rag_repository
        self._mllm = mllm_provider
        self._rag_top_k = rag_top_k
        self._mllm_max_tokens = mllm_max_tokens

    @property
    def knowledge_document_count(self) -> int:
        """Number of documents in the knowledge base -- exposed for the
        `/assistant/status` endpoint without reaching into a private
        attribute."""
        return self._rag.document_count()

    @property
    def mllm_provider(self) -> MLLMProviderInterface:
        """The active MLLM provider -- exposed read-only so callers (and
        tests) can inspect which provider is wired in without reaching
        into a private attribute."""
        return self._mllm

    @property
    def rag_repository(self) -> RAGRepositoryInterface:
        """The active knowledge base -- exposed so the ingestion pipeline
        (Phase 1.4 refinement) can add standalone knowledge entries
        (facts with no associated graph node) after construction, without
        reaching into a private attribute."""
        return self._rag

    def answer(self, query: str, context: AssistantContext) -> AssistantResponse:
        """Produce a grounded, natural-language answer to `query`.

        Never raises: RAG and MLLM failures are both recoverable per the
        core exception hierarchy, and this method converts either into a
        degraded-but-honest response rather than propagating an exception
        into the API layer.
        """
        retrieval_results = []
        try:
            retrieval_results = self._rag.search(query, top_k=self._rag_top_k)
        except RAGError as exc:
            logger.warning(
                "RAG search failed; continuing without retrieved context", extra={"error": str(exc)}
            )

        # The dashboard-relevance check only gates *whether the caller
        # should have fetched dashboard data before calling this method*
        # (see api/routers/assistant.py) -- by the time `context` reaches
        # here, context.dashboard is either already populated or None.
        # This engine does not fetch dashboard data itself; it only
        # reports whether the query looked dashboard-relevant, for the
        # response's `used_dashboard` flag to be meaningful even when the
        # fetch ultimately found nothing.
        dashboard_was_relevant = is_dashboard_relevant(query)

        messages = build_messages(query, context, retrieval_results)

        try:
            mllm_response = self._mllm.generate(messages, max_tokens=self._mllm_max_tokens)
            response_text = mllm_response.text
        except MLLMError as exc:
            logger.warning(
                "MLLM generation failed; returning fallback response", extra={"error": str(exc)}
            )
            response_text = _FALLBACK_RESPONSE_TEXT

        return AssistantResponse(
            text=response_text,
            used_rag=len(retrieval_results) > 0,
            used_dashboard=dashboard_was_relevant and context.dashboard is not None,
            used_navigation_context=context.navigation is not None,
            used_vision_context=context.vision is not None,
            source_document_ids=[r.document.id for r in retrieval_results],
        )
