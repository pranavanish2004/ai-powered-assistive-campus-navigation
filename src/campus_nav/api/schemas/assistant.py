"""Pydantic request/response schemas for the /assistant endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

from campus_nav.assistant.context import AssistantResponse


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    navigation_session_id: str | None = Field(
        default=None,
        description="If provided and a matching active navigation session "
        "exists, its current state (destination, distance remaining, next "
        "instruction) is included as context for this query.",
    )
    include_vision_context: bool = Field(
        default=True,
        description="If True and the vision pipeline is running, current "
        "obstacles/OCR text are included as context.",
    )


class AskResponse(BaseModel):
    text: str
    used_rag: bool
    used_dashboard: bool
    used_navigation_context: bool
    used_vision_context: bool
    source_document_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_domain(cls, response: AssistantResponse) -> AskResponse:
        return cls(
            text=response.text,
            used_rag=response.used_rag,
            used_dashboard=response.used_dashboard,
            used_navigation_context=response.used_navigation_context,
            used_vision_context=response.used_vision_context,
            source_document_ids=response.source_document_ids,
        )


class AssistantStatusResponse(BaseModel):
    mllm_provider: str
    rag_backend: str
    knowledge_document_count: int
