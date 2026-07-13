"""Pydantic request and response contracts for the serving boundary."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ModelInfoResponse(BaseModel):
    requested_provider: str
    active_provider: str
    name: str | None = None
    artifact: str | None = None
    artifact_candidate: str | None = None
    device: str | None = None
    demo_mode: bool
    ready: bool
    model_loaded: bool
    degraded: bool
    degraded_reason: str | None = None
    attempted_artifacts: list[str] = Field(default_factory=list)


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"
    live: bool = True
    version: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded"]
    ready: bool
    version: str
    model: ModelInfoResponse


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    live: bool = True
    ready: bool
    version: str
    model_loaded: bool
    rag_chunks: int
    requested_provider: str
    active_provider: str
    degraded_reason: str | None = None
    model: ModelInfoResponse


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=100_000, examples=["Once upon a time"])
    max_new_tokens: int = Field(128, ge=1, le=4_096)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int | None = Field(50, ge=0)
    top_p: float | None = Field(0.95, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    model: str
    latency_ms: float


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str = Field(..., min_length=1, max_length=100_000)


class ChatRequest(BaseModel):
    """Canonical ``messages`` contract plus the backwards-compatible legacy shape."""

    message: str | None = Field(
        None,
        min_length=1,
        max_length=100_000,
        examples=["What makes dorkLLM different?"],
    )
    history: list[ChatMessage] = Field(default_factory=list, max_length=128)
    messages: list[ChatMessage] = Field(default_factory=list, max_length=128)
    mode: Literal["auto", "rag", "generate"] = "auto"
    retrieval_top_k: int = Field(5, ge=1, le=20)
    max_new_tokens: int = Field(256, ge=1, le=4_096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    sampling_top_k: int | None = Field(50, ge=0)
    top_p: float | None = Field(0.95, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_contract(self) -> ChatRequest:
        if self.messages:
            if self.message is not None or self.history:
                raise ValueError("Use canonical messages or legacy message/history, not both.")
            if self.messages[-1].role != "user":
                raise ValueError("The final canonical message must have role=user.")
        elif self.message is None:
            raise ValueError("Provide either message or canonical messages.")
        return self


class ChatResponse(BaseModel):
    answer: str
    mode: Literal["rag", "generate"]
    citations: list[dict[str, Any]] = Field(default_factory=list)
    model: str
    latency_ms: float
    requested_provider: str
    active_provider: str
    artifact: str | None = None
    device: str | None = None
    degraded: bool
    degraded_reason: str | None = None


class EvalRequest(BaseModel):
    config: str = "configs/eval_default.yaml"


class RagIngestRequest(BaseModel):
    source: str | None = None


class RagQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=100_000)
    top_k: int = Field(5, ge=1, le=20)


class AgentRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=100_000)


class GenericResponse(BaseModel):
    result: dict[str, Any]
