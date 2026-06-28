"""Pydantic request/response models for the FastAPI service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    model_loaded: bool
    rag_chunks: int


class GenerateRequest(BaseModel):
    prompt: str = Field(..., examples=["Once upon a time"])
    max_new_tokens: int = Field(128, ge=1, le=2048)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int | None = Field(50, ge=0)
    top_p: float | None = Field(0.95, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    model: str
    latency_ms: float


class EvalRequest(BaseModel):
    config: str = "configs/eval_default.yaml"


class RagIngestRequest(BaseModel):
    source: str | None = None


class RagQueryRequest(BaseModel):
    question: str
    top_k: int = Field(5, ge=1, le=20)


class AgentRequest(BaseModel):
    task: str


class GenericResponse(BaseModel):
    result: dict[str, Any]
