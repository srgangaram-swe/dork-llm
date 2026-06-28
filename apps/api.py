"""FastAPI service exposing the Dork LLM platform.

Endpoints:
    GET  /health         — liveness + model/RAG status
    POST /generate       — text generation from the tiny GPT (mock fallback)
    POST /evaluate       — run the evaluation harness
    POST /rag/ingest     — ingest documents into the vector store
    POST /rag/query      — grounded, cited RAG answer
    POST /agent/run      — run the agentic research assistant
    GET  /metrics        — in-memory request/latency metrics

Run with: ``uvicorn apps.api:app --reload`` (or ``make api``).
"""

from __future__ import annotations

from dork import __version__
from dork.serving.schemas import (
    AgentRequest,
    EvalRequest,
    GenerateRequest,
    GenerateResponse,
    GenericResponse,
    HealthResponse,
    RagIngestRequest,
    RagQueryRequest,
)
from dork.serving.service import DorkService
from fastapi import FastAPI

app = FastAPI(
    title="Dork LLM",
    version=__version__,
    description="Train, evaluate, retrieve, and serve a compact LLM systems platform.",
)
service = DorkService()


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(**service.health())


@app.get("/metrics", tags=["system"])
def metrics() -> dict:
    return service.metrics.snapshot()


@app.post("/generate", response_model=GenerateResponse, tags=["model"])
def generate(req: GenerateRequest) -> GenerateResponse:
    out = service.generate(
        req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    )
    return GenerateResponse(**out)


@app.post("/evaluate", response_model=GenericResponse, tags=["evaluation"])
def evaluate(req: EvalRequest) -> GenericResponse:
    return GenericResponse(result=service.evaluate(req.config))


@app.post("/rag/ingest", response_model=GenericResponse, tags=["rag"])
def rag_ingest(req: RagIngestRequest) -> GenericResponse:
    return GenericResponse(result=service.rag_ingest(req.source))


@app.post("/rag/query", response_model=GenericResponse, tags=["rag"])
def rag_query(req: RagQueryRequest) -> GenericResponse:
    return GenericResponse(result=service.rag_query(req.question, req.top_k))


@app.post("/agent/run", response_model=GenericResponse, tags=["agents"])
def agent_run(req: AgentRequest) -> GenericResponse:
    return GenericResponse(result=service.run_agent(req.task))
