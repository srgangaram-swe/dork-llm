"""FastAPI application factory for the AxiomStack / DorkLLM service."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any

from dork import __version__
from dork.serving.runtime import ModelUnavailableError
from dork.serving.schemas import (
    AgentRequest,
    ChatRequest,
    ChatResponse,
    EvalRequest,
    GenerateRequest,
    GenerateResponse,
    GenericResponse,
    HealthResponse,
    LivenessResponse,
    ModelInfoResponse,
    RagIngestRequest,
    RagQueryRequest,
)
from dork.serving.service import DorkService, ServingInputError
from dork.serving.settings import ServingSettings
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": (
        "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; "
        "form-action 'self'; img-src 'self' data:; connect-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net"
    ),
}


def _service(request: Request) -> DorkService:
    return request.app.state.dork_service


ServiceDependency = Annotated[DorkService, Depends(_service)]


def _chat_kwargs(payload: ChatRequest) -> dict[str, Any]:
    return {
        "message": payload.message,
        "history": [item.model_dump() for item in payload.history],
        "messages": [item.model_dump() for item in payload.messages],
        "mode": payload.mode,
        "retrieval_top_k": payload.retrieval_top_k,
        "max_new_tokens": payload.max_new_tokens,
        "temperature": payload.temperature,
        "sampling_top_k": payload.sampling_top_k,
        "top_p": payload.top_p,
    }


def _sse_event(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def _completion_chunks(text: str, size: int) -> list[str]:
    return [text[start : start + size] for start in range(0, len(text), size)] or [""]


async def _run_chat_without_blocking(
    request: Request,
    service: DorkService,
    payload: ChatRequest,
) -> dict[str, Any] | None:
    """Run synchronous local inference in a worker and notice client disconnects."""

    task = asyncio.create_task(asyncio.to_thread(service.chat, **_chat_kwargs(payload)))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + service.settings.generation_timeout_seconds
    poll_seconds = service.settings.stream_poll_interval_ms / 1_000
    while not task.done():
        if await request.is_disconnected():
            task.cancel()
            return None
        if loop.time() >= deadline:
            task.cancel()
            raise TimeoutError("generation deadline exceeded")
        await asyncio.sleep(poll_seconds)
    return task.result()


def _request_id(request: Request) -> str:
    value = request.headers.get("X-Request-ID", "")
    return value if _REQUEST_ID.fullmatch(value) else str(uuid.uuid4())


def create_app(
    *,
    service: DorkService | None = None,
    settings: ServingSettings | None = None,
    mount_web: bool = True,
) -> FastAPI:
    """Create an isolated application with an injectable service/runtime."""

    resolved_settings = settings or (service.settings if service is not None else None)
    resolved_settings = resolved_settings or ServingSettings.from_env()
    resolved_service = service or DorkService(settings=resolved_settings)

    application = FastAPI(
        title="AxiomStack / DorkLLM API",
        version=__version__,
        description="Train, measure, ground, and serve a compact language-model system.",
    )
    application.state.dork_service = resolved_service

    @application.middleware("http")
    async def response_metadata(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request.state.request_id = _request_id(request)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        for name, value in _SECURITY_HEADERS.items():
            response.headers[name] = value
        return response

    def public_error(request: Request, code: str, message: str, status_code: int) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=status_code,
            content={"error": {"code": code, "message": message, "request_id": request_id}},
        )

    @application.exception_handler(ServingInputError)
    async def handle_serving_input(request: Request, _: ServingInputError) -> JSONResponse:
        return public_error(request, ServingInputError.code, ServingInputError.public_message, 422)

    @application.exception_handler(ModelUnavailableError)
    async def handle_model_unavailable(
        request: Request,
        _: ModelUnavailableError,
    ) -> JSONResponse:
        return public_error(
            request,
            ModelUnavailableError.code,
            ModelUnavailableError.public_message,
            503,
        )

    @application.get("/health/live", response_model=LivenessResponse, tags=["system"])
    def liveness(service: ServiceDependency) -> LivenessResponse:
        return LivenessResponse(**service.liveness())

    @application.get("/ready", tags=["system"], include_in_schema=False)
    @application.get("/health/ready", tags=["system"])
    def readiness(service: ServiceDependency) -> JSONResponse:
        result = service.readiness()
        return JSONResponse(result, status_code=200 if result["ready"] else 503)

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health(service: ServiceDependency) -> HealthResponse:
        return HealthResponse(**service.health())

    @application.get(
        "/api/v1/model",
        response_model=ModelInfoResponse,
        tags=["system"],
    )
    def model_info(service: ServiceDependency) -> ModelInfoResponse:
        return ModelInfoResponse(**service.model_info())

    @application.get("/metrics", tags=["system"])
    def metrics(service: ServiceDependency) -> dict[str, Any]:
        return service.metrics.snapshot()

    @application.post("/generate", response_model=GenerateResponse, tags=["model"])
    def generate(payload: GenerateRequest, service: ServiceDependency) -> GenerateResponse:
        result = service.generate(
            payload.prompt,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_k=payload.top_k,
            top_p=payload.top_p,
        )
        return GenerateResponse(**result)

    @application.post("/chat", response_model=ChatResponse, tags=["model"])
    def legacy_chat(payload: ChatRequest, service: ServiceDependency) -> ChatResponse:
        return ChatResponse(**service.chat(**_chat_kwargs(payload)))

    @application.post("/api/v1/chat/stream", tags=["model"])
    async def stream_chat(
        payload: ChatRequest,
        request: Request,
        service: ServiceDependency,
    ) -> StreamingResponse:
        request_id = request.state.request_id

        async def events() -> AsyncIterator[str]:
            try:
                runtime = await asyncio.to_thread(service.model_info)
                if await request.is_disconnected():
                    return
                yield _sse_event(
                    "meta",
                    {
                        "request_id": request_id,
                        "delivery_mode": "chunked_completion",
                        "native_token_streaming": False,
                        "requested_provider": runtime["requested_provider"],
                        "active_provider": runtime["active_provider"],
                        "model": runtime["name"],
                        "artifact": runtime["artifact"],
                        "device": runtime["device"],
                        "degraded": runtime["degraded"],
                    },
                )
                result = await _run_chat_without_blocking(request, service, payload)
                if result is None:
                    return
                for index, chunk in enumerate(
                    _completion_chunks(str(result["answer"]), service.settings.stream_chunk_chars)
                ):
                    if await request.is_disconnected():
                        return
                    yield _sse_event(
                        "delta",
                        {"request_id": request_id, "index": index, "delta": chunk},
                    )
                    await asyncio.sleep(0)
                for citation in result.get("citations", []):
                    if await request.is_disconnected():
                        return
                    yield _sse_event(
                        "citation",
                        {"request_id": request_id, "citation": citation},
                    )
                yield _sse_event(
                    "done",
                    {
                        "request_id": request_id,
                        "finish_reason": "stop",
                        "mode": result["mode"],
                        "model": result["model"],
                        "latency_ms": result["latency_ms"],
                        "answer": result["answer"],
                    },
                )
            except ServingInputError:
                yield _sse_event(
                    "error",
                    {
                        "request_id": request_id,
                        "code": ServingInputError.code,
                        "message": ServingInputError.public_message,
                    },
                )
            except ModelUnavailableError:
                yield _sse_event(
                    "error",
                    {
                        "request_id": request_id,
                        "code": ModelUnavailableError.code,
                        "message": ModelUnavailableError.public_message,
                    },
                )
            except TimeoutError:
                yield _sse_event(
                    "error",
                    {
                        "request_id": request_id,
                        "code": "generation_timeout",
                        "message": "The generation deadline was exceeded.",
                    },
                )
            except Exception:
                logger.exception("Streaming chat failed (request_id=%s)", request_id)
                yield _sse_event(
                    "error",
                    {
                        "request_id": request_id,
                        "code": "internal_error",
                        "message": "The chat request could not be completed.",
                    },
                )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @application.post("/evaluate", response_model=GenericResponse, tags=["evaluation"])
    def evaluate(payload: EvalRequest, service: ServiceDependency) -> GenericResponse:
        return GenericResponse(result=service.evaluate(payload.config))

    @application.post("/rag/ingest", response_model=GenericResponse, tags=["rag"])
    def rag_ingest(payload: RagIngestRequest, service: ServiceDependency) -> GenericResponse:
        return GenericResponse(result=service.rag_ingest(payload.source))

    @application.post("/rag/query", response_model=GenericResponse, tags=["rag"])
    def rag_query(payload: RagQueryRequest, service: ServiceDependency) -> GenericResponse:
        return GenericResponse(result=service.rag_query(payload.question, payload.top_k))

    @application.post("/agent/run", response_model=GenericResponse, tags=["agents"])
    def agent_run(payload: AgentRequest, service: ServiceDependency) -> GenericResponse:
        return GenericResponse(result=service.run_agent(payload.task))

    if mount_web and WEB_DIR.exists():
        application.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return application


app = create_app()
service: DorkService = app.state.dork_service
