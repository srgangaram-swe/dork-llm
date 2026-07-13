"""API contract tests with injected deterministic providers."""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("fastapi")
from apps.api import create_app
from dork.generation.providers import LanguageModel
from dork.rag.pipeline import RagPipeline
from dork.serving.service import DorkService
from dork.serving.settings import ServingSettings
from dork.utils.config import RagConfig
from fastapi.testclient import TestClient


class FakeModel(LanguageModel):
    provider = "local_gpt"
    name = "fake-local-gpt"
    artifact = "artifacts/fake"
    device = "cpu"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.prompts: list[str] = []
        self.stops: list[list[str] | None] = []

    def complete(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
    ) -> str:
        self.prompts.append(prompt)
        self.stops.append(stop)
        if self.fail:
            raise RuntimeError("sensitive implementation detail")
        if "Context:" in prompt:
            return "Grounded evidence from the retrieved document. [1]"
        return "A deterministic assistant answer."


def _service(model: FakeModel | None = None, **settings_overrides: Any) -> DorkService:
    settings = ServingSettings(
        provider="local_gpt",
        device="cpu",
        max_new_tokens=512,
        **settings_overrides,
    )
    return DorkService(settings=settings, language_model=model or FakeModel())


def _rag_service() -> DorkService:
    model = FakeModel()
    cfg = RagConfig(
        ingest={
            "source_dir": "data/sample_docs",
            "chunking": {"chunk_size": 80, "chunk_overlap": 16, "min_chunk_chars": 20},
        },
        embeddings={"backend": "hash", "dim": 128},
        vector_store={"backend": "memory"},
        retrieval={"top_k": 3, "rerank": True, "rerank_top_n": 2, "min_score": 0.0},
        generation={"refuse_when_insufficient": True},
    )
    rag = RagPipeline(cfg, model=model)
    rag.ingest()
    return DorkService(
        settings=ServingSettings(provider="local_gpt", device="cpu"),
        language_model=model,
        rag_pipeline=rag,
    )


def _client(service: DorkService, *, mount_web: bool = False) -> TestClient:
    return TestClient(create_app(service=service, mount_web=mount_web))


def _parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    parsed: list[tuple[str, dict[str, Any]]] = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        event = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        data = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        parsed.append((event, json.loads(data)))
    return parsed


def test_health_readiness_model_info_and_security_headers() -> None:
    with _client(_service()) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        assert body["requested_provider"] == "local_gpt"
        assert body["active_provider"] == "local_gpt"
        assert body["model"]["artifact"] == "artifacts/fake"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["content-security-policy"]
        assert response.headers["x-request-id"]

        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/api/v1/model").json()["name"] == "fake-local-gpt"


def test_generate_and_legacy_chat_endpoints() -> None:
    model = FakeModel()
    with _client(_service(model)) as client:
        generated = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 8})
        assert generated.status_code == 200
        assert generated.json()["completion"] == "A deterministic assistant answer."

        chatted = client.post(
            "/chat",
            json={
                "message": "latest",
                "mode": "generate",
                "history": [
                    {"role": "assistant", "content": "prior response"},
                    {"role": "user", "content": "latest"},
                ],
                "max_new_tokens": 8,
            },
        )
        assert chatted.status_code == 200
        assert chatted.json()["answer"] == "A deterministic assistant answer."
        assert chatted.json()["active_provider"] == "local_gpt"
        assert model.prompts[-1].count("latest") == 1
        assert model.prompts[-1].startswith("### Instruction:\n")
        assert model.stops[-1] == ["### Instruction:", "<|endoftext|>"]


def test_canonical_messages_contract_and_validation() -> None:
    with _client(_service()) as client:
        response = client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "reply"},
                    {"role": "user", "content": "second"},
                ],
                "mode": "generate",
                "max_new_tokens": 8,
            },
        )
        assert response.status_code == 200

        ambiguous = client.post(
            "/chat",
            json={
                "message": "legacy",
                "messages": [{"role": "user", "content": "canonical"}],
            },
        )
        assert ambiguous.status_code == 422

        non_user_final = client.post(
            "/chat",
            json={"messages": [{"role": "assistant", "content": "not a request"}]},
        )
        assert non_user_final.status_code == 422


def test_service_input_caps_return_safe_error() -> None:
    service = _service(max_message_chars=5, max_total_message_chars=8)
    with _client(service) as client:
        response = client.post(
            "/chat",
            json={"message": "sixsix", "mode": "generate", "max_new_tokens": 8},
        )
        assert response.status_code == 422
        body = response.json()["error"]
        assert body["code"] == "invalid_request"
        assert body["message"] == "The chat request is invalid."
        assert body["request_id"] == response.headers["x-request-id"]


def test_stream_contract_is_named_json_and_truthful_about_chunking() -> None:
    request_id = "contract-test-123"
    with _client(_rag_service()) as client:
        response = client.post(
            "/api/v1/chat/stream",
            headers={"X-Request-ID": request_id},
            json={
                "messages": [{"role": "user", "content": "What is causal masking?"}],
                "mode": "rag",
                "max_new_tokens": 8,
            },
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-request-id"] == request_id
    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "meta"
    assert "delta" in names
    assert "citation" in names
    assert names[-1] == "done"
    assert "error" not in names
    assert {data["request_id"] for _, data in events} == {request_id}

    meta = events[0][1]
    assert meta["delivery_mode"] == "chunked_completion"
    assert meta["native_token_streaming"] is False
    deltas = "".join(data["delta"] for name, data in events if name == "delta")
    done = events[-1][1]
    assert deltas == done["answer"]
    assert done["finish_reason"] == "stop"


def test_stream_failures_do_not_leak_exception_details() -> None:
    with _client(_service(FakeModel(fail=True))) as client:
        response = client.post(
            "/api/v1/chat/stream",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "mode": "generate",
                "max_new_tokens": 8,
            },
        )
    events = _parse_sse(response.text)
    assert [name for name, _ in events] == ["meta", "error"]
    assert events[-1][1]["code"] == "internal_error"
    assert "sensitive implementation detail" not in response.text


def test_strict_default_is_live_but_not_ready_and_never_silently_mocks() -> None:
    attempts: list[str | None] = []

    def unavailable(provider: str, artifact: str | None, device: str) -> LanguageModel:
        attempts.append(artifact)
        raise FileNotFoundError(artifact)

    service = DorkService(settings=ServingSettings(), model_loader=unavailable)
    with _client(service) as client:
        live = client.get("/health/live")
        assert live.status_code == 200
        assert attempts == []

        ready = client.get("/ready")
        assert ready.status_code == 503
        assert ready.json()["model"]["active_provider"] == "unavailable"
        assert ready.json()["model"]["demo_mode"] is False

        generated = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 8})
        assert generated.status_code == 503
        assert generated.json()["error"]["code"] == "model_unavailable"


def test_explicit_demo_mode_reports_degraded_mock_fallback() -> None:
    def unavailable(provider: str, artifact: str | None, device: str) -> LanguageModel:
        raise FileNotFoundError(artifact)

    service = DorkService(
        settings=ServingSettings(demo_mode=True),
        model_loader=unavailable,
    )
    with _client(service) as client:
        health = client.get("/health").json()
        assert health["requested_provider"] == "auto"
        assert health["active_provider"] == "mock"
        assert health["ready"] is False
        assert health["status"] == "degraded"
        assert "Demo-mode mock fallback" in health["degraded_reason"]


def test_rag_and_agent_endpoints_share_the_injected_model() -> None:
    service = _rag_service()
    with _client(service) as client:
        rag = client.post(
            "/rag/query",
            json={"question": "What does causal masking prevent?", "top_k": 3},
        )
        assert rag.status_code == 200
        assert rag.json()["result"]["model"] == "fake-local-gpt"

        agent = client.post("/agent/run", json={"task": "Calculate 6 * 7"})
        assert agent.status_code == 200
        assert "42" in agent.json()["result"]["answer"]


def test_web_root_is_still_mounted() -> None:
    with _client(_service(), mount_web=True) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert '<script src="/app.js" type="module"></script>' in response.text
