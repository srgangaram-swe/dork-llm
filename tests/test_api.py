"""Smoke tests for the FastAPI service (require the [serve] extra)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from apps.api import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_generate_endpoint():
    r = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 8})
    assert r.status_code == 200
    assert "completion" in r.json()


def test_chat_endpoint_generate_mode():
    r = client.post(
        "/chat",
        json={
            "message": "hello",
            "mode": "generate",
            "history": [{"role": "user", "content": "who are you?"}],
            "max_new_tokens": 16,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "generate"
    assert body["answer"]


def test_web_root_serves_chat_app():
    r = client.get("/")
    assert r.status_code == 200
    assert "dorkLLM" in r.text


def test_rag_query_endpoint():
    client.post("/rag/ingest", json={})
    r = client.post(
        "/rag/query", json={"question": "What does causal masking prevent?", "top_k": 3}
    )
    assert r.status_code == 200
    assert "result" in r.json()


def test_agent_endpoint():
    r = client.post("/agent/run", json={"task": "Calculate 6 * 7"})
    assert r.status_code == 200
    assert "42" in r.json()["result"]["answer"]


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "requests" in r.json()
