"""Typed serving configuration loaded explicitly from environment variables.

The project intentionally avoids a dependency on ``pydantic-settings``.  The
small adapter below keeps environment parsing at the process boundary while the
rest of the serving layer receives an immutable, validated settings object.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ProviderName = Literal["auto", "local_gpt", "hf", "mock"]
DeviceName = Literal["auto", "cpu", "cuda", "mps"]


class ServingSettings(BaseModel):
    """Validated runtime settings for the API and model service."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: ProviderName = "auto"
    artifact: str | None = None
    device: DeviceName = "auto"
    demo_mode: bool = False
    rag_config_path: str = "configs/rag_default.yaml"

    max_messages: int = Field(24, ge=1, le=128)
    max_message_chars: int = Field(12_000, ge=1, le=100_000)
    max_total_message_chars: int = Field(48_000, ge=1, le=500_000)
    max_new_tokens: int = Field(512, ge=1, le=4_096)

    stream_chunk_chars: int = Field(64, ge=1, le=4_096)
    stream_poll_interval_ms: int = Field(50, ge=10, le=1_000)
    generation_timeout_seconds: float = Field(120.0, gt=0.0, le=3_600.0)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ServingSettings:
        """Build settings from ``DORK_*`` variables.

        Passing an explicit mapping makes the behavior deterministic in tests and
        avoids mutating global process state.
        """

        env = os.environ if environ is None else environ
        mapping = {
            "provider": "DORK_MODEL_PROVIDER",
            "device": "DORK_MODEL_DEVICE",
            "demo_mode": "DORK_DEMO_MODE",
            "rag_config_path": "DORK_RAG_CONFIG",
            "max_messages": "DORK_MAX_MESSAGES",
            "max_message_chars": "DORK_MAX_MESSAGE_CHARS",
            "max_total_message_chars": "DORK_MAX_TOTAL_MESSAGE_CHARS",
            "max_new_tokens": "DORK_MAX_NEW_TOKENS",
            "stream_chunk_chars": "DORK_STREAM_CHUNK_CHARS",
            "stream_poll_interval_ms": "DORK_STREAM_POLL_INTERVAL_MS",
            "generation_timeout_seconds": "DORK_GENERATION_TIMEOUT_SECONDS",
        }
        values: dict[str, str] = {
            field: env[name] for field, name in mapping.items() if name in env
        }
        artifact = (
            env.get("DORK_MODEL_ARTIFACT")
            if "DORK_MODEL_ARTIFACT" in env
            else env.get("DORK_MODEL_PATH")
        )
        if artifact is not None and artifact.strip():
            values["artifact"] = artifact
        return cls.model_validate(values)
