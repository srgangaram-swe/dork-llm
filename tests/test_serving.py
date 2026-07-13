"""Unit tests for serving settings, model routing, and chat normalization."""

from __future__ import annotations

from typing import Any

import pytest
from dork.generation.providers import LanguageModel, MockLanguageModel
from dork.serving.runtime import (
    DEFAULT_ARTIFACT_CANDIDATES,
    ModelUnavailableError,
    resolve_model_runtime,
)
from dork.serving.service import DorkService, ServingInputError
from dork.serving.settings import ServingSettings
from dork.utils.config import RagConfig


class RecordingModel(LanguageModel):
    provider = "local_gpt"
    name = "recording-local-gpt"
    artifact = "artifacts/test-model"
    device = "cpu"

    def __init__(self, answer: str = "measured answer") -> None:
        self.answer = answer
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
        return self.answer


class _Store:
    def count(self) -> int:
        return 0


class _Rag:
    def __init__(self, model: LanguageModel) -> None:
        self.model = model
        self.store = _Store()

    def ingest(self, source: str | None = None) -> Any:
        class Stats:
            @staticmethod
            def to_dict() -> dict[str, int]:
                return {"chunks": 0}

        return Stats()


def test_settings_are_strict_by_default_and_parse_environment() -> None:
    defaults = ServingSettings()
    assert defaults.demo_mode is False
    assert defaults.provider == "auto"

    settings = ServingSettings.from_env(
        {
            "DORK_MODEL_PROVIDER": "local_gpt",
            "DORK_MODEL_PATH": "artifacts/compat",
            "DORK_MODEL_DEVICE": "cpu",
            "DORK_DEMO_MODE": "true",
            "DORK_MAX_MESSAGES": "7",
        }
    )
    assert settings.artifact == "artifacts/compat"
    assert settings.demo_mode is True
    assert settings.max_messages == 7


def test_explicit_artifact_environment_variable_wins_over_path_alias() -> None:
    settings = ServingSettings.from_env(
        {
            "DORK_MODEL_ARTIFACT": "artifacts/explicit",
            "DORK_MODEL_PATH": "artifacts/compat",
        }
    )
    assert settings.artifact == "artifacts/explicit"


def test_candidate_resolution_uses_documented_priority() -> None:
    attempts: list[tuple[str, str | None, str]] = []

    def loader(provider: str, artifact: str | None, device: str) -> LanguageModel:
        attempts.append((provider, artifact, device))
        if artifact == "artifacts/tiny_gpt_sft":
            return RecordingModel()
        raise FileNotFoundError(artifact)

    runtime = resolve_model_runtime(
        ServingSettings(provider="auto", device="cpu"),
        loader=loader,
    )
    assert [attempt[1] for attempt in attempts] == [
        candidate.path for candidate in DEFAULT_ARTIFACT_CANDIDATES[:3]
    ]
    assert runtime.artifact_candidate == "tiny_sft"
    assert runtime.active_provider == "local_gpt"
    assert runtime.ready is True
    assert runtime.model_loaded is True


def test_explicit_artifact_disables_candidate_search() -> None:
    attempts: list[str | None] = []

    def loader(provider: str, artifact: str | None, device: str) -> LanguageModel:
        attempts.append(artifact)
        return RecordingModel()

    runtime = resolve_model_runtime(
        ServingSettings(provider="local_gpt", artifact="artifacts/only-this"),
        loader=loader,
    )
    assert attempts == ["artifacts/only-this"]
    assert runtime.artifact_candidate == "explicit"


def test_missing_model_is_unavailable_without_demo_mode() -> None:
    def loader(provider: str, artifact: str | None, device: str) -> LanguageModel:
        raise FileNotFoundError(artifact)

    runtime = resolve_model_runtime(ServingSettings(), loader=loader)
    assert runtime.model is None
    assert runtime.active_provider == "unavailable"
    assert runtime.ready is False
    assert runtime.degraded is True
    with pytest.raises(ModelUnavailableError):
        runtime.require_model()


def test_demo_fallback_is_explicit_and_truthful() -> None:
    def loader(provider: str, artifact: str | None, device: str) -> LanguageModel:
        raise FileNotFoundError(artifact)

    runtime = resolve_model_runtime(ServingSettings(demo_mode=True), loader=loader)
    assert isinstance(runtime.model, MockLanguageModel)
    assert runtime.requested_provider == "auto"
    assert runtime.active_provider == "mock"
    assert runtime.ready is False
    assert runtime.degraded_reason is not None
    assert "Demo-mode mock fallback" in runtime.degraded_reason

    explicit = resolve_model_runtime(ServingSettings(provider="mock", demo_mode=True))
    assert explicit.active_provider == "mock"
    assert explicit.ready is True
    assert explicit.degraded is False


def test_rag_uses_exact_selected_language_model() -> None:
    selected = RecordingModel()
    captured: dict[str, Any] = {}

    def factory(cfg: RagConfig, model: LanguageModel) -> Any:
        captured["cfg"] = cfg
        captured["model"] = model
        return _Rag(model)

    service = DorkService(
        language_model=selected,
        rag_factory=factory,
        rag_config_loader=lambda _: RagConfig(),
    )
    assert service.rag.model is selected
    assert captured["model"] is selected


def test_legacy_duplicate_latest_user_is_removed_once() -> None:
    model = RecordingModel()
    service = DorkService(language_model=model)
    result = service.chat(
        "hello",
        mode="generate",
        history=[
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "hello"},
        ],
        max_new_tokens=8,
    )
    assert result["answer"] == "measured answer"
    assert model.prompts[-1].count("hello") == 1
    assert "User: earlier" in model.prompts[-1]
    assert model.prompts[-1].startswith("### Instruction:\n")
    assert model.prompts[-1].endswith("\n### Response:\n")
    assert model.stops[-1] == ["### Instruction:", "<|endoftext|>"]


def test_canonical_messages_build_one_current_user_turn() -> None:
    model = RecordingModel()
    service = DorkService(language_model=model)
    service.chat(
        mode="generate",
        messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "latest"},
        ],
        max_new_tokens=8,
    )
    prompt = model.prompts[-1]
    assert prompt.count("latest") == 1
    assert "System: Be concise." in prompt


def test_chat_limits_and_ambiguous_contract_are_rejected() -> None:
    service = DorkService(
        settings=ServingSettings(max_messages=2, max_message_chars=5, max_total_message_chars=8),
        language_model=RecordingModel(),
    )
    with pytest.raises(ServingInputError):
        service.chat("abcdef", mode="generate")
    with pytest.raises(ServingInputError):
        service.chat(
            "new",
            mode="generate",
            messages=[{"role": "user", "content": "other"}],
        )
    with pytest.raises(ServingInputError):
        service.chat(
            mode="generate",
            messages=[
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
            ],
        )
