"""Explicit model artifact resolution for the serving process."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dork.generation.providers import HFModel, LanguageModel, LocalGPTModel, MockLanguageModel
from dork.serving.settings import ServingSettings
from dork.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ArtifactCandidate:
    """A named local checkpoint candidate in serving priority order."""

    key: str
    path: str


DEFAULT_ARTIFACT_CANDIDATES: tuple[ArtifactCandidate, ...] = (
    ArtifactCandidate("frontier_sft", "artifacts/dorkllm_frontier_sft"),
    ArtifactCandidate("frontier", "artifacts/dorkllm_frontier"),
    ArtifactCandidate("tiny_sft", "artifacts/tiny_gpt_sft"),
    ArtifactCandidate("base", "artifacts/tiny_gpt"),
)

ModelLoader = Callable[[str, str | None, str], LanguageModel]


class ModelUnavailableError(RuntimeError):
    """Raised when an operation requires a model but none is ready."""

    code = "model_unavailable"
    public_message = "The requested language model is not ready."


def default_model_loader(provider: str, artifact: str | None, device: str) -> LanguageModel:
    """Load one provider without any implicit fallback."""

    if provider == "local_gpt":
        if artifact is None:
            raise ValueError("A local_gpt artifact path is required.")
        return LocalGPTModel.from_artifacts(artifact, device=device)
    if provider == "hf":
        if artifact is None:
            raise ValueError("A Hugging Face model name is required.")
        return HFModel(artifact, device=device)
    if provider == "mock":
        return MockLanguageModel()
    raise ValueError(f"Unsupported serving provider: {provider!r}")


@dataclass(frozen=True)
class ModelRuntime:
    """Truthful result of resolving the requested serving model."""

    model: LanguageModel | None
    requested_provider: str
    active_provider: str
    name: str | None
    artifact: str | None
    artifact_candidate: str | None
    device: str | None
    demo_mode: bool
    degraded_reason: str | None = None
    attempted_artifacts: tuple[str, ...] = ()

    @property
    def degraded(self) -> bool:
        return self.degraded_reason is not None

    @property
    def ready(self) -> bool:
        if self.model is None:
            return False
        return not (self.active_provider == "mock" and self.requested_provider != "mock")

    @property
    def model_loaded(self) -> bool:
        return self.model is not None and self.active_provider not in {"mock", "unavailable"}

    def require_model(self) -> LanguageModel:
        """Return the selected model or fail with a public-safe exception."""

        if self.model is None:
            raise ModelUnavailableError(self.degraded_reason or self.public_summary())
        return self.model

    def public_summary(self) -> str:
        if self.degraded_reason:
            return self.degraded_reason
        return "The requested language model is unavailable."

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_provider": self.requested_provider,
            "active_provider": self.active_provider,
            "name": self.name,
            "artifact": self.artifact,
            "artifact_candidate": self.artifact_candidate,
            "device": self.device,
            "demo_mode": self.demo_mode,
            "ready": self.ready,
            "model_loaded": self.model_loaded,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "attempted_artifacts": list(self.attempted_artifacts),
        }


def _active_metadata(
    model: LanguageModel,
    requested_provider: str,
    fallback_provider: str,
    artifact: str | None,
    candidate: str | None,
    settings: ServingSettings,
    attempted: list[str],
) -> ModelRuntime:
    active_provider = str(getattr(model, "provider", fallback_provider))
    active_artifact = getattr(model, "artifact", None) or artifact
    active_device = getattr(model, "device", None)
    if active_device is None and active_provider != "mock":
        active_device = settings.device
    return ModelRuntime(
        model=model,
        requested_provider=requested_provider,
        active_provider=active_provider,
        name=model.name,
        artifact=str(active_artifact) if active_artifact is not None else None,
        artifact_candidate=candidate,
        device=str(active_device) if active_device is not None else None,
        demo_mode=settings.demo_mode,
        attempted_artifacts=tuple(attempted),
    )


def _unavailable_or_demo_fallback(
    settings: ServingSettings,
    attempted: list[str],
    reason: str,
) -> ModelRuntime:
    if settings.demo_mode:
        model = MockLanguageModel()
        return ModelRuntime(
            model=model,
            requested_provider=settings.provider,
            active_provider=model.provider,
            name=model.name,
            artifact=None,
            artifact_candidate=None,
            device=None,
            demo_mode=True,
            degraded_reason=f"{reason} Demo-mode mock fallback is active.",
            attempted_artifacts=tuple(attempted),
        )
    return ModelRuntime(
        model=None,
        requested_provider=settings.provider,
        active_provider="unavailable",
        name=None,
        artifact=settings.artifact,
        artifact_candidate=None,
        device=settings.device,
        demo_mode=False,
        degraded_reason=reason,
        attempted_artifacts=tuple(attempted),
    )


def resolve_model_runtime(
    settings: ServingSettings,
    loader: ModelLoader = default_model_loader,
) -> ModelRuntime:
    """Resolve the requested model and record every serving-relevant decision."""

    requested = settings.provider
    attempted: list[str] = []

    if requested == "mock":
        if not settings.demo_mode:
            return _unavailable_or_demo_fallback(
                settings,
                attempted,
                "The mock provider is disabled outside demo mode.",
            )
        model = loader("mock", None, settings.device)
        return _active_metadata(model, requested, "mock", None, None, settings, attempted)

    if requested == "hf":
        if settings.artifact is None:
            return _unavailable_or_demo_fallback(
                settings,
                attempted,
                "No Hugging Face model name was configured.",
            )
        attempted.append(settings.artifact)
        try:
            model = loader("hf", settings.artifact, settings.device)
        except Exception as exc:
            logger.warning("Could not load requested Hugging Face model: %s", exc)
            return _unavailable_or_demo_fallback(
                settings,
                attempted,
                f"The requested Hugging Face model failed to load ({type(exc).__name__}).",
            )
        return _active_metadata(
            model, requested, "hf", settings.artifact, "explicit", settings, attempted
        )

    candidates = (
        (ArtifactCandidate("explicit", settings.artifact),)
        if settings.artifact is not None
        else DEFAULT_ARTIFACT_CANDIDATES
    )
    failures: list[str] = []
    for candidate in candidates:
        attempted.append(candidate.path)
        try:
            model = loader("local_gpt", candidate.path, settings.device)
        except Exception as exc:
            failures.append(f"{candidate.key}:{type(exc).__name__}")
            logger.info("Serving candidate %s did not load: %s", candidate.key, exc)
            continue
        return _active_metadata(
            model,
            requested,
            "local_gpt",
            candidate.path,
            candidate.key,
            settings,
            attempted,
        )

    detail = ", ".join(failures) if failures else "no candidates"
    return _unavailable_or_demo_fallback(
        settings,
        attempted,
        f"No local model artifact could be loaded ({detail}).",
    )
