"""Dependency-injected service layer shared by every application surface."""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from dork import __version__
from dork.generation.providers import LanguageModel
from dork.serving.runtime import (
    ModelLoader,
    ModelRuntime,
    ModelUnavailableError,
    resolve_model_runtime,
)
from dork.serving.settings import ServingSettings
from dork.utils.config import RagConfig, load_rag_config
from dork.utils.logging import get_logger

if TYPE_CHECKING:
    from dork.rag.pipeline import RagPipeline

logger = get_logger(__name__)

RagFactory = Callable[[RagConfig, LanguageModel], "RagPipeline"]
_SFT_INSTRUCTION_HEADER = "### Instruction:"
_SFT_RESPONSE_HEADER = "### Response:"
_CHAT_STOP_MARKERS = ["### Instruction:", "<|endoftext|>"]


class ServingInputError(ValueError):
    """Public-safe validation error raised by non-HTTP service callers."""

    code = "invalid_request"
    public_message = "The chat request is invalid."


class Metrics:
    """Thread-safe counters and latency accumulators."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counts: dict[str, int] = defaultdict(int)
        self.latency_ms_sum: dict[str, float] = defaultdict(float)

    def record(self, endpoint: str, latency_ms: float) -> None:
        with self._lock:
            self.counts[endpoint] += 1
            self.latency_ms_sum[endpoint] += latency_ms

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests": dict(self.counts),
                "avg_latency_ms": {
                    key: self.latency_ms_sum[key] / self.counts[key]
                    for key in self.counts
                    if self.counts[key]
                },
            }


def _default_rag_factory(cfg: RagConfig, model: LanguageModel) -> RagPipeline:
    from dork.rag.pipeline import RagPipeline

    return RagPipeline(cfg, model=model)


class DorkService:
    """Own one selected language model and share it with generation and RAG."""

    def __init__(
        self,
        rag_config: str | None = None,
        *,
        settings: ServingSettings | None = None,
        language_model: LanguageModel | None = None,
        model_runtime: ModelRuntime | None = None,
        model_loader: ModelLoader | None = None,
        rag_pipeline: RagPipeline | None = None,
        rag_factory: RagFactory | None = None,
        rag_config_loader: Callable[[str], RagConfig] = load_rag_config,
    ) -> None:
        self.settings = settings or ServingSettings.from_env()
        self.rag_config_path = rag_config or self.settings.rag_config_path
        self.metrics = Metrics()
        self._model_loader = model_loader
        self._runtime = model_runtime
        if language_model is not None:
            self._runtime = self._runtime_for_injected_model(language_model)
        self._rag = rag_pipeline
        self._rag_factory = rag_factory or _default_rag_factory
        self._rag_config_loader = rag_config_loader
        self._lock = threading.RLock()

        if self._rag is not None:
            # The selected serving model is authoritative, even for an injected
            # pipeline. This prevents RAG from quietly using a config-level mock.
            self._rag.model = self.language_model

    def _runtime_for_injected_model(self, model: LanguageModel) -> ModelRuntime:
        provider = str(getattr(model, "provider", "injected"))
        artifact = getattr(model, "artifact", None) or self.settings.artifact
        device = getattr(model, "device", None)
        return ModelRuntime(
            model=model,
            requested_provider=self.settings.provider,
            active_provider=provider,
            name=model.name,
            artifact=str(artifact) if artifact is not None else None,
            artifact_candidate="injected",
            device=str(device) if device is not None else None,
            demo_mode=self.settings.demo_mode,
        )

    @property
    def model_runtime(self) -> ModelRuntime:
        if self._runtime is None:
            with self._lock:
                if self._runtime is None:
                    if self._model_loader is None:
                        self._runtime = resolve_model_runtime(self.settings)
                    else:
                        self._runtime = resolve_model_runtime(
                            self.settings,
                            loader=self._model_loader,
                        )
        return self._runtime

    @property
    def language_model(self) -> LanguageModel:
        return self.model_runtime.require_model()

    @property
    def model_loaded(self) -> bool:
        return self.model_runtime.model_loaded

    @property
    def rag(self) -> RagPipeline:
        if self._rag is None:
            with self._lock:
                if self._rag is None:
                    cfg = self._rag_config_loader(self.rag_config_path)
                    self._rag = self._rag_factory(cfg, self.language_model)
                    if self._rag.store.count() == 0:
                        try:
                            self._rag.ingest()
                        except Exception as exc:  # pragma: no cover - local data failure
                            logger.warning("Automatic RAG ingestion was skipped: %s", exc)
        return self._rag

    def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Generate a completion after enforcing serving input limits."""

        prompt = self._validated_content(prompt, "prompt", preserve_whitespace=True)
        max_new_tokens = int(kwargs.get("max_new_tokens", 128))
        self._validate_max_new_tokens(max_new_tokens)
        started = time.perf_counter()
        completion = self.language_model.complete(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=float(kwargs.get("temperature", 0.8)),
            top_k=kwargs.get("top_k"),
            top_p=kwargs.get("top_p"),
            stop=kwargs.get("stop"),
        )
        latency = (time.perf_counter() - started) * 1_000
        self.metrics.record("generate", latency)
        return {
            "prompt": prompt,
            "completion": completion,
            "model": self.language_model.name,
            "latency_ms": latency,
        }

    def rag_ingest(self, source: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        stats = self.rag.ingest(source).to_dict()
        self.metrics.record("rag_ingest", (time.perf_counter() - started) * 1_000)
        return stats

    def rag_query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        question = self._validated_content(question, "question")
        started = time.perf_counter()
        answer = self.rag.query(question, top_k=top_k).to_dict()
        self.metrics.record("rag_query", (time.perf_counter() - started) * 1_000)
        return answer

    def chat(
        self,
        message: str | None = None,
        mode: str = "auto",
        history: Sequence[Mapping[str, str]] | None = None,
        *,
        messages: Sequence[Mapping[str, str]] | None = None,
        retrieval_top_k: int = 5,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        sampling_top_k: int | None = 50,
        top_p: float | None = 0.95,
    ) -> dict[str, Any]:
        """Return a chat response for either the legacy or canonical contract.

        Canonical clients send ``messages`` with the latest user turn last.
        Legacy clients send ``message`` plus ``history``. If a legacy client has
        already appended its current user turn to history, the exact trailing
        duplicate is removed before prompt construction.
        """

        if mode not in {"auto", "rag", "generate"}:
            raise ServingInputError("mode must be auto, rag, or generate")
        self._validate_max_new_tokens(max_new_tokens)
        current, prior = self.normalize_chat_messages(
            message=message,
            history=history,
            messages=messages,
        )
        started = time.perf_counter()
        citations: list[dict[str, Any]] = []

        if mode in {"auto", "rag"}:
            try:
                rag_out = self.rag_query(current, top_k=retrieval_top_k)
                if rag_out.get("answer") and (not rag_out.get("refused") or mode == "rag"):
                    answer = self._clean_rag_answer(str(rag_out["answer"]))
                    latency = (time.perf_counter() - started) * 1_000
                    self.metrics.record("chat", latency)
                    return self._chat_result(
                        answer=answer,
                        mode="rag",
                        citations=list(rag_out.get("citations", [])),
                        model=str(rag_out.get("model") or self.language_model.name),
                        latency_ms=latency,
                    )
                citations = list(rag_out.get("citations", []))
            except ModelUnavailableError:
                raise
            except Exception as exc:  # pragma: no cover - defensive local fallback
                logger.warning("RAG chat path failed; using direct generation: %s", exc)

        prompt = self._chat_prompt(current, prior)
        generated = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=sampling_top_k,
            top_p=top_p,
            stop=_CHAT_STOP_MARKERS,
        )
        latency = (time.perf_counter() - started) * 1_000
        self.metrics.record("chat", latency)
        return self._chat_result(
            answer=str(generated["completion"]).strip(),
            mode="generate",
            citations=citations,
            model=str(generated["model"]),
            latency_ms=latency,
        )

    def normalize_chat_messages(
        self,
        *,
        message: str | None,
        history: Sequence[Mapping[str, str]] | None,
        messages: Sequence[Mapping[str, str]] | None,
    ) -> tuple[str, list[dict[str, str]]]:
        """Normalize canonical/legacy chat payloads and enforce bounded input."""

        if messages:
            if message is not None or history:
                raise ServingInputError("Use either messages or legacy message/history, not both.")
            normalized = [self._validated_message(item) for item in messages]
            if normalized[-1]["role"] != "user":
                raise ServingInputError("The final canonical message must have role=user.")
            current = normalized[-1]["content"]
            prior = normalized[:-1]
        else:
            if message is None:
                raise ServingInputError("A message or canonical messages list is required.")
            current = self._validated_content(message, "message")
            prior = [self._validated_message(item) for item in (history or [])]
            if prior and prior[-1]["role"] == "user" and prior[-1]["content"] == current:
                prior.pop()

        all_messages = [*prior, {"role": "user", "content": current}]
        if len(all_messages) > self.settings.max_messages:
            raise ServingInputError(
                f"At most {self.settings.max_messages} chat messages are accepted."
            )
        total_chars = sum(len(item["content"]) for item in all_messages)
        if total_chars > self.settings.max_total_message_chars:
            raise ServingInputError(
                "The combined chat history exceeds the configured character limit."
            )
        return current, prior

    def _validated_message(self, item: Mapping[str, str]) -> dict[str, str]:
        role = item.get("role", "")
        if role not in {"system", "user", "assistant"}:
            raise ServingInputError("Message roles must be system, user, or assistant.")
        return {
            "role": role,
            "content": self._validated_content(item.get("content", ""), "message content"),
        }

    def _validated_content(
        self,
        content: str,
        field: str,
        *,
        preserve_whitespace: bool = False,
    ) -> str:
        if not isinstance(content, str):
            raise ServingInputError(f"{field} must be a string.")
        normalized = content.strip()
        if not normalized:
            raise ServingInputError(f"{field} cannot be empty.")
        if len(content) > self.settings.max_message_chars:
            raise ServingInputError(
                f"{field} exceeds the {self.settings.max_message_chars}-character limit."
            )
        return content if preserve_whitespace else normalized

    def _validate_max_new_tokens(self, value: int) -> None:
        if value < 1 or value > self.settings.max_new_tokens:
            raise ServingInputError(
                f"max_new_tokens must be between 1 and {self.settings.max_new_tokens}."
            )

    @staticmethod
    def _chat_prompt(message: str, history: Sequence[Mapping[str, str]]) -> str:
        """Format bounded context with the exact delimiter used during SFT."""

        instruction = [
            "Answer as dorkLLM, a direct and technically honest local assistant.",
            "State uncertainty instead of inventing facts.",
        ]
        if history:
            instruction.append("\nBounded conversation context:")
        for item in history:
            role = item.get("role", "user").title()
            content = item.get("content", "").strip()
            if content:
                instruction.append(f"{role}: {content}")
        instruction.extend(("\nCurrent user request:", message))
        instruction_text = "\n".join(instruction)
        return f"{_SFT_INSTRUCTION_HEADER}\n" f"{instruction_text}\n\n" f"{_SFT_RESPONSE_HEADER}\n"

    @staticmethod
    def _clean_rag_answer(answer: str) -> str:
        return re.sub(r"^\s*\[\d+\]\s*", "", answer).strip()

    def _chat_result(
        self,
        *,
        answer: str,
        mode: str,
        citations: list[dict[str, Any]],
        model: str,
        latency_ms: float,
    ) -> dict[str, Any]:
        runtime = self.model_runtime
        return {
            "answer": answer,
            "mode": mode,
            "citations": citations,
            "model": model,
            "latency_ms": latency_ms,
            "requested_provider": runtime.requested_provider,
            "active_provider": runtime.active_provider,
            "artifact": runtime.artifact,
            "device": runtime.device,
            "degraded": runtime.degraded,
            "degraded_reason": runtime.degraded_reason,
        }

    def run_agent(self, task: str) -> dict[str, Any]:
        from dork.agents.research_agent import ResearchAgent

        task = self._validated_content(task, "task")
        started = time.perf_counter()
        result = ResearchAgent(self.rag).run(task).to_dict()
        self.metrics.record("agent_run", (time.perf_counter() - started) * 1_000)
        return result

    def evaluate(self, config_path: str = "configs/eval_default.yaml") -> dict[str, Any]:
        from dork.evaluation.harness import EvalHarness
        from dork.utils.config import load_eval_config

        started = time.perf_counter()
        report = EvalHarness(load_eval_config(config_path)).run(write=False)
        self.metrics.record("evaluate", (time.perf_counter() - started) * 1_000)
        return {"summary": report["summary"], "gate": report["gate"], "model": report["model"]}

    def liveness(self) -> dict[str, Any]:
        """Return process liveness without forcing heavyweight model loading."""

        return {"status": "ok", "live": True, "version": __version__}

    def model_info(self) -> dict[str, Any]:
        return self.model_runtime.to_dict()

    def readiness(self) -> dict[str, Any]:
        runtime = self.model_runtime
        return {
            "status": "ready" if runtime.ready else "degraded",
            "ready": runtime.ready,
            "version": __version__,
            "model": runtime.to_dict(),
        }

    def health(self) -> dict[str, Any]:
        runtime = self.model_runtime
        rag_chunks = self._rag.store.count() if self._rag is not None else 0
        return {
            "status": "ok" if runtime.ready else "degraded",
            "live": True,
            "ready": runtime.ready,
            "version": __version__,
            "model_loaded": runtime.model_loaded,
            "rag_chunks": rag_chunks,
            "requested_provider": runtime.requested_provider,
            "active_provider": runtime.active_provider,
            "degraded_reason": runtime.degraded_reason,
            "model": runtime.to_dict(),
        }
