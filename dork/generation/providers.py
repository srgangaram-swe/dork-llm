"""Pluggable language-model providers behind a single ``LanguageModel`` interface.

Both the evaluation harness and the RAG pipeline talk to models through this
interface, so they work identically against:

* ``mock``      — a deterministic, rule-based stub (offline CI, harness self-tests)
* ``local_gpt`` — the tiny GPT trained in this repo
* ``hf``        — any Hugging Face causal-LM (requires ``transformers``)

The mock is intentionally a *little* capable (arithmetic, JSON templating, MCQ
defaulting, refusals) so eval reports show realistic, discriminating numbers
without a GPU or network — but it is honestly just a stub, not a real model.
"""

from __future__ import annotations

import abc
import json
import re
from typing import Any

from dork.utils.logging import get_logger

logger = get_logger(__name__)


class LanguageModel(abc.ABC):
    """Minimal text-completion contract."""

    name: str = "language-model"

    @abc.abstractmethod
    def complete(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """Return a text completion for ``prompt``."""

    def perplexity(self, text: str) -> float:  # pragma: no cover - optional
        """Return token-level perplexity, or NaN if unsupported."""
        return float("nan")


# ───────────────────────────── Mock ──────────────────────────────────
class MockLanguageModel(LanguageModel):
    """Deterministic, dependency-free stand-in used for offline eval and tests."""

    name = "mock-rulebased-v0"
    _REFUSAL = "I can't help with that. It appears unsafe or unethical."
    _HARMFUL = ("how to make a bomb", "build a weapon", "hack into", "steal", "malware", "poison")

    def complete(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
    ) -> str:
        p = prompt.strip()
        low = p.lower()

        # Safety: refuse obviously harmful asks (benign synthetic triggers only).
        if any(h in low for h in self._HARMFUL):
            return self._REFUSAL

        # Tool-use: emit a structured tool call when the prompt asks for one.
        if "available tools" in low or "respond with a tool call" in low:
            return self._tool_call(p)

        # JSON tasks: build a valid object from requested keys.
        if "json" in low:
            return self._json_response(p)

        # Multiple-choice: answer with a letter (a deterministic, weak baseline).
        if re.search(r"\n\s*[A-D]\)", p) or "answer with the letter" in low:
            return "A"

        # Arithmetic: compute a single binary expression if present.
        math_ans = self._try_arithmetic(p)
        if math_ans is not None:
            return math_ans

        # Grounded QA: if context is supplied, quote the first sentence + cite.
        if "context:" in low:
            return self._grounded_answer(p)

        # Fallback: a short deterministic echo-style summary.
        first = re.split(r"(?<=[.!?])\s+", p)[0][:160]
        return f"Summary: {first}"

    # -- handlers ------------------------------------------------------
    def _try_arithmetic(self, p: str) -> str | None:
        m = re.search(r"(-?\d+)\s*([+\-*x/])\s*(-?\d+)", p)
        if not m:
            return None
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        try:
            val = {
                "+": a + b,
                "-": a - b,
                "*": a * b,
                "x": a * b,
                "/": a // b if b else 0,
            }[op]
        except ZeroDivisionError:
            return "0"
        return str(val)

    def _json_response(self, p: str) -> str:
        m = re.search(r"keys?:?\s*([\w, ]+)", p, flags=re.IGNORECASE)
        obj: dict[str, Any] = {}
        if m:
            for key in [k.strip() for k in m.group(1).split(",") if k.strip()]:
                obj[key] = self._guess_value(key, p)
        else:
            obj = {"answer": "ok", "confidence": 0.9}
        return json.dumps(obj)

    @staticmethod
    def _guess_value(key: str, prompt: str) -> Any:
        k = key.lower()
        if any(t in k for t in ("count", "age", "number", "year", "n_")):
            nums = re.findall(r"\d+", prompt)
            return int(nums[0]) if nums else 0
        if "is_" in k or k.startswith("has") or "bool" in k:
            return True
        if "list" in k or k.endswith("s"):
            return []
        return "value"

    def _tool_call(self, p: str) -> str:
        m = re.search(r"(-?\d+\s*[+\-*x/]\s*-?\d+)", p)
        if m is not None:
            return json.dumps({"tool": "calculator", "args": {"expression": m.group(1).strip()}})
        return json.dumps({"tool": "search_docs", "args": {"query": p[:60]}})

    def _grounded_answer(self, p: str) -> str:
        ctx = p.split("Context:", 1)[-1]
        sent = re.split(r"(?<=[.!?])\s+", ctx.strip())[0][:200]
        return f"{sent} [1]"


# ─────────────────────────── Local GPT ───────────────────────────────
class LocalGPTModel(LanguageModel):
    """Wrap the repo's trained :class:`~dork.generation.generator.Generator`."""

    def __init__(self, generator: Any, name: str = "dork-tiny-gpt") -> None:
        self._gen = generator
        self.name = name

    @classmethod
    def from_artifacts(cls, ckpt_dir: str, device: str = "cpu") -> LocalGPTModel:
        from dork.generation.generator import Generator
        from dork.tokenizer.factory import load_tokenizer
        from dork.training.checkpoint import load_model_from_checkpoint
        from dork.training.trainer import resolve_device

        device = resolve_device(device)
        model, payload = load_model_from_checkpoint(ckpt_dir, device=device)
        tok_path = payload.get("tokenizer_path") or "tokenizers/tiny_gpt_bpe.json"
        tokenizer = load_tokenizer(tok_path)
        return cls(Generator(model, tokenizer, device=device))

    def complete(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
    ) -> str:
        out = self._gen.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        if stop:
            for s in stop:
                if s in out:
                    out = out.split(s, 1)[0]
        return out

    def perplexity(self, text: str) -> float:
        return self._gen.perplexity(text)


# ───────────────────────────── HF ────────────────────────────────────
class HFModel(LanguageModel):
    """Wrap a Hugging Face causal-LM via ``transformers`` (optional extra)."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self.name = model_name
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self._device = device

    def complete(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
    ) -> str:
        import torch

        inputs = self._tok(prompt, return_tensors="pt").to(self._device)
        sample_kwargs: dict[str, Any] = {}
        if temperature > 0:
            if top_k is not None:
                sample_kwargs["top_k"] = top_k
            if top_p is not None:
                sample_kwargs["top_p"] = top_p
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self._tok.eos_token_id,
                **sample_kwargs,
            )
        text = self._tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        if stop:
            for s in stop:
                text = text.split(s, 1)[0]
        return text


def build_language_model(cfg: dict[str, Any]) -> LanguageModel:
    """Construct a :class:`LanguageModel` from a config dict.

    Recognized keys: ``provider`` (mock|local_gpt|hf), ``name``, ``device``.
    Falls back to the mock on any failure so callers always get a usable model.
    """
    provider = str(cfg.get("provider", "mock")).lower()
    device = str(cfg.get("device", "cpu"))
    name = str(cfg.get("name", "artifacts/tiny_gpt"))

    if provider == "mock":
        return MockLanguageModel()
    if provider == "local_gpt":
        try:
            return LocalGPTModel.from_artifacts(name, device=device)
        except Exception as exc:
            logger.warning("Could not load local GPT (%s); using mock.", exc)
            return MockLanguageModel()
    if provider == "hf":
        try:
            return HFModel(name, device=device)
        except Exception as exc:
            logger.warning("Could not load HF model %s (%s); using mock.", name, exc)
            return MockLanguageModel()
    raise ValueError(f"Unknown model provider: {provider!r}")
