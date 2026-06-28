"""Text generation: sampling primitives and a tokenizer-aware generator."""

from __future__ import annotations

from dork.generation.generator import Generator
from dork.generation.providers import (
    HFModel,
    LanguageModel,
    LocalGPTModel,
    MockLanguageModel,
    build_language_model,
)
from dork.generation.sampling import apply_top_k, apply_top_p, sample_next_token

__all__ = [
    "Generator",
    "HFModel",
    "LanguageModel",
    "LocalGPTModel",
    "MockLanguageModel",
    "apply_top_k",
    "apply_top_p",
    "build_language_model",
    "sample_next_token",
]
