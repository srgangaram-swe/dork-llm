"""Reusable scoring helpers used across evaluation suites."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from typing import Any

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation/articles, and collapse whitespace (SQuAD-style)."""
    s = s.lower().translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> bool:
    """Normalized exact match."""
    return normalize_text(pred) == normalize_text(gold)


def contains_answer(pred: str, gold: str) -> bool:
    """True if the normalized gold answer appears in the normalized prediction."""
    g = normalize_text(gold)
    return bool(g) and g in normalize_text(pred)


def token_f1(pred: str, gold: str) -> float:
    """Token-overlap F1 between prediction and gold (SQuAD-style)."""
    p_tokens = normalize_text(pred).split()
    g_tokens = normalize_text(gold).split()
    if not p_tokens or not g_tokens:
        return float(p_tokens == g_tokens)
    # Multiset intersection caps each repeated token at its count in the other
    # sequence. A membership-only count can produce recall > 1 and therefore an
    # invalid F1 score when the prediction repeats a gold token.
    num_same = sum((Counter(p_tokens) & Counter(g_tokens)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_json(text: str) -> Any | None:
    """Best-effort extraction of the first JSON object/array from ``text``."""
    text = text.strip()
    # Strip common markdown code fences.
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None


def is_valid_json(text: str) -> tuple[bool, Any | None]:
    """Return ``(valid, parsed_obj_or_None)``."""
    obj = extract_json(text)
    return (obj is not None, obj)


def schema_ok(obj: Any, required_keys: list[str]) -> bool:
    """True if ``obj`` is a dict containing every required key."""
    if not isinstance(obj, dict):
        return False
    return all(k in obj for k in required_keys)


def key_coverage(obj: Any, required_keys: list[str]) -> float:
    """Fraction of required keys present in ``obj``."""
    if not required_keys:
        return 1.0
    if not isinstance(obj, dict):
        return 0.0
    return sum(1 for k in required_keys if k in obj) / len(required_keys)


def extract_choice_letter(text: str) -> str | None:
    """Pull a multiple-choice answer letter (A-D) from free-form text."""
    m = re.search(r"\b([A-D])\b", text.strip().upper())
    return m.group(1) if m else None


def extract_citations(text: str) -> list[int]:
    """Extract bracketed citation indices like ``[1]`` or ``[2]`` from ``text``."""
    return [int(x) for x in re.findall(r"\[(\d+)\]", text)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
