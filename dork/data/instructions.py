"""Synthetic, public-safe instruction/response pairs for supervised fine-tuning.

Everything here is generated programmatically or hand-written about this project's
own concepts — no scraped, proprietary, or sensitive data. The set is deliberately
small and simple: at educational scale, SFT demonstrates the *mechanism*
(instruction template + response-only loss masking + adapting to the instruction
distribution), not a capability leap. See ``docs/agent_design.md`` and the model card.
"""

from __future__ import annotations

import random

# Hand-written domain Q&A. Teaches a consistent, grounded answer style.
CONCEPT_QA: list[dict[str, str]] = [
    {
        "instruction": "What does causal masking do in a decoder?",
        "response": "It prevents each position from attending to future positions, "
        "preserving the autoregressive property.",
    },
    {
        "instruction": "What is perplexity?",
        "response": "The exponential of the average negative log-likelihood the model "
        "assigns to held-out text; lower is better.",
    },
    {
        "instruction": "What is retrieval-augmented generation?",
        "response": "A method that retrieves relevant passages and conditions generation "
        "on them so answers are grounded in evidence.",
    },
    {
        "instruction": "What does top-p sampling do?",
        "response": "It keeps the smallest set of tokens whose cumulative probability "
        "reaches p, then samples from that nucleus.",
    },
    {
        "instruction": "Why tie the embedding and output weights?",
        "response": "Weight tying shares parameters between the input embedding and the "
        "output projection, reducing parameters and often improving quality.",
    },
    {
        "instruction": "What is a KV cache used for?",
        "response": "It stores past keys and values so incremental decoding reuses them "
        "instead of recomputing attention over the whole prefix.",
    },
    {
        "instruction": "What optimizer is standard for training transformers?",
        "response": "AdamW, with weight decay applied to weight matrices but not to biases "
        "or layer-norm parameters.",
    },
    {
        "instruction": "When should a RAG assistant refuse to answer?",
        "response": "When retrieval finds no passage above the score threshold, it should "
        "refuse rather than hallucinate.",
    },
]

_WORDS = ["hello", "model", "token", "vector", "tensor", "layer", "attention", "corpus"]


def synthetic_instructions(n_arith: int = 48, seed: int = 0) -> list[dict[str, str]]:
    """Return a list of ``{"instruction", "response"}`` pairs.

    Args:
        n_arith: Number of arithmetic problems to generate.
        seed: RNG seed for reproducible generation.
    """
    rng = random.Random(seed)
    data: list[dict[str, str]] = []

    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
    ]
    for _ in range(n_arith):
        a, b = rng.randint(1, 20), rng.randint(1, 20)
        sym, fn = rng.choice(ops)
        data.append({"instruction": f"Compute {a} {sym} {b}.", "response": str(fn(a, b))})

    for w in _WORDS:
        data.append({"instruction": f"Reverse the word '{w}'.", "response": w[::-1]})
        data.append({"instruction": f"How many letters are in '{w}'?", "response": str(len(w))})

    data.extend(CONCEPT_QA)
    rng.shuffle(data)
    return data


def split_instructions(
    data: list[dict[str, str]], val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Deterministically split pairs into (train, val)."""
    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]
