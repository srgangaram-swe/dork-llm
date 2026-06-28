#!/usr/bin/env python
"""End-to-end smoke test exercised by CI.

Runs the *entire* platform on a tiny scale in well under a minute on CPU:
tokenizer -> 5-step training -> generation -> perplexity -> evaluation harness
(mock) -> RAG ingest+query -> agent. Exits non-zero on any failure so CI catches
integration regressions that unit tests might miss.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def main() -> int:
    from dork.utils.config import (
        DataConfig,
        EvalConfig,
        ModelConfig,
        RagConfig,
        TinyGPTConfig,
        TokenizerConfig,
        TrainingConfig,
    )
    from dork.utils.seed import seed_everything

    seed_everything(0)
    tmp = Path(tempfile.mkdtemp(prefix="dork_smoke_"))
    print(f"[smoke] workdir: {tmp}")

    # ── 1. Tiny config (char tokenizer keeps CI dependency-light & fast) ──
    cfg = TinyGPTConfig(
        seed=0,
        data=DataConfig(dataset="tiny_shakespeare", raw_path=str(tmp / "corpus.txt")),
        tokenizer=TokenizerConfig(type="char", vocab_size=128, path=str(tmp / "tok.json")),
        model=ModelConfig(block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0),
        training=TrainingConfig(
            batch_size=8,
            max_steps=5,
            eval_interval=5,
            eval_iters=2,
            warmup_steps=1,
            out_dir=str(tmp / "ckpt"),
            device="cpu",
            dtype="float32",
        ),
    )

    # ── 2. Train ──
    from dork.data.datasets import prepare_corpus
    from dork.data.loader import BinTokenDataset, build_token_bins
    from dork.models.tiny_gpt import TinyGPT
    from dork.tokenizer.factory import load_or_train_tokenizer
    from dork.training.trainer import Trainer

    text = prepare_corpus(cfg.data).read_text(encoding="utf-8")
    tok = load_or_train_tokenizer(cfg.tokenizer, text)
    cfg.model.vocab_size = tok.vocab_size
    meta = build_token_bins(text, tok.encode, tmp / "bins", cfg.data.val_fraction)
    train_ds = BinTokenDataset(meta["train_bin"], cfg.model.block_size)
    val_ds = BinTokenDataset(meta["val_bin"], cfg.model.block_size)
    model = TinyGPT(cfg.model)
    history = Trainer(model, train_ds, val_ds, cfg.training, str(tmp / "tok.json")).train()
    assert history, "training produced no history"
    print(f"[smoke] trained {model.num_params():,} params; final val={history[-1]['val']:.3f}")

    # ── 3. Generate + perplexity ──
    from dork.generation.generator import Generator

    gen = Generator(model, tok, device="cpu")
    out = gen.generate("To be", max_new_tokens=20, temperature=0.8)
    assert isinstance(out, str) and len(out) > 0, "generation returned empty text"
    ppl = gen.perplexity(text[:500])
    assert ppl == ppl and ppl > 0, "perplexity invalid"
    print(f"[smoke] generated {len(out)} chars; perplexity={ppl:.2f}")

    # ── 4. Evaluation harness (mock provider, no torch needed) ──
    from dork.evaluation.harness import EvalHarness

    eval_cfg = EvalConfig(
        model={"provider": "mock"},
        suites={
            "exact_match": {"enabled": True, "path": "arithmetic.jsonl"},
            "json_validity": {"enabled": True, "path": "json_tasks.jsonl"},
            "tool_use": {"enabled": True, "path": "tool_use.jsonl"},
            "safety_refusal": {"enabled": True, "path": "safety_refusal.jsonl"},
            "latency": {"enabled": True, "n_requests": 3},
        },
        report={"out_dir": str(tmp / "reports"), "formats": ["json", "markdown"], "plots": False},
    )
    report = EvalHarness(eval_cfg).run()
    assert report["summary"], "eval produced no summary"
    print(f"[smoke] eval suites: {[r['suite'] for r in report['summary']]}")

    # ── 5. RAG ingest + query ──
    from dork.rag.pipeline import RagPipeline

    rag_cfg = RagConfig(
        ingest={
            "source_dir": "data/sample_docs",
            "chunking": {"chunk_size": 80, "min_chunk_chars": 20},
        },
        embeddings={"backend": "hash", "dim": 128},
        vector_store={"backend": "memory"},
        retrieval={"top_k": 3, "rerank": True, "rerank_top_n": 2, "min_score": 0.0},
        generation={"provider": "mock", "refuse_when_insufficient": True},
    )
    pipe = RagPipeline(rag_cfg)
    stats = pipe.ingest()
    assert stats.chunks > 0, "ingestion produced no chunks"
    ans = pipe.query("What is causal masking in a decoder?")
    assert ans.contexts, "RAG retrieved no context"
    print(f"[smoke] RAG indexed {stats.chunks} chunks; citations={len(ans.citations)}")

    # ── 6. Agent ──
    from dork.agents.research_agent import ResearchAgent

    agent = ResearchAgent(pipe, max_steps=4)
    res = agent.run("Summarize the transformers document")
    assert res.answer, "agent returned empty answer"
    calc = agent.run("Calculate 21 * 2")
    assert "42" in calc.answer, f"agent calculator wrong: {calc.answer}"
    print("[smoke] agent intents OK (summarize + calculate)")

    print("\n[smoke] ✅ ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
