# Portfolio Summary

Dork LLM is an end-to-end LLM systems portfolio project that combines model
internals, evaluation, retrieval, agents, and serving in one cohesive codebase.

## What It Shows

- A GPT-style decoder-only transformer implemented from scratch in PyTorch.
- Tokenizer training, binary token dataset preparation, training loop,
  validation, checkpointing, and text generation.
- KV-cache inference with a benchmark against the reference decode path.
- Supervised fine-tuning (SFT) with prompt formatting and response-only loss.
- A reusable evaluation harness covering reasoning, structured output,
  instruction following, RAG faithfulness, tool use, safety behavior, and
  latency.
- Reproducible scaling/ablation scripts with committed plots and local
  experiment tracking, plus optional W&B mirroring.
- A production-style RAG assistant with document ingestion, chunking,
  embeddings, vector search, reranking, citations, and refusal on insufficient
  evidence.
- A bounded research agent with tool use, deterministic routing, structured
  outputs, and cited answers.
- FastAPI and Streamlit front ends sharing one service layer.
- Tests, CI, Docker, typed configs, CLI commands, and professional docs.

## Technical Story

Most small LLM demos show only a notebook or one API call. Dork LLM demonstrates
the surrounding engineering needed to ship AI systems: model implementation,
measurement, retrieval grounding, tool orchestration, reproducible workflows,
and operator-friendly interfaces.

The project is intentionally honest about scale. The Tiny GPT is not a frontier
model; it is a compact vehicle for proving deep understanding of transformer
internals and the systems around modern LLM applications.

## Best Demo Path

```bash
make install
make smoke
make eval
make benchmark
make scaling-study
make ingest-docs
make query-rag Q="What does causal masking prevent?"
make run-agent TASK="Compare RAG systems and evaluation"
make api
make dashboard
```

## Interview Talking Points

- Why causal masking matters and how the model enforces it.
- How a KV cache changes generation cost while preserving greedy outputs.
- Why SFT uses response-only loss instead of re-learning the prompt.
- Why evals need multiple suites rather than one aggregate score.
- How citation provenance flows from document offsets to final answers.
- How local experiment tracking and committed plots support reproducible claims.
- Why deterministic fallbacks make LLM infrastructure testable.
- How the same orchestration layer supports CLI, scripts, API, and dashboard.
