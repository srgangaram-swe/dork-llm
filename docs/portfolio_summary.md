# AxiomStack portfolio summary

AxiomStack is an end-to-end language-model systems project that combines model
mathematics, statistical evaluation, retrieval, agents, and full-stack serving
in one cohesive codebase. DorkLLM is the from-scratch model; DorkChat is the
browser research cockpit.

## What It Shows

- A GPT-style decoder-only transformer implemented from scratch in PyTorch.
- Tokenizer training, binary token dataset preparation, training loop,
  validation, checkpointing, and text generation.
- Multi-head/grouped-query attention, QK normalization, and compact KV-cache
  inference with numerical parity tests against the reference path.
- Causal supervised fine-tuning with true next-token response-only loss.
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
- A typed FastAPI/SSE contract, accessible DorkChat UI, and Streamlit research
  dashboard sharing one service layer.
- Python unit/integration tests, frontend unit tests, Playwright browser tests,
  CI, a non-root container, typed configs, CLI commands, and design docs.

## Technical Story

Most small LLM demos show only a notebook or one API call. AxiomStack demonstrates
the surrounding engineering needed to ship AI systems: model implementation,
measurement, retrieval grounding, tool orchestration, reproducible workflows,
and operator-friendly interfaces.

The project is intentionally honest about scale. DorkLLM is not a frontier
model; it is a compact vehicle for proving understanding of transformer
internals and the systems around modern LLM applications.

## Best Demo Path

```bash
make install
make smoke
make check
make eval
make benchmark
make scaling-study
make ingest-docs
make query-rag Q="What does causal masking prevent?"
make run-agent TASK="Compare RAG systems and evaluation"
make web-demo
```

## Interview Talking Points

- Why causal masking matters and how the model enforces it.
- Why GQA changes projection and cache memory without changing query-head count.
- How a KV cache changes generation cost while preserving greedy outputs.
- Why causal SFT shifts targets and masks prompt labels rather than learning
  same-position identity.
- Why evals need multiple suites rather than one aggregate score.
- How citation provenance flows from document offsets to final answers.
- How local experiment tracking and committed plots support reproducible claims.
- Why explicit, labeled demo providers make LLM infrastructure testable without
  corrupting model-readiness claims.
- How the same orchestration layer supports CLI, scripts, API, and dashboard.
