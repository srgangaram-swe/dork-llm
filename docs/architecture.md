# Architecture

Dork LLM is one Python package (`dork`) with clearly separated subsystems, a thin
CLI and script layer, and a serving layer. This document explains how the pieces
fit together and the design decisions behind them.

## High-level data flow

```
                ┌──────────────────────────── dork (package) ───────────────────────────┐
 corpus ─▶ tokenizer ─▶ TinyGPT ─▶ training ─▶ checkpoint ─▶ generation                  │
                                                          │                              │
 docs ─▶ loaders ─▶ chunking ─▶ embeddings ─▶ vector store ─▶ retrieval ─▶ RAG answer    │
                                                          │           │                  │
                                                          ▼           ▼                  │
                                                       agents      evaluation            │
                └────────────────────────────────────────┬─────────────┬────────────────┘
                                                          ▼             ▼
                                                   serving (FastAPI) · dashboard (Streamlit)
```

## Modules

| Package | Responsibility |
|---|---|
| `dork.utils` | Config (pydantic), logging, seeding, paths, I/O, local experiment tracking. No ML deps. |
| `dork.data` | Corpus preparation (offline fallback) and memory-mapped token batching. |
| `dork.tokenizer` | `Tokenizer` interface; char + byte-level BPE backends; factory. |
| `dork.models` | `TinyGPT` and its transformer blocks (attention, GELU/SwiGLU MLPs, LayerNorm/RMSNorm, RoPE). |
| `dork.training` | Trainer loop, cosine LR schedule, checkpointing. |
| `dork.generation` | Sampling primitives, `Generator`, and `LanguageModel` providers. |
| `dork.evaluation` | Evaluator ABC + registry, suites, reporting, harness. |
| `dork.rag` | Loaders, chunking, embeddings, vector stores, reranker, pipeline. |
| `dork.agents` | Tools + the `ResearchAgent`. |
| `dork.serving` | Shared `DorkService` + metrics + API schemas. |
| `dork.pipelines` | High-level orchestration shared by CLI, scripts, and service. |

## Key design decisions

### 0. Small-model track with modern architecture switches
The default config remains intentionally tiny for laptop demos, but the model
code now supports a stronger training track inspired by modern decoder LLMs:
RoPE positional encoding, RMSNorm, SwiGLU feed-forward blocks, and per-layer
stochastic depth. These are exposed through typed config fields, so experiments
can compare baseline GPT-2-style blocks against the stronger profile without
forking model code.

`configs/dorkllm_frontier.yaml` is the canonical "make it actually better"
profile: TinyStories-oriented data, larger BPE vocabulary, longer context,
8-layer/512-width transformer, gradient accumulation, and separate SFT artifacts.

### 1. Local-first with graceful fallbacks
Every subsystem has a zero-dependency path so the platform runs and self-tests
offline, then upgrades transparently:

| Subsystem | Offline default | Production backend |
|---|---|---|
| Language model | `MockLanguageModel` (deterministic) | trained `TinyGPT` / HF model |
| Corpus | bundled public-domain text | TinyShakespeare/TinyStories download |
| Embeddings | `HashEmbedder` (hashing vectorizer) | sentence-transformers |
| Vector store | `MemoryVectorStore` (NumPy) | ChromaDB |

This makes CI fast and hermetic, lets reviewers run everything without a GPU, and
keeps the abstractions honest (each interface has ≥2 implementations).

### 2. One orchestration layer, four front-ends
`dork.pipelines` holds the workflow logic. The Typer CLI (`dork`), the `scripts/`
entry points (what the Makefile calls), the `DorkService` behind the API, and the
static Matrix chat web app all call the same serving/generation/RAG paths — no
duplicated logic, identical behavior everywhere.

### 3. Typed configs, validated at the edge
Training configs are fully typed pydantic models (`TinyGPTConfig`); a malformed
value fails at load time with a clear error, not deep in a training loop. Eval and
RAG configs are validated but permissive (`extra="allow"`) so they're easy to extend.

### 4. Lazy heavy imports
`torch` is imported lazily inside the functions that need it, so the eval/RAG/agent
stack imports and runs without the `[train]` extra installed. The package's public
API stays convenient without forcing a 2 GB dependency on every user.

### 5. Provenance everywhere
RAG chunks carry their source path and character offsets, so every citation points
back to an exact span. Checkpoints embed the model config and tokenizer path, so a
model is self-describing and reproducible.

### 6. Local-first experiment tracking
Training, SFT, evaluation, benchmarking, and scaling runs write JSON metadata,
metrics, and summaries under `experiments/`. W&B mirroring is optional and only
used when explicitly enabled, so CI and offline demos stay dependency-light.

## Request lifecycle (RAG query example)

1. `POST /rag/query` → `DorkService.rag_query`
2. `RagPipeline.retrieve`: embed query → vector search top-k → lexical rerank → score filter
3. If nothing clears the threshold and `refuse_when_insufficient` → **refuse**
4. Otherwise build a grounded prompt and call the configured `LanguageModel`
5. Extract `[n]` citations, map them back to source chunks, return `RagAnswer`

## Testing & CI

- **Unit tests** cover tokenizers, sampling, chunking, embeddings, the vector
  store, metrics, the agent tools, and the harness — all offline.
- **Integration tests** train a tiny model for a few steps and assert the loss
  decreases and a checkpoint round-trips.
- **`scripts/smoke_test.py`** exercises the *entire* platform end-to-end in CI.
- CI matrix runs on Python 3.11 and 3.12 with ruff, black, mypy, and pytest.

See [eval_harness.md](eval_harness.md), [rag_design.md](rag_design.md), and
[agent_design.md](agent_design.md) for subsystem deep-dives.
