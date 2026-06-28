# Limitations

Dork LLM is a compact LLM systems project. It is meant to be impressive because
it is complete, tested, reproducible, and honest, not because it pretends to be a
frontier model.

## Model Scale

The Tiny GPT is millions of parameters, not billions. It can learn local text
style and short continuation patterns, but it is not a general assistant, a
factual model, or an instruction-tuned system.

Expected weaknesses:

- repetitive generations;
- malformed words;
- weak long-range coherence;
- no reliable factual knowledge;
- no strong safety alignment;
- sensitivity to small-data training choices.

## Data

The default data is Tiny Shakespeare or a public-domain fallback excerpt. This is
useful for local training and architecture demonstration, but it does not cover
modern factual knowledge, instruction following, coding, multilingual behavior,
or production support conversations.

## Evaluation

The evaluation datasets are intentionally small and synthetic. They are useful
for regression tests and showing harness design, but they are not broad enough
to certify deployment readiness.

RAG faithfulness uses heuristic overlap and citation checks. It can catch some
obvious failures, but it is not a substitute for human review, entailment models,
or production monitoring.

## RAG

The default hash embedder is deterministic and offline-friendly, but semantic
retrieval quality is limited. For more realistic retrieval, install the `[rag]`
extra and use `sentence_transformers`.

The local vector store is simple and appropriate for demos. At larger scale, use
ChromaDB, FAISS, LanceDB, or a managed vector database with observability,
access control, and reindexing workflows.

## Agent

The agent uses deterministic routing rather than a general planner. This makes
it reliable and testable for a portfolio project, but it does not demonstrate
complex autonomous planning. The local Python execution tool is not a production
security boundary.

## Serving

The FastAPI service keeps metrics in memory and runs a single local process. A
production deployment would need persistent metrics, tracing, auth, rate limits,
queueing, autoscaling, model warmup, batching, and stronger error isolation.

## What This Project Demonstrates

It demonstrates practical LLM engineering:

- transformer internals;
- tokenizer and training pipeline;
- eval harness design;
- retrieval and citation plumbing;
- agent tool orchestration;
- API/dashboard serving;
- tests, CI, Docker, configs, and docs.

It does not demonstrate frontier-scale pretraining, RLHF, large distributed
training, proprietary data work, or production security hardening.
