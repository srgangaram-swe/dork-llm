# Agent Design

## Purpose

The research agent wraps the RAG pipeline with bounded tool use. It can answer,
summarize, compare, extract claims, draft experiment plans, and calculate simple
arithmetic. It is built to demonstrate agentic AI engineering without depending
on an unreliable open-ended planner.

## Entry Points

```bash
make run-agent TASK="Summarize the transformers document"
dork agent --task "Compare RAG systems and evaluation"
```

The FastAPI endpoint is `POST /agent/run`.

## Routing

`ResearchAgent.classify_intent` uses deterministic routing:

- arithmetic expressions -> `calculate`
- "compare", "vs", "difference between" -> `compare`
- "experiment plan", "ablation" -> `experiment_plan`
- "claim", "extract", "key points" -> `extract_claims`
- "summarize", "overview", "tl;dr" -> `summarize`
- fallback -> `answer`

This is deliberate. Deterministic routing is easier to test and gives a stable
baseline. A future LLM planner can reuse the same tools and result schema.

## Tools

Standalone tools:

- `calculator`: AST-based arithmetic evaluator. It never calls Python `eval`.
- `python_exec`: restricted import-free Python snippet runner for local demos.

RAG-bound capabilities:

- search indexed docs;
- summarize retrieved chunks;
- compare retrieved evidence;
- extract claims;
- generate experiment plans.

The Python tool is a convenience sandbox, not a security boundary. It blocks
imports, dunder access, file IO patterns, `eval`, `exec`, and common unsafe
constructs, but it should not be exposed to untrusted users in production.

## Result Schema

Every run returns an `AgentResult`:

- `task`: original request.
- `intent`: routed intent.
- `answer`: final response.
- `steps`: tool trajectory.
- `citations`: source-backed evidence.
- `structured`: optional JSON-like payload.
- `refused`: whether evidence was insufficient.

This makes the agent inspectable in tests, the API, and the dashboard.

## Safety and Grounding

The agent delegates factual document questions to the RAG pipeline. If retrieval
does not find enough evidence, the RAG pipeline refuses instead of inventing an
answer. Calculation tasks do not require retrieval and return a direct tool
observation.

## Future Work

- Add an LLM planner behind the same `Tool` interface.
- Add step budgets, retry policies, and structured tool-call validation.
- Add stronger sandboxing or remove local code execution for deployed use.
- Add citation-level claim verification.
