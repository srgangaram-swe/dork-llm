# LinkedIn Post Draft

I built AxiomStack — “Proof. Probability. Production.” — an end-to-end language-
model systems project that brings together three parts of the stack:

1. DorkLLM, a small GPT-style model trained from scratch in PyTorch.
2. A reusable LLM evaluation harness.
3. DorkChat, a full-stack RAG and model research cockpit with citations.

The tiny GPT is intentionally compact. It is not trying to be a frontier model.
The point is to implement the internals directly: causal self-attention,
transformer blocks, tokenization, checkpointing, validation, and sampling. I
also added grouped-query attention, QK normalization, compact KV-cache decoding,
and numerical comparisons against the reference generation path.

The post-training path demonstrates causal supervised fine-tuning with an
instruction template and next-token response-only loss. At this scale it is
about showing the mechanism correctly, not pretending a tiny model becomes a
polished assistant.

The eval harness is the part I think matters most in real AI systems. It covers
perplexity, exact match, multiple choice, JSON validity, instruction following,
RAG faithfulness, tool use, refusal behavior, and latency. It writes JSON, CSV,
Markdown, and plots so results can be used by humans, CI, and dashboards.

The RAG side ingests local documents, chunks them, embeds them, retrieves and
reranks evidence, and answers with citations. The research agent can summarize,
compare, extract claims, draft experiment plans, calculate, and refuse when the
evidence is not there.

I also treated it like a real software project: typed contracts, FastAPI and
server-sent events, an accessible browser client, Python unit/integration tests,
frontend unit tests, Playwright, CI, a non-root container, local experiment
tracking, model/system cards, and an explicit limitations section.

This was a compact project, but it touched a lot of the engineering that makes
LLM products reliable: internals, evaluation, grounding, agents, serving, and
developer experience.

Repository: https://github.com/srgangaram-swe/dork-llm
