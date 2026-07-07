# LinkedIn Post Draft

I built Dork LLM, an end-to-end LLM systems project that brings together three
parts of the stack I wanted to understand deeply:

1. A small GPT-style language model trained from scratch in PyTorch.
2. A reusable LLM evaluation harness.
3. A RAG + agentic research assistant with citations.

The tiny GPT is intentionally compact. It is not trying to be a frontier model.
The point is to implement the internals directly: causal self-attention,
transformer blocks, tokenization, checkpointing, validation, and text generation
with temperature/top-k/top-p sampling. I also added KV-cache decoding and a
benchmark that compares it against the reference generation path.

The post-training path demonstrates supervised fine-tuning with an instruction
template and response-only loss masking. At this scale it is about showing the
mechanism clearly, not pretending a tiny model becomes a polished assistant.

The eval harness is the part I think matters most in real AI systems. It covers
perplexity, exact match, multiple choice, JSON validity, instruction following,
RAG faithfulness, tool use, refusal behavior, and latency. It writes JSON, CSV,
Markdown, and plots so results can be used by humans, CI, and dashboards.

The RAG side ingests local documents, chunks them, embeds them, retrieves and
reranks evidence, and answers with citations. The research agent can summarize,
compare, extract claims, draft experiment plans, calculate, and refuse when the
evidence is not there.

I also treated it like a real software project: typed configs, tests, CI, Docker,
FastAPI, Streamlit, CLI commands, notebooks, local experiment tracking with
optional W&B, a model card, architecture docs, limitations, and reproducible
local workflows.

This was a compact project, but it touched a lot of the engineering that makes
LLM products reliable: internals, evaluation, grounding, agents, serving, and
developer experience.

Repository: https://github.com/srgangaram-swe/dork-llm
