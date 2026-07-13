# Resume Bullets

Use or adapt these bullets depending on the role.

## LLM Engineer / Research Engineer

- Implemented a compact GPT-style decoder-only transformer from scratch in
  PyTorch, including causal multi-head/grouped-query attention, RoPE, RMSNorm,
  QK normalization, SwiGLU, weight tying, AdamW, compact KV-cache decoding, and
  temperature/top-k/top-p generation.
- Built a reproducible tiny language-model training pipeline with public-data
  ingestion, tokenizer training, binary token datasets, validation loss tracking,
  checkpoint metadata, deterministic seeds, and CPU/GPU device selection.
- Added supervised fine-tuning (SFT) with instruction prompts, response-only
  causal next-token loss masking, response-preserving truncation, and an
  isolated checkpoint path; added regression tests for label alignment.
- Created a reusable LLM evaluation harness spanning perplexity, exact match,
  multiple choice, instruction following, JSON validity, RAG faithfulness, tool
  use, safety/refusal behavior, and latency/throughput reporting.

## Applied AI / RAG / Agents

- Built a source-grounded RAG assistant with Markdown/text/PDF ingestion,
  document chunking, deterministic embeddings, local vector search, lexical
  reranking, citation tracking, and refusal when retrieved evidence is
  insufficient.
- Implemented an agentic research assistant that routes tasks to search,
  summarization, comparison, claim extraction, experiment planning, calculator,
  and restricted local-code tools while returning structured outputs and
  citations.
- Served model, evaluation, RAG, and agent workflows through a typed FastAPI/SSE
  contract, an accessible browser chat client, and a Streamlit dashboard backed
  by a shared service layer with explicit runtime provenance.

## AI Systems / Infrastructure

- Packaged an end-to-end LLM platform with typed configs, Typer CLI, Makefile
  workflows, a non-root container, GitHub Actions across Python 3.11–3.13,
  pytest, frontend unit tests, Playwright, Ruff, Black, and mypy.
- Added local JSON experiment tracking with optional W&B mirroring for training,
  SFT, evaluation, KV-cache benchmarking, and scaling studies.
- Designed strict model readiness plus an explicit labeled demo provider,
  enabling offline CI without silently substituting a mock for a trained model.
- Authored production-style architecture, model card, evaluation, RAG, agent,
  limitations, notebooks, and portfolio documentation for a clean-room public
  GitHub project.
