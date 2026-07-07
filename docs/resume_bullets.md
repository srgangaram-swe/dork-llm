# Resume Bullets

Use or adapt these bullets depending on the role.

## LLM Engineer / Research Engineer

- Implemented a compact GPT-style decoder-only transformer from scratch in
  PyTorch, including token and positional embeddings, causal multi-head
  attention, residual blocks, LayerNorm, weight tying, AdamW, cosine scheduling,
  checkpointing, KV-cache decoding, and temperature/top-k/top-p generation.
- Built a reproducible tiny language-model training pipeline with public-data
  ingestion, tokenizer training, binary token datasets, validation loss tracking,
  checkpoint metadata, deterministic seeds, and CPU/GPU device selection.
- Added supervised fine-tuning (SFT) with instruction prompts, response-only
  loss masking, before/after response perplexity, and a dedicated checkpoint
  path.
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
- Served model, evaluation, RAG, and agent workflows through FastAPI and a
  Streamlit dashboard backed by a shared service layer and in-memory metrics.

## AI Systems / Infrastructure

- Packaged an end-to-end LLM platform with typed YAML configs, Typer CLI,
  Makefile workflows, Docker, GitHub Actions CI, pytest coverage, Ruff, Black,
  mypy, and pre-commit hooks.
- Added local JSON experiment tracking with optional W&B mirroring for training,
  SFT, evaluation, KV-cache benchmarking, and scaling studies.
- Designed local-first fallbacks for heavy LLM dependencies, enabling offline
  CI and deterministic smoke tests while preserving upgrade paths to PyTorch,
  Hugging Face models, sentence-transformers, and ChromaDB.
- Authored production-style architecture, model card, evaluation, RAG, agent,
  limitations, notebooks, and portfolio documentation for a clean-room public
  GitHub project.
