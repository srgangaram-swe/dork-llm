# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Stronger dorkLLM track**: `configs/dorkllm_frontier.yaml` adds a
  TinyStories-oriented profile with longer context, 8k BPE, RMSNorm, SwiGLU,
  RoPE, stochastic depth, gradient accumulation, and a dedicated SFT path.
- **Modern block variants**: `TinyGPT` now supports LayerNorm/RMSNorm,
  GELU/SwiGLU feed-forward blocks, and per-layer stochastic-depth scheduling.
- **Matrix chat web app**: `make web` serves a polished purple chat UI at `/`
  backed by the new `/chat` endpoint with RAG-first and generation fallback.
- **KV-cache inference**: incremental decoding with a per-layer key/value cache,
  numerically identical to the reference path and ~9× faster generation on CPU.
- **Supervised fine-tuning (SFT)**: instruction-tuning with a prompt template and
  response-only loss masking; `dork sft` / `make sft`. Demonstrates the
  pretrain→finetune paradigm (held-out response perplexity drops sharply).
- **Reproducible scaling study**: `scripts/scaling_study.py` trains tiny GPTs
  across sizes, fits a power law (loss ~ params^b), and emits a committed plot.
- **Experiment tracking**: local JSON metadata/metrics/summaries for train, SFT,
  eval, benchmark, and scaling runs, with optional W&B mirroring.
- **Notebook demos**: train, evaluate, and RAG walkthroughs under `notebooks/`.
- Community health files: CONTRIBUTING, SECURITY, CITATION.cff, CODE_OF_CONDUCT,
  issue/PR templates.

### Fixed
- Model loader now rejects a tokenizer/model vocab mismatch at load time, so the
  service falls back to the mock model instead of crashing during generation.
- Removed a duplicated README section header.

## [0.1.0] - 2026-07-01

### Added
- Initial platform: tiny GPT (decoder-only transformer) trained from scratch,
  a reusable evaluation harness with CI gating, and a cited RAG + agentic
  research assistant, served via FastAPI and a Streamlit dashboard.
- Tokenizers (char + byte-level BPE), typed configs, CLI, tests, Docker, CI.
