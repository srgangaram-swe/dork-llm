# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **AxiomStack product hierarchy**: AxiomStack is the platform, DorkLLM the
  from-scratch model family, and DorkChat the browser research cockpit.
- **Modern-small DorkLLM track**: the compatibility-named
  `configs/dorkllm_frontier.yaml` profile adds RoPE, RMSNorm, SwiGLU, grouped-
  query attention, QK normalization, stochastic depth, gradient accumulation,
  and a dedicated SFT path.
- **Grouped-query attention**: configurable KV-head count with compact caches,
  legacy MHA compatibility, and cache-parity tests.
- **DorkChat research cockpit**: responsive, accessible streaming chat with
  runtime provenance, generation controls, cancellation, persistence, prompt
  starters, and evidence-rich citations.
- **Versioned streaming API**: typed SSE events, bounded conversations, request
  IDs, readiness/model metadata, and dependency-injected service tests.
- **Full-stack verification**: frontend unit tests, Playwright browser
  integration, API dependencies in CI, Python 3.11–3.13, and dev/main/prod
  branch triggers.
- **Forward delivery roadmap**: five GitHub milestones and 26 assigned issues
  covering statistics, deep-learning systems, production grounding, and v1.0.
- **KV-cache inference**: incremental decoding with a per-layer key/value cache,
  numerically identical to the reference path and ~9× faster generation on CPU.
- **Supervised fine-tuning (SFT)**: instruction-tuning with a prompt template and
  causal next-token response-only loss masking; `dork sft` / `make sft`.
- **Reproducible scaling study**: `scripts/scaling_study.py` trains tiny GPTs
  across sizes, fits a power law (loss ~ params^b), and emits a committed plot.
- **Experiment tracking**: local JSON metadata/metrics/summaries for train, SFT,
  eval, benchmark, and scaling runs, with optional W&B mirroring.
- **Notebook demos**: train, evaluate, and RAG walkthroughs under `notebooks/`.
- Community health files: CONTRIBUTING, SECURITY, CITATION.cff, CODE_OF_CONDUCT,
  issue/PR templates.

### Fixed
- SFT now predicts the next response token instead of the same-position input;
  tests no longer write checkpoints into the developer artifact directory.
- Token-overlap F1 now uses bounded multiset intersection and cannot exceed 1.
- Cached multi-token attention now uses an offset-aware causal mask.
- Chat no longer duplicates the newest user turn in its model prompt.
- Runtime model failures are reported as readiness degradation; mock fallback
  requires explicit demo mode and is never labeled as DorkLLM.
- Model loader rejects tokenizer/checkpoint vocabulary mismatch before serving.
- Removed a duplicated README section header.

## [0.1.0] - 2026-07-01

### Added
- Initial platform: tiny GPT (decoder-only transformer) trained from scratch,
  a reusable evaluation harness with CI gating, and a cited RAG + agentic
  research assistant, served via FastAPI and a Streamlit dashboard.
- Tokenizers (char + byte-level BPE), typed configs, CLI, tests, Docker, CI.
