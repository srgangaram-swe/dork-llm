# Dork LLM — developer workflow commands.
# Run `make help` to see all targets.

.DEFAULT_GOAL := help
PY ?= python
PIP ?= $(PY) -m pip
PKG := dork

# Config file defaults (override on the command line, e.g. `make train-small-gpt TRAIN_CONFIG=...`)
TOKENIZER_CONFIG ?= configs/train_tiny_gpt.yaml
TRAIN_CONFIG     ?= configs/train_tiny_gpt.yaml
FRONTIER_CONFIG  ?= configs/dorkllm_frontier.yaml
EVAL_CONFIG      ?= configs/eval_default.yaml
RAG_CONFIG       ?= configs/rag_default.yaml

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ───────────────────────────── Setup ─────────────────────────────
.PHONY: install
install: ## Install the package with all extras (editable)
	$(PIP) install -e ".[all]"
	pre-commit install || true

.PHONY: install-core
install-core: ## Install only the lightweight core (no torch/rag/serve)
	$(PIP) install -e ".[dev]"

# ──────────────────────── Code quality ───────────────────────────
.PHONY: lint
lint: ## Lint with ruff
	ruff check $(PKG) scripts apps tests

.PHONY: format
format: ## Auto-format with black + ruff --fix
	black $(PKG) scripts apps tests
	ruff check --fix $(PKG) scripts apps tests

.PHONY: format-check
format-check: ## Check formatting with black
	black --check $(PKG) scripts apps tests

.PHONY: typecheck
typecheck: ## Static type check with mypy
	mypy $(PKG)

.PHONY: test
test: ## Run the test suite
	pytest

.PHONY: test-fast
test-fast: ## Run tests excluding slow ones
	pytest -m "not slow"

.PHONY: check
check: lint format-check typecheck test ## Run lint + format check + typecheck + tests (CI parity)

# ──────────────────────── ML pipelines ───────────────────────────
.PHONY: prepare-data
prepare-data: ## Download/prepare the training corpus
	$(PY) scripts/prepare_dataset.py --config $(TRAIN_CONFIG)

.PHONY: train-tokenizer
train-tokenizer: ## Train the BPE tokenizer
	$(PY) scripts/train_tokenizer.py --config $(TOKENIZER_CONFIG)

.PHONY: train-small-gpt
train-small-gpt: ## Train the tiny GPT from scratch
	$(PY) scripts/train_tiny_gpt.py --config $(TRAIN_CONFIG)

.PHONY: train-frontier
train-frontier: ## Train the stronger dorkLLM profile (TinyStories + RMSNorm/SwiGLU/RoPE)
	$(PY) scripts/prepare_dataset.py --config $(FRONTIER_CONFIG)
	$(PY) scripts/train_tokenizer.py --config $(FRONTIER_CONFIG)
	$(PY) scripts/train_tiny_gpt.py --config $(FRONTIER_CONFIG)

.PHONY: sft
sft: ## Instruction-tune (SFT) the base model
	$(PY) scripts/finetune_sft.py --config $(TRAIN_CONFIG)

.PHONY: generate
generate: ## Generate text from the trained model
	$(PY) scripts/generate_text.py --config $(TRAIN_CONFIG) --prompt "Once upon a time"

.PHONY: eval
eval: ## Run the evaluation harness
	$(PY) scripts/evaluate_model.py --config $(EVAL_CONFIG)

# ──────────────────────────── RAG ────────────────────────────────
.PHONY: ingest-docs
ingest-docs: ## Ingest documents into the vector store
	$(PY) scripts/ingest_documents.py --config $(RAG_CONFIG) --source data/sample_docs

.PHONY: query-rag
query-rag: ## Query the RAG system (set Q="your question")
	$(PY) scripts/query_rag.py --config $(RAG_CONFIG) --question "$(Q)"

.PHONY: run-agent
run-agent: ## Run the agentic research assistant (set TASK="...")
	$(PY) scripts/run_agent.py --config $(RAG_CONFIG) --task "$(TASK)"

.PHONY: benchmark
benchmark: ## Benchmark inference latency/throughput
	$(PY) scripts/benchmark_inference.py --config $(TRAIN_CONFIG)

.PHONY: benchmark-inference
benchmark-inference: benchmark ## Alias for `make benchmark`

.PHONY: benchmark_inference
benchmark_inference: benchmark ## Alias for `make benchmark`

.PHONY: scaling-study
scaling-study: ## Run the reproducible scaling study (params vs. loss + plot)
	$(PY) scripts/scaling_study.py

.PHONY: experiments
experiments: ## List local experiment tracking runs
	$(PY) -m dork.utils.tracking --out-dir experiments

.PHONY: smoke
smoke: ## Run the end-to-end smoke test used by CI
	$(PY) scripts/smoke_test.py

# ─────────────────────────── Serving ─────────────────────────────
.PHONY: api
api: ## Run the FastAPI service
	uvicorn apps.api:app --reload --host 0.0.0.0 --port 8000

.PHONY: web
web: ## Run the Matrix chat web app + API
	uvicorn apps.api:app --reload --host 127.0.0.1 --port 8790

.PHONY: dashboard
dashboard: ## Run the Streamlit dashboard
	streamlit run apps/dashboard.py

# ─────────────────────────── Docker ──────────────────────────────
.PHONY: docker-build
docker-build: ## Build the Docker image
	docker build -t dork-llm:latest .

.PHONY: docker-run
docker-run: ## Run the API inside Docker
	docker run --rm -p 8000:8000 dork-llm:latest

# ─────────────────────────── Cleanup ─────────────────────────────
.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
