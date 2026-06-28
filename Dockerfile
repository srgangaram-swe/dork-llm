# syntax=docker/dockerfile:1
# ── Dork LLM reproducible runtime ────────────────────────────────────────
# Builds a CPU-only image that can train the tiny GPT, run evals, serve the
# API, and host the dashboard. Kept slim; heavy GPU stacks are out of scope.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps: build tools for native wheels, git for datasets/tokenizers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first to avoid pulling CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1"

# Leverage layer caching: copy metadata, install deps, then copy source.
COPY pyproject.toml README.md ./
COPY dork ./dork
RUN pip install -e ".[train,rag,eval,serve]"

# Copy the rest of the project (configs, apps, scripts, sample data).
COPY . .

EXPOSE 8000 8501

# Default: serve the API. Override CMD to run training/eval/dashboard.
CMD ["uvicorn", "apps.api:app", "--host", "0.0.0.0", "--port", "8000"]
