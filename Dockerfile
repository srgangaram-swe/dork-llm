# syntax=docker/dockerfile:1
# AxiomStack CPU runtime. Local checkpoints are mounted at runtime, never copied
# from a developer workstation into the image.
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

# Install CPU-only PyTorch first so package resolution cannot pull CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1"

WORKDIR /build
COPY pyproject.toml README.md ./
COPY dork ./dork
RUN pip install ".[train,rag,eval,serve]"

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    DORK_MODEL_DEVICE=cpu

RUN groupadd --system axiom && useradd --system --gid axiom --create-home axiom

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=axiom:axiom apps ./apps
COPY --chown=axiom:axiom configs ./configs
COPY --chown=axiom:axiom data/sample_docs ./data/sample_docs

USER axiom

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "apps.api:app", "--host", "0.0.0.0", "--port", "8000"]
