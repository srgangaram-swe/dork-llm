"""High-level orchestration used by both the CLI and the ``scripts/`` entry points.

Keeping the workflow logic here (rather than duplicated across scripts) means the
``dork`` CLI, the standalone scripts, the FastAPI service and the dashboard all
share one implementation. Heavy imports (torch) are deferred into the functions
that need them so the light-weight workflows (eval/RAG/agent) run without the
``[train]`` extra installed.
"""

from __future__ import annotations

from typing import Any

from dork.utils.config import (
    load_eval_config,
    load_rag_config,
    load_tiny_gpt_config,
)
from dork.utils.logging import get_logger
from dork.utils.seed import seed_everything

logger = get_logger(__name__)


# ───────────────────────── Tiny GPT pipeline ─────────────────────────
def prepare_data(config_path: str) -> dict[str, Any]:
    """Download/prepare the corpus and report basic stats."""
    cfg = load_tiny_gpt_config(config_path)
    seed_everything(cfg.seed)
    from dork.data.datasets import prepare_corpus

    path = prepare_corpus(cfg.data)
    text = path.read_text(encoding="utf-8")
    stats = {"corpus_path": str(path), "chars": len(text), "lines": text.count("\n") + 1}
    logger.info("Prepared corpus: %s", stats)
    return stats


def train_tokenizer(config_path: str) -> dict[str, Any]:
    """Train (or rebuild) the tokenizer on the prepared corpus."""
    cfg = load_tiny_gpt_config(config_path)
    seed_everything(cfg.seed)
    from dork.data.datasets import prepare_corpus
    from dork.tokenizer.factory import train_tokenizer as _train

    text = prepare_corpus(cfg.data).read_text(encoding="utf-8")
    tok = _train(cfg.tokenizer, text)
    path = tok.save(cfg.tokenizer.path)
    return {"tokenizer_path": str(path), "vocab_size": tok.vocab_size, "type": cfg.tokenizer.type}


def train_model(config_path: str) -> dict[str, Any]:
    """Run the full tiny-GPT training pipeline end to end."""
    cfg = load_tiny_gpt_config(config_path)
    seed_everything(cfg.seed)

    from dork.data.datasets import prepare_corpus
    from dork.data.loader import BinTokenDataset, build_token_bins
    from dork.models.tiny_gpt import TinyGPT
    from dork.tokenizer.factory import load_or_train_tokenizer
    from dork.training.trainer import Trainer

    text = prepare_corpus(cfg.data).read_text(encoding="utf-8")
    tokenizer = load_or_train_tokenizer(cfg.tokenizer, text)
    # Sync the real vocab (BPE may round to a slightly different size).
    cfg.model.vocab_size = tokenizer.vocab_size

    bins_dir = f"{cfg.training.out_dir}/data"
    meta = build_token_bins(text, tokenizer.encode, bins_dir, cfg.data.val_fraction)
    train_ds = BinTokenDataset(str(meta["train_bin"]), cfg.model.block_size)
    val_ds = BinTokenDataset(str(meta["val_bin"]), cfg.model.block_size)

    model = TinyGPT(cfg.model)
    trainer = Trainer(model, train_ds, val_ds, cfg.training, tokenizer_path=cfg.tokenizer.path)
    history = trainer.train()
    best_val = min((h["val"] for h in history), default=float("nan"))
    return {
        "params": model.num_params(),
        "best_val_loss": best_val,
        "steps": cfg.training.max_steps,
        "out_dir": cfg.training.out_dir,
        "device": trainer.device,
    }


def generate(config_path: str, prompt: str, **overrides: Any) -> str:
    """Generate a continuation from the trained model."""
    cfg = load_tiny_gpt_config(config_path)
    from dork.generation.generator import Generator
    from dork.tokenizer.factory import load_tokenizer
    from dork.training.checkpoint import load_model_from_checkpoint
    from dork.training.trainer import resolve_device

    device = resolve_device(cfg.training.device)
    model, payload = load_model_from_checkpoint(cfg.training.out_dir, device=device)
    tok_path = payload.get("tokenizer_path") or cfg.tokenizer.path
    tokenizer = load_tokenizer(tok_path)
    gen = Generator(model, tokenizer, device=device)

    gcfg = cfg.generation.model_copy(update={k: v for k, v in overrides.items() if v is not None})
    return gen.generate(prompt, cfg=gcfg)


def benchmark(config_path: str, n_requests: int = 20) -> dict[str, Any]:
    """Benchmark generation latency/throughput and the KV-cache speedup.

    Times the same decode both with the KV cache and with the O(T²) reference
    path so the report quantifies the inference optimization, not just raw speed.
    """
    import time

    cfg = load_tiny_gpt_config(config_path)
    from dork.generation.generator import Generator
    from dork.tokenizer.factory import load_tokenizer
    from dork.training.checkpoint import load_model_from_checkpoint
    from dork.training.trainer import resolve_device

    device = resolve_device(cfg.training.device)
    model, payload = load_model_from_checkpoint(cfg.training.out_dir, device=device)
    tokenizer = load_tokenizer(payload.get("tokenizer_path") or cfg.tokenizer.path)
    gen = Generator(model, tokenizer, device=device)

    prompt = "Once upon a time"
    tokens = cfg.generation.max_new_tokens
    gen.generate(prompt, max_new_tokens=8, temperature=0.8)  # warmup

    def _time(use_cache: bool) -> list[float]:
        lat = []
        for _ in range(n_requests):
            t0 = time.perf_counter()
            gen.generate(prompt, max_new_tokens=tokens, temperature=0.8, use_cache=use_cache)
            lat.append(time.perf_counter() - t0)
        return sorted(lat)

    def _summary(lat: list[float]) -> dict[str, float]:
        mean_s = sum(lat) / len(lat)
        return {
            "mean_ms": mean_s * 1000,
            "p50_ms": lat[len(lat) // 2] * 1000,
            "p95_ms": lat[min(len(lat) - 1, int(0.95 * len(lat)))] * 1000,
            "tokens_per_sec": tokens / mean_s if mean_s else float("inf"),
        }

    kv = _summary(_time(use_cache=True))
    ref = _summary(_time(use_cache=False))
    return {
        "device": device,
        "params": model.num_params(),
        "n_requests": n_requests,
        "new_tokens": tokens,
        "kv_cache": kv,
        "reference": ref,
        "speedup": ref["mean_ms"] / kv["mean_ms"] if kv["mean_ms"] else float("inf"),
        # Flattened aliases for backwards-compatible dashboards.
        "mean_ms": kv["mean_ms"],
        "tokens_per_sec": kv["tokens_per_sec"],
    }


# ───────────────────────── Evaluation pipeline ───────────────────────
def run_eval(config_path: str) -> dict[str, Any]:
    """Run the evaluation harness and write reports."""
    from dork.evaluation.harness import EvalHarness

    cfg = load_eval_config(config_path)
    return EvalHarness(cfg).run()


# ─────────────────────────── RAG pipeline ────────────────────────────
def ingest(config_path: str, source: str | None = None) -> dict[str, Any]:
    """Ingest documents into the vector store."""
    from dork.rag.pipeline import RagPipeline

    cfg = load_rag_config(config_path)
    seed_everything(cfg.seed)
    return RagPipeline(cfg).ingest(source).to_dict()


def query_rag(config_path: str, question: str) -> dict[str, Any]:
    """Answer a question with the RAG pipeline (re-ingests for a fresh index)."""
    from dork.rag.pipeline import RagPipeline

    cfg = load_rag_config(config_path)
    seed_everything(cfg.seed)
    pipeline = RagPipeline(cfg)
    if pipeline.store.count() == 0:
        pipeline.ingest()
    return pipeline.query(question).to_dict()


def run_agent(config_path: str, task: str) -> dict[str, Any]:
    """Run the research agent on a task."""
    from dork.agents.research_agent import ResearchAgent
    from dork.rag.pipeline import RagPipeline

    cfg = load_rag_config(config_path)
    seed_everything(cfg.seed)
    pipeline = RagPipeline(cfg)
    if pipeline.store.count() == 0:
        pipeline.ingest()
    agent_cfg = cfg.agent or {}
    agent = ResearchAgent(
        pipeline,
        max_steps=int(agent_cfg.get("max_steps", 6)),
        allow_code_exec=bool(agent_cfg.get("allow_code_exec", True)),
    )
    return agent.run(task).to_dict()
