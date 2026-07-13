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
from dork.utils.tracking import start_tracker

logger = get_logger(__name__)


def _synchronize_device(device: str) -> None:
    """Wait for queued accelerator work before reading the benchmark clock."""
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()
    elif device.startswith("mps"):
        import torch

        torch.mps.synchronize()


def _latency_summary(latencies_s: list[float], new_tokens: int) -> dict[str, float]:
    """Summarize positive request latencies without discarding pair order."""
    import numpy as np

    if not latencies_s or any(value <= 0.0 for value in latencies_s):
        raise ValueError("latencies must be a non-empty sequence of positive values")
    latencies = np.asarray(latencies_s, dtype=np.float64)
    mean_s = float(np.mean(latencies))
    p50_s, p95_s = np.quantile(latencies, [0.5, 0.95])
    std_s = float(np.std(latencies, ddof=1)) if latencies.size > 1 else 0.0
    return {
        "mean_ms": mean_s * 1000.0,
        "p50_ms": float(p50_s) * 1000.0,
        "p95_ms": float(p95_s) * 1000.0,
        "std_ms": std_s * 1000.0,
        "coefficient_of_variation": std_s / mean_s,
        "tokens_per_sec": new_tokens / mean_s,
    }


def _eval_summary_metrics(report: dict[str, Any]) -> dict[str, float]:
    """Flatten eval summary rows into ``suite.metric -> value`` scalars."""
    metrics: dict[str, float] = {}
    for row in report.get("summary", []):
        suite = row.get("suite")
        metric = row.get("metric")
        value = row.get("value")
        if suite and metric and isinstance(value, (int, float)):
            metrics[f"{suite}.{metric}"] = float(value)
    return metrics


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
    tracker = start_tracker(cfg.tracking, "train-tiny-gpt", config=cfg, tags=["train"])
    trainer = Trainer(
        model,
        train_ds,
        val_ds,
        cfg.training,
        tokenizer_path=cfg.tokenizer.path,
        tracker=tracker,
    )
    history = trainer.train()
    best_val = min((h["val"] for h in history), default=float("nan"))
    result = {
        "params": model.num_params(),
        "best_val_loss": best_val,
        "steps": cfg.training.max_steps,
        "out_dir": cfg.training.out_dir,
        "device": trainer.device,
    }
    if tracker is not None:
        result["tracking_run_dir"] = str(tracker.run_dir)
        tracker.finish(result)
    return result


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


def benchmark(
    config_path: str,
    n_requests: int = 20,
    *,
    warmup_requests: int = 2,
    bootstrap_resamples: int = 5_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Benchmark generation latency/throughput and the KV-cache speedup.

    The protocol runs identical seeded decodes as adjacent pairs and alternates
    which implementation runs first.  This controls generation workload and
    reduces bias from thermal or background-load drift.  CUDA and MPS queues are
    synchronized around every timed region.
    """
    import time

    if isinstance(n_requests, bool) or not isinstance(n_requests, int) or n_requests <= 0:
        raise ValueError("n_requests must be a positive integer")
    if (
        isinstance(warmup_requests, bool)
        or not isinstance(warmup_requests, int)
        or warmup_requests <= 0
    ):
        raise ValueError("warmup_requests must be a positive integer")

    cfg = load_tiny_gpt_config(config_path)
    from dork.evaluation.statistics import paired_bootstrap
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
    warmup_tokens = min(tokens, 8)

    def _generate(*, use_cache: bool, new_tokens: int, seed: int) -> str:
        return gen.generate(
            prompt,
            max_new_tokens=new_tokens,
            temperature=0.8,
            seed=seed,
            use_cache=use_cache,
        )

    # Warm both implementations equally.  Alternate order here and below so a
    # consistently first implementation does not inherit a systematic bias.
    for warmup_index in range(warmup_requests):
        modes = (True, False) if warmup_index % 2 == 0 else (False, True)
        for use_cache in modes:
            _generate(
                use_cache=use_cache,
                new_tokens=warmup_tokens,
                seed=int(cfg.seed) + warmup_index,
            )
            _synchronize_device(device)

    kv_latencies: list[float] = []
    reference_latencies: list[float] = []
    pairs: list[dict[str, Any]] = []
    for pair_index in range(n_requests):
        order = ("kv_cache", "reference") if pair_index % 2 == 0 else ("reference", "kv_cache")
        pair_seed = int(cfg.seed) + warmup_requests + pair_index
        durations: dict[str, float] = {}
        outputs: dict[str, str] = {}
        for mode in order:
            _synchronize_device(device)
            start_ns = time.perf_counter_ns()
            outputs[mode] = _generate(
                use_cache=mode == "kv_cache",
                new_tokens=tokens,
                seed=pair_seed,
            )
            _synchronize_device(device)
            durations[mode] = (time.perf_counter_ns() - start_ns) / 1_000_000_000.0

        kv_s = durations["kv_cache"]
        reference_s = durations["reference"]
        if kv_s <= 0.0 or reference_s <= 0.0:
            raise RuntimeError("benchmark clock did not advance during generation")
        kv_latencies.append(kv_s)
        reference_latencies.append(reference_s)
        pairs.append(
            {
                "pair": pair_index + 1,
                "seed": pair_seed,
                "order": list(order),
                "kv_cache_ms": kv_s * 1000.0,
                "reference_ms": reference_s * 1000.0,
                "reference_minus_kv_ms": (reference_s - kv_s) * 1000.0,
                "speedup": reference_s / kv_s,
                "outputs_match": outputs["kv_cache"] == outputs["reference"],
            }
        )

    kv = _latency_summary(kv_latencies, tokens)
    ref = _latency_summary(reference_latencies, tokens)
    speedup = paired_bootstrap(
        reference_latencies,
        kv_latencies,
        statistic="ratio_of_means",
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    savings = paired_bootstrap(
        reference_latencies,
        kv_latencies,
        statistic="mean_delta",
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    result = {
        "device": device,
        "params": model.num_params(),
        "n_requests": n_requests,
        "new_tokens": tokens,
        "kv_cache": kv,
        "reference": ref,
        "speedup": speedup.estimate,
        "paired_speedup": speedup.as_dict(),
        "paired_latency_savings_ms": {
            **savings.as_dict(),
            "estimate": savings.estimate * 1000.0,
            "ci_low": savings.ci_low * 1000.0,
            "ci_high": savings.ci_high * 1000.0,
            "unit": "ms",
        },
        "pairs": pairs,
        "protocol": {
            "clock": "time.perf_counter_ns",
            "warmup_requests_per_implementation": warmup_requests,
            "warmup_new_tokens": warmup_tokens,
            "paired_interleaving": True,
            "counterbalanced_order": True,
            "identical_seed_within_pair": True,
            "device_synchronization": device.startswith(("cuda", "mps")),
        },
        # Backwards-compatible throughput alias; mean latency lives under
        # ``kv_cache`` only so the report has a single authoritative mean_ms.
        "tokens_per_sec": kv["tokens_per_sec"],
    }
    tracker = start_tracker(cfg.tracking, "benchmark-inference", config=cfg, tags=["benchmark"])
    if tracker is not None:
        tracker.log_metrics(
            {
                "kv_mean_ms": kv["mean_ms"],
                "reference_mean_ms": ref["mean_ms"],
                "kv_tokens_per_sec": kv["tokens_per_sec"],
                "speedup": result["speedup"],
                "speedup_ci_low": speedup.ci_low,
                "speedup_ci_high": speedup.ci_high,
            }
        )
        result["tracking_run_dir"] = str(tracker.run_dir)
        tracker.finish(result)
    return result


# ─────────────────────── Post-training (SFT) ─────────────────────────
def finetune_sft(config_path: str) -> dict[str, Any]:
    """Instruction-tune the base tiny GPT on synthetic instruction data.

    Loads the pretrained checkpoint (if present; otherwise fine-tunes a fresh
    model so the path is always runnable), builds a masked instruction dataset,
    fine-tunes, and reports before/after held-out response perplexity.
    """
    cfg = load_tiny_gpt_config(config_path)
    seed_everything(cfg.seed)

    from dork.data.datasets import prepare_corpus
    from dork.data.instructions import split_instructions, synthetic_instructions
    from dork.models.tiny_gpt import TinyGPT
    from dork.tokenizer.factory import load_or_train_tokenizer
    from dork.training.checkpoint import load_model_from_checkpoint
    from dork.training.sft import SFTTrainer, build_sft_dataset
    from dork.utils.config import TrainingConfig

    sft = cfg.sft
    text = prepare_corpus(cfg.data).read_text(encoding="utf-8")
    tokenizer = load_or_train_tokenizer(cfg.tokenizer, text)

    # Start from the pretrained base if it exists; otherwise a fresh model.
    try:
        model, _ = load_model_from_checkpoint(sft.base_out_dir, device="cpu")
        base_source = sft.base_out_dir
    except FileNotFoundError:
        logger.warning("No base checkpoint at %s; fine-tuning a fresh model.", sft.base_out_dir)
        cfg.model.vocab_size = tokenizer.vocab_size
        model = TinyGPT(cfg.model)
        base_source = "(fresh init)"

    pairs = synthetic_instructions(n_arith=sft.n_arith, seed=cfg.seed)
    train_pairs, val_pairs = split_instructions(pairs, sft.val_fraction, cfg.seed)
    train_xy = build_sft_dataset(train_pairs, tokenizer, cfg.model.block_size)
    val_xy = build_sft_dataset(val_pairs, tokenizer, cfg.model.block_size)

    train_cfg = TrainingConfig(
        batch_size=sft.batch_size,
        max_steps=sft.max_steps,
        eval_interval=sft.eval_interval,
        warmup_steps=sft.warmup_steps,
        learning_rate=sft.learning_rate,
        min_lr=sft.min_lr,
        out_dir=sft.out_dir,
        device=cfg.training.device,
        dtype="float32",
    )
    tracker = start_tracker(cfg.tracking, "sft-instruction-tune", config=cfg, tags=["sft"])
    trainer = SFTTrainer(
        model,
        train_xy,
        val_xy,
        train_cfg,
        tokenizer_path=cfg.tokenizer.path,
        tracker=tracker,
    )
    before = trainer._eval_response_loss(val_xy)
    history = trainer.train()
    after = min((h["val"] for h in history), default=before)
    import math

    result = {
        "base_source": base_source,
        "out_dir": sft.out_dir,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "val_response_loss_before": before,
        "val_response_loss_after": after,
        "val_ppl_before": math.exp(min(before, 20)),
        "val_ppl_after": math.exp(min(after, 20)),
        "steps": sft.max_steps,
    }
    if tracker is not None:
        result["tracking_run_dir"] = str(tracker.run_dir)
        tracker.finish(result)
    return result


# ───────────────────────── Evaluation pipeline ───────────────────────
def run_eval(config_path: str) -> dict[str, Any]:
    """Run the evaluation harness and write reports."""
    from dork.evaluation.harness import EvalHarness

    cfg = load_eval_config(config_path)
    report = EvalHarness(cfg).run()
    tracker = start_tracker(
        getattr(cfg, "tracking", {"enabled": False}),
        "eval-harness",
        config=cfg,
        tags=["eval"],
    )
    if tracker is not None:
        metrics = _eval_summary_metrics(report)
        if metrics:
            tracker.log_metrics(metrics)
        gate = report.get("gate", {})
        result = {
            "gate_passed": bool(gate.get("passed", True)),
            "n_suites": len(report.get("summary", [])),
            "out_dir": (cfg.report or {}).get("out_dir", "reports"),
        }
        report["tracking_run_dir"] = str(tracker.run_dir)
        tracker.finish(result)
    return report


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
