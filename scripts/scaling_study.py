#!/usr/bin/env python
"""Reproducible scaling study: does validation loss fall as the model grows?

Trains several tiny GPTs of increasing width/depth on the same corpus and
tokenizer for the same number of steps, then records parameters vs. validation
loss and fits a power law. This is a *miniature* of the empirical scaling-laws
methodology used to plan large training runs — not a claim about frontier scale.

Outputs:
    reports/scaling_study.json   — per-run metrics + fitted exponent
    docs/assets/scaling_study.png — loss-vs-parameters plot (committed)

Usage:
    python scripts/scaling_study.py --steps 250 --sizes 64 128 192 256
"""

from __future__ import annotations

import argparse
import json
import math
import time

from dork.data.datasets import prepare_corpus
from dork.data.loader import BinTokenDataset, build_token_bins
from dork.models.tiny_gpt import TinyGPT
from dork.tokenizer.factory import load_or_train_tokenizer
from dork.training.trainer import Trainer
from dork.utils.config import DataConfig, ModelConfig, TokenizerConfig, TrainingConfig
from dork.utils.io import save_json
from dork.utils.logging import get_logger
from dork.utils.paths import resolve_path
from dork.utils.seed import seed_everything

logger = get_logger(__name__)


def run_scaling_study(
    sizes: list[int],
    steps: int = 250,
    block_size: int = 128,
    seed: int = 1337,
    vocab_size: int = 1024,
) -> dict:
    """Train one model per ``n_embd`` in ``sizes`` and record params vs. val loss."""
    seed_everything(seed)
    data_cfg = DataConfig(dataset="tiny_shakespeare")
    text = prepare_corpus(data_cfg).read_text(encoding="utf-8")
    tok = load_or_train_tokenizer(
        TokenizerConfig(type="bpe", vocab_size=vocab_size, path="tokenizers/scaling_bpe.json"),
        text,
    )
    meta = build_token_bins(text, tok.encode, "artifacts/scaling/data", data_cfg.val_fraction)
    train_ds = BinTokenDataset(meta["train_bin"], block_size)
    val_ds = BinTokenDataset(meta["val_bin"], block_size)

    runs = []
    for n_embd in sizes:
        seed_everything(seed)
        n_head = max(2, n_embd // 64)
        n_layer = max(2, n_embd // 96)
        model = TinyGPT(
            ModelConfig(
                vocab_size=tok.vocab_size,
                block_size=block_size,
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
                dropout=0.0,
            )
        )
        cfg = TrainingConfig(
            batch_size=16,
            max_steps=steps,
            eval_interval=max(steps, 1),
            eval_iters=50,
            warmup_steps=max(1, steps // 10),
            out_dir=f"artifacts/scaling/e{n_embd}",
            device="cpu",
            dtype="float32",
            learning_rate=3e-3,
        )
        t0 = time.time()
        history = Trainer(model, train_ds, val_ds, cfg).train()
        val = history[-1]["val"]
        params = model.num_params()
        runs.append(
            {
                "n_embd": n_embd,
                "n_layer": n_layer,
                "n_head": n_head,
                "params": params,
                "val_loss": val,
                "val_ppl": math.exp(min(val, 20)),
                "minutes": round((time.time() - t0) / 60, 2),
            }
        )
        logger.info("size n_embd=%d params=%d val_loss=%.4f", n_embd, params, val)

    # Fit a power law: val_loss ~= a * params^b (line in log-log space).
    xs = [math.log(r["params"]) for r in runs]
    ys = [math.log(r["val_loss"]) for r in runs]
    exponent, intercept = _linfit(xs, ys)
    result = {
        "runs": runs,
        "power_law": {"exponent_b": exponent, "intercept_a": math.exp(intercept)},
        "note": "Educational-scale trend, not a frontier scaling claim.",
        "config": {"steps": steps, "block_size": block_size, "vocab_size": tok.vocab_size},
    }
    save_json("reports/scaling_study.json", result)
    _plot(runs, exponent, "docs/assets/scaling_study.png")
    return result


def _linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Ordinary least squares slope/intercept."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False))
    var = sum((x - mx) ** 2 for x in xs) or 1.0
    slope = cov / var
    return slope, my - slope * mx


def _plot(runs: list[dict], exponent: float, path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        logger.warning("matplotlib unavailable; skipping scaling plot.")
        return

    params = [r["params"] for r in runs]
    losses = [r["val_loss"] for r in runs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(params, losses, "o-", color="#4C78A8")
    for r in runs:
        ax.annotate(
            f"{r['n_embd']}d",
            (r["params"], r["val_loss"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("parameters (log scale)")
    ax.set_ylabel("validation loss")
    ax.set_title(f"Dork LLM scaling trend (fitted exponent b={exponent:.3f})")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    out = resolve_path(path, create_parent=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Saved scaling plot to %s", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 192, 256])
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--block-size", type=int, default=128)
    args = ap.parse_args()
    result = run_scaling_study(args.sizes, steps=args.steps, block_size=args.block_size)
    print(json.dumps({"runs": result["runs"], "power_law": result["power_law"]}, indent=2))


if __name__ == "__main__":
    main()
