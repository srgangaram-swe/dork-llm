#!/usr/bin/env python
"""Generate text from the trained model. Thin wrapper over dork.pipelines."""

from __future__ import annotations

import argparse

from dork.pipelines import generate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/train_tiny_gpt.yaml")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    args = ap.parse_args()
    text = generate(
        args.config,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(args.prompt + text)


if __name__ == "__main__":
    main()
