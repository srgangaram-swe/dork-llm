#!/usr/bin/env python
"""Train the tokenizer. Thin wrapper over dork.pipelines."""

from __future__ import annotations

import argparse
import json

from dork.pipelines import train_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/train_tiny_gpt.yaml")
    args = ap.parse_args()
    print(json.dumps(train_tokenizer(args.config), indent=2))


if __name__ == "__main__":
    main()
