#!/usr/bin/env python
"""Benchmark inference latency/throughput. Thin wrapper over dork.pipelines."""

from __future__ import annotations

import argparse
import json

from dork.pipelines import benchmark


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/train_tiny_gpt.yaml")
    ap.add_argument("--n-requests", type=int, default=20)
    args = ap.parse_args()
    print(json.dumps(benchmark(args.config, args.n_requests), indent=2))


if __name__ == "__main__":
    main()
