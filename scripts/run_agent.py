#!/usr/bin/env python
"""Run the agentic research assistant. Thin wrapper over dork.pipelines."""

from __future__ import annotations

import argparse
import json

from dork.pipelines import run_agent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/rag_default.yaml")
    ap.add_argument("--task", required=True)
    args = ap.parse_args()
    print(json.dumps(run_agent(args.config, args.task), indent=2))


if __name__ == "__main__":
    main()
