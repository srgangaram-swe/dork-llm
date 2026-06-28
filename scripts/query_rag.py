#!/usr/bin/env python
"""Query the RAG assistant. Thin wrapper over dork.pipelines."""

from __future__ import annotations

import argparse
import json

from dork.pipelines import query_rag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/rag_default.yaml")
    ap.add_argument("--question", required=True)
    args = ap.parse_args()
    print(json.dumps(query_rag(args.config, args.question), indent=2))


if __name__ == "__main__":
    main()
