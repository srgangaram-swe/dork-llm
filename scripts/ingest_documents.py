#!/usr/bin/env python
"""Ingest documents into the vector store. Thin wrapper over dork.pipelines."""

from __future__ import annotations

import argparse
import json

from dork.pipelines import ingest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/rag_default.yaml")
    ap.add_argument("--source", default=None, help="Source directory of documents.")
    args = ap.parse_args()
    print(json.dumps(ingest(args.config, args.source), indent=2))


if __name__ == "__main__":
    main()
