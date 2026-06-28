#!/usr/bin/env python
"""Run the evaluation harness. Thin wrapper over dork.pipelines."""

from __future__ import annotations

import argparse
import json
import sys

from dork.pipelines import run_eval


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/eval_default.yaml")
    ap.add_argument(
        "--fail-on-gate", action="store_true", help="Exit non-zero if the CI gate fails."
    )
    args = ap.parse_args()
    report = run_eval(args.config)
    print(json.dumps(report["summary"], indent=2))
    if args.fail_on_gate and not report.get("gate", {}).get("passed", True):
        print("CI gate FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
