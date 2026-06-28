#!/usr/bin/env python
"""Download/prepare the training corpus. Thin wrapper over dork.pipelines."""

from __future__ import annotations

import argparse
import json

from dork.pipelines import prepare_data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/train_tiny_gpt.yaml")
    args = ap.parse_args()
    print(json.dumps(prepare_data(args.config), indent=2))


if __name__ == "__main__":
    main()
