"""Small, dependency-light I/O helpers for YAML/JSON/JSONL/text."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from dork.utils.paths import resolve_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict (resolved against the project root)."""
    p = resolve_path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the top of {p}, got {type(data).__name__}")
    return data


def read_text(path: str | Path) -> str:
    """Read a UTF-8 text file."""
    return resolve_path(path).read_text(encoding="utf-8")


def save_text(path: str | Path, text: str) -> Path:
    """Write a UTF-8 text file, creating parent dirs as needed."""
    p = resolve_path(path, create_parent=True)
    p.write_text(text, encoding="utf-8")
    return p


def save_json(path: str | Path, obj: Any, *, indent: int = 2) -> Path:
    """Serialize ``obj`` to JSON, creating parent dirs as needed."""
    p = resolve_path(path, create_parent=True)
    p.write_text(json.dumps(obj, indent=indent, ensure_ascii=False, default=str), encoding="utf-8")
    return p


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    return json.loads(read_text(path))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    rows: list[dict[str, Any]] = []
    for line in read_text(path).splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Lazily iterate a JSONL file."""
    p = resolve_path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write an iterable of dicts to JSONL."""
    p = resolve_path(path, create_parent=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return p
