"""Lightweight experiment tracking.

Every run writes a local directory containing:

- ``metadata.json``: run identity, tags, config, and environment hints.
- ``metrics.jsonl``: append-only scalar metrics, one JSON row per log call.
- ``summary.json``: final status and headline metrics.

Weights & Biases is optional and only used when explicitly enabled. This keeps
CI/offline workflows dependency-free while preserving an upgrade path for richer
experiment dashboards.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from dork.utils.io import save_json
from dork.utils.logging import get_logger
from dork.utils.paths import resolve_path

logger = get_logger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", text.strip().lower()).strip("-")
    return slug or "run"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@dataclass
class ExperimentTracker:
    """Append local metrics and optionally mirror them to W&B."""

    run_name: str
    project: str = "dork-llm"
    out_dir: str | Path = "experiments"
    config: Mapping[str, Any] | None = None
    tags: Iterable[str] = field(default_factory=list)
    use_wandb: bool = False

    def __post_init__(self) -> None:
        self.started_at = _now()
        self.run_id = f"{self.started_at.replace(':', '').replace('-', '')}-{_slug(self.run_name)}-{uuid.uuid4().hex[:8]}"
        self.run_dir = resolve_path(self.out_dir, create_parent=True) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self._wandb_run: Any | None = None

        metadata = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "project": self.project,
            "tags": list(self.tags),
            "started_at": self.started_at,
            "config": _jsonable(self.config or {}),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        }
        save_json(self.run_dir / "metadata.json", metadata)
        if self.use_wandb:
            self._start_wandb(metadata)
        logger.info("Tracking run %s at %s", self.run_name, self.run_dir)

    def _start_wandb(self, metadata: dict[str, Any]) -> None:
        try:
            import wandb  # type: ignore[import-not-found]
        except Exception as exc:
            logger.warning("W&B tracking requested but unavailable: %s", exc)
            return
        self._wandb_run = wandb.init(
            project=self.project,
            name=self.run_name,
            id=self.run_id,
            tags=metadata["tags"],
            config=metadata["config"],
            reinit=True,
        )

    def log_metrics(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        """Append one metric row locally and mirror to W&B when enabled."""
        row = {"time": _now(), "step": step, "metrics": _jsonable(metrics)}
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        if self._wandb_run is not None:
            self._wandb_run.log(row["metrics"], step=step)

    def finish(self, summary: Mapping[str, Any] | None = None, *, status: str = "finished") -> None:
        """Write the final run summary and close the optional W&B run."""
        payload = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "status": status,
            "started_at": self.started_at,
            "finished_at": _now(),
            "summary": _jsonable(summary or {}),
        }
        save_json(self.summary_path, payload)
        if self._wandb_run is not None:
            self._wandb_run.summary.update(payload["summary"])
            self._wandb_run.finish()
            self._wandb_run = None


def start_tracker(
    tracking_cfg: Any,
    run_name: str,
    *,
    config: Mapping[str, Any] | BaseModel | None = None,
    tags: Iterable[str] = (),
) -> ExperimentTracker | None:
    """Create an experiment tracker from a config object/dict, or return ``None``."""
    enabled = bool(_cfg_get(tracking_cfg, "enabled", False))
    if not enabled:
        return None

    cfg_tags = list(_cfg_get(tracking_cfg, "tags", []))
    use_wandb = bool(_cfg_get(tracking_cfg, "wandb", False)) or os.getenv("DORK_WANDB") == "1"
    return ExperimentTracker(
        run_name=run_name,
        project=str(_cfg_get(tracking_cfg, "project", "dork-llm")),
        out_dir=_cfg_get(tracking_cfg, "out_dir", "experiments"),
        config=_jsonable(config or {}),
        tags=[*cfg_tags, *list(tags)],
        use_wandb=use_wandb,
    )


def list_runs(out_dir: str | Path = "experiments") -> list[dict[str, Any]]:
    """Return local run summaries, newest first."""
    root = resolve_path(out_dir)
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for meta_path in root.glob("*/metadata.json"):
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        summary_path = meta_path.parent / "summary.json"
        summary = {}
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        rows.append(
            {
                "run_id": meta.get("run_id"),
                "run_name": meta.get("run_name"),
                "project": meta.get("project"),
                "started_at": meta.get("started_at"),
                "status": summary.get("status", "running"),
                "path": str(meta_path.parent),
            }
        )
    return sorted(rows, key=lambda r: str(r.get("started_at", "")), reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="List local Dork LLM experiment runs.")
    ap.add_argument("--out-dir", default="experiments")
    args = ap.parse_args()
    print(json.dumps(list_runs(args.out_dir), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
