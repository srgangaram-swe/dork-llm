"""Tests for dependency-light local experiment tracking."""

from __future__ import annotations

import json

from dork.utils.tracking import list_runs, start_tracker


def test_tracker_can_be_disabled(tmp_path):
    tracker = start_tracker({"enabled": False, "out_dir": str(tmp_path)}, "disabled")
    assert tracker is None
    assert not list(tmp_path.iterdir())


def test_local_tracker_writes_metadata_metrics_and_summary(tmp_path):
    tracker = start_tracker(
        {
            "enabled": True,
            "out_dir": str(tmp_path),
            "project": "dork-tests",
            "wandb": False,
            "tags": ["unit"],
        },
        "train",
        config={"lr": 1e-3},
        tags=["tracking"],
    )
    assert tracker is not None

    tracker.log_metrics({"loss": 2.5, "accuracy": 0.25}, step=3)
    tracker.finish({"best_loss": 2.0})

    metadata = json.loads((tracker.run_dir / "metadata.json").read_text(encoding="utf-8"))
    metrics = [
        json.loads(line)
        for line in (tracker.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads((tracker.run_dir / "summary.json").read_text(encoding="utf-8"))

    assert metadata["run_name"] == "train"
    assert metadata["project"] == "dork-tests"
    assert metadata["config"]["lr"] == 1e-3
    assert metadata["tags"] == ["unit", "tracking"]
    assert metrics[0]["step"] == 3
    assert metrics[0]["metrics"]["loss"] == 2.5
    assert summary["summary"]["best_loss"] == 2.0

    runs = list_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["run_id"] == tracker.run_id
    assert runs[0]["status"] == "finished"
