"""Render evaluation results to JSON, CSV, Markdown and (optional) plots.

The Markdown report is the human-facing artifact; the JSON is machine-readable
for dashboards/CI; the CSV is a flat summary for spreadsheets. Plots are written
only when matplotlib is installed.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dork.utils.io import save_json
from dork.utils.logging import get_logger
from dork.utils.paths import resolve_path

logger = get_logger(__name__)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        return f"{value:.4f}"
    return str(value)


def write_reports(
    report: dict[str, Any],
    out_dir: str | Path,
    formats: list[str] | None = None,
    plots: bool = True,
) -> dict[str, Path]:
    """Write requested report formats and return a mapping of format -> path."""
    formats = formats or ["json", "csv", "markdown"]
    out = resolve_path(out_dir, create_parent=True)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if "json" in formats:
        written["json"] = save_json(out / "eval_report.json", report)

    if "csv" in formats:
        written["csv"] = _write_csv(out / "eval_summary.csv", report)

    if "markdown" in formats:
        written["markdown"] = _write_markdown(out / "eval_report.md", report)

    if plots:
        plot_path = _write_plot(out / "eval_metrics.png", report)
        if plot_path:
            written["plot"] = plot_path

    logger.info("Wrote eval reports: %s", {k: str(v) for k, v in written.items()})
    return written


def _write_csv(path: Path, report: dict[str, Any]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["suite", "category", "n", "metric", "value"])
        for row in report.get("summary", []):
            writer.writerow(
                [row["suite"], row["category"], row["n"], row["metric"], _fmt(row["value"])]
            )
    return path


def _write_markdown(path: Path, report: dict[str, Any]) -> Path:
    lines: list[str] = []
    model = report.get("model", {})
    gate = report.get("gate", {})
    lines.append("# Dork LLM — Evaluation Report\n")
    lines.append(
        f"- **Model:** `{model.get('name', '?')}` (provider: `{model.get('provider', '?')}`)"
    )
    lines.append(f"- **Generated:** {report.get('timestamp', '')}")
    lines.append(f"- **Seed:** {report.get('seed', '')}")
    status = "✅ PASS" if gate.get("passed", True) else "❌ FAIL"
    lines.append(f"- **CI gate:** {status}\n")

    lines.append("## Summary\n")
    lines.append("| Suite | Category | N | Metric | Value |")
    lines.append("|---|---|---:|---|---:|")
    for row in report.get("summary", []):
        lines.append(
            f"| {row['suite']} | {row['category']} | {row['n']} | "
            f"{row['metric']} | {_fmt(row['value'])} |"
        )
    lines.append("")

    # Full metrics per suite.
    lines.append("## Detailed metrics\n")
    for name, suite in report.get("suites", {}).items():
        metric_str = ", ".join(f"`{k}={_fmt(v)}`" for k, v in suite.get("metrics", {}).items())
        lines.append(f"- **{name}** ({suite.get('category')}, n={suite.get('n')}): {metric_str}")
    lines.append("")

    # Threshold gating detail.
    if gate.get("checks"):
        lines.append("## CI gate checks\n")
        lines.append("| Check | Threshold | Actual | Pass |")
        lines.append("|---|---:|---:|:---:|")
        for chk in gate["checks"]:
            mark = "✅" if chk["passed"] else "❌"
            lines.append(
                f"| {chk['key']} | {_fmt(chk['threshold'])} | {_fmt(chk['actual'])} | {mark} |"
            )
        lines.append("")

    # A few representative failure cases for qualitative review.
    lines.append("## Sample failure cases\n")
    any_fail = False
    for name, suite in report.get("suites", {}).items():
        fails = [c for c in suite.get("cases", []) if not c.get("passed")]
        for case in fails[:2]:
            any_fail = True
            lines.append(f"**[{name}] {case['case_id']}**")
            lines.append(f"- prompt: `{_truncate(case.get('prompt', ''))}`")
            lines.append(f"- output: `{_truncate(case.get('output', ''))}`")
            lines.append(f"- expected: `{_truncate(str(case.get('expected', '')))}`\n")
    if not any_fail:
        lines.append("_No failures recorded._\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _truncate(s: str, n: int = 120) -> str:
    s = s.replace("\n", " ⏎ ")
    return s if len(s) <= n else s[:n] + "…"


def _write_plot(path: Path, report: dict[str, Any]) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        logger.info("matplotlib not available; skipping plot.")
        return None

    rows = [
        r
        for r in report.get("summary", [])
        if isinstance(r["value"], (int, float))
        and r["value"] == r["value"]
        and r["category"] != "performance"
    ]
    if not rows:
        return None

    labels = [r["suite"] for r in rows]
    values = [float(r["value"]) for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4))
    bars = ax.bar(labels, values, color="#4C78A8")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(f"Dork LLM eval — {report.get('model', {}).get('name', '')}")
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, values, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
