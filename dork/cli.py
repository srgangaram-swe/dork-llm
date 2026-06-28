"""The ``dork`` command-line interface.

A single Typer app exposing every workflow. Each command is a thin wrapper over
:mod:`dork.pipelines`, so the CLI, the ``scripts/`` entry points and the service
layer stay in lock-step.

Examples:
    dork train-tokenizer --config configs/train_tiny_gpt.yaml
    dork train --config configs/train_tiny_gpt.yaml
    dork generate --prompt "Once upon a time"
    dork eval --config configs/eval_default.yaml
    dork ingest --source data/sample_docs
    dork query --question "What is causal masking?"
    dork agent --task "Summarize the transformers document"
"""

from __future__ import annotations

import json

import typer
from rich.console import Console

from dork import __version__
from dork import pipelines as P

app = typer.Typer(add_completion=False, help="Dork LLM — train, evaluate, retrieve, and serve.")
console = Console()

TRAIN_CFG = "configs/train_tiny_gpt.yaml"
EVAL_CFG = "configs/eval_default.yaml"
RAG_CFG = "configs/rag_default.yaml"


def _echo(obj: object) -> None:
    console.print_json(json.dumps(obj, default=str))


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"dork-llm {__version__}")


@app.command("prepare-data")
def prepare_data(config: str = TRAIN_CFG) -> None:
    """Download/prepare the training corpus."""
    _echo(P.prepare_data(config))


@app.command("train-tokenizer")
def train_tokenizer(config: str = TRAIN_CFG) -> None:
    """Train the tokenizer on the corpus."""
    _echo(P.train_tokenizer(config))


@app.command()
def train(config: str = TRAIN_CFG) -> None:
    """Train the tiny GPT from scratch."""
    _echo(P.train_model(config))


@app.command()
def generate(
    prompt: str = typer.Option("Once upon a time", help="Prompt to continue."),
    config: str = TRAIN_CFG,
    max_new_tokens: int = typer.Option(None, help="Override max new tokens."),
    temperature: float = typer.Option(None, help="Override temperature."),
    top_k: int = typer.Option(None, help="Override top-k."),
    top_p: float = typer.Option(None, help="Override top-p."),
) -> None:
    """Generate text from the trained model."""
    text = P.generate(
        config,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    console.print(f"[bold cyan]{prompt}[/bold cyan]{text}")


@app.command("eval")
def evaluate(config: str = EVAL_CFG) -> None:
    """Run the evaluation harness and write reports."""
    report = P.run_eval(config)
    console.print("[bold]Eval summary[/bold]")
    for row in report["summary"]:
        console.print(f"  {row['suite']:<22} {row['metric']:<18} = {row['value']}")
    gate = report.get("gate", {})
    status = "PASS" if gate.get("passed", True) else "FAIL"
    console.print(f"[bold]CI gate:[/bold] {status}")


@app.command()
def benchmark(config: str = TRAIN_CFG, n_requests: int = 20) -> None:
    """Benchmark generation latency/throughput."""
    _echo(P.benchmark(config, n_requests))


@app.command()
def ingest(config: str = RAG_CFG, source: str = typer.Option(None, help="Source dir.")) -> None:
    """Ingest documents into the vector store."""
    _echo(P.ingest(config, source))


@app.command()
def query(
    question: str = typer.Option(..., help="Question to answer."), config: str = RAG_CFG
) -> None:
    """Ask the RAG assistant a question."""
    ans = P.query_rag(config, question)
    console.print(f"[bold green]Answer:[/bold green] {ans['answer']}")
    if ans["citations"]:
        console.print("[bold]Citations:[/bold]")
        for c in ans["citations"]:
            console.print(f"  [{c['marker']}] {c['source']} ({c['score']:.3f})")


@app.command()
def agent(task: str = typer.Option(..., help="Research task."), config: str = RAG_CFG) -> None:
    """Run the agentic research assistant."""
    result = P.run_agent(config, task)
    console.print(
        f"[bold]Intent:[/bold] {result['intent']}  [bold]Tools:[/bold] {result['tools_used']}"
    )
    console.print(f"[bold green]Answer:[/bold green]\n{result['answer']}")


if __name__ == "__main__":  # pragma: no cover
    app()
