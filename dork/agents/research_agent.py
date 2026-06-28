"""An agentic research assistant over the RAG index.

The agent plans a short sequence of tool calls to satisfy a research task —
searching the document index, summarizing sources, extracting claims, comparing
documents, drafting experiment plans, or doing arithmetic — and returns a
structured, citation-bearing result. It refuses when the evidence is insufficient.

Routing is a deterministic intent classifier by default (robust with small/mock
backends); the same tools can be driven by an LLM planner when a stronger model
is configured. See ``docs/agent_design.md``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from dork.agents.tools import Tool, default_tools
from dork.rag.pipeline import RagPipeline
from dork.rag.schema import Citation, ScoredChunk
from dork.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentStep:
    """One tool invocation in the agent's trajectory."""

    tool: str
    args: dict[str, Any]
    observation: str

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args, "observation": self.observation}


@dataclass
class AgentResult:
    """The agent's final, structured, cited output."""

    task: str
    answer: str
    intent: str
    steps: list[AgentStep] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    structured: dict[str, Any] | None = None
    refused: bool = False

    @property
    def tools_used(self) -> list[str]:
        return [s.tool for s in self.steps]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "intent": self.intent,
            "answer": self.answer,
            "refused": self.refused,
            "tools_used": self.tools_used,
            "steps": [s.to_dict() for s in self.steps],
            "citations": [c.to_dict() for c in self.citations],
            "structured": self.structured,
        }


class ResearchAgent:
    """A bounded, tool-using research assistant grounded in the RAG index."""

    def __init__(
        self, pipeline: RagPipeline, max_steps: int = 6, allow_code_exec: bool = True
    ) -> None:
        self.pipeline = pipeline
        self.model = pipeline.model
        self.max_steps = max_steps
        self.tools: dict[str, Tool] = {}
        if allow_code_exec:
            self.tools = default_tools()
        else:
            self.tools = {"calculator": default_tools()["calculator"]}
        self._last_contexts: list[ScoredChunk] = []

    # ── Public entry point ───────────────────────────────────────────
    def run(self, task: str) -> AgentResult:
        """Route ``task`` to the right capability and return a structured result."""
        intent = self.classify_intent(task)
        logger.info("Agent task -> intent=%s", intent)
        handler = {
            "calculate": self._do_calculate,
            "compare": self._do_compare,
            "experiment_plan": self._do_experiment_plan,
            "extract_claims": self._do_extract_claims,
            "summarize": self._do_summarize,
            "answer": self._do_answer,
        }[intent]
        return handler(task, intent)

    @staticmethod
    def classify_intent(task: str) -> str:
        """Lightweight intent router (deterministic, model-agnostic)."""
        t = task.lower()
        if re.search(r"-?\d+\s*[+\-*/x]\s*-?\d+", t) or t.startswith(("calculate", "compute")):
            return "calculate"
        if "compare" in t or " vs " in t or "difference between" in t:
            return "compare"
        if "experiment plan" in t or "design an experiment" in t or "ablation" in t:
            return "experiment_plan"
        if "claim" in t or "key points" in t or "extract" in t:
            return "extract_claims"
        if "summar" in t or "tl;dr" in t or "overview" in t:
            return "summarize"
        return "answer"

    # ── Capability handlers ──────────────────────────────────────────
    def _do_calculate(self, task: str, intent: str) -> AgentResult:
        m = re.search(r"(-?\d+(?:\.\d+)?\s*[+\-*/x]\s*-?\d+(?:\.\d+)?)", task)
        expr = (m.group(1) if m else task).replace("x", "*")
        obs = self.tools["calculator"].run({"expression": expr})
        step = AgentStep("calculator", {"expression": expr}, obs)
        return AgentResult(task, f"The result is {obs}.", intent, steps=[step])

    def _do_answer(self, task: str, intent: str) -> AgentResult:
        chunks = self._search(task)
        if not chunks:
            return AgentResult(task, self.pipeline.query(task).answer, intent, refused=True)
        ans = self.pipeline.query(task)
        step = AgentStep("search_docs", {"query": task}, self._format_hits(chunks))
        return AgentResult(
            task,
            ans.answer,
            intent,
            steps=[step],
            citations=ans.citations,
            refused=ans.refused,
        )

    def _do_summarize(self, task: str, intent: str) -> AgentResult:
        topic = self._strip_verb(task, ("summarize", "summary of", "overview of", "tl;dr"))
        chunks = self._search(topic or task)
        steps = [AgentStep("search_docs", {"query": topic or task}, self._format_hits(chunks))]
        if not chunks:
            return AgentResult(
                task, self.pipeline.query(task).answer, intent, steps=steps, refused=True
            )
        context = "\n".join(f"[{i + 1}] {c.chunk.text}" for i, c in enumerate(chunks))
        prompt = (
            "Summarize the following context concisely. Cite sources as [n].\n\n"
            f"Context:\n{context}\n\nSummary:"
        )
        summary = self.model.complete(prompt, max_new_tokens=180, temperature=0.0).strip()
        steps.append(AgentStep("summarize", {"topic": topic}, summary))
        return AgentResult(
            task, summary, intent, steps=steps, citations=self._cite(summary, chunks)
        )

    def _do_extract_claims(self, task: str, intent: str) -> AgentResult:
        topic = self._strip_verb(
            task, ("extract claims from", "extract", "key claims in", "key points in")
        )
        chunks = self._search(topic or task)
        steps = [AgentStep("search_docs", {"query": topic or task}, self._format_hits(chunks))]
        claims = []
        for i, c in enumerate(chunks):
            sentence = re.split(r"(?<=[.!?])\s+", c.chunk.text.strip())[0]
            claims.append({"claim": sentence, "source": c.chunk.source, "citation": i + 1})
        structured = {"topic": topic or task, "claims": claims}
        answer = f"Extracted {len(claims)} claim(s):\n" + "\n".join(
            f"- {cl['claim']} [{cl['citation']}]" for cl in claims
        )
        steps.append(AgentStep("extract_claims", {"topic": topic}, json.dumps(structured)))
        return AgentResult(
            task,
            answer,
            intent,
            steps=steps,
            citations=self._cite(answer, chunks),
            structured=structured,
        )

    def _do_compare(self, task: str, intent: str) -> AgentResult:
        topics = self._split_comparison(task)
        steps: list[AgentStep] = []
        sections: list[str] = []
        all_chunks: list[ScoredChunk] = []
        for topic in topics:
            hits = self._search(topic)
            steps.append(AgentStep("search_docs", {"query": topic}, self._format_hits(hits)))
            if hits:
                sections.append(f"**{topic}**: {hits[0].chunk.text[:200]}")
                all_chunks.append(hits[0])
        if not all_chunks:
            return AgentResult(
                task, self.pipeline.query(task).answer, intent, steps=steps, refused=True
            )
        answer = "Comparison:\n" + "\n".join(f"- {s} [{i + 1}]" for i, s in enumerate(sections))
        structured = {"compared": topics, "n_sources": len(all_chunks)}
        steps.append(AgentStep("compare_docs", {"topics": topics}, answer))
        return AgentResult(
            task,
            answer,
            intent,
            steps=steps,
            citations=self._cite(answer, all_chunks),
            structured=structured,
        )

    def _do_experiment_plan(self, task: str, intent: str) -> AgentResult:
        topic = self._strip_verb(
            task, ("design an experiment for", "experiment plan for", "experiment plan")
        )
        chunks = self._search(topic or task)
        steps = [AgentStep("search_docs", {"query": topic or task}, self._format_hits(chunks))]
        plan = {
            "objective": f"Evaluate {topic or 'the approach'} against a baseline.",
            "hypothesis": "The proposed change improves the primary metric without regressions.",
            "datasets": ["held-out validation split", "synthetic eval suites"],
            "metrics": ["perplexity", "task accuracy", "latency"],
            "method": [
                "Establish a baseline measurement.",
                "Apply the change with a fixed random seed.",
                "Run the evaluation harness and compare via the gate thresholds.",
            ],
            "ablations": ["vary one hyperparameter at a time", "remove the component under test"],
            "risks": ["overfitting to the eval set", "insufficient sample size"],
            "grounded_in": [c.chunk.source for c in chunks],
        }
        answer = json.dumps(plan, indent=2)
        steps.append(AgentStep("experiment_plan", {"topic": topic}, answer))
        return AgentResult(
            task,
            answer,
            intent,
            steps=steps,
            citations=self._cite("[1]" if chunks else "", chunks),
            structured=plan,
        )

    # ── Helpers ──────────────────────────────────────────────────────
    def _search(self, query: str) -> list[ScoredChunk]:
        self._last_contexts = self.pipeline.retrieve(query)
        return self._last_contexts

    @staticmethod
    def _format_hits(chunks: list[ScoredChunk]) -> str:
        if not chunks:
            return "(no relevant chunks found)"
        return "\n".join(
            f"[{i + 1}] ({c.score:.3f}) {c.chunk.source}: {c.chunk.text[:120]}"
            for i, c in enumerate(chunks)
        )

    @staticmethod
    def _cite(text: str, chunks: list[ScoredChunk]) -> list[Citation]:
        from dork.evaluation.metrics import extract_citations

        markers = sorted(set(extract_citations(text))) or list(range(1, len(chunks) + 1))
        out: list[Citation] = []
        for m in markers:
            if 1 <= m <= len(chunks):
                c = chunks[m - 1]
                out.append(
                    Citation(m, c.chunk.source, c.chunk.chunk_id, c.chunk.text[:200], c.score)
                )
        return out

    @staticmethod
    def _strip_verb(task: str, verbs: tuple[str, ...]) -> str:
        t = task.strip()
        low = t.lower()
        for v in verbs:
            if low.startswith(v):
                return t[len(v) :].strip(" :.")
            if v in low:
                return t[low.index(v) + len(v) :].strip(" :.")
        return t

    @staticmethod
    def _split_comparison(task: str) -> list[str]:
        t = re.sub(r"^\s*compare\s+", "", task, flags=re.IGNORECASE)
        t = re.sub(r"difference between", "", t, flags=re.IGNORECASE)
        parts = re.split(r"\b(?:vs|versus|and|with)\b|,", t)
        topics = [p.strip(" .?") for p in parts if p.strip(" .?")]
        return topics[:3] if topics else [task]
