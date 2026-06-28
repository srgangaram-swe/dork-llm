"""Tests for the research agent and its tools."""

from __future__ import annotations

import pytest
from dork.agents.research_agent import ResearchAgent
from dork.agents.tools import default_tools, safe_eval
from dork.rag.pipeline import RagPipeline


def test_safe_eval_arithmetic():
    assert safe_eval("2 + 3 * 4") == 14
    assert safe_eval("(10 - 2) / 4") == 2


def test_safe_eval_rejects_unsafe():
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('echo hi')")


def test_calculator_tool():
    tools = default_tools()
    assert tools["calculator"].run({"expression": "21 * 2"}) == "42"


def test_python_exec_blocks_imports():
    tools = default_tools()
    out = tools["python_exec"].run({"code": "import os"})
    assert "forbidden" in out.lower()


def test_python_exec_runs_safe_code():
    tools = default_tools()
    out = tools["python_exec"].run({"code": "result = sum(range(5))"})
    assert "10" in out


@pytest.fixture
def agent(rag_config) -> ResearchAgent:
    pipe = RagPipeline(rag_config)
    pipe.ingest()
    return ResearchAgent(pipe)


def test_agent_intent_classification():
    assert ResearchAgent.classify_intent("Calculate 2 + 2") == "calculate"
    assert ResearchAgent.classify_intent("Compare A and B") == "compare"
    assert ResearchAgent.classify_intent("Summarize the doc") == "summarize"
    assert ResearchAgent.classify_intent("Design an experiment for X") == "experiment_plan"


def test_agent_calculate(agent):
    res = agent.run("Calculate 21 * 2")
    assert "42" in res.answer
    assert res.tools_used == ["calculator"]


def test_agent_extract_claims_structured(agent):
    res = agent.run("Extract claims from the transformers document")
    assert res.structured is not None
    assert "claims" in res.structured
    assert res.intent == "extract_claims"


def test_agent_experiment_plan_is_json(agent):
    res = agent.run("Design an experiment for evaluating retrieval")
    assert res.structured is not None
    assert "objective" in res.structured and "metrics" in res.structured
