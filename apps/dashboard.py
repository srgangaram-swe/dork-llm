"""Streamlit dashboard for the Dork LLM platform.

Five tabs map onto the platform's subsystems: text generation, the evaluation
harness, the RAG assistant, the research agent, and serving metrics. The
dashboard talks to :class:`~dork.serving.service.DorkService` directly (in-process)
so it runs without a separate API server.

Run with: ``streamlit run apps/dashboard.py`` (or ``make dashboard``).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from dork.serving.service import DorkService


@st.cache_resource
def get_service() -> DorkService:
    return DorkService()


service = get_service()

st.set_page_config(page_title="Dork LLM", page_icon="🧠", layout="wide")
st.title("🧠 Dork LLM — LLM systems platform")
st.caption(
    "A tiny GPT trained from scratch · a reusable eval harness · a cited RAG + agent. "
    "Educational-scale, local-first, and honest about its limits."
)

health = service.health()
c1, c2, c3 = st.columns(3)
c1.metric("Version", health["version"])
c2.metric("Trained model loaded", "yes" if health["model_loaded"] else "no (mock)")
c3.metric("RAG chunks indexed", health["rag_chunks"])

gen_tab, eval_tab, rag_tab, agent_tab, metrics_tab = st.tabs(
    ["✍️ Generate", "📊 Evaluate", "🔎 RAG", "🤖 Agent", "📈 Metrics"]
)

with gen_tab:
    st.subheader("Generate text from the tiny GPT")
    prompt = st.text_area("Prompt", "Once upon a time")
    col = st.columns(3)
    temperature = col[0].slider("Temperature", 0.0, 1.5, 0.8, 0.05)
    max_new = col[1].slider("Max new tokens", 8, 512, 128, 8)
    top_k = col[2].slider("Top-k", 0, 200, 50, 5)
    if st.button("Generate", type="primary"):
        with st.spinner("Generating…"):
            out = service.generate(
                prompt, max_new_tokens=max_new, temperature=temperature, top_k=top_k
            )
        st.markdown(f"**{prompt}**{out['completion']}")
        st.caption(f"model={out['model']} · {out['latency_ms']:.1f} ms")

with eval_tab:
    st.subheader("Run the evaluation harness")
    cfg_path = st.text_input("Eval config", "configs/eval_default.yaml")
    if st.button("Run evaluation"):
        with st.spinner("Evaluating…"):
            report = service.evaluate(cfg_path)
        df = pd.DataFrame(report["summary"])
        st.dataframe(df, use_container_width=True)
        gate = report["gate"]
        st.success("CI gate: PASS") if gate["passed"] else st.error("CI gate: FAIL")
        chart_df = df[df["category"] != "performance"].set_index("suite")["value"]
        if not chart_df.empty:
            st.bar_chart(chart_df)

with rag_tab:
    st.subheader("Ask the RAG assistant")
    if st.button("Ingest sample documents"):
        with st.spinner("Ingesting…"):
            stats = service.rag_ingest()
        st.json(stats)
    question = st.text_input("Question", "What does causal masking prevent?")
    top_k = st.slider("Retrieved chunks (k)", 1, 10, 5)
    if st.button("Ask", type="primary"):
        with st.spinner("Retrieving + answering…"):
            ans = service.rag_query(question, top_k)
        st.markdown(f"**Answer:** {ans['answer']}")
        if ans["refused"]:
            st.warning("The assistant refused (insufficient evidence).")
        if ans["citations"]:
            st.markdown("**Citations**")
            st.dataframe(pd.DataFrame(ans["citations"]), use_container_width=True)
        with st.expander("Retrieved context"):
            for ctx in ans["contexts"]:
                st.markdown(f"`[{ctx['score']:.3f}]` **{ctx['source']}** — {ctx['text'][:300]}…")

with agent_tab:
    st.subheader("Run the research agent")
    task = st.text_input("Task", "Compare RAG systems and evaluation")
    if st.button("Run agent", type="primary"):
        with st.spinner("Planning + acting…"):
            res = service.run_agent(task)
        st.markdown(f"**Intent:** `{res['intent']}` · **Tools:** {res['tools_used']}")
        st.markdown(f"**Answer**\n\n{res['answer']}")
        if res.get("structured"):
            with st.expander("Structured output"):
                st.json(res["structured"])
        with st.expander("Trajectory"):
            for i, step in enumerate(res["steps"], 1):
                st.markdown(f"**Step {i} — `{step['tool']}`**")
                st.code(step["observation"][:500])

with metrics_tab:
    st.subheader("Serving metrics")
    st.json(service.metrics.snapshot())
    st.caption("In-memory counters; resets when the dashboard restarts.")
