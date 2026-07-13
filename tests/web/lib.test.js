import assert from "node:assert/strict";
import test from "node:test";

import {
  LIMITS,
  consumeChatResponse,
  createChatPayload,
  formatLatency,
  formatScore,
  historyForRequest,
  loadStoredJson,
  normalizeHealth,
  normalizeSettings,
  parseSseFrame,
  textBlocks,
  trimHistory,
} from "../../apps/web/lib.js";

test("normalizes settings into supported UI bounds", () => {
  assert.deepEqual(
    normalizeSettings({ mode: "invalid", temperature: 9, maxTokens: 17 }),
    {
      mode: "auto",
      temperature: 1.4,
      maxTokens: 32,
    },
  );
  assert.deepEqual(
    normalizeSettings({ mode: "rag", temperature: 0.34, max_tokens: 301 }),
    {
      mode: "rag",
      temperature: 0.3,
      maxTokens: 288,
    },
  );
});

test("builds a bounded canonical payload with the current turn exactly once", () => {
  const messages = Array.from(
    { length: LIMITS.requestHistory + 4 },
    (_, index) => ({
      id: String(index),
      role: index % 2 ? "assistant" : "user",
      content: `turn ${index}`,
    }),
  );
  const payload = createChatPayload(
    "current question",
    { mode: "auto" },
    messages,
  );

  assert.equal(payload.messages.length, LIMITS.requestHistory + 1);
  assert.equal(payload.messages.at(-2).content, `turn ${messages.length - 1}`);
  assert.deepEqual(payload.messages.at(-1), {
    role: "user",
    content: "current question",
  });
  assert.equal(
    payload.messages.filter(({ content }) => content === "current question")
      .length,
    1,
  );
});

test("sanitizes and bounds persisted history", () => {
  const source = [
    { role: "system", content: "not persisted" },
    { role: "user", content: "hello" },
    {
      role: "assistant",
      content: "answer",
      citations: [{ source: "doc.md", score: 0.91 }],
    },
    { role: "assistant", content: "   " },
  ];
  assert.deepEqual(historyForRequest(source), [
    { role: "user", content: "hello" },
    { role: "assistant", content: "answer" },
  ]);
  assert.equal(trimHistory(source, 1)[0].citations[0].source, "doc.md");
});

test("fails closed when browser storage contains invalid JSON", () => {
  const storage = { getItem: () => "{not-json" };
  assert.deepEqual(loadStoredJson(storage, "key", []), []);
});

test("normalizes legacy and structured runtime truth without hiding fallback state", () => {
  const fallback = normalizeHealth({
    status: "degraded",
    ready: false,
    version: "0.1.0",
    model_loaded: false,
    rag_chunks: 12,
  });
  assert.equal(fallback.state, "degraded");
  assert.equal(fallback.modelName, "Mock fallback");

  const ready = normalizeHealth({
    status: "ok",
    ready: true,
    version: "0.2.0",
    model_loaded: true,
    active_provider: "local_gpt",
    rag_chunks: 23,
    model: { name: "DorkLLM-SFT", degraded: false },
  });
  assert.equal(ready.state, "ready");
  assert.equal(ready.modelName, "DorkLLM-SFT");
  assert.equal(ready.provider, "local_gpt");
});

test("parses SSE fields and safely separates text from fenced code", () => {
  assert.deepEqual(parseSseFrame('event: delta\nid: 7\ndata: {"delta":"hi"}'), {
    event: "delta",
    id: "7",
    data: '{"delta":"hi"}',
  });
  assert.deepEqual(
    textBlocks('<img src=x onerror="alert(1)">\n\n```py\nprint("safe")\n```'),
    [
      { type: "text", content: '<img src=x onerror="alert(1)">' },
      { type: "code", language: "py", content: 'print("safe")' },
    ],
  );
});

test("consumes named chat SSE events without losing token whitespace", async () => {
  const stream = [
    'event: meta\ndata: {"request_id":"r-1","model":"DorkLLM","active_provider":"local"}\n\n',
    'event: delta\ndata: {"delta":"Grounded"}\n\n',
    'event: delta\ndata: {"delta":" answer"}\n\n',
    'event: citation\ndata: {"citation":{"marker":1,"source":"docs/rag.md","snippet":"Evidence","score":0.92}}\n\n',
    'event: done\ndata: {"mode":"rag","model":"DorkLLM","latency_ms":42}\n\n',
  ].join("");
  const deltas = [];
  const response = new Response(stream, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
  const result = await consumeChatResponse(response, {
    onDelta: (delta) => deltas.push(delta),
  });

  assert.equal(result.answer, "Grounded answer");
  assert.deepEqual(deltas, ["Grounded", " answer"]);
  assert.equal(result.citations[0].snippet, "Evidence");
  assert.equal(result.metadata.mode, "rag");
  assert.equal(result.metadata.provider, "local");
  assert.equal(result.metadata.latencyMs, 42);
});

test("supports the existing non-streaming chat response", async () => {
  const response = new Response(
    JSON.stringify({
      answer: "JSON fallback",
      mode: "generate",
      model: "test-model",
      latency_ms: 12.4,
      citations: [],
    }),
    { status: 200, headers: { "content-type": "application/json" } },
  );
  const result = await consumeChatResponse(response);
  assert.equal(result.answer, "JSON fallback");
  assert.equal(result.metadata.model, "test-model");
  assert.equal(formatLatency(result.metadata.latencyMs), "12 ms");
  assert.equal(formatScore(0.87), "87% match");
});

test("turns public SSE error events into rejected requests", async () => {
  const response = new Response(
    'event: error\ndata: {"code":"unavailable","message":"Model is not ready."}\n\n',
    { status: 200, headers: { "content-type": "text/event-stream" } },
  );
  await assert.rejects(
    () => consumeChatResponse(response),
    /Model is not ready/,
  );
});
