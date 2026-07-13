export const STORAGE_KEYS = Object.freeze({
  history: "axiomstack.dorkchat.history.v1",
  settings: "axiomstack.dorkchat.settings.v1",
});

export const DEFAULT_SETTINGS = Object.freeze({
  mode: "auto",
  temperature: 0.7,
  maxTokens: 256,
});

export const LIMITS = Object.freeze({
  persistedMessages: 24,
  requestHistory: 12,
  messageCharacters: 12_000,
  citations: 6,
});

const MODES = new Set(["auto", "rag", "generate"]);

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function finiteNumber(value) {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

function safeText(value, fallback = "", maximum = LIMITS.messageCharacters) {
  if (typeof value !== "string") return fallback;
  return value.slice(0, maximum);
}

function firstText(...values) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return "";
}

export function normalizeSettings(value) {
  const raw = isRecord(value) ? value : {};
  const mode = MODES.has(raw.mode) ? raw.mode : DEFAULT_SETTINGS.mode;
  const temperatureValue = finiteNumber(raw.temperature);
  const tokenValue = finiteNumber(raw.maxTokens ?? raw.max_tokens);
  const temperature =
    temperatureValue === null
      ? DEFAULT_SETTINGS.temperature
      : Math.round(clamp(temperatureValue, 0, 1.4) * 10) / 10;
  const maxTokens =
    tokenValue === null
      ? DEFAULT_SETTINGS.maxTokens
      : clamp(Math.round(tokenValue / 32) * 32, 32, 512);

  return { mode, temperature, maxTokens };
}

export function normalizeCitation(value, index = 0) {
  const raw = isRecord(value) ? value : {};
  const markerValue = finiteNumber(raw.marker ?? raw.index);
  const scoreValue = finiteNumber(
    raw.score ?? raw.relevance_score ?? raw.similarity,
  );
  return {
    marker:
      markerValue === null ? index + 1 : Math.max(1, Math.round(markerValue)),
    source:
      firstText(raw.source, raw.title, raw.document, raw.chunk_id) ||
      "Retrieved context",
    chunkId: firstText(raw.chunk_id, raw.chunkId, raw.id),
    snippet: safeText(firstText(raw.snippet, raw.text, raw.content), "", 600),
    score: scoreValue,
  };
}

function normalizeMetadata(value) {
  const raw = isRecord(value) ? value : {};
  const latency = finiteNumber(raw.latencyMs ?? raw.latency_ms);
  return {
    model: firstText(raw.model, raw.model_name, raw.model_id),
    mode: MODES.has(raw.mode) ? raw.mode : firstText(raw.mode),
    provider: firstText(
      raw.active_provider,
      raw.provider,
      raw.requested_provider,
    ),
    artifact: firstText(raw.artifact),
    deliveryMode: firstText(raw.deliveryMode, raw.delivery_mode),
    nativeTokenStreaming:
      typeof raw.nativeTokenStreaming === "boolean"
        ? raw.nativeTokenStreaming
        : typeof raw.native_token_streaming === "boolean"
          ? raw.native_token_streaming
          : null,
    degraded: typeof raw.degraded === "boolean" ? raw.degraded : null,
    requestId: firstText(raw.requestId, raw.request_id),
    latencyMs: latency,
    status: firstText(raw.status),
    timestamp: firstText(raw.timestamp),
  };
}

export function normalizeMessage(value) {
  if (!isRecord(value) || !["user", "assistant"].includes(value.role))
    return null;
  const content = safeText(value.content);
  if (!content.trim()) return null;
  const citations = Array.isArray(value.citations)
    ? value.citations.slice(0, LIMITS.citations).map(normalizeCitation)
    : [];
  return {
    id: safeText(value.id, "", 100),
    role: value.role,
    content,
    citations,
    metadata: normalizeMetadata(value.metadata),
  };
}

export function trimHistory(messages, maximum = LIMITS.persistedMessages) {
  if (!Array.isArray(messages)) return [];
  return messages
    .map(normalizeMessage)
    .filter(Boolean)
    .slice(-Math.max(0, maximum));
}

export function historyForRequest(messages, maximum = LIMITS.requestHistory) {
  return trimHistory(messages, maximum).map(({ role, content }) => ({
    role,
    content,
  }));
}

export function createChatPayload(message, settings, previousMessages) {
  const normalized = normalizeSettings(settings);
  const currentMessage = safeText(String(message ?? "")).trim();
  return {
    messages: [
      ...historyForRequest(previousMessages),
      { role: "user", content: currentMessage },
    ],
    mode: normalized.mode,
    retrieval_top_k: 5,
    max_new_tokens: normalized.maxTokens,
    temperature: normalized.temperature,
    sampling_top_k: 50,
    top_p: 0.95,
  };
}

export function loadStoredJson(storage, key, fallback) {
  try {
    const raw = storage?.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

export function storeJson(storage, key, value) {
  try {
    storage?.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

export function normalizeHealth(value) {
  const raw = isRecord(value) ? value : {};
  const modelObject = isRecord(raw.model) ? raw.model : {};
  const runtimeObject = isRecord(raw.runtime) ? raw.runtime : {};
  const explicitModelLoaded =
    typeof raw.model_loaded === "boolean"
      ? raw.model_loaded
      : typeof modelObject.loaded === "boolean"
        ? modelObject.loaded
        : null;
  const provider =
    firstText(
      raw.active_provider,
      raw.provider,
      modelObject.active_provider,
      modelObject.provider,
      raw.requested_provider,
      modelObject.requested_provider,
      runtimeObject.provider,
    ) || (explicitModelLoaded ? "local" : "mock");
  const fallback =
    typeof raw.degraded === "boolean"
      ? raw.degraded
      : typeof modelObject.degraded === "boolean"
        ? modelObject.degraded
        : typeof raw.fallback === "boolean"
          ? raw.fallback
          : typeof modelObject.fallback === "boolean"
            ? modelObject.fallback
            : provider.toLowerCase().includes("mock") ||
              explicitModelLoaded === false;
  const status = firstText(raw.status, runtimeObject.status) || "unknown";
  const ready =
    typeof raw.ready === "boolean"
      ? raw.ready
      : ["ok", "ready", "healthy", "degraded"].includes(status.toLowerCase());
  const modelName =
    firstText(
      typeof raw.model === "string" ? raw.model : "",
      raw.model_name,
      raw.model_id,
      modelObject.name,
      modelObject.id,
      runtimeObject.model,
    ) || (fallback ? "Mock fallback" : "DorkLLM local checkpoint");
  const ragChunks = finiteNumber(
    raw.rag_chunks ?? raw.rag_count ?? raw.indexed_chunks,
  );
  const state =
    !ready || fallback || status.toLowerCase() === "degraded"
      ? "degraded"
      : "ready";

  return {
    state,
    statusLabel: state === "ready" ? "Ready" : "Fallback",
    modelName,
    provider,
    ragChunks: ragChunks === null ? null : Math.max(0, Math.round(ragChunks)),
    version: firstText(raw.version, raw.api_version) || "unknown",
    note:
      firstText(
        raw.degraded_reason,
        modelObject.degraded_reason,
        raw.detail,
        raw.message,
        runtimeObject.detail,
      ) ||
      (state === "ready"
        ? "A validated model artifact is serving requests."
        : "The API is reachable, but responses use a fallback runtime."),
  };
}

export function normalizeChatResponse(value) {
  const raw = isRecord(value) ? value : {};
  const response = isRecord(raw.response) ? raw.response : raw;
  const message = isRecord(response.message) ? response.message : {};
  const rawMetadata = isRecord(raw.metadata) ? raw.metadata : {};
  const responseMetadata = isRecord(response.metadata) ? response.metadata : {};
  const citationsValue = response.citations ?? raw.citations;
  return {
    answer: safeText(
      firstText(
        response.answer,
        response.content,
        message.content,
        response.text,
      ),
    ),
    citations: Array.isArray(citationsValue)
      ? citationsValue.slice(0, LIMITS.citations).map(normalizeCitation)
      : [],
    metadata: normalizeMetadata({
      ...rawMetadata,
      ...responseMetadata,
      ...raw,
      ...response,
      model:
        response.model ??
        raw.model ??
        responseMetadata.model ??
        rawMetadata.model,
      mode:
        response.mode ?? raw.mode ?? responseMetadata.mode ?? rawMetadata.mode,
      latencyMs:
        response.latencyMs ??
        response.latency_ms ??
        raw.latencyMs ??
        raw.latency_ms ??
        responseMetadata.latencyMs ??
        responseMetadata.latency_ms ??
        rawMetadata.latencyMs ??
        rawMetadata.latency_ms,
    }),
  };
}

export function parseSseFrame(frame) {
  const normalized = String(frame ?? "").replaceAll("\r\n", "\n");
  if (!normalized.trim()) return null;
  let event = "message";
  let id = "";
  const data = [];

  for (const line of normalized.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    const separator = line.indexOf(":");
    const field = separator < 0 ? line : line.slice(0, separator);
    let content = separator < 0 ? "" : line.slice(separator + 1);
    if (content.startsWith(" ")) content = content.slice(1);
    if (field === "event") event = content || "message";
    if (field === "data") data.push(content);
    if (field === "id") id = content;
  }

  if (!data.length) return null;
  return { event, data: data.join("\n"), id };
}

async function* readSseEvents(body) {
  if (!body?.getReader)
    throw new Error("Streaming response body is unavailable.");
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    buffer = buffer.replaceAll("\r\n", "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const frame = parseSseFrame(buffer.slice(0, boundary));
      buffer = buffer.slice(boundary + 2);
      if (frame) yield frame;
      boundary = buffer.indexOf("\n\n");
    }
    if (done) break;
  }

  const finalFrame = parseSseFrame(buffer);
  if (finalFrame) yield finalFrame;
}

function parseEventData(data) {
  if (data === "[DONE]") return data;
  try {
    return JSON.parse(data);
  } catch {
    return data;
  }
}

function eventError(value) {
  if (typeof value === "string") return value;
  if (!isRecord(value)) return "The streaming request failed.";
  const detail = value.detail;
  if (typeof detail === "string") return detail;
  return (
    firstText(value.error, value.message) || "The streaming request failed."
  );
}

function mergeMetadata(current, incoming) {
  const merged = { ...current };
  for (const [key, value] of Object.entries(incoming)) {
    if (value !== "" && value !== null && value !== undefined)
      merged[key] = value;
  }
  return merged;
}

async function responseError(response) {
  const contentType =
    response.headers?.get?.("content-type")?.toLowerCase() || "";
  try {
    if (contentType.includes("application/json")) {
      const payload = await response.json();
      if (typeof payload?.detail === "string") return payload.detail;
      if (Array.isArray(payload?.detail))
        return "The request did not pass API validation.";
      return (
        firstText(payload?.error, payload?.message) ||
        `Request failed (${response.status}).`
      );
    }
    const text = await response.text();
    return text.trim().slice(0, 300) || `Request failed (${response.status}).`;
  } catch {
    return `Request failed (${response.status}).`;
  }
}

export async function consumeChatResponse(
  response,
  { onDelta = () => {}, onCitation = () => {}, onMeta = () => {} } = {},
) {
  if (!response.ok) {
    const error = new Error(await responseError(response));
    error.status = response.status;
    throw error;
  }

  const contentType =
    response.headers?.get?.("content-type")?.toLowerCase() || "";
  if (!contentType.includes("text/event-stream")) {
    return normalizeChatResponse(await response.json());
  }

  const aggregate = { answer: "", citations: [], metadata: {} };
  for await (const frame of readSseEvents(response.body)) {
    const payload = parseEventData(frame.data);
    if (payload === "[DONE]") break;

    if (frame.event === "error") throw new Error(eventError(payload));

    if (frame.event === "delta" || frame.event === "message") {
      const deltaValue =
        typeof payload === "string"
          ? payload
          : (payload?.delta ??
            payload?.text ??
            payload?.token ??
            payload?.content);
      const delta = typeof deltaValue === "string" ? deltaValue : "";
      if (delta) {
        aggregate.answer += delta;
        onDelta(delta, aggregate.answer);
      }
      continue;
    }

    if (frame.event === "citation") {
      const citation = normalizeCitation(
        payload?.citation ?? payload,
        aggregate.citations.length,
      );
      const key = `${citation.marker}:${citation.source}:${citation.chunkId}`;
      const duplicate = aggregate.citations.some(
        (item) => `${item.marker}:${item.source}:${item.chunkId}` === key,
      );
      if (!duplicate && aggregate.citations.length < LIMITS.citations) {
        aggregate.citations.push(citation);
        onCitation(citation, [...aggregate.citations]);
      }
      continue;
    }

    if (frame.event === "meta" || frame.event === "done") {
      const normalized = normalizeChatResponse(payload);
      aggregate.metadata = mergeMetadata(
        aggregate.metadata,
        normalized.metadata,
      );
      if (!aggregate.answer && normalized.answer)
        aggregate.answer = normalized.answer;
      if (normalized.citations.length)
        aggregate.citations = normalized.citations;
      onMeta({ ...aggregate.metadata }, frame.event);
      if (frame.event === "done") break;
    }
  }

  return normalizeChatResponse(aggregate);
}

export function formatLatency(value) {
  const milliseconds = finiteNumber(value);
  if (milliseconds === null || milliseconds < 0) return "No requests yet";
  return milliseconds < 1000
    ? `${Math.round(milliseconds)} ms`
    : `${(milliseconds / 1000).toFixed(1)} s`;
}

export function formatScore(value) {
  const score = finiteNumber(value);
  if (score === null) return "";
  if (score >= 0 && score <= 1) return `${Math.round(score * 100)}% match`;
  return `score ${score.toFixed(2)}`;
}

export function textBlocks(value) {
  const text = safeText(String(value ?? ""));
  const blocks = [];
  const fence = /```([^\n`]*)\n?([\s\S]*?)```/g;
  let cursor = 0;
  let match;

  const addText = (content) => {
    for (const paragraph of content.split(/\n{2,}/)) {
      if (paragraph) blocks.push({ type: "text", content: paragraph });
    }
  };

  while ((match = fence.exec(text)) !== null) {
    addText(text.slice(cursor, match.index));
    blocks.push({
      type: "code",
      language: safeText(match[1].trim(), "", 40),
      content: match[2].replace(/\n$/, ""),
    });
    cursor = match.index + match[0].length;
  }
  addText(text.slice(cursor));
  return blocks;
}
