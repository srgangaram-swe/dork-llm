import {
  DEFAULT_SETTINGS,
  LIMITS,
  STORAGE_KEYS,
  consumeChatResponse,
  createChatPayload,
  formatLatency,
  formatScore,
  loadStoredJson,
  normalizeHealth,
  normalizeSettings,
  storeJson,
  textBlocks,
  trimHistory,
} from "/lib.js";

const STREAM_ENDPOINT = "/api/v1/chat/stream";
const CHAT_ENDPOINT = "/chat";
const STREAM_FALLBACK_STATUSES = new Set([404, 405, 501]);
const HEALTH_INTERVAL_MS = 20_000;
const REQUEST_TIMEOUT_MS = 120_000;

function required(selector) {
  const element = document.querySelector(selector);
  if (!element) throw new Error(`DorkChat could not find ${selector}.`);
  return element;
}

const elements = {
  announcer: required("#announcer"),
  characterCount: required("#character-count"),
  clearButton: required("#clear-button"),
  composer: required("#composer"),
  controlsToggle: required("#controls-toggle"),
  emptyState: required("#empty-state"),
  errorBanner: required("#error-banner"),
  errorMessage: required("#error-message"),
  latency: required("#latency"),
  maxTokens: required("#max-tokens"),
  maxTokensValue: required("#max-tokens-value"),
  messages: required("#messages"),
  modeDescription: required("#mode-description"),
  modelName: required("#model-name"),
  providerName: required("#provider-name"),
  prompt: required("#prompt"),
  ragCount: required("#rag-count"),
  rail: required(".workspace-rail"),
  retryButton: required("#retry-button"),
  runtimeBadge: required("#runtime-badge"),
  runtimeNote: required("#runtime-note"),
  runtimeStatus: required("#runtime-status"),
  samplingControls: required("#sampling-controls"),
  samplingHelp: required("#sampling-help"),
  sendButton: required("#send-button"),
  settingsForm: required("#settings-form"),
  stopButton: required("#stop-button"),
  temperature: required("#temperature"),
  temperatureValue: required("#temperature-value"),
  version: required("#api-version"),
};

const state = {
  abortReason: null,
  activeController: null,
  healthInFlight: false,
  healthTimer: null,
  isSending: false,
  lastRequest: null,
  messages: trimHistory(loadStoredJson(localStorage, STORAGE_KEYS.history, [])),
  settings: normalizeSettings(
    loadStoredJson(localStorage, STORAGE_KEYS.settings, DEFAULT_SETTINGS),
  ),
};

function newId(prefix) {
  if (globalThis.crypto?.randomUUID)
    return `${prefix}-${globalThis.crypto.randomUUID()}`;
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function ensureMessageIds() {
  for (const message of state.messages) {
    if (!message.id) message.id = newId(message.role);
  }
}

function announce(message) {
  elements.announcer.textContent = "";
  window.setTimeout(() => {
    elements.announcer.textContent = message;
  }, 25);
}

function scrollConversation() {
  const reduceMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;
  elements.messages.lastElementChild?.scrollIntoView({
    behavior: reduceMotion ? "auto" : "smooth",
    block: "end",
  });
}

function modeLabel(mode) {
  return (
    { auto: "auto route", rag: "evidence", generate: "model" }[mode] || mode
  );
}

function formatTimestamp(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function renderTextContent(container, content) {
  const fragment = document.createDocumentFragment();
  for (const block of textBlocks(content)) {
    if (block.type === "code") {
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      if (block.language) code.dataset.language = block.language;
      code.textContent = block.content;
      pre.append(code);
      fragment.append(pre);
    } else {
      const paragraph = document.createElement("p");
      paragraph.textContent = block.content;
      fragment.append(paragraph);
    }
  }
  container.replaceChildren(fragment);
}

function createThinkingIndicator() {
  const indicator = document.createElement("div");
  indicator.className = "thinking";
  indicator.setAttribute("role", "status");
  const label = document.createElement("span");
  label.className = "sr-only";
  label.textContent = "DorkLLM is generating a response.";
  indicator.append(label);
  for (let index = 0; index < 3; index += 1) {
    const dot = document.createElement("span");
    dot.setAttribute("aria-hidden", "true");
    indicator.append(dot);
  }
  return indicator;
}

function createCitationSection(citations) {
  const section = document.createElement("section");
  section.className = "citation-section";
  section.setAttribute("aria-label", "Supporting evidence");

  const heading = document.createElement("p");
  heading.className = "citation-heading";
  heading.textContent = `Supporting evidence · ${citations.length}`;
  section.append(heading);

  const list = document.createElement("ol");
  list.className = "citations";
  for (const citation of citations) {
    const item = document.createElement("li");
    item.className = "citation-card";

    const header = document.createElement("header");
    const source = document.createElement("span");
    source.className = "citation-source";
    source.textContent = `[${citation.marker}] ${citation.source}`;
    header.append(source);

    const scoreText = formatScore(citation.score);
    if (scoreText) {
      const score = document.createElement("span");
      score.className = "citation-score";
      score.textContent = scoreText;
      header.append(score);
    }
    item.append(header);

    if (citation.snippet) {
      const snippet = document.createElement("p");
      snippet.className = "citation-snippet";
      snippet.textContent = citation.snippet;
      item.append(snippet);
    }
    list.append(item);
  }
  section.append(list);
  return section;
}

function createMessageElement(message) {
  const item = document.createElement("li");
  item.className = `message ${message.role}`;
  item.dataset.messageId = message.id;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = message.role === "assistant" ? "DL" : "YOU";

  const card = document.createElement("article");
  card.className = "message-card";
  card.setAttribute(
    "aria-label",
    message.role === "assistant" ? "DorkLLM response" : "Your message",
  );

  const meta = document.createElement("div");
  meta.className = "message-meta";
  const author = document.createElement("span");
  author.className = "message-author";
  author.textContent = message.role === "assistant" ? "DorkLLM" : "You";
  meta.append(author);

  if (message.metadata?.mode) {
    const mode = document.createElement("span");
    mode.className = "message-tag";
    mode.textContent = modeLabel(message.metadata.mode);
    meta.append(mode);
  }
  if (message.metadata?.model) {
    const model = document.createElement("span");
    model.textContent = message.metadata.model;
    meta.append(model);
  }
  if (message.metadata?.provider) {
    const provider = document.createElement("span");
    provider.textContent = message.metadata.provider;
    meta.append(provider);
  }
  if (message.metadata?.deliveryMode) {
    const delivery = document.createElement("span");
    delivery.className = "message-tag";
    delivery.textContent = message.metadata.deliveryMode.replaceAll("_", " ");
    meta.append(delivery);
  }
  if (message.metadata?.status === "stopped") {
    const status = document.createElement("span");
    status.className = "message-tag";
    status.textContent = "stopped";
    meta.append(status);
  }
  const time = formatTimestamp(message.metadata?.timestamp);
  if (time) {
    const timestamp = document.createElement("time");
    timestamp.dateTime = message.metadata.timestamp;
    timestamp.textContent = time;
    meta.append(timestamp);
  }
  card.append(meta);

  const body = document.createElement("div");
  body.className = "message-body";
  if (message.pending && !message.content)
    body.append(createThinkingIndicator());
  else renderTextContent(body, message.content);
  card.append(body);

  if (message.citations?.length)
    card.append(createCitationSection(message.citations));

  if (message.role === "assistant" && !message.pending && message.content) {
    const actions = document.createElement("div");
    actions.className = "message-actions";

    const copy = document.createElement("button");
    copy.className = "message-action";
    copy.type = "button";
    copy.dataset.action = "copy";
    copy.dataset.messageId = message.id;
    copy.textContent = "Copy";
    actions.append(copy);

    const retry = document.createElement("button");
    retry.className = "message-action";
    retry.type = "button";
    retry.dataset.action = "retry";
    retry.dataset.messageId = message.id;
    retry.textContent = "Retry";
    actions.append(retry);
    card.append(actions);
  }

  item.append(avatar, card);
  return item;
}

function syncEmptyState() {
  elements.emptyState.hidden = state.messages.length > 0;
  elements.messages.hidden = state.messages.length === 0;
}

function renderMessages() {
  const fragment = document.createDocumentFragment();
  for (const message of state.messages)
    fragment.append(createMessageElement(message));
  elements.messages.replaceChildren(fragment);
  syncEmptyState();
}

function appendMessage(message) {
  state.messages.push(message);
  elements.messages.append(createMessageElement(message));
  syncEmptyState();
  scrollConversation();
}

function refreshMessage(message) {
  const current = Array.from(elements.messages.children).find(
    (element) => element.dataset.messageId === message.id,
  );
  if (current) current.replaceWith(createMessageElement(message));
  else renderMessages();
}

function removeMessage(message) {
  state.messages = state.messages.filter((item) => item.id !== message.id);
  const current = Array.from(elements.messages.children).find(
    (element) => element.dataset.messageId === message.id,
  );
  current?.remove();
  syncEmptyState();
}

function persistMessages() {
  const bounded = trimHistory(state.messages, LIMITS.persistedMessages);
  state.messages = bounded.map((message) => ({
    ...message,
    id: message.id || newId(message.role),
  }));
  storeJson(localStorage, STORAGE_KEYS.history, state.messages);
  renderMessages();
}

function hideError() {
  elements.errorBanner.hidden = true;
  elements.errorMessage.textContent = "";
}

function showError(message) {
  elements.errorMessage.textContent = message;
  elements.errorBanner.hidden = false;
}

function setSending(isSending) {
  state.isSending = isSending;
  elements.messages.setAttribute("aria-busy", String(isSending));
  elements.prompt.disabled = isSending;
  elements.sendButton.disabled = isSending;
  elements.stopButton.hidden = !isSending;
  elements.clearButton.disabled = isSending;
}

function readableError(error) {
  if (error?.name === "AbortError") return "The request was stopped.";
  if (typeof error?.message === "string" && error.message.trim())
    return error.message.trim();
  return "DorkChat could not complete the request. Confirm the API is available and try again.";
}

async function requestChat(payload, signal, callbacks) {
  const request = {
    method: "POST",
    headers: {
      Accept: "text/event-stream, application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal,
  };

  let response = await fetch(STREAM_ENDPOINT, request);
  if (STREAM_FALLBACK_STATUSES.has(response.status)) {
    response = await fetch(CHAT_ENDPOINT, {
      ...request,
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
    });
  }
  return consumeChatResponse(response, callbacks);
}

async function executeChat(payload) {
  if (state.isSending) return;
  hideError();
  setSending(true);

  const assistant = {
    id: newId("assistant"),
    role: "assistant",
    content: "",
    citations: [],
    metadata: {
      mode: payload.mode,
      model: "",
      provider: "",
      latencyMs: null,
      requestId: "",
      status: "",
      timestamp: new Date().toISOString(),
    },
    pending: true,
  };
  appendMessage(assistant);

  const controller = new AbortController();
  state.activeController = controller;
  state.abortReason = null;
  const timeout = window.setTimeout(() => {
    state.abortReason = "timeout";
    controller.abort();
  }, REQUEST_TIMEOUT_MS);

  try {
    const result = await requestChat(payload, controller.signal, {
      onDelta(_delta, fullAnswer) {
        assistant.content = fullAnswer;
        refreshMessage(assistant);
        scrollConversation();
      },
      onCitation(_citation, citations) {
        assistant.citations = citations;
        refreshMessage(assistant);
      },
      onMeta(metadata) {
        assistant.metadata = { ...assistant.metadata, ...metadata };
        refreshMessage(assistant);
      },
    });

    assistant.content =
      result.answer ||
      assistant.content ||
      "The runtime returned an empty response.";
    assistant.citations = result.citations.length
      ? result.citations
      : assistant.citations;
    assistant.metadata = { ...assistant.metadata, ...result.metadata };
    assistant.pending = false;
    refreshMessage(assistant);
    elements.latency.textContent = formatLatency(assistant.metadata.latencyMs);
    persistMessages();
    announce("DorkLLM response complete.");
    void refreshHealth();
  } catch (error) {
    const wasStopped =
      controller.signal.aborted || error?.name === "AbortError";
    if (assistant.content) {
      assistant.pending = false;
      assistant.metadata.status = "stopped";
      refreshMessage(assistant);
      persistMessages();
    } else {
      removeMessage(assistant);
    }

    if (wasStopped) {
      const timedOut = state.abortReason === "timeout";
      if (timedOut)
        showError("The request exceeded the two-minute client timeout.");
      announce(timedOut ? "Request timed out." : "Generation stopped.");
    } else {
      showError(readableError(error));
      announce("Request failed. A retry is available.");
    }
  } finally {
    window.clearTimeout(timeout);
    state.abortReason = null;
    state.activeController = null;
    setSending(false);
    elements.prompt.focus();
  }
}

function submitPrompt(message) {
  const content = message.trim();
  if (!content || state.isSending) return;
  const payload = createChatPayload(content, state.settings, state.messages);
  const user = {
    id: newId("user"),
    role: "user",
    content,
    citations: [],
    metadata: { timestamp: new Date().toISOString() },
  };
  appendMessage(user);
  persistMessages();
  state.lastRequest = payload;
  elements.prompt.value = "";
  updateComposerSize();
  void executeChat(payload);
}

function retryAssistant(messageId) {
  if (state.isSending) return;
  const assistantIndex = state.messages.findIndex(
    (message) => message.id === messageId,
  );
  if (assistantIndex < 0) return;
  let userIndex = assistantIndex - 1;
  while (userIndex >= 0 && state.messages[userIndex].role !== "user")
    userIndex -= 1;
  if (userIndex < 0) return;

  const user = state.messages[userIndex];
  const previousMessages = state.messages.slice(0, userIndex);
  const payload = createChatPayload(
    user.content,
    state.settings,
    previousMessages,
  );
  state.messages = state.messages.slice(0, userIndex + 1);
  renderMessages();
  persistMessages();
  state.lastRequest = payload;
  void executeChat(payload);
}

async function copyMessage(messageId, button) {
  const message = state.messages.find((item) => item.id === messageId);
  if (!message?.content) return;
  try {
    await navigator.clipboard.writeText(message.content);
  } catch {
    const helper = document.createElement("textarea");
    helper.value = message.content;
    helper.setAttribute("readonly", "");
    helper.className = "sr-only";
    document.body.append(helper);
    helper.select();
    document.execCommand("copy");
    helper.remove();
  }
  const original = button.textContent;
  button.textContent = "Copied";
  announce("Response copied to the clipboard.");
  window.setTimeout(() => {
    button.textContent = original;
  }, 1500);
}

function updateComposerSize() {
  elements.prompt.style.height = "auto";
  elements.prompt.style.height = `${Math.min(elements.prompt.scrollHeight, 180)}px`;
  elements.characterCount.textContent = `${elements.prompt.value.length} / 6000`;
}

function updateSettings() {
  const selectedMode = elements.settingsForm.querySelector(
    'input[name="mode"]:checked',
  )?.value;
  state.settings = normalizeSettings({
    mode: selectedMode,
    temperature: elements.temperature.value,
    maxTokens: elements.maxTokens.value,
  });
  elements.temperature.value = String(state.settings.temperature);
  elements.maxTokens.value = String(state.settings.maxTokens);
  elements.temperatureValue.textContent = state.settings.temperature.toFixed(1);
  elements.maxTokensValue.textContent = `${state.settings.maxTokens} tokens`;

  const descriptions = {
    auto: "Routes grounded questions through retrieval, then falls back to model generation.",
    rag: "Requires retrieved evidence and returns source-level provenance.",
    generate: "Uses the active DorkLLM runtime without document retrieval.",
  };
  elements.modeDescription.textContent = descriptions[state.settings.mode];
  elements.samplingControls.disabled = state.settings.mode === "rag";
  elements.samplingHelp.textContent =
    state.settings.mode === "rag"
      ? "Evidence mode uses the server-managed grounded generation profile."
      : "Sampling applies whenever the model generation path is active.";
  storeJson(localStorage, STORAGE_KEYS.settings, state.settings);
}

function syncSettings() {
  const modeInput = elements.settingsForm.querySelector(
    `input[name="mode"][value="${state.settings.mode}"]`,
  );
  if (modeInput) modeInput.checked = true;
  elements.temperature.value = String(state.settings.temperature);
  elements.maxTokens.value = String(state.settings.maxTokens);
  updateSettings();
}

function renderHealth(health) {
  elements.runtimeBadge.className = `runtime-badge ${health.state}`;
  elements.runtimeStatus.textContent = health.statusLabel;
  elements.modelName.textContent = health.modelName;
  elements.providerName.textContent = health.provider;
  elements.ragCount.textContent =
    health.ragChunks === null ? "—" : String(health.ragChunks);
  elements.version.textContent = health.version;
  elements.runtimeNote.textContent = health.note;
}

function scheduleHealthRefresh() {
  window.clearTimeout(state.healthTimer);
  if (!document.hidden) {
    state.healthTimer = window.setTimeout(
      () => void refreshHealth(),
      HEALTH_INTERVAL_MS,
    );
  }
}

async function refreshHealth() {
  if (state.healthInFlight) return;
  state.healthInFlight = true;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 6000);
  try {
    const response = await fetch("/health", {
      headers: { Accept: "application/json" },
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok)
      throw new Error(`Health check failed (${response.status}).`);
    renderHealth(normalizeHealth(await response.json()));
  } catch {
    renderHealth({
      state: "offline",
      statusLabel: "Offline",
      modelName: "Runtime unavailable",
      provider: "—",
      ragChunks: null,
      version: "unknown",
      note: "DorkChat cannot reach the AxiomStack API. Start the service and retry.",
    });
  } finally {
    window.clearTimeout(timeout);
    state.healthInFlight = false;
    scheduleHealthRefresh();
  }
}

elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  submitPrompt(elements.prompt.value);
});

elements.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});

elements.prompt.addEventListener("input", updateComposerSize);
elements.settingsForm.addEventListener("input", updateSettings);
elements.settingsForm.addEventListener("submit", (event) =>
  event.preventDefault(),
);

elements.stopButton.addEventListener("click", () => {
  state.abortReason = "user";
  state.activeController?.abort();
});

elements.retryButton.addEventListener("click", () => {
  if (state.lastRequest && !state.isSending) {
    hideError();
    void executeChat(state.lastRequest);
  }
});

elements.clearButton.addEventListener("click", () => {
  if (
    state.messages.length &&
    !window.confirm("Clear this conversation from this browser?")
  )
    return;
  state.abortReason = "clear";
  state.activeController?.abort();
  state.messages = [];
  state.lastRequest = null;
  localStorage.removeItem(STORAGE_KEYS.history);
  hideError();
  renderMessages();
  elements.latency.textContent = "No requests yet";
  elements.prompt.focus();
  announce("Conversation cleared.");
});

elements.messages.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  if (button.dataset.action === "copy")
    void copyMessage(button.dataset.messageId, button);
  if (button.dataset.action === "retry")
    retryAssistant(button.dataset.messageId);
});

for (const starter of document.querySelectorAll(".prompt-starter")) {
  starter.addEventListener("click", () => {
    elements.prompt.value = starter.dataset.prompt || "";
    updateComposerSize();
    elements.prompt.focus();
    elements.prompt.setSelectionRange(
      elements.prompt.value.length,
      elements.prompt.value.length,
    );
  });
}

elements.controlsToggle.addEventListener("click", () => {
  const expanded = elements.rail.dataset.expanded !== "true";
  elements.rail.dataset.expanded = String(expanded);
  elements.controlsToggle.setAttribute("aria-expanded", String(expanded));
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) window.clearTimeout(state.healthTimer);
  else void refreshHealth();
});

ensureMessageIds();
syncSettings();
renderMessages();
updateComposerSize();
void refreshHealth();
