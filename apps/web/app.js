const canvas = document.querySelector("#matrix-rain");
const ctx = canvas.getContext("2d");
const messagesEl = document.querySelector("#messages");
const composer = document.querySelector("#composer");
const promptEl = document.querySelector("#prompt");
const sendButton = document.querySelector("#send-button");
const statusPill = document.querySelector("#status-pill");
const statusText = document.querySelector("#status-text");
const modelName = document.querySelector("#model-name");
const ragCount = document.querySelector("#rag-count");
const latencyEl = document.querySelector("#latency");
const tempSlider = document.querySelector("#temperature");
const tempValue = document.querySelector("#temperature-value");
const tokenSlider = document.querySelector("#max-tokens");
const tokenValue = document.querySelector("#max-tokens-value");
const modeButtons = Array.from(document.querySelectorAll(".mode-button"));

const glyphs = "dorkLLM01$#@{}[]<>/\\|+=*abcdefghijklmnopqrstuvwxyz".split("");
const history = [];
let columns = [];
let activeMode = "auto";
let isSending = false;

function resizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(window.innerWidth * ratio);
  canvas.height = Math.floor(window.innerHeight * ratio);
  canvas.style.width = `${window.innerWidth}px`;
  canvas.style.height = `${window.innerHeight}px`;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

  const fontSize = Math.max(14, Math.floor(window.innerWidth / 92));
  const count = Math.ceil(window.innerWidth / fontSize);
  columns = Array.from({ length: count }, (_, i) => ({
    x: i * fontSize,
    y: Math.random() * window.innerHeight,
    speed: 6 + Math.random() * 12,
    fontSize,
  }));
}

function drawRain() {
  ctx.fillStyle = "rgba(5, 3, 13, 0.18)";
  ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);

  for (const col of columns) {
    const char = glyphs[Math.floor(Math.random() * glyphs.length)];
    ctx.font = `${col.fontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.fillStyle = Math.random() > 0.82 ? "#d45cff" : "#48ffbf";
    ctx.fillText(char, col.x, col.y);
    col.y += col.speed;
    if (col.y > window.innerHeight + 40) {
      col.y = -20 - Math.random() * 120;
      col.speed = 6 + Math.random() * 12;
    }
  }

  requestAnimationFrame(drawRain);
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function addMessage(role, content, metadata = {}) {
  const node = document.createElement("article");
  node.className = `message ${role}`;

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = metadata.label || role;
  node.append(meta);

  const body = document.createElement("div");
  body.innerHTML = escapeHtml(content);
  node.append(body);

  if (metadata.citations?.length) {
    const citations = document.createElement("div");
    citations.className = "citations";
    for (const citation of metadata.citations.slice(0, 4)) {
      const item = document.createElement("div");
      item.className = "citation";
      const marker = citation.marker ? `[${citation.marker}]` : "[source]";
      const source = citation.source || citation.chunk_id || "retrieved context";
      item.textContent = `${marker} ${source}`;
      citations.append(item);
    }
    node.append(citations);
  }

  messagesEl.append(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setStatus(ok, text) {
  statusPill.classList.toggle("offline", !ok);
  statusText.textContent = text;
}

async function refreshHealth() {
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error(`health ${res.status}`);
    const body = await res.json();
    setStatus(true, body.model_loaded ? "model live" : "mock live");
    modelName.textContent = body.model_loaded ? "local gpt" : "mock fallback";
    ragCount.textContent = body.rag_chunks ?? 0;
  } catch {
    setStatus(false, "offline");
    modelName.textContent = "unreachable";
  }
}

async function sendMessage(message) {
  isSending = true;
  sendButton.disabled = true;
  sendButton.textContent = "Thinking";
  const pending = document.createElement("article");
  pending.className = "message assistant";
  pending.innerHTML = '<div class="meta">dorkLLM</div><div>...</div>';
  messagesEl.append(pending);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  try {
    const payload = {
      message,
      mode: activeMode,
      history,
      retrieval_top_k: 5,
      max_new_tokens: Number(tokenSlider.value),
      temperature: Number(tempSlider.value),
      sampling_top_k: 50,
      top_p: 0.95,
    };

    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(detail || `chat ${res.status}`);
    }
    const body = await res.json();
    pending.remove();
    addMessage("assistant", body.answer || "(empty response)", {
      label: `dorkLLM / ${body.mode} / ${body.model}`,
      citations: body.citations,
    });
    history.push({ role: "assistant", content: body.answer || "" });
    latencyEl.textContent = `${Math.round(body.latency_ms)} ms`;
    await refreshHealth();
  } catch (err) {
    pending.remove();
    addMessage("assistant", `Request failed: ${err.message}`, { label: "dorkLLM / error" });
  } finally {
    isSending = false;
    sendButton.disabled = false;
    sendButton.textContent = "Send";
    promptEl.focus();
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  if (isSending) return;
  const message = promptEl.value.trim();
  if (!message) return;
  promptEl.value = "";
  addMessage("user", message, { label: "you" });
  history.push({ role: "user", content: message });
  sendMessage(message);
});

promptEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    activeMode = button.dataset.mode;
    modeButtons.forEach((item) => item.classList.toggle("active", item === button));
  });
});

tempSlider.addEventListener("input", () => {
  tempValue.textContent = tempSlider.value;
});

tokenSlider.addEventListener("input", () => {
  tokenValue.textContent = tokenSlider.value;
});

window.addEventListener("resize", resizeCanvas);

resizeCanvas();
drawRain();
refreshHealth();
setInterval(refreshHealth, 15000);

addMessage(
  "assistant",
  "I am dorkLLM. Ask me about the repo, transformer internals, RAG, evals, or training plans.",
  { label: "dorkLLM / ready" },
);
