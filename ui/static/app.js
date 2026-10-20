const $ = (selector) => document.querySelector(selector);
const state = {
  jobs: [], library: [], settings: null, selectedKey: "", selectedJob: null,
  preview: null, log: "", pollTimer: null, previewSignature: "",
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function toast(message, type = "") {
  const node = document.createElement("div");
  node.className = `toast ${type}`;
  node.textContent = message;
  $("#toast-region").append(node);
  window.setTimeout(() => node.remove(), 3400);
}

function setFormError(message = "") {
  const box = $("#form-error");
  box.textContent = message;
  box.hidden = !message;
}

function safeText(value) { return String(value ?? ""); }
function escapeHtml(value) {
  return safeText(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}
function fileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
function relativeTime(value) {
  if (!value) return "Waiting";
  const seconds = Math.max(0, Math.round((Date.now() - new Date(value).getTime()) / 1000));
  if (seconds < 60) return "Just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return new Date(value).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
function shortLabel(job) {
  if (!job) return "Untitled run";
  if (job.mode === "regenerate") return `Refresh · ${job.label}`;
  try {
    const url = new URL(job.label);
    const host = url.hostname.replace(/^www\./, "");
    const id = url.searchParams.get("v") || url.pathname.split("/").filter(Boolean).at(-1) || host;
    return `${host} · ${id}`;
  } catch { return job.label || "Video run"; }
}

function initSettings(settings) {
  if (!settings) return;
  const keyConfigured = Boolean(settings.gemini_api_key_configured);
  const keyStatus = $("#gemini-key-status");
  keyStatus.textContent = keyConfigured ? "Configured" : "Not configured";
  keyStatus.dataset.configured = String(keyConfigured);
  $("#remove-gemini-key").hidden = !keyConfigured;
  if (state.settings) return;
  state.settings = settings;
  const model = $("#transcription-model");
  model.replaceChildren(...settings.transcription_models.map((name) => new Option(name, name)));
  model.value = settings.transcription_model;
  $("#batch-size").value = String(settings.batch_size);
  $("#gemini-model").value = settings.gemini_model;
  $("#temperature").value = String(settings.temperature);
  $("#temperature-value").value = Number(settings.temperature).toFixed(1);
  $("#temperature-value").textContent = Number(settings.temperature).toFixed(1);
  $("#generation-retries").value = String(settings.generation_retries);
  $("#output-folder-hint").textContent = `Default: ${settings.output_dir}`;
}

function setTheme(theme, persist = true) {
  const dark = theme === "dark";
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  const button = $("#theme-toggle");
  if (button) {
    button.textContent = dark ? "☀" : "☾";
    button.title = dark ? "Switch to light mode" : "Switch to dark mode";
    button.setAttribute("aria-label", button.title);
  }
  const meta = $("#theme-color-meta");
  if (meta) meta.content = dark ? "#171b18" : "#f4f5f1";
  if (persist) {
    try { localStorage.setItem("autotrim-theme", dark ? "dark" : "light"); } catch { /* Theme still applies for this session. */ }
  }
}

function initTheme() {
  let theme = "light";
  try { theme = localStorage.getItem("autotrim-theme") || "light"; } catch { /* Use light mode when storage is unavailable. */ }
  setTheme(theme, false);
}

async function updateGeminiKey(remove = false) {
  const input = $("#gemini-api-key");
  const button = remove ? $("#remove-gemini-key") : $("#save-gemini-key");
  const original = button.textContent;
  button.disabled = true;
  button.textContent = remove ? "Removing…" : "Saving…";
  try {
    const result = await api(remove ? "/api/settings/gemini-key/remove" : "/api/settings/gemini-key", {
      method: "POST",
      body: JSON.stringify(remove ? {} : { api_key: input.value }),
    });
    const status = $("#gemini-key-status");
    status.textContent = result.configured ? "Configured" : "Not configured";
    status.dataset.configured = String(result.configured);
    $("#remove-gemini-key").hidden = !result.configured;
    if (!remove) input.value = "";
    toast(remove ? "Gemini API key removed from this computer" : "Gemini API key saved locally");
  } catch (error) { toast(error.message, "error"); }
  finally { button.disabled = false; button.textContent = original; }
}

function renderRecentRuns() {
  const recent = $("#recent-list");
  const meaningful = state.jobs.filter((job) => job.status !== "cancelled").slice(0, 7);
  recent.replaceChildren();
  if (!meaningful.length) {
    const empty = document.createElement("div");
    empty.className = "sidebar-empty";
    empty.textContent = "Your runs will appear here.";
    recent.append(empty);
  }
  for (const job of meaningful) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `recent-run ${state.selectedKey === `job:${job.id}` ? "selected" : ""}`;
    button.dataset.status = job.status;
    const dot = document.createElement("span"); dot.className = "recent-run-icon";
    const text = document.createElement("span"); text.className = "recent-run-copy";
    const title = document.createElement("b"); title.textContent = shortLabel(job);
    const detail = document.createElement("small"); detail.textContent = `${job.stage} · ${relativeTime(job.created_at)}`;
    text.append(title, detail); button.append(dot, text);
    button.addEventListener("click", () => selectJob(job.id));
    recent.append(button);
  }
  $("#run-count").textContent = String(state.jobs.length);
}

function renderReviewOptions() {
  const select = $("#review-select");
  const previous = state.selectedKey;
  const fragment = document.createDocumentFragment();
  const placeholder = new Option("Choose a run or transcript to review", "");
  fragment.append(placeholder);
  if (state.jobs.length) {
    const group = document.createElement("optgroup"); group.label = "Recent runs";
    for (const job of state.jobs) group.append(new Option(`${shortLabel(job)} — ${job.status}`, `job:${job.id}`));
    fragment.append(group);
  }
  if (state.library.length) {
    const group = document.createElement("optgroup"); group.label = "Saved transcripts";
    for (const item of state.library) group.append(new Option(item.title, `file:${item.transcript_path}`));
    fragment.append(group);
  }
  select.replaceChildren(fragment);
  if ([...select.options].some((option) => option.value === previous)) select.value = previous;
}

function renderQueue() {
  const active = state.jobs.find((job) => job.status === "running");
  const queued = state.jobs.filter((job) => job.status === "queued").sort((a, b) => a.created_at.localeCompare(b.created_at));
  const card = $("#active-run");
  const pill = $("#queue-pill");
  const live = active || queued[0];
  card.className = `active-run ${live ? (active ? "running" : "queued") : "empty-active"} ${live?.status || ""}`;
  if (live) {
    card.innerHTML = `<div class="empty-orbit"><span>${active ? "◌" : "·"}</span></div><div class="active-run-copy"><b>${escapeHtml(shortLabel(live))}</b><p>${escapeHtml(active ? live.stage : "Waiting its turn")} · ${escapeHtml(live.status)}</p></div><button class="cancel-job" data-cancel="${live.id}">Cancel</button>`;
  } else {
    card.innerHTML = '<div class="empty-orbit"><span>◌</span></div><div><b>Nothing running just yet</b><p>Start a run and watch each step unfold here.</p></div>';
  }
  pill.textContent = active ? "IN PROGRESS" : queued.length ? "IN QUEUE" : "ALL CLEAR";
  pill.classList.toggle("busy", Boolean(active || queued.length));
  $("#monitor-label").classList.toggle("is-live", Boolean(active));
  $("#monitor-label").innerHTML = `<i></i> ${active ? "LIVE" : "IDLE"}`;
  const queueList = $("#queue-list"); queueList.replaceChildren();
  queued.slice(0, 3).forEach((job, index) => {
    const row = document.createElement("div"); row.className = "queue-row";
    const marker = document.createElement("span"); marker.className = "queued-marker"; marker.textContent = String(index + 1).padStart(2, "0");
    const label = document.createElement("span"); label.className = "queue-row-label"; label.textContent = shortLabel(job);
    const cancel = document.createElement("button"); cancel.className = "queue-cancel"; cancel.dataset.cancel = job.id; cancel.textContent = "×"; cancel.title = "Remove from queue";
    row.append(marker, label, cancel); queueList.append(row);
  });
  $("#queue-total").textContent = `${queued.length} waiting`;
  updateResources(active?.resources);
}

function updateResources(resources) {
  const cpu = resources?.process_cpu_percent;
  const memory = resources?.process_memory_mb;
  const memoryPercent = resources?.process_memory_percent;
  const gpu = resources?.gpu_percent;
  const gpuMemory = resources?.gpu_memory_mb;
  const gpuTotal = resources?.gpu_memory_total_mb;
  $("#cpu-value").textContent = cpu == null ? "—" : `${Math.round(cpu)}%`;
  $("#cpu-meter").style.width = cpu == null ? "0%" : `${Math.min(100, Math.max(2, cpu))}%`;
  $("#cpu-caption").textContent = cpu == null ? "Waiting for a run" : "AutoTrim process tree";
  $("#memory-value").textContent = memory == null ? "—" : `${(memory / 1024).toFixed(1)} GB`;
  $("#memory-meter").style.width = memoryPercent == null ? "0%" : `${Math.min(100, Math.max(1, memoryPercent))}%`;
  $("#memory-caption").textContent = memoryPercent == null ? "Process memory" : `${memoryPercent.toFixed(1)}% of system RAM`;
  $("#gpu-value").textContent = gpu == null ? "—" : `${gpu}%`;
  $("#gpu-meter").style.width = gpu == null ? "0%" : `${gpu}%`;
  $("#gpu-caption").textContent = gpu == null ? "NVIDIA GPU telemetry unavailable" : "System GPU utilization";
  $("#gpu-memory-value").textContent = gpuMemory == null ? "—" : `${(gpuMemory / 1024).toFixed(1)} GB`;
  $("#gpu-memory-meter").style.width = !gpuMemory || !gpuTotal ? "0%" : `${Math.min(100, gpuMemory / gpuTotal * 100)}%`;
  $("#gpu-memory-caption").textContent = gpuMemory == null ? "Dedicated VRAM" : `${(gpuTotal / 1024).toFixed(1)} GB available`;
}

function markdownInline(value) {
  return escapeHtml(value)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>");
}
function renderMarkdown(value) {
  const lines = safeText(value).replace(/\r/g, "").split("\n");
  const html = []; let paragraph = []; let list = [];
  const flushParagraph = () => { if (paragraph.length) { html.push(`<p>${paragraph.map(markdownInline).join("<br>")}</p>`); paragraph = []; } };
  const flushList = () => { if (list.length) { html.push(`<ul>${list.map((item) => `<li>${markdownInline(item)}</li>`).join("")}</ul>`); list = []; } };
  for (const line of lines) {
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    const bullet = line.match(/^\s*[-*]\s+(.+)$/);
    if (!line.trim()) { flushParagraph(); flushList(); continue; }
    if (heading) { flushParagraph(); flushList(); const level = Math.min(heading[1].length, 3); html.push(`<h${level}>${markdownInline(heading[2])}</h${level}>`); continue; }
    if (bullet) { flushParagraph(); list.push(bullet[1]); continue; }
    flushList(); paragraph.push(line);
  }
  flushParagraph(); flushList();
  return html.join("");
}

function outputData() {
  const root = state.preview?.content;
  return root?.content && typeof root.content === "object" ? root.content : root;
}

function renderOutput() {
  const target = $("#output-content");
  const data = outputData();
  if (!data) {
    target.innerHTML = '<div class="review-placeholder"><span>✳</span><p>No Substack article for this transcript yet.<br>Regenerate from transcript to create one.</p></div>';
    return;
  }
  const title = data.substack_title;
  const body = data.substack_content;
  if (!body && !title) { target.innerHTML = '<p class="output-empty">The Substack article is not available in this output.</p>'; return; }
  const copyText = `${title ? `# ${title}\n\n` : ""}${body || ""}`;
  target.innerHTML = `<div class="content-actions"><button class="copy-button" data-copy="full-output">Copy Substack article</button></div><div class="markdown-body">${title ? `<h1>${markdownInline(title)}</h1>` : ""}${renderMarkdown(body)}</div>`;
  target.querySelector("[data-copy='full-output']")?.addEventListener("click", () => copyToClipboard(copyText, "Article copied"));
}

function renderFiles(files = []) {
  const list = $("#file-list"); list.replaceChildren();
  if (!files.length) {
    const empty = document.createElement("span"); empty.className = "muted-empty"; empty.textContent = "No files yet"; list.append(empty);
    $("#files-caption").textContent = "Downloads appear when a transcript or article is ready.";
    return;
  }
  $("#files-caption").textContent = `${files.length} file${files.length === 1 ? "" : "s"} ready to download`;
  for (const file of files) {
    const chip = document.createElement("div"); chip.className = "file-chip";
    const icon = document.createElement("span"); icon.className = "file-chip-icon"; icon.textContent = file.name.endsWith(".json") ? "{}" : file.name.endsWith(".md") ? "¶" : "▤";
    const copy = document.createElement("span"); copy.className = "file-chip-copy";
    const name = document.createElement("b"); name.textContent = file.name;
    const size = document.createElement("small"); size.textContent = fileSize(file.size);
    copy.append(name, size);
    const download = document.createElement("a"); download.className = "file-download"; download.href = `/api/download?path=${encodeURIComponent(file.path)}`; download.download = file.name; download.title = `Download ${file.name}`; download.textContent = "↓";
    chip.append(icon, copy, download); list.append(chip);
  }
}

function renderPreview(preview) {
  state.preview = preview;
  const transcript = $("#transcript-content");
  if (preview?.transcript) {
    transcript.textContent = preview.transcript;
    $("#source-name").textContent = preview.transcript_txt.split(/[\\/]/).at(-2) || "Transcript";
    $("#copy-transcript").disabled = false;
  } else {
    transcript.innerHTML = '<div class="review-placeholder"><span>▤</span><p>Select a completed run or transcript<br>to read the source here.</p></div>';
    $("#source-name").textContent = "Your original, in context";
    $("#copy-transcript").disabled = true;
  }
  $("#output-name").textContent = preview?.content ? "Generated from this transcript" : "No generated content yet";
  $("#regenerate-button").disabled = !preview?.transcript_txt;
  renderOutput();
  renderFiles(preview?.files || []);
}

async function selectJob(jobId) {
  state.selectedKey = `job:${jobId}`;
  state.selectedJob = state.jobs.find((job) => job.id === jobId) || null;
  state.previewSignature = "";
  renderRecentRuns(); renderReviewOptions();
  $("#review-select").value = state.selectedKey;
  $("#log-job-label").textContent = state.selectedJob ? `${shortLabel(state.selectedJob)} · ${state.selectedJob.status}` : "";
  await refreshSelected(true);
}

async function selectLibrary(path) {
  state.selectedKey = `file:${path}`;
  state.selectedJob = null;
  state.previewSignature = "library";
  renderReviewOptions(); $("#review-select").value = state.selectedKey;
  $("#log-job-label").textContent = "Saved transcript · not tied to a run";
  $("#copy-log").disabled = true;
  $("#log-content").textContent = "This transcript is from your existing output library. Queue a regeneration to see generation logs here.";
  try {
    const preview = await api(`/api/library/preview?path=${encodeURIComponent(path)}`);
    renderPreview(preview);
  } catch (error) { toast(error.message, "error"); }
}

async function refreshSelected(forcePreview = false) {
  if (!state.selectedKey.startsWith("job:")) return;
  const jobId = state.selectedKey.slice(4);
  const job = state.jobs.find((item) => item.id === jobId);
  if (!job) return;
  state.selectedJob = job;
  $("#log-job-label").textContent = `${shortLabel(job)} · ${job.status}${job.error ? ` · ${job.error}` : ""}`;
  $("#copy-log").disabled = false;
  const signature = [job.status, job.stage, job.transcript_txt, job.content_dir].join("|");
  if (forcePreview || signature !== state.previewSignature) {
    state.previewSignature = signature;
    try { renderPreview(await api(`/api/jobs/${jobId}/preview`)); }
    catch (error) { if (job.status !== "queued" && job.status !== "running") toast(error.message, "error"); }
  }
  try {
    const data = await api(`/api/jobs/${jobId}/log`);
    state.log = data.log;
    const logNode = $("#log-content");
    const atBottom = logNode.scrollHeight - logNode.scrollTop - logNode.clientHeight < 35;
    logNode.textContent = state.log || "Waiting for the first line of output…";
    if (atBottom) logNode.scrollTop = logNode.scrollHeight;
  } catch { /* Job may have disappeared between polls. */ }
}

function renderState(data) {
  const previousSelection = state.selectedKey;
  state.jobs = data.jobs || [];
  state.library = data.library || [];
  initSettings(data.settings);
  renderRecentRuns(); renderReviewOptions(); renderQueue();
  if (!state.selectedKey) {
    const preferred = state.jobs.find((job) => job.status === "running") || state.jobs.find((job) => job.status === "queued") || state.jobs.find((job) => job.status === "completed") || state.jobs[0];
    if (preferred) state.selectedKey = `job:${preferred.id}`;
    else if (state.library[0]) state.selectedKey = `file:${state.library[0].transcript_path}`;
  }
  if (state.selectedKey !== previousSelection) state.previewSignature = "";
  if (state.selectedKey.startsWith("job:")) {
    const id = state.selectedKey.slice(4);
    if ($("#review-select").value !== state.selectedKey) $("#review-select").value = state.selectedKey;
    refreshSelected(false);
  } else if (state.selectedKey.startsWith("file:")) {
    const path = state.selectedKey.slice(5);
    if ($("#review-select").value !== state.selectedKey) selectLibrary(path);
  }
}

async function poll() {
  try {
    const indicator = $(".server-indicator");
    indicator.classList.remove("disconnected");
    indicator.lastChild.textContent = " Local server connected";
    renderState(await api("/api/state"));
  }
  catch (error) { $(".server-indicator").classList.add("disconnected"); $(".server-indicator").lastChild.textContent = " Reconnecting"; }
}

async function startRun(event) {
  event.preventDefault(); setFormError("");
  const button = $("#start-button"); button.disabled = true; button.querySelector("span:first-child").textContent = "Adding to queue…";
  const urls = $("#video-url").value.split(/\r?\n/).map((value) => value.trim()).filter(Boolean);
  if (!urls.length || urls.length > 10) { setFormError("Paste between 1 and 10 video links, one per line."); button.disabled = false; button.querySelector("span:first-child").textContent = "Start content run"; return; }
  const payload = {
    urls,
    transcription_model: $("#transcription-model").value,
    batch_size: Number($("#batch-size").value),
    gemini_model: $("#gemini-model").value.trim(),
    temperature: Number($("#temperature").value),
    generation_retries: Number($("#generation-retries").value),
    output_folder: $("#output-folder").value.trim(),
  };
  try {
    const result = await api("/api/jobs/batch", { method: "POST", body: JSON.stringify(payload) });
    const job = result.jobs[0];
    $("#video-url").value = ""; $("#clear-url").hidden = true;
    toast(`${result.jobs.length} link${result.jobs.length === 1 ? "" : "s"} added to the queue`);
    state.selectedKey = `job:${job.id}`; state.previewSignature = "";
    await poll(); await selectJob(job.id);
    document.querySelector("#runs").scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (error) { setFormError(error.message); }
  finally { button.disabled = false; button.querySelector("span:first-child").textContent = "Start content run"; }
}

async function regenerate() {
  if (!state.preview?.transcript_txt) return;
  const button = $("#regenerate-button"); button.disabled = true;
  try {
    const job = await api("/api/regenerate", { method: "POST", body: JSON.stringify({ transcript_path: state.preview.transcript_txt, gemini_model: $("#gemini-model").value.trim(), temperature: Number($("#temperature").value), generation_retries: Number($("#generation-retries").value) }) });
    state.selectedKey = `job:${job.id}`; state.previewSignature = "";
    toast("Regeneration added to the queue");
    await poll(); await selectJob(job.id);
  } catch (error) { toast(error.message, "error"); }
  finally { button.disabled = !state.preview?.transcript_txt; }
}

async function copyToClipboard(text, success = "Copied to clipboard") {
  try { await navigator.clipboard.writeText(text); toast(success); }
  catch { toast("Clipboard access was blocked by the browser.", "error"); }
}

async function cancelJob(jobId) {
  try { await api(`/api/jobs/${jobId}/cancel`, { method: "POST", body: "{}" }); toast("Job cancelled"); await poll(); }
  catch (error) { toast(error.message, "error"); }
}

function bindEvents() {
  $("#theme-toggle").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
  $("#save-gemini-key").addEventListener("click", () => updateGeminiKey());
  $("#remove-gemini-key").addEventListener("click", () => updateGeminiKey(true));
  $("#gemini-api-key").addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); updateGeminiKey(); } });
  $("#run-form").addEventListener("submit", startRun);
  $("#refresh-button").addEventListener("click", poll);
  $("#active-run").addEventListener("click", (event) => { const button = event.target.closest("[data-cancel]"); if (button) cancelJob(button.dataset.cancel); });
  $("#queue-list").addEventListener("click", (event) => { const button = event.target.closest("[data-cancel]"); if (button) cancelJob(button.dataset.cancel); });
  $("#review-select").addEventListener("change", (event) => {
    const value = event.target.value;
    if (value.startsWith("job:")) selectJob(value.slice(4));
    else if (value.startsWith("file:")) selectLibrary(value.slice(5));
    else { state.selectedKey = ""; renderPreview(null); }
  });
  $("#regenerate-button").addEventListener("click", regenerate);
  $("#copy-transcript").addEventListener("click", () => copyToClipboard(state.preview?.transcript || "", "Transcript copied"));
  $("#copy-log").addEventListener("click", () => copyToClipboard(state.log, "Run log copied"));
  $("#clear-url").addEventListener("click", () => { $("#video-url").value = ""; $("#clear-url").hidden = true; $("#video-url").focus(); });
  $("#video-url").addEventListener("input", (event) => { $("#clear-url").hidden = !event.target.value; setFormError(""); });
  $("#temperature").addEventListener("input", (event) => { $("#temperature-value").value = Number(event.target.value).toFixed(1); $("#temperature-value").textContent = Number(event.target.value).toFixed(1); });
}

initTheme();
bindEvents();
poll();
state.pollTimer = window.setInterval(poll, 1500);
