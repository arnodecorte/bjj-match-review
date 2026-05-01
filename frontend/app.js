/**
 * BJJ Match Review — app.js
 * Handles upload → polling → results display.
 */

"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let currentJobId = null;
let pollTimer = null;
let positionLog = [];    // [{timestamp, position, display_name, confidence, color}]
let videoDuration = 0;
let serverMeta = {};     // labels, colors, display_names from /api/meta

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const $ = id => document.getElementById(id);

const dropZone        = $("drop-zone");
const fileInput       = $("file-input");
const browseBtn       = $("browse-btn");
const uploadBtn       = $("upload-btn");
const changeBtn       = $("change-btn");
const fileNameEl      = $("file-name");
const fileInfoEl      = $("file-info");
const uploadSection   = $("upload-section");
const processingSection = $("processing-section");
const resultsSection  = $("results-section");
const progressBar     = $("progress-bar");
const progressLabel   = $("progress-label");
const statusMessage   = $("status-message");
const modelBanner     = $("model-banner");
const videoPlayer     = $("video-player");
const posOverlay      = $("pos-overlay");
const overlayLabel    = $("overlay-label");
const overlayConf     = $("overlay-conf");
const timelineEl      = $("timeline");
const legendEl        = $("timeline-legend");
const logBody         = $("log-body");
const newBtn          = $("new-btn");

// ---------------------------------------------------------------------------
// Bootstrap: fetch server metadata
// ---------------------------------------------------------------------------
(async () => {
  try {
    const res = await fetch("/api/meta");
    if (res.ok) serverMeta = await res.json();
  } catch (_) {
    // Server not available yet — that's fine for local file open
  }
})();

// ---------------------------------------------------------------------------
// Drop-zone interactions
// ---------------------------------------------------------------------------
dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("dragging");
  const file = e.dataTransfer.files[0];
  if (file) selectFile(file);
});
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
browseBtn.addEventListener("click", e => { e.stopPropagation(); fileInput.click(); });
fileInput.addEventListener("change", () => { if (fileInput.files[0]) selectFile(fileInput.files[0]); });

changeBtn.addEventListener("click", () => {
  fileInput.value = "";
  fileInfoEl.classList.add("hidden");
  dropZone.classList.remove("hidden");
  uploadBtn.disabled = true;
});

let selectedFile = null;

function selectFile(file) {
  if (!file.type.startsWith("video/")) {
    alert("Please choose a video file.");
    return;
  }
  selectedFile = file;
  fileNameEl.textContent = file.name;
  fileInfoEl.classList.remove("hidden");
  dropZone.classList.add("hidden");
  uploadBtn.disabled = false;
}

// ---------------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------------
uploadBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  uploadBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", selectedFile);

  showProcessing("Uploading…");

  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Upload failed");
    }
    const data = await res.json();
    currentJobId = data.job_id;
    startPolling();
  } catch (err) {
    alert("Upload error: " + err.message);
    uploadBtn.disabled = false;
    showUpload();
  }
});

// ---------------------------------------------------------------------------
// Polling
// ---------------------------------------------------------------------------
function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(poll, 1500);
  poll(); // immediate first check
}

async function poll() {
  if (!currentJobId) return;
  try {
    const res = await fetch(`/api/status/${currentJobId}`);
    if (!res.ok) return;
    const data = await res.json();

    const pct = data.progress || 0;
    progressBar.style.width = pct + "%";
    progressLabel.textContent = pct + "%";
    const statusMessages = {
      queued:     "Queued — waiting for worker…",
      processing: `Processing video (${pct}%)…`,
      done:       "Done!",
      error:      "Error: " + (data.error || "unknown"),
    };
    statusMessage.textContent = statusMessages[data.status] ?? `Status: ${data.status}`;

    if (data.status === "done") {
      clearInterval(pollTimer);
      await loadResults();
    } else if (data.status === "error") {
      clearInterval(pollTimer);
      alert("Processing failed: " + (data.error || "unknown error"));
      showUpload();
    }
  } catch (_) { /* network blip, ignore */ }
}

// ---------------------------------------------------------------------------
// Load & render results
// ---------------------------------------------------------------------------
async function loadResults() {
  const res = await fetch(`/api/results/${currentJobId}`);
  if (!res.ok) { alert("Could not load results."); showUpload(); return; }
  const data = await res.json();

  positionLog = data.results || [];

  // Banner: heuristic vs trained model
  modelBanner.textContent = data.using_heuristic
    ? "⚠️  Using heuristic classifier (geometry-based). Train a model with the ViCoS dataset for higher accuracy."
    : "✅  Using trained MLP classifier.";
  modelBanner.className = "model-banner " + (data.using_heuristic ? "heuristic" : "mlp");

  // Video
  videoPlayer.src = `/api/video/${currentJobId}`;
  videoPlayer.load();
  videoPlayer.onloadedmetadata = () => { videoDuration = videoPlayer.duration; };

  renderTimeline();
  renderLogTable();

  showResults();
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------
function renderTimeline() {
  timelineEl.innerHTML = "";
  legendEl.innerHTML = "";
  if (!positionLog.length) return;

  const duration = positionLog[positionLog.length - 1].timestamp || 1;
  const seen = new Set();

  positionLog.forEach((entry, i) => {
    const next = positionLog[i + 1];
    const segDuration = next ? (next.timestamp - entry.timestamp) : 5; // last segment
    const widthPct = (segDuration / duration) * 100;

    const seg = document.createElement("div");
    seg.className = "timeline-segment";
    seg.style.width = widthPct + "%";
    seg.style.background = entry.color || "#888";
    seg.title = `${fmtTime(entry.timestamp)} — ${entry.display_name} (${pct(entry.confidence)}%)`;
    seg.dataset.index = i;

    seg.addEventListener("click", () => {
      videoPlayer.currentTime = entry.timestamp;
      videoPlayer.play();
      highlightRow(i);
    });

    timelineEl.appendChild(seg);

    // Legend
    if (!seen.has(entry.position)) {
      seen.add(entry.position);
      const item = document.createElement("div");
      item.className = "legend-item";
      item.innerHTML = `
        <span class="legend-dot" style="background:${entry.color}"></span>
        ${entry.display_name}
      `;
      legendEl.appendChild(item);
    }
  });
}

// ---------------------------------------------------------------------------
// Log table
// ---------------------------------------------------------------------------
function renderLogTable() {
  logBody.innerHTML = "";
  positionLog.forEach((entry, i) => {
    const tr = document.createElement("tr");
    tr.id = "row-" + i;
    tr.innerHTML = `
      <td>${fmtTime(entry.timestamp)}</td>
      <td>
        <span class="pos-chip" style="background:${entry.color || "#555"}">
          ${entry.display_name}
        </span>
      </td>
      <td>
        <div class="conf-bar-wrap">
          <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:${pct(entry.confidence)}%;background:${entry.color}"></div>
          </div>
          <span class="conf-pct">${pct(entry.confidence)}%</span>
        </div>
      </td>
    `;
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => {
      videoPlayer.currentTime = entry.timestamp;
      videoPlayer.play();
    });
    logBody.appendChild(tr);
  });
}

function highlightRow(index) {
  document.querySelectorAll(".active-row").forEach(el => el.classList.remove("active-row"));
  const row = $("row-" + index);
  if (row) {
    row.classList.add("active-row");
    row.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

// ---------------------------------------------------------------------------
// Video timeupdate → position overlay
// ---------------------------------------------------------------------------
videoPlayer.addEventListener("timeupdate", () => {
  const t = videoPlayer.currentTime;
  const entry = currentEntry(t);
  if (entry) {
    posOverlay.classList.remove("hidden");
    posOverlay.style.background = hexToRgba(entry.color || "#333", 0.72);
    overlayLabel.textContent = entry.display_name;
    overlayConf.textContent = pct(entry.confidence) + "%";

    // Highlight corresponding row
    const idx = positionLog.indexOf(entry);
    if (idx >= 0) highlightRow(idx);
  } else {
    posOverlay.classList.add("hidden");
  }
});

function currentEntry(time) {
  let result = null;
  for (const entry of positionLog) {
    if (entry.timestamp <= time + 0.01) result = entry;
    else break;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Navigation helpers
// ---------------------------------------------------------------------------
newBtn.addEventListener("click", reset);

function reset() {
  selectedFile = null;
  currentJobId = null;
  positionLog = [];
  fileInput.value = "";
  fileInfoEl.classList.add("hidden");
  dropZone.classList.remove("hidden");
  uploadBtn.disabled = true;
  videoPlayer.src = "";
  posOverlay.classList.add("hidden");
  showUpload();
}

function showUpload() {
  uploadSection.classList.remove("hidden");
  processingSection.classList.add("hidden");
  resultsSection.classList.add("hidden");
}

function showProcessing(msg) {
  uploadSection.classList.add("hidden");
  processingSection.classList.remove("hidden");
  resultsSection.classList.add("hidden");
  statusMessage.textContent = msg || "";
  progressBar.style.width = "0%";
  progressLabel.textContent = "0%";
}

function showResults() {
  uploadSection.classList.add("hidden");
  processingSection.classList.add("hidden");
  resultsSection.classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
function fmtTime(secs) {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function pct(conf) {
  return Math.round((conf || 0) * 100);
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
