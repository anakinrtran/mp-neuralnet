// Workspace 1 page: block editor, generated code, runs and results.
// API paths are relative so the page also works behind PrairieLearn's workspace proxy.

import { defineBlocks } from "./blocks.js";
import { checkProgram, generateCode } from "./generators.js";

const $ = (id) => document.getElementById(id);

let cfg;
let ws; // this workspace's module: workspaces/ws1.js, ...
let workspace;
let check = { problems: [], hints: [], notes: [] };
let history = [];
let running = false;
let source = null; // EventSource of the run being shown
let shownRun = null;
let runStarted = 0;
let weightsKind = "templates";
const weightsCache = new Map();
let practiceGrid = null;
let charsImage = null;
let submission = null;

async function api(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

// ---------- blocks and code ----------

// Blocks are saved in the workspace's home folder, so they survive a restart or another computer.
let saveTimer = null;
function saveBlocks() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    api("api/blocks", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ blocks: Blockly.serialization.workspaces.save(workspace) }),
    }).catch(() => { $("status").textContent = "Couldn't save your blocks. Check your connection."; });
  }, 400);
}

function loadBlocks(state) {
  try {
    Blockly.serialization.workspaces.load(state, workspace);
  } catch {
    Blockly.serialization.workspaces.load(ws.START_STATE, workspace);
  }
}

function highlight(code) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const re = /(#.*$)|\b(from|import)\b|\b(\d+)\b|\b(load_data|train|evaluate|decode_practice)(?=\()|(nn\.\w+)/gm;
  let out = "";
  let last = 0;
  for (const m of code.matchAll(re)) {
    out += esc(code.slice(last, m.index));
    const cls = m[1] ? "c" : m[2] ? "k" : m[3] ? "n" : "f";
    out += `<span class="${cls}">${esc(m[0])}</span>`;
    last = m.index + m[0].length;
  }
  return out + esc(code.slice(last));
}

function updateCode() {
  $("code").innerHTML = highlight(generateCode(workspace));
  check = checkProgram(workspace, cfg, ws.checkModel);
  const lines = check.problems.length
    ? check.problems.map((p) => `<p class="problem">⚠ ${p}</p>`)
    : ['<p class="ok">✓ Ready to run.</p>'];
  lines.push(...check.hints.map((h) => `<p class="hint">💡 ${h}</p>`));
  lines.push(...check.notes.map((n) => `<p class="muted">${n}</p>`));
  $("checks").innerHTML = lines.join("");
  updateButtons();
}

function updateButtons() {
  $("run").disabled = running || check.problems.length > 0;
  $("run").hidden = running;
  $("stop").hidden = !running;
  $("submit").disabled = running || !latestFinished();
}

// ---------- tabs ----------

function showTab(name) {
  for (const b of document.querySelectorAll(".tabs button")) {
    const on = b.dataset.tab === name;
    b.setAttribute("aria-selected", on);
    if (on) b.classList.remove("fresh");
  }
  for (const p of document.querySelectorAll(".panel")) p.hidden = p.id !== `tab-${name}`;
  if (name === "training") drawCharts();
  if (name === "weights") renderWeights();
}

function markFresh(name) {
  const b = document.querySelector(`.tabs button[data-tab="${name}"]`);
  if (b.getAttribute("aria-selected") !== "true") b.classList.add("fresh");
}

// ---------- runs ----------

async function startRun() {
  if (running || check.problems.length) return;
  try {
    const run = await api("api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: generateCode(workspace) }),
    });
    setRunning(true);
    showTab("training");
    watchRun(run.id, true);
  } catch (e) {
    $("status").textContent = e.message;
  }
}

function setRunning(on) {
  running = on;
  if (on) runStarted = Date.now();
  updateButtons();
}

function resetRunView() {
  charts.loss.points = [];
  charts.acc.points = [];
  $("loss-value").textContent = "";
  $("acc-value").textContent = "";
  $("run-error").hidden = true;
  $("run-output").hidden = true;
  $("run-output").textContent = "";
  $("practice-view").hidden = true;
  $("practice-empty").hidden = false;
  practiceGrid = null;
}

/** Show a run's events. Finished runs replay their saved events. */
function watchRun(id, live) {
  if (source) source.close();
  shownRun = id;
  resetRunView();
  source = new EventSource(`api/runs/${id}/events`);
  source.onmessage = (m) => handleEvent(JSON.parse(m.data), live);
  source.onerror = () => {
    // the browser would reconnect and replay from the start; do that ourselves, cleanly
    source.close();
    if (running) setTimeout(() => watchRun(id, live), 1000);
  };
}

function handleEvent(e, live) {
  switch (e.event) {
    case "train_start":
      charts.loss.xMax = charts.acc.xMax = e.epochs;
      $("train-progress").textContent = `Training: epoch 1 of ${e.epochs}…`;
      break;
    case "step":
      charts.loss.points.push([e.progress, e.loss]);
      $("loss-value").textContent = e.loss.toFixed(3);
      scheduleDraw();
      break;
    case "val":
      charts.acc.points.push([e.progress, e.accuracy]);
      $("acc-value").textContent = pct(e.accuracy);
      scheduleDraw();
      break;
    case "epoch":
      if (e.epoch < charts.acc.xMax) {
        $("train-progress").textContent = `Training: epoch ${e.epoch + 1} of ${charts.acc.xMax}…`;
      }
      break;
    case "evaluate":
      $("acc-value").textContent = pct(e.accuracy);
      break;
    case "decoded":
      showPractice(e.grid);
      if (live) markFresh("practice");
      break;
    case "output":
      $("run-output").hidden = false;
      $("run-output").textContent += e.text + "\n";
      break;
    case "error":
      $("run-error").hidden = false;
      $("run-error").textContent = e.message;
      break;
    case "end":
      source.close();
      finishRun(e.status, live);
      break;
  }
}

function finishRun(status, live) {
  const words = { done: "Finished", stopped: "Stopped", error: "Stopped with an error" };
  const secs = Math.round((Date.now() - runStarted) / 1000);
  $("train-progress").textContent = live ? `${words[status]} after ${secs}s.` : `${words[status]} (latest run).`;
  if (live) {
    setRunning(false);
    $("status").textContent = words[status];
    if (status === "done") {
      markFresh("compare");
      markFresh("weights");
    }
  }
  refreshHistory();
}

setInterval(() => {
  if (running) $("status").textContent = `Running… ${Math.round((Date.now() - runStarted) / 1000)}s`;
}, 500);

// ---------- submit ----------

const latestFinished = () => history.filter((r) => r.status === "done").pop();
const runNumber = (id) => history.findIndex((r) => r.id === id) + 1;

async function submitLatest() {
  const run = latestFinished();
  if (!run) return;
  try {
    ({ submission } = await api("api/submission", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: run.id }),
    }));
  } catch (e) {
    $("status").textContent = e.message;
    return;
  }
  $("status").textContent = `Submitted run ${runNumber(run.id)}.`;
  showSubmission();
  refreshHistory();
  showTab("steps");
}

function showSubmission() {
  const note = $("submit-note");
  if (!submission) {
    note.innerHTML = "Click <b>Submit</b> at the top. It saves your latest finished run for grading.";
    return;
  }
  const n = runNumber(submission.run_id);
  note.innerHTML = `<span class="ok">✓ Submitted run ${n || ""} ` +
    `(${layersText(submission.layers)}, ${pct(submission.val_accuracy)} validation accuracy).</span> ` +
    "Now go back to the PrairieLearn question page and click <b>Save &amp; Grade</b>. " +
    "To submit a better model later, run it and click Submit again.";
}

// ---------- charts ----------

const pct = (v) => `${(v * 100).toFixed(1)}%`;

const charts = {
  loss: { canvas: null, points: [], xMax: 1, hover: null, fmt: (v) => v.toFixed(2), yMax: null },
  acc: { canvas: null, points: [], xMax: 1, hover: null, fmt: (v) => `${Math.round(v * 100)}%`, yMax: 1 },
};

let drawPending = false;
function scheduleDraw() {
  if (drawPending) return;
  drawPending = true;
  requestAnimationFrame(() => { drawPending = false; drawCharts(); });
}

function drawCharts() {
  if ($("tab-training").hidden) return;
  for (const c of Object.values(charts)) drawChart(c);
}

function drawChart(chart) {
  const { canvas, points } = chart;
  const css = getComputedStyle(document.documentElement);
  const color = (name) => css.getPropertyValue(name).trim();
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = 160;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const pad = { l: 42, r: 10, t: 8, b: 36 };
  const yMax = chart.yMax ?? Math.max(0.5, ...points.map((p) => p[1])) * 1.1;
  const x = (v) => pad.l + (v / chart.xMax) * (w - pad.l - pad.r);
  const y = (v) => pad.t + (1 - v / yMax) * (h - pad.t - pad.b);

  ctx.font = "11px system-ui, sans-serif";
  ctx.fillStyle = color("--ink-muted");
  ctx.strokeStyle = color("--chart-grid");
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const v = (yMax * i) / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y(v)); ctx.lineTo(w - pad.r, y(v)); ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(chart.fmt(v), pad.l - 6, y(v) + 4);
  }
  ctx.textAlign = "center";
  const every = Math.ceil(chart.xMax / 10);
  for (let e = 0; e <= chart.xMax; e += every) ctx.fillText(e, x(e), h - pad.b + 14);
  ctx.fillText("epochs", (pad.l + w - pad.r) / 2, h - 4);

  if (points.length) {
    ctx.strokeStyle = color("--chart-line");
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    points.forEach(([px, py], i) => (i ? ctx.lineTo(x(px), y(py)) : ctx.moveTo(x(px), y(py))));
    ctx.stroke();
  }

  const tip = chart.tip;
  if (chart.hover != null && points[chart.hover]) {
    const [px, py] = points[chart.hover];
    ctx.strokeStyle = color("--ink-muted");
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x(px), pad.t); ctx.lineTo(x(px), h - pad.b); ctx.stroke();
    ctx.fillStyle = color("--chart-line");
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x(px), y(py), 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    tip.hidden = false;
    tip.textContent = `epoch ${px.toFixed(2)}: ${chart === charts.acc ? pct(py) : py.toFixed(3)}`;
    tip.style.left = `${Math.min(x(px) + 8, w - tip.offsetWidth)}px`;
    tip.style.top = `${y(py) + 4}px`;
  } else {
    tip.hidden = true;
  }
}

function setupChart(chart, canvasId) {
  chart.canvas = $(canvasId);
  chart.tip = document.createElement("div");
  chart.tip.className = "tooltip";
  chart.tip.hidden = true;
  chart.canvas.parentElement.appendChild(chart.tip);
  chart.canvas.addEventListener("mousemove", (ev) => {
    if (!chart.points.length) return;
    const w = chart.canvas.clientWidth;
    const xv = ((ev.offsetX - 42) / (w - 52)) * chart.xMax;
    let best = 0;
    chart.points.forEach(([px], i) => {
      if (Math.abs(px - xv) < Math.abs(chart.points[best][0] - xv)) best = i;
    });
    chart.hover = best;
    drawChart(chart);
  });
  chart.canvas.addEventListener("mouseleave", () => { chart.hover = null; drawChart(chart); });
}

// ---------- compare runs ----------

const layersText = (layers) => (layers ? layers.filter((l) => l !== "Flatten").join(" → ") : "—");

async function refreshHistory() {
  try {
    history = await api("api/runs");
  } catch {
    return;
  }
  const finished = history.filter((r) => r.status === "done" && r.val_accuracy != null);
  const card = (title, run, delta) => `
    <div class="card">
      <h3>${title}</h3>
      <div class="big">${pct(run.val_accuracy)}</div>
      ${delta ?? ""}
      <div class="layers">${layersText(run.layers)} · ${run.epochs} epochs</div>
    </div>`;
  if (finished.length === 0) {
    $("compare-cards").innerHTML = '<p class="muted">Finished runs show up here.</p>';
  } else if (finished.length === 1) {
    $("compare-cards").innerHTML = card("First run", finished[0]) +
      '<p class="muted">Add a hidden layer and run again to compare.</p>';
  } else {
    const first = finished[0];
    const latest = finished[finished.length - 1];
    const d = (latest.val_accuracy - first.val_accuracy) * 100;
    const delta = `<div class="delta">${d >= 0 ? "▲" : "▼"} ${Math.abs(d).toFixed(1)} points vs. first run</div>`;
    $("compare-cards").innerHTML = card("First run", first) + card("Latest run", latest, delta);
  }
  $("history").querySelector("tbody").innerHTML = history.map((r, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${layersText(r.layers)}</td>
      <td class="num">${r.epochs ?? "—"}</td>
      <td class="num">${r.val_accuracy != null ? pct(r.val_accuracy) : "—"}</td>
      <td>${r.status}${submission?.run_id === r.id ? ' <span class="badge">submitted</span>' : ""}</td>
    </tr>`).join("");
  updateButtons();
  if (!$("tab-weights").hidden) renderWeights();
}

// ---------- weights ----------

const linearCount = (run) => (run.layers || []).filter((l) => l.startsWith("Linear")).length;

async function renderWeights() {
  for (const b of document.querySelectorAll(".subtabs button")) {
    b.setAttribute("aria-pressed", b.dataset.kind === weightsKind);
  }
  const done = history.filter((r) => r.status === "done");
  const grid = $("weights-grid");
  const note = $("weights-note");
  let run;
  if (weightsKind === "templates") {
    run = done.filter((r) => linearCount(r) === 1).pop();
    if (!run) {
      grid.innerHTML = "";
      note.textContent = "Run a model with just one Linear layer (like the starting model) to see class templates.";
      return;
    }
    note.textContent = "Each picture shows the weights of the linear model for one class. Red pixels raise that " +
      "class's score when there is ink there; blue pixels lower it. The model picks the class whose template " +
      "matches the image best.";
  } else {
    run = done[done.length - 1];
    if (!run || linearCount(run) < 2) {
      grid.innerHTML = "";
      note.textContent = "Your latest model has no hidden layer. Add Linear and ReLU blocks before Linear 36, " +
        "run it, then come back.";
      return;
    }
  }

  const key = `${run.id}:${weightsKind}`;
  if (!weightsCache.has(key)) {
    try {
      weightsCache.set(key, await api(`api/runs/${run.id}/weights?kind=${weightsKind}`));
    } catch (e) {
      note.textContent = e.message;
      return;
    }
  }
  const data = weightsCache.get(key);
  if (weightsKind === "features") {
    note.textContent = `Each picture is one unit of the hidden layer (showing ${data.units.length} of ` +
      `${data.total_units}). These are not classes: each unit responds to a stroke or shape, and the ` +
      "output layer combines them into class scores.";
  }
  grid.innerHTML = "";
  data.units.forEach((values, i) => {
    const fig = document.createElement("figure");
    fig.appendChild(weightCanvas(values));
    fig.append(weightsKind === "templates" ? cfg.classes[i] : `${i + 1}`);
    grid.appendChild(fig);
  });
}

function weightCanvas(values) {
  const c = document.createElement("canvas");
  c.width = 28;
  c.height = 28;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(28, 28);
  const pos = [192, 57, 43];
  const neg = [46, 111, 181];
  values.forEach((v, i) => {
    const t = Math.abs(v) / 100;
    const col = v >= 0 ? pos : neg;
    for (let k = 0; k < 3; k++) img.data[i * 4 + k] = Math.round(255 + (col[k] - 255) * t);
    img.data[i * 4 + 3] = 255;
  });
  ctx.putImageData(img, 0, 0);
  return c;
}

// ---------- practice image ----------

const hexToRgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

function showPractice(grid) {
  practiceGrid = grid;
  $("practice-empty").hidden = true;
  $("practice-view").hidden = false;
  const canvas = $("practice-canvas");
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(32, 32);
  const rgb = cfg.colors.map(hexToRgb);
  grid.flat().forEach((c, i) => {
    img.data.set([...rgb[c], 255], i * 4);
  });
  ctx.putImageData(img, 0, 0);
}

function hoverPractice(ev) {
  if (!practiceGrid) return;
  const canvas = $("practice-canvas");
  const col = Math.min(31, Math.floor((ev.offsetX / canvas.clientWidth) * 32));
  const row = Math.min(31, Math.floor((ev.offsetY / canvas.clientHeight) * 32));
  if (!charsImage) {
    charsImage = new Image();
    charsImage.src = "api/practice/characters.png";
    charsImage.onload = () => hoverPractice(ev);
    return;
  }
  if (!charsImage.complete) return;
  const ctx = $("char-canvas").getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(charsImage, col * 28, row * 28, 28, 28, 0, 0, 28, 28);
  const cls = practiceGrid[row][col];
  $("char-label").innerHTML = `Your model read:<br><span class="class-chip" style="background:${cfg.colors[cls]}">` +
    `</span><b>${cfg.classes[cls]}</b>`;
}

// ---------- start up ----------

/** Fill in the parts of the page that differ between workspaces. */
function setUpPage() {
  const number = cfg.workspace.replace("ws", "");
  document.title = `Workspace ${number}: ${cfg.title}`;
  $("ws-number").textContent = number;
  $("ws-title").textContent = cfg.title;
  $("tab-steps").innerHTML = ws.STEPS;
  for (const b of document.querySelectorAll(".tabs button")) b.hidden = !ws.TABS.includes(b.dataset.tab);
}

async function main() {
  cfg = await api("api/config");
  ws = await import(`./workspaces/${cfg.workspace}.js`);
  setUpPage();
  defineBlocks(cfg);
  workspace = Blockly.inject("blockly", {
    toolbox: ws.TOOLBOX,
    renderer: "zelos",
    media: "vendor/blockly/media/",
    sounds: false,
    trashcan: true,
    maxInstances: ws.MAX_INSTANCES,
    zoom: { controls: true, wheel: false, startScale: 0.75 },
    move: { scrollbars: true, drag: true, wheel: true },
    grid: { spacing: 24, length: 2, colour: "#e3e6ea", snap: false },
  });
  const saved = await api("api/blocks").catch(() => ({ blocks: null }));
  loadBlocks(saved.blocks || ws.START_STATE);
  workspace.addChangeListener(Blockly.Events.disableOrphans);
  workspace.addChangeListener((e) => {
    if (e.isUiEvent) return;
    updateCode();
    saveBlocks();
  });
  new ResizeObserver(() => Blockly.svgResize(workspace)).observe($("blockly"));
  updateCode();

  setupChart(charts.loss, "loss-chart");
  setupChart(charts.acc, "acc-chart");
  window.addEventListener("resize", drawCharts);

  for (const b of document.querySelectorAll(".tabs button")) b.onclick = () => showTab(b.dataset.tab);
  for (const b of document.querySelectorAll(".subtabs button")) {
    b.onclick = () => { weightsKind = b.dataset.kind; renderWeights(); };
  }
  $("run").onclick = startRun;
  $("stop").onclick = () => shownRun && api(`api/runs/${shownRun}/stop`, { method: "POST" });
  $("reset").onclick = () => {
    if (confirm("Replace your blocks with the starting linear model?")) {
      workspace.clear();
      loadBlocks(ws.START_STATE);
    }
  };
  $("practice-canvas").addEventListener("mousemove", hoverPractice);
  $("submit").onclick = submitLatest;

  // Pick up where we left off: reattach to a run in progress, or show the latest one.
  ({ submission } = await api("api/submission").catch(() => ({ submission: null })));
  await refreshHistory();
  showSubmission();
  const active = history.find((r) => r.status === "running");
  const latest = history[history.length - 1];
  if (active) {
    setRunning(true);
    runStarted = active.started * 1000;
    watchRun(active.id, true);
  } else if (latest) {
    watchRun(latest.id, false);
  }
}

main().catch((e) => {
  $("checks").innerHTML = `<p class="problem">Could not start the workspace: ${e.message}</p>`;
});
