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
let runInfo = {}; // the shown run's events that the Confusion, Worst mistakes and Model card tabs use

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
  $("code").innerHTML = highlight(generateCode(workspace, cfg));
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
  if (name === "filters") renderFilters();
  if (name === "onestep") renderOneStep();
  if (name === "gradients") renderGradients();
  if (name === "compare") renderOverlay();
  if (name === "confusion") renderConfusion();
  if (name === "mistakes") renderMistakes();
  if (name === "card") renderCard();
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
      body: JSON.stringify({ code: generateCode(workspace, cfg) }),
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
  charts.acc.points2 = [];
  runInfo = {};
  $("acc-legend").hidden = true;
  $("acc-title").textContent = "Validation accuracy";
  $("stop-note").hidden = true;
  gradients = { names: [], xMax: 1, points: [] };
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
    case "model":
    case "settings":
    case "data":
    case "train_done":
      runInfo[e.event] = e;
      break;
    case "train_start":
      runInfo.train_start = e;
      charts.loss.xMax = charts.acc.xMax = e.epochs;
      gradients = { names: e.weighted || [], xMax: e.epochs, perEpoch: e.steps_per_epoch, points: [] };
      $("train-progress").textContent = `Training: epoch 1 of ${e.epochs}…`;
      break;
    case "step":
      charts.loss.points.push([e.progress, e.loss]);
      if (e.updates) gradients.points.push([e.progress, e.updates]);
      if (!$("tab-gradients").hidden) renderGradients();
      $("loss-value").textContent = e.loss.toFixed(3);
      scheduleDraw();
      break;
    case "val":
      charts.acc.points.push([e.progress, e.accuracy]);
      $("acc-value").textContent = pct(e.accuracy);
      if (e.train_accuracy != null) {
        charts.acc.points2.push([e.progress, e.train_accuracy]);
        $("acc-legend").hidden = false;
        $("acc-title").textContent = "Accuracy";
        $("acc-value").textContent = `validation ${pct(e.accuracy)} · training ${pct(e.train_accuracy)}`;
      }
      scheduleDraw();
      break;
    case "early_stop":
      $("stop-note").hidden = false;
      $("stop-note").textContent = `Early stopping: validation accuracy didn't improve for a while, so training ` +
        `stopped after epoch ${e.epoch}. The model keeps its weights from epoch ${e.best_epoch} ` +
        `(${pct(e.accuracy)} validation accuracy).`;
      break;
    case "best_weights":
      if ($("stop-note").hidden) {
        $("stop-note").hidden = false;
        $("stop-note").textContent = `Early stopping never ran out of patience. The model keeps its weights ` +
          `from its best epoch, ${e.epoch} (${pct(e.accuracy)} validation accuracy).`;
      }
      break;
    case "confusion":
    case "mistakes":
      runInfo[e.event] = e;
      if (live) markFresh(e.event);
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
  for (const tab of ["confusion", "mistakes", "card"]) {
    if (!$(`tab-${tab}`).hidden) showTab(tab);  // redraw with the run just shown
  }
  const words = { done: "Finished", stopped: "Stopped", error: "Stopped with an error" };
  const secs = Math.round((Date.now() - runStarted) / 1000);
  $("train-progress").textContent = live ? `${words[status]} after ${secs}s.` : `${words[status]} (latest run).`;
  if (live) {
    setRunning(false);
    if (status !== "stopped") overlayRuns.add(shownRun);  // its curve joins the Compare runs overlay
    $("status").textContent = words[status];
    if (status === "done") {
      markFresh("card");
      markFresh("compare");
      markFresh("weights");
      markFresh("filters");
      markFresh("gradients");
    }
  }
  refreshHistory().then(refreshSubmission);
}

setInterval(() => {
  if (running) $("status").textContent = `Running… ${Math.round((Date.now() - runStarted) / 1000)}s`;
}, 500);

// ---------- grading ----------
// Every finished run is saved for grading by the server; this shows which one.

const runNumber = (id) => history.findIndex((r) => r.id === id) + 1;

async function refreshSubmission() {
  ({ submission } = await api("api/submission").catch(() => ({ submission: null })));
  const chip = $("graded");
  const note = $("submit-note");
  if (!submission) {
    chip.textContent = "";
    if (note) note.textContent = "Finish a run first. Every finished run is saved for grading.";
    return;
  }
  const n = runNumber(submission.run_id);
  chip.textContent = `Save & Grade grades run ${n} · ${pct(submission.val_accuracy)}`;
  if (note) {
    note.innerHTML = `<span class="ok">✓ Run ${n} is saved for grading ` +
      `(${layersText(submission.layers)}, ${pct(submission.val_accuracy)} validation accuracy).</span> ` +
      "Go back to the PrairieLearn question page and click <b>Save &amp; Grade</b>. " +
      "Each new finished run replaces it.";
  }
  for (const td of document.querySelectorAll("#history .badge")) td.remove();
  document.querySelector(`#history tr[data-run="${submission.run_id}"] td:last-child`)
    ?.insertAdjacentHTML("beforeend", ' <span class="badge">graded</span>');
}

// ---------- charts ----------

const pct = (v) => `${(v * 100).toFixed(1)}%`;

const charts = {
  loss: { canvas: null, points: [], xMax: 1, hover: null, fmt: (v) => v.toFixed(2), yMax: null },
  acc: { canvas: null, points: [], points2: [], xMax: 1, hover: null, fmt: (v) => `${Math.round(v * 100)}%`, yMax: 1 },
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

  // points2: a second line (training accuracy, Workspace 4 on), drawn under the first
  for (const [line, name] of [[chart.points2 || [], "--chart-line-2"], [points, "--chart-line"]]) {
    if (!line.length) continue;
    ctx.strokeStyle = color(name);
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    line.forEach(([px, py], i) => (i ? ctx.lineTo(x(px), y(py)) : ctx.moveTo(x(px), y(py))));
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
    const train = chart.points2?.[chart.hover];
    tip.textContent = `epoch ${px.toFixed(2)}: ${chart === charts.acc ? pct(py) : py.toFixed(3)}` +
      (train ? ` (training ${pct(train[1])})` : "");
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
      '<p class="muted">Change something and run again to compare.</p>';
  } else {
    const first = finished[0];
    const latest = finished[finished.length - 1];
    const d = (latest.val_accuracy - first.val_accuracy) * 100;
    const delta = `<div class="delta">${d >= 0 ? "▲" : "▼"} ${Math.abs(d).toFixed(1)} points vs. first run</div>`;
    $("compare-cards").innerHTML = card("First run", first) + card("Latest run", latest, delta);
  }
  const curves = !!ws.COMPARE_CURVES;
  for (const th of document.querySelectorAll("#history .settings-col")) th.hidden = !curves;
  $("history").querySelector("tbody").innerHTML = history.map((r, i) => `
    <tr data-run="${r.id}">
      <td>${curves ? `<label><input type="checkbox" class="pick" data-run="${r.id}" ` +
        `${overlayRuns.has(r.id) ? "checked" : ""} ${r.status === "running" ? "disabled" : ""}> ${i + 1}</label>` : i + 1}</td>
      <td>${layersText(r.layers)}</td>
      ${curves ? `<td>${r.settings ?? "—"}</td>` : ""}
      <td class="num">${r.epochs ?? "—"}</td>
      <td class="num">${r.val_accuracy != null ? pct(r.val_accuracy) : "—"}</td>
      <td>${r.status}${submission?.run_id === r.id ? ' <span class="badge">graded</span>' : ""}</td>
    </tr>`).join("");
  updateButtons();
  if (!$("tab-compare").hidden) renderOverlay();
  if (!$("tab-weights").hidden) renderWeights();
  if (!$("tab-filters").hidden) renderFilters();
}

// ---------- line charts with several lines (overlay, gradients) ----------

const LINE_COLORS = ["#2f6fd6", "#d65f2f", "#2f9a5a", "#9b4fc4", "#b5901f", "#c43f7a", "#3a9cb5", "#6b6b6b"];

/** Draws lines [{ name, points: [[x, y], ...] }] on a canvas; y is log-scaled when `log`,
 * and shown as 0-100% when `percent`. */
function drawLines(canvas, lines, { xMax, log = false, percent = false, height = 160, xLabel = "epochs" }) {
  const css = getComputedStyle(document.documentElement);
  const color = (name) => css.getPropertyValue(name).trim();
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = height;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const pad = { l: 46, r: 10, t: 8, b: 36 };
  const ys = lines.flatMap((l) => l.points.map((p) => p[1])).filter((v) => v != null && Number.isFinite(v));
  // log: decades from 1e-6 (shown as "no change") up to the largest value
  const lo = log ? -6 : 0;
  const hi = log ? Math.max(0, Math.ceil(Math.log10(Math.max(...ys, 1e-6)))) : percent ? 1 : Math.max(0.5, ...ys) * 1.1;
  const t = (v) => (log ? Math.log10(Math.max(v, 1e-6)) : v);
  const x = (v) => pad.l + (v / xMax) * (w - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (t(v) - lo) / (hi - lo)) * (h - pad.t - pad.b);

  ctx.font = "11px system-ui, sans-serif";
  ctx.fillStyle = color("--ink-muted");
  ctx.strokeStyle = color("--chart-grid");
  ctx.lineWidth = 1;
  const ticks = log ? Array.from({ length: hi - lo + 1 }, (_, i) => 10 ** (lo + i)).filter((_, i, a) => a.length < 8 || i % 2 === 0)
    : [0, 1, 2, 3, 4].map((i) => (hi * i) / 4);
  for (const v of ticks) {
    ctx.beginPath(); ctx.moveTo(pad.l, y(v)); ctx.lineTo(w - pad.r, y(v)); ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(log ? (v <= 1e-6 ? "none" : String(v)) : percent ? `${Math.round(v * 100)}%` : v.toFixed(2),
      pad.l - 6, y(v) + 4);
  }
  ctx.textAlign = "center";
  const every = Math.ceil(xMax / 10);
  for (let e = 0; e <= xMax; e += every) ctx.fillText(e, x(e), h - pad.b + 14);
  ctx.fillText(xLabel, (pad.l + w - pad.r) / 2, h - 4);

  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  lines.forEach((line, k) => {
    ctx.strokeStyle = LINE_COLORS[k % LINE_COLORS.length];
    ctx.beginPath();
    let started = false;
    for (const [px, py] of line.points) {
      if (py == null || !Number.isFinite(py)) { started = false; continue; }
      if (started) ctx.lineTo(x(px), y(py)); else ctx.moveTo(x(px), y(py));
      started = true;
    }
    ctx.stroke();
  });
}

function lineLegend(el, names) {
  el.innerHTML = names.map((n, k) =>
    `<span style="--line-color:${LINE_COLORS[k % LINE_COLORS.length]}">${n}</span>`).join("");
}

// ---------- gradients (update size per layer) ----------

let gradients = { names: [], xMax: 1, points: [] };

function renderGradients() {
  const note = $("gradients-note");
  if (!gradients.points.length) {
    $("gradients").hidden = true;
    note.textContent = "Run your model, then come back to see how much each layer changes during training.";
    return;
  }
  note.textContent = shownRun ? `Run ${runNumber(shownRun) || history.length + 1}.` : "";
  $("gradients").hidden = false;
  const lines = gradients.names.map((name, k) => ({ name, points: gradients.points.map(([p, u]) => [p, u[k]]) }));
  drawLines($("gradients-chart"), lines, { xMax: gradients.xMax, log: true, height: 200 });
  lineLegend($("gradients-legend"), gradients.names);
  // the same lines in steps, cut to the start of training
  const zoom = 50;
  const early = lines.map((l) => ({ ...l, points: l.points.map(([p, u]) => [p * gradients.perEpoch, u])
    .filter(([step]) => step <= zoom) }));
  $("gradients-zoom").hidden = !gradients.perEpoch;
  if (gradients.perEpoch) drawLines($("gradients-zoom-chart"), early, { xMax: zoom, log: true, height: 200, xLabel: "steps" });
}

// ---------- loss-curve overlay (Compare runs) ----------

const overlayRuns = new Set();
const curveCache = new Map();

async function renderOverlay() {
  if (!ws.COMPARE_CURVES) return;
  const picked = history.filter((r) => overlayRuns.has(r.id) && r.status !== "running");
  $("overlay").hidden = picked.length === 0;
  if (!picked.length) return;
  // Workspace 4 compares validation accuracy instead of the training loss
  const kind = ws.COMPARE_CURVES === "val" ? "val" : "loss";
  $("overlay-title").textContent = kind === "val" ? "Validation accuracy of the runs you tick below"
    : "Training loss of the runs you tick below";
  for (const r of picked) {
    const key = `${r.id}:${kind}`;
    if (!curveCache.has(key)) curveCache.set(key, await api(`api/runs/${r.id}/curve?kind=${kind}`).catch(() => []));
  }
  const lines = picked.map((r) => ({ name: `Run ${runNumber(r.id)}`, points: curveCache.get(`${r.id}:${kind}`) }));
  const xMax = Math.max(1, ...picked.map((r) => r.epochs || 1));
  drawLines($("overlay-chart"), lines, { xMax, percent: kind === "val" });
  lineLegend($("overlay-legend"), picked.map((r) => `Run ${runNumber(r.id)}: ${r.settings ?? layersText(r.layers)}`));
}

// ---------- confusion matrix and worst mistakes (Workspace 4 on) ----------

const LABEL_SPACE = 14;
let confusionCell = 10;  // cell size in the tab, fitted to the panel's height

function renderConfusion() {
  const data = runInfo.confusion;
  $("confusion-view").hidden = !data;
  $("confusion-note").innerHTML = data
    ? `Run ${runNumber(shownRun) || history.length}: what your model guessed for each validation character.`
    : "Add <b>Confusion matrix</b> (in Evaluation) to the end of your program and run it.";
  if (!data) return;
  const panel = $("tab-confusion");
  const room = panel.clientHeight - $("confusion-note").offsetHeight - 40 - LABEL_SPACE;
  confusionCell = room > 0 ? Math.max(7, Math.min(12, Math.floor(room / cfg.num_classes))) : 10;
  drawConfusion($("confusion-canvas"), data.matrix, confusionCell);
}

function drawConfusion(canvas, matrix, CELL = 10) {
  const n = cfg.num_classes;
  const size = LABEL_SPACE + n * CELL;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.height = size * dpr;
  canvas.style.width = canvas.style.height = `${size}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.font = `${Math.min(9, CELL)}px system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.fillStyle = "#5f6b7a";
  for (let i = 0; i < n; i++) {
    ctx.fillText(cfg.classes[i], LABEL_SPACE + i * CELL + CELL / 2, 11);  // guessed (columns)
    ctx.fillText(cfg.classes[i], LABEL_SPACE / 2, LABEL_SPACE + i * CELL + CELL / 2 + 3);  // true (rows)
  }
  matrix.forEach((row, t) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    row.forEach((count, g) => {
      // square root, so a few mix-ups still show
      const k = Math.sqrt(count / total);
      const [r, gr, b] = t === g ? [47, 154, 90] : [214, 48, 49];
      ctx.fillStyle = `rgb(${255 + (r - 255) * k}, ${255 + (gr - 255) * k}, ${255 + (b - 255) * k})`;
      ctx.fillRect(LABEL_SPACE + g * CELL, LABEL_SPACE + t * CELL, CELL - 1, CELL - 1);
    });
  });
}

let confusionTip;

function hoverConfusion(ev) {
  const data = runInfo.confusion;
  if (!confusionTip) {
    confusionTip = document.createElement("div");
    confusionTip.className = "tooltip";
    $("confusion-view").appendChild(confusionTip);
  }
  const g = Math.floor((ev.offsetX - LABEL_SPACE) / confusionCell);
  const t = Math.floor((ev.offsetY - LABEL_SPACE) / confusionCell);
  confusionTip.hidden = !data || g < 0 || t < 0 || g >= cfg.num_classes || t >= cfg.num_classes;
  if (confusionTip.hidden) return;
  const count = data.matrix[t][g];
  const [tc, gc] = [cfg.classes[t], cfg.classes[g]];
  confusionTip.innerHTML = t === g
    ? `<b>${tc}</b> read as ${tc}: <b>${count}</b> of 100 right`
    : `<b>${tc}</b> read as <b>${gc}</b>: ${count} time${count === 1 ? "" : "s"}`;
  confusionTip.style.left = `${ev.offsetX + 12}px`;
  confusionTip.style.top = `${ev.offsetY + 12}px`;
}

function renderMistakes() {
  const data = runInfo.mistakes;
  const grid = $("mistakes-grid");
  grid.innerHTML = "";
  if (!data) {
    $("mistakes-note").innerHTML = "Add <b>Worst mistakes</b> (in Evaluation) to the end of your program and run it.";
    return;
  }
  $("mistakes-note").textContent = `Run ${runNumber(shownRun) || history.length}: ${data.count} of ` +
    `${data.total} validation characters read wrong (look-alike mix-ups such as 0/O not counted). ` +
    "These are the ones your model was most sure about. Would you have read them right?";
  for (const m of data.items) {
    const fig = document.createElement("figure");
    fig.appendChild(grayCanvas(m.image));
    const cap = document.createElement("figcaption");
    cap.innerHTML = `is <b>${m.label}</b><br>read <b>${m.guess}</b> · ${pct(m.confidence)}`;
    fig.appendChild(cap);
    grid.appendChild(fig);
  }
}

// ---------- your image (Workspace 5) ----------

async function uploadImage(file) {
  if (!file) return;
  $("upload-note").textContent = "Encoding your picture and decoding it with your model…";
  $("upload-view").hidden = true;
  $("upload-download").hidden = true;
  let r;
  try {
    r = await api("api/upload", { method: "POST", body: file });
  } catch (e) {
    $("upload-note").textContent = e.message;
    return;
  }
  const [w, h] = r.size;
  const fig = (src, text, cls = "") => `<figure><img class="${cls}" src="${src}" alt=""><figcaption>${text}</figcaption></figure>`;
  $("upload-note").textContent = `Decoded with your model from run ${runNumber(r.run_id)}.`;
  $("upload-view").innerHTML =
    fig(r.images.original, "1. Your picture") +
    fig(r.images.small, `2. Shrunk to ${w}×${h} pixels`, "pixels") +
    fig(r.images.palette, `3. Reduced to ${r.colors} colors` +
      (r.colors_before > r.colors ? ` (from ${r.colors_before.toLocaleString("en-US")})` : "") +
      ", each one a color group", "pixels") +
    fig(r.images.characters, `4. Written in ${(w * h).toLocaleString("en-US")} handwritten characters: ` +
      "each pixel is a character from its color's group", "chars") +
    fig(r.images.decoded, `5. What your model read: ${pct(r.accuracy)} of pixels right`, "pixels") +
    (r.reused.length
      ? `<p class="note">Some characters appear more than once: there are only 100 of each character here ` +
        `(200 for look-alike pairs), and ${r.reused.length} of your colors cover more pixels than that. ` +
        "A repeated character is read the same way every time, so its mistakes " +
        "repeat too.</p>"
      : "");
  $("upload-view").hidden = false;
  $("upload-download").href = r.images.side_by_side;
  $("upload-download").hidden = false;
}

// ---------- model card (Workspace 5) ----------

const LOOKALIKE_PAIRS = ["0O", "1I", "2Z", "8B", "6G", "5S"];
const isLookalike = (a, b) => LOOKALIKE_PAIRS.some((pair) => pair.includes(a) && pair.includes(b));

function renderCard() {
  const run = history.find((r) => r.id === shownRun);
  const view = $("card-view");
  if (!run || run.status !== "done" || !runInfo.model) {
    view.hidden = true;
    $("card-download").hidden = true;
    $("card-note").textContent = "The model card describes your latest finished run. Run your model to the end, " +
      "then come back.";
    return;
  }
  const n = (v) => v.toLocaleString("en-US");
  const { model, settings, train_start: start, data, confusion } = runInfo;
  const val = charts.acc.points.at(-1)?.[1];
  const stop = $("stop-note").hidden ? "" : ` ${$("stop-note").textContent}`;
  let weak = "<p>Add <b>Confusion matrix</b> to your program to list them.</p>";
  if (confusion) {
    const cells = [];
    confusion.matrix.forEach((row, t) => row.forEach((c, g) => { if (t !== g && c) cells.push([c, t, g]); }));
    cells.sort((a, b) => b[0] - a[0]);
    const right = confusion.matrix.map((row, t) => [row[t], t]).sort((a, b) => a[0] - b[0]).slice(0, 5);
    const look = cells.filter(([, t, g]) => isLookalike(cfg.classes[t], cfg.classes[g]));
    const mixup = ([c, t, g]) => `<b>${cfg.classes[t]}</b> as <b>${cfg.classes[g]}</b> ${c} times out of 100`;
    weak = "<ul>" + cells.filter((cell) => !look.includes(cell)).slice(0, 5)
      .map((cell) => `<li>Reads ${mixup(cell)}</li>`).join("") + "</ul>" +
      (look.length ? `<p>Look-alike pairs, which share a color in the decoded image: reads ` +
        `${look.slice(0, 3).map(mixup).join("; ")}.</p>` : "") +
      `<p>Hardest characters: ${right.map(([c, t]) => `<b>${cfg.classes[t]}</b> (${c}% right)`).join(", ")}.</p>`;
  }
  view.innerHTML = `
    <h2>Model card: handwritten character reader</h2>
    <p class="muted">Workspace 5, run ${runNumber(run.id)}, ${new Date(run.started * 1000).toLocaleDateString()}</p>
    <h3>What it does</h3>
    <p>Reads one handwritten character, a 28×28 grayscale image, and says which of 36 characters it is:
      a digit 0–9 or an uppercase letter A–Z.</p>
    <h3>Model</h3>
    <p>${model.layers.join(" → ")}</p>
    <p>${n(start.parameters)} weights.</p>
    <h3>Training</h3>
    <p>${n(data.train)} handwritten characters from EMNIST, the same number of each class.
      ${settings ? `${settings.text}.` : ""} ${start.epochs} epochs${runInfo.train_done ? `, ${Math.round(runInfo.train_done.seconds)}s to train` : ""}.${stop}</p>
    <h3>How well it does</h3>
    <p>Validation accuracy: <b>${val != null ? pct(val) : "—"}</b> on ${n(data.val)} characters it never trained on
      (100 of each class). Mixing up look-alikes such as 0/O counts as wrong here.</p>
    <figure class="chart"><figcaption>Training loss</figcaption><canvas id="card-loss"></canvas></figure>
    <figure class="chart"><figcaption>Accuracy (blue: validation, orange: training)</figcaption><canvas id="card-acc"></canvas></figure>
    <h3>Confusion matrix</h3>
    ${confusion ? '<canvas id="card-confusion"></canvas>' : "<p>Not measured in this run.</p>"}
    <h3>Known weaknesses</h3>
    <p>Its most common mix-ups on the validation characters:</p>
    ${weak}
    <h3>Limits</h3>
    <ul>
      <li>It only knows digits and uppercase letters. Anything else (a lowercase letter, a symbol, a drawing)
        still gets one of the 36 answers.</li>
      <li>It was trained and tested on EMNIST: centered characters, light ink on a dark background, scanned
        from forms. Other handwriting, cameras or line widths may do worse.</li>
      <li>Some characters look the same in many people's handwriting (0/O, 1/I, 2/Z, 5/S, 6/G, 8/B): no model
        can always tell them apart from the character alone.</li>
    </ul>`;
  view.hidden = false;
  $("card-download").hidden = false;
  $("card-note").textContent = "Generated from your latest finished run. Run again and it updates.";
  drawLines($("card-loss"), [{ points: charts.loss.points }], { xMax: charts.loss.xMax });
  drawLines($("card-acc"), [{ points: charts.acc.points }, { points: charts.acc.points2 }],
    { xMax: charts.acc.xMax, percent: true });
  if (confusion) drawConfusion($("card-confusion"), confusion.matrix);
}

function downloadCard() {
  const copy = $("card-view").cloneNode(true);
  for (const c of copy.querySelectorAll("canvas")) {
    const img = document.createElement("img");
    img.src = $(c.id).toDataURL("image/png");
    img.style.width = $(c.id).style.width || "100%";
    c.replaceWith(img);
  }
  const html = "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Model card</title>" +
    "<style>body{font:15px/1.5 system-ui,sans-serif;max-width:720px;margin:24px auto;padding:0 16px;color:#1d232b}" +
    ".muted{color:#5f6b7a}img{max-width:100%}figure{margin:0 0 12px}figcaption{font-weight:600}</style>" +
    `</head><body>${copy.innerHTML}</body></html>`;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([html], { type: "text/html" }));
  a.download = "model-card.html";
  a.click();
  URL.revokeObjectURL(a.href);
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

function weightCanvas(values, size = 28) {
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(size, size);
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

// ---------- filter timelapse ----------

let timelapse = null; // { runId, data }

async function renderFilters() {
  const run = history.filter((r) => r.status === "done" && (r.layers || []).some((l) => l.startsWith("Conv"))).pop();
  const note = $("filters-note");
  if (!run) {
    $("filters-view").hidden = true;
    note.textContent = "Run a model with a Conv block, then come back to watch its filters learn.";
    return;
  }
  if (timelapse?.runId !== run.id) {
    try {
      timelapse = { runId: run.id, data: await api(`api/runs/${run.id}/timelapse`) };
    } catch (e) {
      note.textContent = e.message;
      return;
    }
    const slider = $("filters-slider");
    slider.max = timelapse.data.progress.length - 1;
    slider.value = slider.max;
    drawSample(timelapse.data.sample);
  }
  note.textContent = `Run ${runNumber(run.id)}: the first Conv layer's filters (small squares) and what each ` +
    "one finds in the sample character (larger squares). Red means a strong match, blue the opposite. " +
    "Drag the slider to watch them change during training.";
  $("filters-view").hidden = false;
  drawSnapshot();
}

function drawSample(pixels) {
  const ctx = $("filters-sample").getContext("2d");
  const img = ctx.createImageData(28, 28);
  pixels.flat().forEach((v, i) => img.data.set([255 - v, 255 - v, 255 - v, 255], i * 4));
  ctx.putImageData(img, 0, 0);
}

function drawSnapshot() {
  const { progress, filters, outputs } = timelapse.data;
  const k = Number($("filters-slider").value);
  $("filters-when").textContent = k === 0 ? "before training" : `after ${progress[k].toFixed(2)} epochs`;
  const grid = $("filters-grid");
  grid.innerHTML = "";
  filters[k].forEach((kernel, f) => {
    const fig = document.createElement("figure");
    const kc = weightCanvas(kernel.flat(), 3);
    const oc = weightCanvas(outputs[k][f].flat(), 28);
    kc.className = "kernel";
    oc.className = "output";
    fig.append(kc, oc, `${f + 1}`);
    grid.appendChild(fig);
  });
}

// ---------- one training step ----------

let oneStep = null; // { runId, data, stage }

async function renderOneStep(start = 0) {
  const run = history.filter((r) => r.status === "done").pop();
  if (!run) {
    $("onestep-view").hidden = true;
    $("onestep-note").textContent = "Finish a run first, then come back to take one training step apart.";
    return;
  }
  if (oneStep?.runId !== run.id || start) {
    $("onestep-note").textContent = "Loading…";
    try {
      oneStep = { runId: run.id, data: await api(`api/runs/${run.id}/onestep?start=${start}`), stage: 1 };
    } catch (e) {
      $("onestep-note").textContent = e.message;
      return;
    }
  }
  $("onestep-note").textContent = "Training is a loop repeated thousands of times: guess, measure the " +
    "mistake, find which weights caused it, nudge them. Here is one pass of that loop, using your " +
    `model from run ${runNumber(run.id)} on a validation character it gets wrong. Click Next ▶ to go on.`;
  $("onestep-view").hidden = false;
  drawStages();
}

function grayCanvas(pixels, color = null) {
  const c = document.createElement("canvas");
  c.width = c.height = 28;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(28, 28);
  pixels.flat().forEach((v, i) => {
    // color: a heatmap from white to red (v in 0..100, square root to show weak spots too);
    // else dark ink on white (v in 0..255)
    const heat = 255 * (1 - Math.sqrt(v / 100));
    const px = color ? [255, heat, heat] : [255 - v, 255 - v, 255 - v];
    img.data.set([...px, 255], i * 4);
  });
  ctx.putImageData(img, 0, 0);
  return c;
}

const caption = (text) => `<figure><figcaption>${text}</figcaption></figure>`;

function guessBars(top, label) {
  return '<div class="bars">' + top.map(([cls, p]) =>
    `<span class="${cls === label ? "right" : ""}">${cls}</span><div class="bar" style="width:${p * 100}%"></div>` +
    `<span>${pct(p)}</span>`).join("") + "</div>";
}

function drawStages() {
  const d = oneStep.data;
  const max = Math.max(...d.layers.flatMap((l) => [l.change, l.batch_change])) || 1;
  // a dead network (e.g. after a far too high learning rate): no blame reaches any weight
  const frozen = d.layers.every((l) => l.change === 0 && l.batch_change === 0);
  const n = (v) => v.toLocaleString("en-US");
  const guess = d.before[0][0];
  const steps = Math.ceil(d.train_size / d.batch);
  // Words here are limited to what lectures 1–2 teach (LECTURES.md): no softmax, batch,
  // learning rate or optimizer yet.
  const stages = [
    "<b>Forward: the guess.</b> The character flows down through the blocks in your Model, " +
      "top to bottom. The Conv layers look for small patterns like edges and curves, Flatten lines " +
      "everything up, and the Linear layers combine the patterns into a score for each of the 36 " +
      "characters. Here are the five it's most sure about:" +
      `<div class="figs">${caption("the character")}</div>${guessBars(d.before, d.label)}` +
      `<p>It says <b>${guess}</b>, but this is a <b>${d.label}</b>. Look at the character: can you ` +
      "see why it might be confused?</p>",
    "<b>Score the mistake (the loss).</b> To get better, the model needs to know how wrong it was, " +
      "as a single number: the <i>loss</i>. It's close to 0 when the model is sure of the right " +
      "answer, and it grows the less sure the model is." +
      `<p>The model was only ${pct(d.true_before)} sure of ${d.label}, so the loss is ` +
      `<b>${d.loss.toFixed(2)}</b>. This is the same loss you watched go down in the Training tab, ` +
      "averaged over many characters there. Training's only goal is to push that number down.</p>",
    `<b>Backward: who's to blame? (backpropagation)</b> Your model has <b>${n(d.weights)} weights</b> ` +
      "spread across its layers. Which ones caused this mistake? Backpropagation sends the blame " +
      "backward through the same blocks, from the last layer up to the first. Every weight gets a " +
      "<i>gradient</i>, which says which way the weight should move to lower the loss and how much " +
      "it matters." +
      "<p>The blame can be traced all the way back to the character's pixels. Red shows where the " +
      `model was “looking” when it went wrong. Are those the strokes that tell a ${d.label} apart ` +
      `from a ${guess}?</p>` +
      `<div class="figs">${caption("the character")}${caption("where the blame lands")}</div>`,
    "<b>Nudge every weight.</b> Every weight moves a small step the way its gradient points, so " +
      "next time the guess should be a little better. Every layer gets nudged, from the Conv filters " +
      "at the top to the last Linear layer. " +
      (ws.TABS.includes("filters")
        ? "That's how the filters in the <b>Filters</b> tab grew from noise into pattern finders: "
        : "That's how Conv filters grow from noise into pattern finders: ") +
      "one tiny nudge at a time." +
      `<p>The model doesn't learn from one character at a time, though. It looks at a group of ` +
      `${d.batch} characters together and moves the way that helps the whole group. ` +
      "Each pair of bars shows how far one of your layers moves: " +
      "<span class=\"key alt\"></span> for this character alone, " +
      "<span class=\"key\"></span> for the group that includes it.</p>" +
      '<div class="bars">' + d.layers.map((l) =>
        `<span>${l.name}</span><div class="bar alt" style="width:${(l.change / max) * 100}%"></div>` +
        "<span>alone</span>" +
        `<span></span><div class="bar" style="width:${(l.batch_change / max) * 100}%"></div>` +
        "<span>in the group</span>").join("") + "</div>" +
      (frozen
        ? "<p><b>Here, no layer moves at all.</b> Every gradient is zero: the blame never reaches a single " +
          "weight, so there's nothing to nudge. What could leave a network in that state?</p>"
        : "<p>Alone, one odd character would yank the weights toward itself and could undo what the model " +
          `learned about other characters. In the group, it's one voice out of ${d.batch}. Do the first ` +
          "layers move as much as the last ones?</p>"),
    `<b>Did it help?</b> The model went from ${pct(d.true_before)} to <b>${pct(d.true_after)}</b> sure ` +
      `of ${d.label}. ` +
      (frozen
        ? "Nothing changed, because no weight moved. A network stuck like this can't learn, no matter " +
          "how many more steps it takes."
        : pct(d.true_after) === pct(d.true_before)
        ? "It barely moved: one step is small."
        : d.true_after < d.true_before
        ? "It got worse! The step was the best move for the group, not for this one character. " +
          "Mistakes that keep coming up still get fixed over many steps."
        : "It moved the right way: a step that helps the group usually helps most of its characters too.") +
      `<div>${guessBars(d.after, d.label)}</div>` +
      `<p>That's one step. One epoch is ${n(steps)} steps like it, covering all ${n(d.train_size)} ` +
      "training characters once. Every run you've started was this loop over and over: guess, score " +
      "the mistake, send the blame backward, nudge every weight. " +
      (cfg.workspace === "ws2" ? "Workspace 3 is about" : "This workspace is about") + " the choices " +
      "inside this loop, like how big each nudge is and how many characters go in a group.</p>",
  ];
  const list = $("onestep-stages");
  list.innerHTML = stages.slice(0, oneStep.stage).map((s) => `<li>${s}</li>`).join("");
  const figs = list.querySelectorAll(".figs figure");
  figs[0].prepend(grayCanvas(d.image));
  figs[1]?.prepend(grayCanvas(d.image));
  figs[2]?.prepend(grayCanvas(d.saliency, "red"));
  $("onestep-next").disabled = oneStep.stage >= stages.length;
}

// ---------- practice image ----------

const hexToRgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

function showPractice(grid) {
  practiceGrid = grid;
  $("practice-empty").hidden = true;
  $("practice-view").hidden = false;
  const canvas = $("practice-canvas");
  canvas.width = canvas.height = grid.length;  // 32 or 64 characters a side
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(grid.length, grid.length);
  const rgb = cfg.colors.map(hexToRgb);
  grid.flat().forEach((c, i) => {
    img.data.set([...rgb[c], 255], i * 4);
  });
  ctx.putImageData(img, 0, 0);
}

function hoverPractice(ev) {
  if (!practiceGrid) return;
  const canvas = $("practice-canvas");
  const n = practiceGrid.length;
  const col = Math.min(n - 1, Math.floor((ev.offsetX / canvas.clientWidth) * n));
  const row = Math.min(n - 1, Math.floor((ev.offsetY / canvas.clientHeight) * n));
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
  defineBlocks(cfg, ws.BLOCK_OPTIONS);
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
  $("history").addEventListener("change", (ev) => {
    if (!ev.target.matches("input.pick")) return;
    if (ev.target.checked) overlayRuns.add(ev.target.dataset.run);
    else overlayRuns.delete(ev.target.dataset.run);
    renderOverlay();
  });
  $("reset").onclick = () => {
    if (confirm("Replace your blocks with this workspace's starting blocks?")) {
      workspace.clear();
      loadBlocks(ws.START_STATE);
    }
  };
  $("practice-canvas").addEventListener("mousemove", hoverPractice);
  $("confusion-canvas").addEventListener("mousemove", hoverConfusion);
  $("confusion-canvas").addEventListener("mouseleave", () => { if (confusionTip) confusionTip.hidden = true; });
  $("upload-file").addEventListener("change", (ev) => uploadImage(ev.target.files[0]));
  $("card-download").onclick = downloadCard;
  $("filters-slider").addEventListener("input", drawSnapshot);
  $("onestep-next").onclick = () => {
    oneStep.stage++;
    drawStages();
    // the buttons sit under the stages: bring the new stage's start into view
    $("onestep-stages").lastElementChild.scrollIntoView({ block: "start", behavior: "smooth" });
  };
  $("onestep-another").onclick = () => renderOneStep(oneStep.data.index + 1);

  // Pick up where we left off: reattach to a run in progress, or show the latest one.
  await refreshHistory();
  await refreshSubmission();
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
