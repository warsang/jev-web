// jev-web demo — live typed decisions vs. raw text parsing.
//
// Everything below is either (a) the public jev-web API, (b) the naive keyword
// baseline in ./naive.mjs, or (c) DOM plumbing. No faked numbers: latency comes
// from performance.now() around the real call, and the network panel counts real
// requests through the Performance API.

import {
  createDecider,
  detectWebGPU,
  resolveDevice,
  resolveDtype,
  DEFAULT_MODEL,
  DEFAULT_REVISION,
} from "../src/index.mjs";
import { parseAll } from "./naive.mjs";
import { PRESETS } from "./presets.mjs";

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

// ── theme ──────────────────────────────────────────────────────────────
const savedTheme = localStorage.getItem("jev-theme");
if (savedTheme) document.documentElement.dataset.theme = savedTheme;
$("theme").onclick = () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("jev-theme", next);
};

// ── quickstart copy ────────────────────────────────────────────────────
$("copy-quickstart").onclick = async (e) => {
  const text = $("quickstart").code.textContent.replace(/\s*$/, "");
  try {
    await navigator.clipboard.writeText(text);
    e.target.textContent = "Copied ✓";
  } catch {
    e.target.textContent = "Press Ctrl+C";
  }
  setTimeout(() => { e.target.textContent = "Copy 5-line quickstart"; }, 1600);
};

// ── state ──────────────────────────────────────────────────────────────
let decider = null;
let loading = false;
let running = false;
let questions = structuredClone(PRESETS[0].questions);
let debounce = 0;
// NOT `history`: that name shadows window.history, and syncUrl() below calls
// history.replaceState(). The sparkline array and the History API are both
// "history" and only one of them is a Web API.
const runHistory = [];

// URL state: ?preset=negation links straight to a scenario so a disagreement
// between the model and the rules can be pointed at directly.
function readUrlState() {
  const q = new URLSearchParams(location.search);
  const p = q.get("preset");
  if (p && PRESETS.some((x) => x.id === p)) return p;
  return PRESETS[0].id;
}

function syncUrl() {
  const active = [...$("presets").children].find((b) => b.getAttribute("aria-pressed") === "true");
  const id = active?.dataset.id ?? PRESETS[0].id;
  // globalThis.history explicitly: a bare `history` in this module is exactly
  // the name that got shadowed before, and a wrong answer here should never be
  // able to take the inference path down with it.
  globalThis.history.replaceState(
    null,
    "",
    id === PRESETS[0].id ? location.pathname + location.hash : `?preset=${id}${location.hash}`,
  );
}

// ── network monitor (real requests, via the Performance API) ───────────
const net = { cold: 0, warm: 0, hosts: new Map(), watching: false };
try {
  new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      if (entry.initiatorType === "beacon" || entry.initiatorType === "navigation") continue;
      let host;
      try { host = new URL(entry.name, location.href).host; } catch { continue; }
      if (host === location.host) continue; // our own bundle
      net.hosts.set(host, (net.hosts.get(host) ?? 0) + 1);
      // Requests before the model is warm = weight download; after = inference
      // escaping to the network, which is the thing we claim never happens.
      if (decider && !loading) net.warm++; else net.cold++;
    }
    renderNet();
  }).observe({ type: "resource", buffered: false });
} catch { /* PerformanceObserver unsupported: panel just stays empty */ }

function renderNet() {
  $("net-cold").textContent = net.cold;
  $("net-warm").textContent = net.warm;
  const warm = $("net-warm");
  warm.className = `net-v ${net.warm === 0 ? "zero" : "nonzero"}`;
  $("net-warm-sub").textContent = net.warm === 0
    ? "0 — the forward pass is entirely local"
    : `${net.warm} — your text left the device`;
  $("net-leak").textContent = net.warm === 0 ? "nothing" : `${net.warm} request(s)`;
  $("net-leak").className = `net-v ${net.warm === 0 ? "zero" : "nonzero"}`;
  const list = $("host-list");
  list.innerHTML = "";
  if (!net.hosts.size) list.append(el("em", null, "none yet"));
  else for (const [h, n] of [...net.hosts].sort((a, b) => b[1] - a[1])) list.append(el("div", null, `${h} — ${n} request${n === 1 ? "" : "s"}`));
  $("m-net").textContent = net.warm;
}

function renderConn() {
  const on = navigator.onLine;
  $("conn").textContent = on ? "online" : "offline — still works";
  $("conn").className = `pill ${on ? "" : "good"}`;
}
addEventListener("online", renderConn);
addEventListener("offline", renderConn);
renderConn();

// ── question editor ────────────────────────────────────────────────────
const DEFAULTS = {
  choice: ["option A", "option B", "option C"],
  score: ["very negative", "negative", "neutral", "positive", "very positive"],
  noul: ["no", "yes"],
};

function renderQuestions() {
  const host = $("questions");
  host.innerHTML = "";
  questions.forEach((q, qi) => {
    const card = el("div", "q");
    const top = el("div", "q-top");
    top.append(el("span", `q-type ${q.type}`, q.type));
    const input = el("input");
    input.type = "text";
    input.value = q.instructions;
    input.setAttribute("aria-label", "question instructions");
    input.oninput = () => { q.instructions = input.value; schedule(); };
    top.append(input);
    const rm = el("button", "q-remove", "×");
    rm.type = "button";
    rm.title = "remove question";
    rm.onclick = () => { questions.splice(qi, 1); renderQuestions(); schedule(); };
    top.append(rm);
    card.append(top);

    if (q.type !== "noul") {
      const opts = el("div", "q-opts");
      q.options.forEach((o, oi) => {
        const b = el("button", "opt", o);
        b.type = "button";
        b.onclick = () => {
          const next = prompt("Option label", o);
          if (next != null && next.trim()) { q.options[oi] = next.trim(); renderQuestions(); schedule(); }
        };
        opts.append(b);
      });
      const add = el("button", "opt opt-add", "+ option");
      add.type = "button";
      add.onclick = () => { q.options.push(`option ${String.fromCharCode(65 + q.options.length)}`); renderQuestions(); schedule(); };
      opts.append(add);
      card.append(opts);
    }
    host.append(card);
  });
  if (!questions.length) host.append(el("p", "empty", "Add at least one question."));
}

for (const btn of document.querySelectorAll("[data-add]")) {
  btn.onclick = () => {
    const type = btn.dataset.add;
    questions.push({ type, instructions: `A ${type} question…`, options: [...DEFAULTS[type]] });
    renderQuestions();
    schedule();
  };
}

function loadPreset(id) {
  const p = PRESETS.find((x) => x.id === id) ?? PRESETS[0];
  $("state").value = p.state;
  questions = structuredClone(p.questions);
  for (const b of $("presets").children) b.setAttribute("aria-pressed", String(b.dataset.id === p.id));
  renderQuestions();
  // Answer first, write the URL second: the address bar is a nicety and must
  // never be able to stop the demo from producing results.
  schedule();
  syncUrl();
}

const presetHost = $("presets");
for (const p of PRESETS) {
  const b = el("button", null, p.label);
  b.type = "button";
  b.dataset.id = p.id;
  b.title = p.blurb;
  b.onclick = () => loadPreset(p.id);
  presetHost.append(b);
}

$("state").oninput = schedule;

// ── latency sparkline ──────────────────────────────────────────────────
function drawSpark() {
  const svg = $("spark");
  svg.innerHTML = "";
  if (runHistory.length < 2) return;
  const w = 200, h = 44, pad = 4;
  const max = Math.max(...runHistory, 1);
  const x = (i) => pad + (i * (w - pad * 2)) / (runHistory.length - 1);
  const y = (v) => h - pad - (v / max) * (h - pad * 2);
  const pts = runHistory.map((v, i) => `${x(i)},${y(v)}`).join(" ");
  const area = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
  area.setAttribute("points", `${pad},${h - pad} ${pts} ${x(runHistory.length - 1)},${h - pad}`);
  area.setAttribute("fill", "currentColor");
  area.setAttribute("opacity", ".13");
  const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  line.setAttribute("points", pts);
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", "var(--accent)");
  line.setAttribute("stroke-width", "1.5");
  line.setAttribute("vector-effect", "non-scaling-stroke");
  const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  dot.setAttribute("cx", x(runHistory.length - 1));
  dot.setAttribute("cy", y(runHistory[runHistory.length - 1]));
  dot.setAttribute("r", "2.5");
  dot.setAttribute("fill", "var(--accent)");
  svg.style.color = "var(--accent)";
  svg.append(area, line, dot);
}

function flash(node) {
  node.classList.remove("flash");
  void node.offsetWidth;
  node.classList.add("flash");
}

// ── WebGPU that exists but cannot run this model ────────────────────────
//
// navigator.gpu.requestAdapter() resolving is NOT evidence that WebGPU works.
// onnxruntime-web's WebGPU backend generates WGSL per subgraph, and some
// browsers reject it for particular models — Firefox currently fails on
// DeBERTa-v3's "Clip" subgraph with
//   "Failed to create a WebGPU compute pipeline: ShaderModule with 'Clip'
//    label is invalid"
// which surfaces as `failed to call OrtRun()`. The adapter is there; the
// kernels are not. So the demo treats the *first inference* as the real probe
// and falls back to WASM, which is slower but works everywhere.
const WEBGPU_FAILURES = [
  /failed to call OrtRun/i,
  /WebGPU compute pipeline/i,
  /ShaderModule/i,
  /shader module/i,
  /webgpu.*invalid/i,
  /adapter.*lost|Navigator.*not.*support/i,
];

function isWebGpuFailure(err) {
  const msg = String(err?.message ?? err ?? "");
  return WEBGPU_FAILURES.some((re) => re.test(msg));
}

function setNotice(text, kind = "warn") {
  const box = $("notice");
  box.hidden = !text;
  box.className = `notice ${kind}`;
  box.textContent = text ?? "";
}

// ── rendering answers ──────────────────────────────────────────────────
function answerHead(a) {
  const head = el("div", "ans-head");
  head.append(el("span", "ans-q", a.instructions ?? ""));
  let value, tag;
  if (a.type === "choice") { value = a.choice ?? "—"; tag = `argmax of ${Object.keys(a.probabilities).length}`; }
  else if (a.type === "score") { value = a.score?.toFixed?.(3) ?? "—"; tag = `expected level ${a.level}`; }
  else { value = a.noul?.toFixed?.(3) ?? "—"; tag = "p(yes)"; }
  head.append(el("span", "ans-v", value));
  head.append(el("span", "tag", tag));
  return head;
}

function answerBody(a) {
  const box = el("div");
  const probs = a.probabilities ?? {};
  const best = Math.max(...Object.values(probs).map(Number), -Infinity);
  const bars = el("div", "bars");
  for (const [label, p] of Object.entries(probs)) {
    const row = el("div", `bar${Number(p) === best ? " win" : ""}`);
    row.append(el("span", "n", label));
    const t = el("div", "t");
    const f = el("i", "f");
    f.style.width = `${Math.max(0.5, Number(p) * 100)}%`;
    t.append(f);
    row.append(t);
    row.append(el("span", "p", `${(Number(p) * 100).toFixed(1)}%`));
    bars.append(row);
  }
  box.append(bars);
  const conf = el("div", "conf");
  conf.append(el("span", null, `confidence ${(a.confidence * 100).toFixed(1)}%`));
  box.append(conf);
  return box;
}

function renderTyped(result, questions) {
  const host = $("answers");
  host.innerHTML = "";
  result.answers.forEach((a, i) => {
    const q = questions[i] ?? {};
    const card = el("div", "ans live");
    card.append(answerHead({ ...a, instructions: q.instructions }));
    card.append(answerBody(a));
    host.append(card);
  });
  if (result.truncated) {
    host.append(el("p", "note", `⚠ state truncated to ${result.length} tokens (model cap).`));
  }
}

// ── running ────────────────────────────────────────────────────────────
function schedule() {
  clearTimeout(debounce);
  debounce = setTimeout(run, 320);
}

async function run() {
  const state = $("state").value;
  const qs = questions.filter((q) => q.instructions?.trim());
  if (!qs.length) return;

  // The naive path always runs: it costs microseconds and needs no model.
  const naive = parseAll(state, qs);
  renderNaive(naive, qs);
  $("naive-when").textContent = `${naive.ms.toFixed(2)} ms · ${naive.passes} rule set${naive.passes === 1 ? "" : "s"}`;

  if (!decider || running) return;
  running = true;
  const warmBefore = net.warm;
  try {
    let result;
    try {
      result = await decider.decide(state, qs);
    } catch (err) {
      // The common WebGPU failure happens here, not at load: the weights
      // fetch fine, then the first OrtRun() cannot build a compute pipeline
      // for one of the model's subgraphs. Rebuild on WASM and retry once.
      if (decider.info.device !== "wasm" && isWebGpuFailure(err)) {
        noteBrokenDevice(decider.info.device);
        setNotice(
          `WebGPU could not run this model in your browser (${shortErr(err)}). ` +
          `Rebuilt on WASM — slower, still entirely on-device.`,
        );
        $("typed-when").textContent = "WebGPU failed — rebuilding on WASM…";
        $("typed-when").className = "pill warn";
        decider.dispose?.();
        decider = await buildDecider("wasm", "auto");
        result = await decider.decide(state, qs);
      } else {
        throw err;
      }
    }
    renderTyped(result, qs);

    const ms = result.timings?.totalMs ?? 0;
    $("typed-when").textContent = `${ms} ms · 1 pass · ${qs.length} question${qs.length === 1 ? "" : "s"}`;
    $("typed-when").className = "pill good";

    runHistory.push(ms);
    if (runHistory.length > 40) runHistory.shift();
    $("m-latency").textContent = `${ms} ms`;
    $("m-passes").textContent = "1";
    $("m-questions").textContent = String(qs.length);
    $("mx-latency").textContent = `~${ms} ms`;
    flash($("m-latency"));
    drawSpark();

    // The claim, checked: a warm decide() must not touch the network.
    if (net.warm === warmBefore) {
      $("net-warm-sub").textContent = "0 — the forward pass is entirely local";
    }
  } catch (err) {
    $("error").hidden = false;
    $("error").textContent = String(err?.message ?? err);
  } finally {
    running = false;
  }
}

function renderNaive(result, qs) {
  const host = $("naive");
  host.innerHTML = "";
  result.answers.forEach((a, i) => {
    const q = qs[i] ?? {};
    const card = el("div", "ans");
    const head = el("div", "ans-head");
    head.append(el("span", "ans-q", q.instructions ?? ""));
    let value;
    if (a.type === "choice") value = a.value ?? "no answer";
    else if (a.type === "score") value = a.value == null ? "no answer" : `${a.value} (${q.options?.[a.value]})`;
    else value = a.value == null ? "no answer" : a.value ? "yes" : "no";
    head.append(el("span", `ans-v${a.ruled ? "" : " dim"}`, value));
    head.append(el("span", `tag${a.ruled ? "" : " err"}`, a.ruled ? "hard label" : "no rule"));
    card.append(head);
    const conf = el("div", "conf");
    conf.append(el("span", null, `rule: ${a.rule}`));
    if (!a.ruled) conf.append(el("span", null, "no confidence available"));
    card.append(conf);
    host.append(card);
  });
  $("m-questions").title = "";
}

// ── model loading ──────────────────────────────────────────────────────
function fmtBytes(n) {
  if (!Number.isFinite(n) || n <= 0) return "";
  const mb = n / 1e6;
  return mb > 1000 ? `${(mb / 1000).toFixed(2)} GB` : `${mb.toFixed(0)} MB`;
}

// Remember a device that failed, so a reload does not re-download and
// re-crash before falling back again.
const BROKEN_DEVICE_KEY = "jev-web.brokenDevice";
const noteBrokenDevice = (d) => { try { sessionStorage.setItem(BROKEN_DEVICE_KEY, d); } catch { /* private mode */ } };
const readBrokenDevice = () => { try { return sessionStorage.getItem(BROKEN_DEVICE_KEY); } catch { return null; } };

async function buildDecider(device, dtype) {
  return createDecider({
    model: DEFAULT_MODEL,
    revision: DEFAULT_REVISION,
    device,
    dtype,
    onProgress: (p) => {
      const pct = Number.isFinite(p.progress) ? Math.round(p.progress) : null;
      const meta = $("progress-meta");
      meta.textContent = `[${p.phase}] ${p.status ?? ""} ${p.file ?? ""} ${
        p.total ? `${fmtBytes(p.loaded)} / ${fmtBytes(p.total)}` : fmtBytes(p.loaded)
      } ${pct != null ? `${pct}%` : ""}`.trim();
      // nested data files report their own progress; take the max
      if (pct != null) $("progress-fill").style.width = `${pct}%`;
    },
  });
}

async function load() {
  if (loading || decider) return;
  loading = true;
  $("load").disabled = true;
  $("load").textContent = "Loading…";
  $("error").hidden = true;
  $("progress").hidden = false;
  const t0 = performance.now();

  let wanted = $("device").value;
  // A device that already failed this session never gets chosen again.
  if (wanted === "auto" && readBrokenDevice()) wanted = "wasm";

  try {
    try {
      decider = await buildDecider(wanted, $("dtype").value);
    } catch (err) {
      // The model itself may fail to load on WebGPU. Rebuild on WASM rather
      // than showing a dead page.
      if (wanted !== "wasm" && isWebGpuFailure(err)) {
        noteBrokenDevice(wanted);
        setNotice(
          `WebGPU failed on this browser while loading the model (${shortErr(err)}). ` +
          `Retrying on WASM — slower, still entirely on-device.`,
        );
        decider?.dispose?.();
        decider = null;
        $("progress-meta").textContent = "WebGPU unavailable, retrying on WASM…";
        decider = await buildDecider("wasm", $("dtype").value === "auto" ? "auto" : $("dtype").value);
      } else {
        throw err;
      }
    }
    $("progress-fill").style.width = "100%";
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    const i = decider.info;
    $("model-state").textContent = "model ready";
    $("model-state").className = "pill good";
    $("info").textContent =
      `Loaded in ${secs}s — ${i.model}@${(i.revision ?? "main").slice(0, 8)} · ${i.device}/${i.dtype} · ` +
      `temperature ${i.temperature} (${i.configSource}). Cached by the browser: reloads and offline visits are instant.`;
    if (i.device === "wasm" && !readBrokenDevice()) {
      setNotice("Running on WASM (CPU) — no usable WebGPU here, so expect higher latency.");
    }
    $("run").disabled = false;
    $("load").textContent = "Model loaded ✓";
    $("load").disabled = true;
    renderNet();
    schedule();
  } catch (err) {
    $("error").hidden = false;
    $("error").textContent = String(err?.message ?? err);
    $("load").disabled = false;
    $("load").textContent = "Load model";
  } finally {
    loading = false;
  }
}

function shortErr(err) {
  const m = String(err?.message ?? err ?? "");
  const line = m.split("\n").find((l) => /shader|pipeline|OrtRun|webgpu/i.test(l)) ?? m;
  return line.replace(/\s+/g, " ").trim().slice(0, 180) || m.slice(0, 180);
}

$("load").onclick = load;
$("run").onclick = run;
$("device").onchange = () => { if (decider) { decider = null; $("load").disabled = false; $("load").textContent = "Load model"; $("model-state").textContent = "settings changed — reload"; } };
$("dtype").onchange = $("device").onchange;

// ── boot ───────────────────────────────────────────────────────────────
(async () => {
  renderQuestions();
  loadPreset(readUrlState());

  // Report the resolved backend before anyone spends 340 MB finding out.
  // Statically imported on purpose: a dynamic import of a module that is also
  // statically imported makes Rollup emit a broken interop wrapper, and the
  // failure is a silent unhandled rejection rather than a build error.
  const gpu = await detectWebGPU();
  const dev = resolveDevice({ device: "auto", webgpu: gpu });
  const broken = readBrokenDevice();
  if (broken) {
    setNotice(`This browser already failed on ${broken}. Loading on WASM instead — you can retry WebGPU by clearing the "device" selector.`);
  }
  $("info").textContent = gpu && !broken
    ? `WebGPU detected — will load ${resolveDtype(dev, "auto")} on ${dev}. ` +
      `If the first run fails, this page falls back to WASM automatically.`
    : "No WebGPU here — will load q4 on WASM (slower, still fully local).";
})();
