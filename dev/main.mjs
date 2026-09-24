// jev-web playground — generic examples only.
import { createDecider, DEFAULT_MODEL, DEFAULT_REVISION } from "../src/index.mjs";

const $ = (id) => document.getElementById(id);

const PRESETS = {
  banking: {
    state: "I was charged twice for the same order and nobody answers my emails. I want my money back now.",
    questions: [
      {
        type: "choice",
        instructions: "Which product area is the message about?",
        options: ["fees & charges", "pin & security", "refund & dispute", "top-up", "exchange & fiat", "atm & cash", "transfer", "card", "account & identity", "other"],
      },
      { type: "noul", instructions: "The customer is asking for a refund." },
    ],
  },
  review: {
    state: "The plot was slow and the acting felt stiff, but the soundtrack was genuinely beautiful.",
    questions: [
      { type: "score", instructions: "How positive is this review?", options: ["very negative", "negative", "neutral", "positive", "very positive"] },
      { type: "noul", instructions: "The reviewer would recommend this to a friend." },
    ],
  },
};

let decider = null;
let running = false;

function setPreset(name) {
  const p = PRESETS[name] ?? { state: "", questions: [] };
  $("state").value = p.state;
  $("questions").value = JSON.stringify(p.questions, null, 2);
}

function log(text) {
  const el = $("progress");
  el.textContent += `${text}\n`;
  el.scrollTop = el.scrollHeight;
}

function fmtBytes(n) {
  if (!Number.isFinite(n) || n <= 0) return "";
  const mb = n / 1e6;
  return mb > 1000 ? `${(mb / 1000).toFixed(2)} GB` : `${mb.toFixed(1)} MB`;
}

function renderAnswers(result, elapsedMs) {
  const host = $("answers");
  host.innerHTML = "";
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent =
    `${decider.info.device}/${decider.info.dtype} · temperature ${decider.info.temperature} · ` +
    `${result.length} tokens${result.truncated ? " (state truncated)" : ""} · ${elapsedMs.toFixed(0)} ms`;
  host.append(meta);

  for (const a of result.answers) {
    const box = document.createElement("div");
    box.className = "answer";
    const head = document.createElement("h3");
    if (a.type === "choice") {
      head.innerHTML = `choice — <span class="verdict"></span>`;
      head.querySelector(".verdict").textContent = a.choice;
    } else if (a.type === "score") {
      head.innerHTML = `score — <span class="verdict"></span>`;
      head.querySelector(".verdict").textContent = a.score.toFixed(3);
    } else {
      head.innerHTML = `noul — <span class="verdict"></span>`;
      head.querySelector(".verdict").textContent = a.noul.toFixed(3);
    }
    const conf = document.createElement("span");
    conf.className = "badge";
    conf.textContent = `confidence ${(a.confidence * 100).toFixed(1)}%`;
    head.append(conf);
    box.append(head);

    const best = Math.max(...Object.values(a.probabilities));
    for (const [label, p] of Object.entries(a.probabilities)) {
      const row = document.createElement("div");
      row.className = "bar" + (p === best ? " win" : "");
      const name = document.createElement("span");
      name.textContent = label;
      name.style.overflow = "hidden";
      name.style.textOverflow = "ellipsis";
      name.style.whiteSpace = "nowrap";
      const track = document.createElement("div");
      track.className = "track";
      const fill = document.createElement("div");
      fill.className = "fill";
      fill.style.width = `${Math.max(1, p * 100)}%`;
      track.append(fill);
      const pct = document.createElement("span");
      pct.className = "pct";
      pct.textContent = `${(p * 100).toFixed(1)}%`;
      row.append(name, track, pct);
      box.append(row);
    }
    host.append(box);
  }
}

async function loadModel() {
  if (running) return;
  running = true;
  $("load").disabled = true;
  $("error").textContent = "";
  $("progress").textContent = "";
  decider?.dispose?.();
  decider = null;
  try {
    const t0 = performance.now();
    decider = await createDecider({
      model: $("model").value.trim() || DEFAULT_MODEL,
      revision: $("revision").value.trim() || null,
      device: $("device").value,
      dtype: $("dtype").value,
      onProgress: (p) => {
        const loaded = fmtBytes(p.loaded);
        const total = fmtBytes(p.total);
        log(`[${p.phase}] ${p.status ?? ""} ${p.file ?? ""} ${loaded && total ? `${loaded}/${total}` : loaded} ${p.progress != null ? `${Math.round(p.progress)}%` : ""}`.trim());
      },
    });
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    $("info").textContent = `loaded in ${secs}s — ${decider.info.model}@${(decider.info.revision ?? "main").slice(0, 8)} · ${decider.info.device}/${decider.info.dtype} · temperature ${decider.info.temperature} (${decider.info.configSource})`;
    $("run").disabled = false;
    log("ready");
  } catch (err) {
    $("error").textContent = String(err?.stack ?? err);
  } finally {
    running = false;
    $("load").disabled = false;
  }
}

async function run() {
  if (!decider || running) return;
  running = true;
  $("run").disabled = true;
  $("error").textContent = "";
  try {
    const questions = JSON.parse($("questions").value);
    const t0 = performance.now();
    const result = await decider.decide($("state").value, questions);
    renderAnswers(result, performance.now() - t0);
  } catch (err) {
    $("error").textContent = String(err?.stack ?? err);
  } finally {
    running = false;
    $("run").disabled = false;
  }
}

async function resetCache() {
  if (!globalThis.caches) return;
  const keys = await caches.keys();
  await Promise.all(keys.map((k) => caches.delete(k)));
  log(`cleared ${keys.length} cache bucket(s)`);
}

$("model").value = DEFAULT_MODEL;
$("revision").value = DEFAULT_REVISION;
$("load").onclick = loadModel;
$("run").onclick = run;
$("reset").onclick = resetCache;
$("preset-banking").onclick = () => setPreset("banking");
$("preset-review").onclick = () => setPreset("review");
$("preset-empty").onclick = () => setPreset("empty");
setPreset("banking");
