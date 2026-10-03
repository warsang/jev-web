/**
 * laya.mjs — Laya System-1 decision models (ModernBERT encoder + typed head).
 *
 * The Laya family is a *different graph* from open-jev:
 *   encoder: input_ids, attention_mask -> hidden  (last_hidden_state)
 *   head:    hidden, marker_pos, marker_mask, qtype -> logits, act_logits
 *
 * Sequence layout (mirrors the reference `laya` Python runtime,
 * `laya/common.py build_sequence`):
 *   [CLS] <type> question: <instructions> [SEP] [MASK]opt0 [MASK]opt1 … [SEP] <state> [SEP]
 * `marker_pos` points at each option's [MASK]; `qtype` is 0 choice / 1 score /
 * 2 noul. Logits are calibrated per question type *and* option-count bucket
 * (rl_agent_config.json `temperature` / `temperature_by_options`), clamped to
 * [0.5, 5] like the Python runtime.
 *
 * The browser export used here is a two-file q8 ONNX split
 * (alfred361/laya-typed-decisions-web-q8, pinned by revision).
 */

import { normalizeQuestions } from "./questions.mjs";
import { softmaxWithTemperature } from "./answers.mjs";

export const LAYA_DEFAULT_MODEL = "alfred361/laya-typed-decisions-web-q8";
export const LAYA_DEFAULT_REVISION = "96f637c0bee046dce5c61a3174de9f7cdcf09ace";
export const LAYA_DEFAULT_SUBFOLDER = "v1";

/** Reference calibration from the typed-decisions rl_agent_config.json. */
export const LAYA_DEFAULT_TEMPERATURES = [1.0148024559020996, 1.0374259948730469, 1.0575125217437744];
export const LAYA_DEFAULT_TEMPERATURES_BY_OPTIONS = {
  "choice:3-5": 1.7601518630981445,
  "choice:6-10": 1.0000158548355103,
  "score:3-5": 1.2514300346374512,
  "noul:2": 1.983399510383606,
  "choice:11+": 0.10058280825614929,
  "choice:2": 1.9063563346862793,
};
export const LAYA_DEFAULT_MAX_LEN = 1024;
export const LAYA_DEFAULT_HEAD_MAX_LEN = 256;
export const LAYA_DEFAULT_MAX_PREFIXES = 6;

export const LAYA_QTYPE = { choice: 0, score: 1, noul: 2 };
const QTYPE_NAMES = { 0: "choice", 1: "score", 2: "noul" };

// The Python runtime refuses temperatures outside [0.5, 5]: the shipped
// `choice:11+` bucket (0.1006) would sharpen logits ~10x and publish a coin
// flip as near-certainty.
export const LAYA_TEMP_MIN = 0.5;
export const LAYA_TEMP_MAX = 5;

export function clampTemperature(t, lo = LAYA_TEMP_MIN, hi = LAYA_TEMP_MAX) {
  const n = Number(t);
  if (!Number.isFinite(n)) return 1;
  return Math.min(hi, Math.max(lo, n));
}

export function layaTempBucket(qtype, k) {
  const size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
  return `${QTYPE_NAMES[qtype]}:${size}`;
}

/**
 * Option texts exactly as the reference runtime renders them.
 * Our normalized questions carry no per-option criteria, so choice options
 * render as their labels and noul falls back to the reference wording.
 */
export function renderLayaOptions(question) {
  if (question.type === "choice") return question.options.map(String);
  if (question.type === "score") return question.options.map((c, i) => `level ${i}: ${c}`);
  // noul
  const [no, yes] = question.options;
  if (no === "no" && yes === "yes") {
    return ["false: no, the statement does not hold", "true: yes, the statement holds"];
  }
  return [`false: ${no}`, `true: ${yes}`];
}

/**
 * Build one Laya sequence + marker positions.
 * @param {{encode:(text:string)=>number[], ids:{cls:number,sep:number,mask:number},
 *   state:string, question:object, maxLen?:number, headMaxLen?:number,
 *   maxPrefixes?:number}} spec
 * @returns {{ids:number[], markers:number[], qtype:number}}
 */
export function buildLayaSequence({
  encode,
  ids,
  state,
  question,
  maxLen = LAYA_DEFAULT_MAX_LEN,
  headMaxLen = LAYA_DEFAULT_HEAD_MAX_LEN,
  maxPrefixes = LAYA_DEFAULT_MAX_PREFIXES,
}) {
  const options = renderLayaOptions(question);
  if (options.length > maxPrefixes) {
    throw new RangeError(
      `laya: ${question.type} with ${options.length} options exceeds max_prefixes=${maxPrefixes}`);
  }
  const qtype = LAYA_QTYPE[question.type];
  if (qtype === undefined) throw new TypeError(`laya: unsupported question type ${question.type}`);

  const strip = (s) => String(s).replace(/\[MASK\]/g, " ");
  let headIds = encode(`${question.type} question: ${strip(question.instructions)}`);
  let optIds = options.map((o) => [ids.mask, ...encode(` ${strip(o)}`).slice(0, 48)]);

  let optBudget = headMaxLen - optIds.reduce((n, o) => n + o.length, 0);
  if (optBudget < 16) {
    const per = Math.max(4, Math.floor((headMaxLen - 16) / Math.max(1, optIds.length)));
    optIds = optIds.map((o) => o.slice(0, per));
    optBudget = headMaxLen - optIds.reduce((n, o) => n + o.length, 0);
  }
  headIds = headIds.slice(0, Math.max(8, optBudget));

  let seq = [ids.cls, ...headIds, ids.sep];
  const markers = [];
  for (const o of optIds) {
    markers.push(seq.length);
    seq.push(...o);
  }
  seq.push(ids.sep);
  const room = Math.max(0, maxLen - seq.length - 1);
  const stateIds = encode(String(state ?? "")).slice(0, room);
  seq = [...seq, ...stateIds, ids.sep];

  const clipped = seq.slice(0, maxLen);
  return {
    ids: clipped,
    markers: markers.filter((m) => m < maxLen),
    qtype,
  };
}

/** Pad a batch of Laya items into int64/bool tensors. */
export function collateLayaItems(items, padId = 0) {
  const n = items.length;
  const L = Math.max(...items.map((it) => it.ids.length));
  const M = Math.max(...items.map((it) => it.markers.length));
  const inputIds = new BigInt64Array(n * L).fill(BigInt(padId));
  const attention = new BigInt64Array(n * L);
  const markerPos = new BigInt64Array(n * M);
  const markerMask = new Uint8Array(n * M);
  const qtype = new BigInt64Array(n);
  items.forEach((it, i) => {
    it.ids.forEach((v, j) => { inputIds[i * L + j] = BigInt(v); attention[i * L + j] = 1n; });
    it.markers.forEach((m, j) => { markerPos[i * M + j] = BigInt(m); markerMask[i * M + j] = 1; });
    qtype[i] = BigInt(it.qtype);
  });
  return { n, L, M, inputIds, attention, markerPos, markerMask, qtype };
}

/**
 * Decode head logits (row-major [question][marker]) into jev-web answers.
 * @param {{logits:number[]|Float32Array, questions:object[], markerCounts:number[],
 *   temperatures?:number[], temperatureByOptions?:object}} spec
 */
export function layaAnswersFromLogits({
  logits,
  questions,
  markerCounts,
  temperatures = LAYA_DEFAULT_TEMPERATURES,
  temperatureByOptions = LAYA_DEFAULT_TEMPERATURES_BY_OPTIONS,
}) {
  const answers = [];
  let offset = 0;
  questions.forEach((q, i) => {
    const k = markerCounts[i];
    const qtype = LAYA_QTYPE[q.type];
    const t = clampTemperature(
      temperatureByOptions[layaTempBucket(qtype, k)] ?? temperatures[qtype] ?? 1,
    );
    const probs = softmaxWithTemperature(Array.from(logits.slice(offset, offset + k), Number), t);
    offset += k;

    const entropy = k > 1
      ? -probs.reduce((acc, p) => acc + p * Math.log(Math.max(p, 1e-12)), 0) / Math.log(k)
      : 0;
    const confidence = Math.min(1, Math.max(0, 1 - entropy));
    const best = probs.indexOf(Math.max(...probs));

    if (q.type === "choice") {
      answers.push({
        type: "choice",
        choice: q.options[best],
        index: best,
        probabilities: Object.fromEntries(q.options.map((o, j) => [o, probs[j]])),
        confidence,
      });
    } else if (q.type === "score") {
      const expected = probs.reduce((acc, p, j) => acc + p * j, 0);
      answers.push({
        type: "score",
        score: expected,
        level: best,
        probabilities: Object.fromEntries(q.options.map((o, j) => [o, probs[j]])),
        confidence,
      });
    } else {
      const yes = probs[1];
      answers.push({
        type: "noul",
        noul: yes,
        probabilities: { [q.options[0]]: probs[0], [q.options[1]]: probs[1] },
        confidence: Math.max(yes, 1 - yes),
      });
    }
  });
  return answers;
}

/**
 * Fetch a JSON file with optional byte progress (fetch/Response compatible).
 * Falls back to `res.json()` when the body isn't streamable (Node fakes/tests).
 */
async function fetchJson(url, file, { fetchImpl, onProgress }) {
  const bytes = await fetchCachedBytes(url, {
    fetchImpl,
    onProgress: (p) => onProgress?.({ ...p, phase: "tokenizer", file }),
  });
  return JSON.parse(new TextDecoder().decode(bytes));
}

/**
 * Load a Laya tokenizer directly from its files.
 *
 * transformers.js `AutoTokenizer.from_pretrained` ignores `subfolder`, and the
 * Laya web export keeps its tokenizer under `v1/` (and its root has no
 * `tokenizer_config.json`), so the library path throws
 * "Cannot read properties of undefined (reading 'tokenizer_class')".
 * Fetch the two JSON files ourselves and instantiate the class the config
 * names (e.g. `PreTrainedTokenizerFast` -> `PreTrainedTokenizer`).
 *
 * @param {{urlBase:string, fetchImpl?:Function, transformers?:object,
 *   onProgress?:Function}} spec `urlBase` must end with "/".
 * @returns {Promise<object>} tokenizer instance
 */
export async function loadLayaTokenizer({
  urlBase,
  fetchImpl = fetch,
  transformers = null,
  onProgress = null,
} = {}) {
  if (!urlBase) throw new TypeError("loadLayaTokenizer: urlBase is required");
  const base = urlBase.endsWith("/") ? urlBase : `${urlBase}/`;
  const [tokenizerJSON, tokenizerConfig] = await Promise.all([
    fetchJson(`${base}tokenizer.json`, "tokenizer.json", { fetchImpl, onProgress }),
    fetchJson(`${base}tokenizer_config.json`, "tokenizer_config.json", { fetchImpl, onProgress }),
  ]);
  const tf = transformers ?? (await import("@huggingface/transformers"));
  const className = String(tokenizerConfig?.tokenizer_class ?? "PreTrainedTokenizer")
    .replace(/Fast$/, "");
  const Cls = tf[className] ?? tf.PreTrainedTokenizer;
  if (typeof Cls !== "function") {
    throw new Error(`loadLayaTokenizer: no tokenizer class for "${className}"`);
  }
  return new Cls(tokenizerJSON, tokenizerConfig);
}

/** Wrap a transformers.js tokenizer into the tiny adapter laya needs. */
export function layaTokenizerAdapter(tok) {
  const idOf = (prop, text) => {
    const direct = tok?.[prop];
    if (Number.isFinite(direct)) return Number(direct);
    const ids = tok.encode?.(text) ?? tok(text, { add_special_tokens: true }).input_ids?.data;
    return Number(Array.from(ids ?? [])[0] ?? 0);
  };
  return {
    encode: (text) => Array.from(tok(text, { add_special_tokens: false }).input_ids.data, Number),
    decode: (ids) => {
      try {
        return tok.decode(Array.from(ids), { skip_special_tokens: false });
      } catch {
        return Array.from(ids).join(" ");
      }
    },
    ids: {
      cls: idOf("cls_token_id", "[CLS]"),
      sep: idOf("sep_token_id", "[SEP]"),
      mask: idOf("mask_token_id", "[MASK]"),
    },
    pad: idOf("pad_token_id", "[PAD]"),
  };
}

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

const hfUrl = (model, revision, file) =>
  `https://huggingface.co/${model}/resolve/${revision ? `${revision}/` : ""}${file}`;

const MODEL_CACHE = "kf-models-v1";

async function modelCacheStore() {
  try {
    return typeof caches !== "undefined" ? caches : null;
  } catch {
    return null;
  }
}

/** Ask for persistent storage once so big weights survive eviction pressure. */
export async function requestPersistentStorage() {
  try {
    return (await globalThis.navigator?.storage?.persist?.()) ?? false;
  } catch {
    return false;
  }
}

/**
 * Fetch bytes with a Cache-API layer. Hugging Face redirects to a *signed*
 * CDN URL that changes per request, so the browser HTTP cache never hits for
 * the 468 MB encoder weights; this caches them under the stable model URL
 * instead (Cache API), so a page refresh is free.
 *
 * @returns {Promise<Uint8Array>}
 */
export async function fetchCachedBytes(url, {
  fetchImpl = fetch,
  onProgress = null,
  cacheName = MODEL_CACHE,
  force = false,
  persist = false,
} = {}) {
  const store = await modelCacheStore();
  if (persist) await requestPersistentStorage();
  if (store && !force) {
    try {
      const hit = await store.match(url);
      if (hit) {
        const data = new Uint8Array(await hit.arrayBuffer());
        onProgress?.({ phase: "cache", file: String(url).split("/").pop(), loaded: data.length, total: data.length });
        return data;
      }
    } catch {
      /* cache read failed — fall through to the network */
    }
  }
  const res = await fetchImpl(url);
  if (!res?.ok) throw new Error(`fetch ${url} failed (HTTP ${res?.status ?? "no response"})`);
  const data = await readBytesWithProgress(res, url, onProgress);
  if (store) {
    try {
      await store.put(url, new Response(data, {
        headers: { "content-type": "application/octet-stream", "content-length": String(data.length) },
      }));
    } catch {
      /* quota exceeded / private mode — caching is best-effort */
    }
  }
  return data;
}

async function readBytesWithProgress(res, url, onProgress) {
  const total = Number(res.headers?.get?.("content-length")) || 0;
  const file = String(url).split("/").pop();
  if (res.body?.getReader) {
    const reader = res.body.getReader();
    let received = 0;
    let data = total ? new Uint8Array(total) : null;
    const chunks = total ? null : [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (data) {
        if (received + value.length > data.length) {
          // content-length can describe the *encoded* size (gzip/br), so the
          // decoded stream may be larger — grow instead of overflowing.
          const grown = new Uint8Array(Math.max(data.length * 2, received + value.length));
          grown.set(data.subarray(0, received));
          data = grown;
        }
        data.set(value, received);
      } else {
        chunks.push(value);
      }
      received += value.length;
      onProgress?.({ phase: "download", file, loaded: received, total: total || null });
    }
    if (!data) {
      data = new Uint8Array(received);
      let off = 0;
      for (const c of chunks) { data.set(c, off); off += c.length; }
    } else {
      // the buffer may have grown past the decoded length (content-length is
      // the *encoded* size) — always trim to what was actually received
      data = data.subarray(0, received);
    }
    return data;
  }
  let data;
  if (typeof res.arrayBuffer === "function") {
    data = new Uint8Array(await res.arrayBuffer());
  } else if (typeof res.text === "function") {
    data = new TextEncoder().encode(await res.text());
  } else if (typeof res.json === "function") {
    data = new TextEncoder().encode(JSON.stringify(await res.json()));
  } else {
    throw new Error(`fetch ${url}: response has no readable body`);
  }
  onProgress?.({ phase: "download", file, loaded: data.length, total: data.length });
  return data;
}

/**
 * ONNX external-data loader.
 *
 * The pinned Laya export stores weights in `<model>.onnx.data` siblings
 * (encoder ~468 MB, head ~53 MB). onnxruntime-web cannot resolve those from a
 * URL by itself (`Module.MountedFiles is not available`); it needs the bytes
 * passed via `session_options.externalData`. Fetch the sibling (streaming,
 * with progress) and hand it over.
 *
 * @returns {Promise<{path:string, data:Uint8Array}|null>} null when the model
 *   has no external data (404) or the fetch fails.
 */
export async function fetchOnnxExternalData(url, { fetchImpl = fetch, onProgress = null, force = false } = {}) {
  // Probe `<model>.onnx.data` (laya web-q8 layout) then `<model>.onnx_data`
  // (onnx-community layout); a 404 on both means the model has no external data.
  for (const suffix of [".data", "_data"]) {
    const dataUrl = `${url}${suffix}`;
    const path = `${String(url).split("/").pop()}${suffix}`;
    try {
      // Probe cheaply first: a 404 means "try the next suffix".
      const head = await fetchImpl(dataUrl, { method: "HEAD" }).catch(() => null);
      if (head && !head.ok) continue;
      const data = await fetchCachedBytes(dataUrl, { fetchImpl, onProgress, force, persist: true });
      return { path, data };
    } catch {
      // fall through to the next suffix
    }
  }
  return null;
}

/**
 * Create a Laya decider.
 *
 * @param {object} [opts]
 *  model/revision/subfolder — pinned web-q8 ONNX export (or your own mirror)
 *  encoderUrl/headUrl       — override the resolved ONNX URLs
 *  ort                      — injected onnxruntime-web (lazy import otherwise)
 *  transformers             — injected @huggingface/transformers (tokenizer only)
 *  wasmPaths                — ORT wasm/loader directory (self-hosted builds)
 *  device                   — "auto" | "wasm" | "webgpu"
 */
export async function createLayaDecider({
  model = LAYA_DEFAULT_MODEL,
  revision = LAYA_DEFAULT_REVISION,
  subfolder = LAYA_DEFAULT_SUBFOLDER,
  encoderUrl = null,
  headUrl = null,
  tokenizerModel = null,
  tokenizerSubfolder = null,
  ort = null,
  transformers = null,
  fetchImpl = fetch,
  wasmPaths = null,
  device = "auto",
  // (url, sessionOptions) => Promise<InferenceSession>. Hosts that need custom
  // loading (local files, bundled weights, offline mirrors) override this.
  sessionFactory = null,
  onProgress = null,
  maxLen = LAYA_DEFAULT_MAX_LEN,
  headMaxLen = LAYA_DEFAULT_HEAD_MAX_LEN,
  maxPrefixes = LAYA_DEFAULT_MAX_PREFIXES,
  temperatures = LAYA_DEFAULT_TEMPERATURES,
  temperatureByOptions = LAYA_DEFAULT_TEMPERATURES_BY_OPTIONS,
  scope = globalThis,
} = {}) {
  const ortMod = ort ?? (await import("onnxruntime-web"));
  const tf = transformers ?? (await import("@huggingface/transformers"));
  if (wasmPaths) {
    try { ortMod.env.wasm.wasmPaths = wasmPaths; } catch { /* older runtime */ }
  }

  const resolvedDevice = device !== "auto"
    ? device
    : (scope?.navigator?.gpu ? "webgpu" : "wasm");
  const executionProviders = resolvedDevice === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];

  const report = (phase) => (p) =>
    onProgress?.({ phase, file: p?.file ?? null, loaded: p?.loaded ?? null, total: p?.total ?? null });

  let tok;
  if (tokenizerModel) {
    // Host-provided path/repo (local dir in Node, or a repo whose tokenizer
    // lives at the root) — use the library loader.
    const tokSubfolder = tokenizerSubfolder === null
      ? undefined
      : (tokenizerSubfolder ?? subfolder);
    tok = await tf.AutoTokenizer.from_pretrained(tokenizerModel, {
      revision: tokenizerSubfolder ? undefined : revision,
      ...(tokSubfolder ? { subfolder: tokSubfolder } : {}),
      progress_callback: report("tokenizer"),
    });
  } else {
    // Default: the pinned web export keeps the tokenizer under `v1/`, which
    // AutoTokenizer cannot address (no `subfolder` support) — load it directly.
    tok = await loadLayaTokenizer({
      urlBase: hfUrl(model, revision, `${subfolder}/`),
      transformers: tf,
      fetchImpl,
      onProgress,
    });
  }
  const adapter = layaTokenizerAdapter(tok);

  const encUrl = encoderUrl ?? hfUrl(model, revision, `${subfolder}/encoder_q8.onnx`);
  const headUrlResolved = headUrl ?? hfUrl(model, revision, `${subfolder}/head_q8.onnx`);
  const createSession = sessionFactory
    ?? (async (url, opts) => {
      const report2 = (p) => onProgress?.(p);
      const modelBytes = await fetchCachedBytes(url, { fetchImpl, onProgress: report2, persist: true });
      const external = await fetchOnnxExternalData(url, { fetchImpl, onProgress: report2 });
      const sessionOpts = external ? { ...opts, externalData: [external] } : opts;
      return ortMod.InferenceSession.create(modelBytes, sessionOpts);
    });
  const loadPair = async (providers) => {
    const opts = { executionProviders: providers, graphOptimizationLevel: "all" };
    const enc = await createSession(encUrl, { ...opts, progress_callback: report("encoder") });
    const hd = await createSession(headUrlResolved, { ...opts, progress_callback: report("head") });
    return { enc, hd };
  };
  let encoder, head, activeDevice = resolvedDevice;
  try {
    ({ enc: encoder, hd: head } = await loadPair(executionProviders));
  } catch (e) {
    // A quantized graph can be rejected by the WebGPU EP at session creation;
    // retry wasm-only before giving up (the caller may still fall back).
    if (resolvedDevice === "webgpu") {
      onProgress?.({ phase: `webgpu unavailable (${e?.message ?? e}) — retrying on wasm` });
      ({ enc: encoder, hd: head } = await loadPair(["wasm"]));
      activeDevice = "wasm";
    } else {
      throw e;
    }
  }

  const int64 = (arr, dims) => new ortMod.Tensor("int64", arr, dims);
  const bool = (arr, dims) => new ortMod.Tensor("bool", arr, dims);

  return {
    info: {
      family: "laya",
      model,
      revision,
      device: activeDevice,
      dtype: "q8",
      maxLen,
      headMaxLen,
      maxPrefixes,
      // Single representative temperature for UIs that show one value
      // (noul:2 bucket, else the noul type temperature); per-bucket values
      // below are what the decoder actually applies.
      temperature: clampTemperature(temperatureByOptions["noul:2"] ?? temperatures[2] ?? 1),
      temperatures: temperatures.map(clampTemperature),
      temperatureByOptions: Object.fromEntries(
        Object.entries(temperatureByOptions).map(([k, v]) => [k, clampTemperature(v)])),
    },
    async decide(state, questions) {
      const t0 = now();
      const normalized = normalizeQuestions(questions);
      const items = normalized.map((q) =>
        buildLayaSequence({ encode: adapter.encode, ids: adapter.ids, state, question: q, maxLen, headMaxLen, maxPrefixes }));
      const b = collateLayaItems(items, adapter.pad);

      const tEncode = now();
      const encOut = await encoder.run({
        input_ids: int64(b.inputIds, [b.n, b.L]),
        attention_mask: int64(b.attention, [b.n, b.L]),
      });
      const encoderMs = now() - tEncode;
      const hidden = encOut.hidden ?? encOut.last_hidden_state ?? encOut[Object.keys(encOut)[0]];

      // The q8 web head is exported batch-1 only (despite dynamic batch
      // axes), so run it once per question over the shared encoder output.
      // The encoder — the expensive half — still runs as one batched pass.
      const headLogits = [];
      const headPerQuestionMs = [];
      const markerCounts = items.map((it) => it.markers.length);
      const tHead = now();
      for (let i = 0; i < b.n; i++) {
        const k = markerCounts[i];
        const row = hidden.data.subarray(i * b.L * hidden.dims[2], (i + 1) * b.L * hidden.dims[2]);
        const rowHidden = new ortMod.Tensor(hidden.type, row, [1, b.L, hidden.dims[2]]);
        const rowMask = b.attention.subarray(i * b.L, (i + 1) * b.L);
        const rowPos = b.markerPos.subarray(i * b.M, i * b.M + k);
        const rowMarkerMask = b.markerMask.subarray(i * b.M, i * b.M + k);
        const tHeadTotal = now();
        const headOut = await head.run({
          hidden: rowHidden,
          attention_mask: int64(rowMask, [1, b.L]),
          marker_pos: int64(rowPos, [1, k]),
          marker_mask: bool(rowMarkerMask, [1, k]),
          qtype: int64(b.qtype.subarray(i, i + 1), [1]),
        });
        const rowLogits = headOut.logits ?? headOut[Object.keys(headOut)[0]];
        headLogits.push(...Array.from(rowLogits.data, Number));
        headPerQuestionMs.push(Math.round(now() - tHeadTotal));
      }

      const answers = layaAnswersFromLogits({
        logits: headLogits,
        questions: normalized,
        markerCounts,
        temperatures,
        temperatureByOptions,
      });
      const length = Number(b.attention.reduce((a, v) => a + v, 0n));
      // Exact, human-readable view of what the encoder consumed: each
      // question's token sequence decoded back to text (special tokens kept).
      const prompts = items.map((it) => adapter.decode(it.ids));
      const headMs = now() - tHead;
      return {
        answers,
        truncated: items.some((it) => it.ids.length >= maxLen),
        length,
        prompts,
        timings: {
          totalMs: Math.round(now() - t0),
          encoderMs: Math.round(encoderMs),
          headMs: Math.round(headMs),
          headPerQuestionMs,
        },
      };
    },
    async dispose() {
      try { await encoder.release?.(); } catch { /* optional */ }
      try { await head.release?.(); } catch { /* optional */ }
    },
  };
}
