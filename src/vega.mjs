/**
 * vega.mjs — Vega 0.8B family (frozen Qwen3.5 backbone + latent decision-physics engine).
 *
 * Vega is the "System One" decision model by the Laya author: the language model
 * is frozen and never trained; everything learned lives in a small physics
 * engine on top of it (github.com/nandakishorm/vega-08b-public-intents ships
 * the reference `vega_common.py` / `vega_api.py`):
 *
 *   prompt (template 0, built by buildVegaPrompt — mirrors build_prompt):
 *     [BOS] Observation:\n <state> \n\nMeasurement (choice|score|boolean): <instructions>
 *     \nPossible outcomes: \n* <opt> … \nOutcome:
 *   perception: one causal read up to the deepest captured layer (13 and 19 of
 *     24), each span mean-pooled -> item feats [B, 2, 3, d] (state / question /
 *     last token) + option feats [B, K, 1, d] + state chunk means for the
 *     memory reader;
 *   engine: the state becomes a world latent, the question drops a particle
 *     into a 64-d space whose landscape holds one Gaussian well per option, a
 *     damped symplectic integrator with a selective-SSM thermostat runs up to
 *     n_steps steps (settled particles freeze), and the Boltzmann occupancy of
 *     the wells is the answer distribution;
 *   calibration: thermal — tau depends on the particle's physical state
 *     (residual kinetic energy, distance to the nearest well, steps used,
 *     number of options), fitted per question type and shipped in
 *     vega_config.json; split-conformal prediction sets and an abstain flag
 *     come from the same energies, plus an "unbound particle" OOD signal.
 *
 * Two ONNX graph layouts ship:
 *
 *   fused (warsang/vega-08b-web, the default) — one fused torso that pools
 *     in-graph + the fp32 engine, 1.5 GB fp16 torso, base engine only;
 *   split (nvkudva/vega-onnx-q8) — an 8-bit embed graph, a cut decoder that
 *     emits the residual stream at layers 13/19 (the LM head and layers 20-23
 *     never run), the fp32 engine with both gated task adapters, and JS
 *     pooling; 840 MB, wasm-only (8-bit MatMulNBits has no WebGPU kernel).
 *
 * JS owns the protocol, the packing, the thermal calibration and the answer
 * decoding in both.
 */
import { normalizeQuestions } from "./questions.mjs";
import { softmaxWithTemperature } from "./answers.mjs";
import { now, hfUrl, argmax, createSerialQueue } from "./util.mjs";
import { fetchCachedJson, fetchCachedBytes, fetchOnnxExternalData, HttpError } from "./fetch.mjs";

export const VEGA_DEFAULT_MODEL = "warsang/vega-08b-web";
export const VEGA_DEFAULT_REVISION = "d2bd4604ed770ca614745f07f623b66e89f406e4";
// The exact fp16 torso ships by default: q8 on this graph drifts on the
// GatedDeltaNet delta rule (export/vega/04_repo.py measures ~0.46 in the
// pooled features), so torso_quantized.onnx stays an opt-in experiment.
export const VEGA_TORSO_FILE = "onnx/torso_fp16.onnx";
export const VEGA_ENGINE_FILE = "onnx/engine.onnx";

// ---------------------------------------------------------------------------
// Split export: nvkudva/vega-onnx-q8 (8-bit embed + cut decoder + fp32 engine)
// ---------------------------------------------------------------------------
export const VEGA_Q8_DEFAULT_MODEL = "nvkudva/vega-onnx-q8";
export const VEGA_Q8_DEFAULT_REVISION = "017c096d369ac698087f8c35a6ad8c867a7fe0f4";
export const VEGA_Q8_DEFAULT_SUBFOLDER = "v2";
export const VEGA_Q8_EMBED_FILE = "embed_q8.onnx";
export const VEGA_Q8_DECODER_FILE = "decoder_q8.onnx";
export const VEGA_Q8_ENGINE_FILE = "engine.onnx";

/**
 * Named exports. `fp16` is this repo's own fused export (default); `q8` is
 * nvkudva's split 8-bit export of the same checkpoint, which adds the two
 * gated task adapters (public_intents / typed_decisions) and is wasm-only.
 */
export const VEGA_VARIANTS = {
  fp16: {
    model: VEGA_DEFAULT_MODEL,
    revision: VEGA_DEFAULT_REVISION,
    format: "fused",
    description: "fused fp16 torso + fp32 engine, 1.7 GB (the default)",
  },
  q8: {
    model: VEGA_Q8_DEFAULT_MODEL,
    revision: VEGA_Q8_DEFAULT_REVISION,
    subfolder: VEGA_Q8_DEFAULT_SUBFOLDER,
    format: "split",
    description: "8-bit split embed + cut decoder + fp32 engine with adapters, 840 MB, wasm-only",
  },
};

/** Reference prompt template 0 (the inference surface form). */
const VEGA_HEAD = "Observation:\n";
const VEGA_QUESTION = "\n\nMeasurement ({t}): {ins}\nPossible outcomes:";
const VEGA_OPTION = "\n* {o}";
const VEGA_END = "\nOutcome:";
/** Prefixed to short states so the frozen model reads knowing what is asked. */
const VEGA_QUERY_FIRST = "Question to keep in mind while reading ({t}): {ins}\n\n";

/** jev-web question types -> Vega qtypes (choice=0, score=1, boolean=2). */
export const VEGA_QTYPE = { choice: 0, score: 1, boolean: 2 };
export const VEGA_QTYPE_NAMES = ["choice", "score", "boolean"];

export const VEGA_DEFAULTS = {
  optMaxTokens: 32,
  insMaxTokens: 256,
  memoryChunks: 16,
  memoryMinTokens: 128,
  memoryChunkTokens: 128,
  prefixMinTokens: 4096,
  queryFirstMaxState: 4096,
  maxLen: 73728,
  nSteps: 12,
  risk: "0.1",
  alpha: "0.1",
};

// ---------------------------------------------------------------------------
// Protocol: question defs, option texts, prompt building
// ---------------------------------------------------------------------------

/**
 * Map a normalized jev-web question to Vega's internal question def.
 * Vega's `boolean` is the noul primitive; its slots are fixed [no, yes].
 */
export function vegaQuestionDef(question) {
  // the reference json.dumps non-string instructions with ensure_ascii=True
  const ins = typeof question.instructions === "string"
    ? question.instructions
    : vegaJsonDumps(question.instructions, true);
  if (question.type === "noul") {
    return { t: "boolean", ins, crit: { ...(question.criteria ?? {}) } };
  }
  if (question.type === "score") {
    return { t: "score", ins, crit: [...question.options] };
  }
  // choice: criteria {label: description} when provided, else bare options.
  const crit = question.criteria && typeof question.criteria === "object"
    && !Array.isArray(question.criteria)
    ? question.criteria
    : Object.fromEntries(question.options.map((o) => [String(o), null]));
  return { t: "choice", ins, crit };
}

/**
 * Python `json.dumps(x, ensure_ascii)` — ", " / ": " separators, control
 * characters escaped, and optionally \uXXXX for everything above ASCII. The
 * reference runtime serializes dict states, non-string instructions and
 * non-string criteria values with it, so the prompt stays byte-identical to
 * the Python original (JSON.stringify uses compact separators and never
 * escapes non-ASCII).
 */
export function vegaJsonDumps(value, ensureAscii = false) {
  const esc = (s) => {
    let out = '"';
    for (const ch of s) {
      const c = ch.codePointAt(0);
      if (ch === '"') out += '\\"';
      else if (ch === "\\") out += "\\\\";
      else if (ch === "\n") out += "\\n";
      else if (ch === "\r") out += "\\r";
      else if (ch === "\t") out += "\\t";
      else if (c < 0x20) out += "\\u" + c.toString(16).padStart(4, "0");
      else if (ensureAscii && c > 0x7f) {
        if (c > 0xffff) {
          const u = c - 0x10000;
          out += "\\u" + (0xd800 + (u >> 10)).toString(16).padStart(4, "0");
          out += "\\u" + (0xdc00 + (u & 0x3ff)).toString(16).padStart(4, "0");
        } else out += "\\u" + c.toString(16).padStart(4, "0");
      } else out += ch;
    }
    return out + '"';
  };
  const go = (x) => {
    if (x === null || x === undefined) return "null";
    if (typeof x === "string") return esc(x);
    if (typeof x === "boolean") return x ? "true" : "false";
    if (typeof x === "number") return String(x);
    if (Array.isArray(x)) return "[" + x.map(go).join(", ") + "]";
    return "{" + Object.entries(x).map(([k, v]) => esc(k) + ": " + go(v)).join(", ") + "}";
  };
  return go(value);
}

/** json.dumps of a non-string criterion value (strings pass through). */
function vegaCritText(v) {
  return typeof v === "string" ? v : vegaJsonDumps(v, false);
}

/** The reference render_text: strings verbatim, anything else json.dumps. */
export function renderVegaText(content) {
  return typeof content === "string" ? content : vegaJsonDumps(content, false);
}

/** Option texts in label-index order, exactly as option_texts() renders them. */
export function vegaOptionTexts(qdef) {
  if (qdef.t === "choice") {
    return Object.entries(qdef.crit).map(([k, v]) => {
      const d = v === null || v === undefined ? "" : vegaCritText(v);
      return d ? `${k} - ${d}` : String(k);
    });
  }
  if (qdef.t === "score") {
    return qdef.crit.map((c, i) => `level ${i} - ${vegaCritText(c)}`);
  }
  const crit = qdef.crit ?? {};
  const no = crit.false === null || crit.false === undefined ? "" : vegaCritText(crit.false);
  const yes = crit.true === null || crit.true === undefined ? "" : vegaCritText(crit.true);
  return [
    `no - ${no || "the statement does not hold"}`,
    `yes - ${yes || "the statement holds"}`,
  ];
}

/**
 * One prompt: BOS + [query-first prefix for short states] + head + state +
 * question + options + end, with the token span of every segment.
 *
 * The state comes first so its span never sees the question (causal
 * attention); the question, options and final token have read the whole
 * state. Options are capped per option, and an over-long option list is
 * shrunk evenly before the instructions are (mirrors build_prompt).
 *
 * `truncateState` defaults to the reference's silent truncation; the split
 * export refuses the decision instead (a cut-off record is worse than none),
 * so its runtime passes false and gets a RangeError.
 *
 * The shared truncation contract applies: `truncated` is true iff any user
 * text was shortened — an option over `optMaxTokens`, instructions over
 * `insMaxTokens`, the query-first prefix over its 128-token cap, a state that
 * did not fit its room, or the fixed-size re-fit cutting any of the above.
 *
 * @returns {{ids:number[], spans:{state:[number,number], question:[number,number],
 *   options:[number,number][], last:number}, truncated:boolean}}
 */
export function buildVegaPrompt({
  encode,
  stateText,
  qdef,
  maxLen,
  bosId = null,
  queryFirstMaxState = VEGA_DEFAULTS.queryFirstMaxState,
  optMaxTokens = VEGA_DEFAULTS.optMaxTokens,
  insMaxTokens = VEGA_DEFAULTS.insMaxTokens,
  truncateState = true,
}) {
  const options = vegaOptionTexts(qdef);
  const enc = (s) => encode(s, false);
  const bos = bosId === null ? [] : [bosId];
  let truncated = false;
  let head = enc(VEGA_HEAD);
  const qFull = enc(VEGA_QUESTION.replace("{t}", qdef.t).replace("{ins}", qdef.ins));
  if (qFull.length > insMaxTokens) truncated = true;
  let qids = qFull.slice(0, insMaxTokens);
  let oids = options.map((o) => {
    const full = enc(VEGA_OPTION.replace("{o}", o));
    if (full.length > optMaxTokens) truncated = true;
    return full.slice(0, optMaxTokens);
  });
  const end = enc(VEGA_END);
  let fixed = bos.length + head.length + qids.length
    + oids.reduce((n, o) => n + o.length, 0) + end.length;
  if (fixed > maxLen - 16) {
    const per = Math.max(3, Math.floor(
      (maxLen - 16 - bos.length - head.length - end.length - Math.min(qids.length, 64))
      / Math.max(1, oids.length)));
    oids = oids.map((o) => {
      if (o.length > per) truncated = true;
      return o.slice(0, per);
    });
    const qRoom = Math.max(
      8, maxLen - 16 - bos.length - head.length - end.length
        - oids.reduce((n, o) => n + o.length, 0));
    if (qids.length > qRoom) truncated = true;
    qids = qids.slice(0, qRoom);
    fixed = bos.length + head.length + qids.length
      + oids.reduce((n, o) => n + o.length, 0) + end.length;
  }
  let st = enc(stateText);
  let pre = [];
  if (queryFirstMaxState && st.length < queryFirstMaxState) {
    const preFull = enc(VEGA_QUERY_FIRST.replace("{t}", qdef.t).replace("{ins}", qdef.ins));
    const cap = Math.min(128, insMaxTokens);
    if (preFull.length > cap) truncated = true;
    pre = preFull.slice(0, cap);
    fixed += pre.length;
  }
  const room = Math.max(0, maxLen - fixed);
  if (!truncateState && st.length > room) {
    throw new RangeError(
      `vega: state is ${st.length} tokens, only ${room} fit in maxLen=${maxLen}`
      + " (this export refuses a truncated state; raise maxLen)");
  }
  if (st.length > room) truncated = true;
  st = st.slice(0, room); // truncate_left = false: the head of the state is kept
  const ids = [...bos, ...pre, ...head];
  const s0 = ids.length;
  ids.push(...st);
  const spans = { state: st.length ? [s0, ids.length] : [bos.length, s0] };
  const q0 = ids.length;
  ids.push(...qids);
  spans.question = [q0, ids.length];
  spans.options = [];
  for (const o of oids) {
    const a = ids.length;
    ids.push(...o);
    spans.options.push([a, ids.length]);
  }
  ids.push(...end);
  spans.last = ids.length - 1;
  return { ids, spans, truncated };
}

// ---------------------------------------------------------------------------
// Batch packing: padding, segment means (one-hot pooling), state chunks
// ---------------------------------------------------------------------------

/**
 * Pack rendered prompts into tensors + the pooling matrix the torso graph
 * consumes: one row per segment (per item: state, question, options, the
 * final token, then C state-chunk rows), each segment's row is 1/len over its
 * token span, zeros elsewhere. The graph computes pool @ hidden, i.e. exact
 * span means, so no hidden states ever cross into JS.
 *
 * @returns {{inputIds:BigInt64Array, attention:BigInt64Array,
 *   pool:Float32Array, poolDims:[number,number,number], n:number, L:number,
 *   K:number, qtypes:number[], optMask:Uint8Array, nseg:number}}
 */
export function packVegaBatch(items, { padId = 0, chunks: C = VEGA_DEFAULTS.memoryChunks } = {}) {
  const n = items.length;
  const L = Math.max(...items.map((it) => it.ids.length));
  const K = Math.max(...items.map((it) => it.options.length));
  const nseg = 3 + K;
  const rows = nseg + C;
  const inputIds = new BigInt64Array(n * L);
  const attention = new BigInt64Array(n * L);
  const pool = new Float32Array(n * rows * L); // [B, nseg + C, L]
  const optMask = new Uint8Array(n * K);
  const qtypes = [];
  items.forEach((it, b) => {
    it.ids.forEach((v, j) => {
      inputIds[b * L + j] = BigInt(v);
      attention[b * L + j] = 1n;
    });
    const setSpan = (row, [a, e]) => {
      const len = Math.max(1, e - a);
      for (let t = a; t < e; t++) pool[(b * rows + row) * L + t] = 1 / len;
    };
    setSpan(0, it.spans.state);
    setSpan(1, it.spans.question);
    it.spans.options.forEach((span, k) => {
      if (k >= K) return;
      setSpan(3 + k, span);
      optMask[b * K + k] = 1;
    });
    pool[(b * rows + 2) * L + it.spans.last] = 1;
    qtypes.push(it.qtype);
  });
  return {
    inputIds, attention, pool, poolDims: [n, rows, L],
    n, L, K, qtypes, optMask, nseg,
  };
}

/**
 * State chunk means for the memory reader: the state span cut into k equal
 * pieces (about one per memoryChunkTokens tokens, at least 2, at most C;
 * states shorter than memoryMinTokens get none) — mirrors _chunk_pool.
 *
 * Fills rows [nseg, nseg + C) of the SAME pooling matrix the segment means
 * use, so the torso graph emits chunk rows in the one forward it already
 * does (a second run would re-read the whole record).
 *
 * @param {Float32Array} pool  the [B, nseg + C, L] pooling matrix
 * @param {object[]} items     rendered prompts (with spans)
 * @param {number} L           padded sequence length
 */
export function packVegaChunks(pool, items, L, {
  chunks: C = VEGA_DEFAULTS.memoryChunks,
  chunkMinTokens = VEGA_DEFAULTS.memoryMinTokens,
  chunkTokens = VEGA_DEFAULTS.memoryChunkTokens,
  nseg = 3 + Math.max(...items.map((it) => it.options.length)),
} = {}) {
  const n = items.length;
  const chunkMask = new Uint8Array(n * C);
  items.forEach((it, b) => {
    const [s0, s1] = it.spans.state;
    const len = s1 - s0;
    if (len < Math.max(chunkMinTokens, 2)) return;
    const k = Math.max(2, Math.min(C, Math.floor(len / Math.max(1, chunkTokens))));
    for (let c = 0; c < k; c++) {
      // token t of the state belongs to chunk floor(t * k / len) (the cid map)
      const row = b * (nseg + C) + nseg + c;
      let count = 0;
      for (let t = s0; t < s1; t++) {
        if (Math.floor(((t - s0) * k) / len) === c) { pool[row * L + t] = 1; count++; }
      }
      if (count > 0) {
        for (let t = s0; t < s1; t++) {
          if (pool[row * L + t]) pool[row * L + t] = 1 / count;
        }
        chunkMask[b * C + c] = 1;
      }
    }
  });
  return chunkMask;
}

// ---------------------------------------------------------------------------
// Thermal calibration + answer decoding (mirrors thermal_tau / decide)
// ---------------------------------------------------------------------------

/** Fallback when a checkpoint ships no calibration block (probabilities raw). */
export const VEGA_DEFAULT_THETA = [[0], [0], [0]];

/**
 * Per-type thermal calibration from the checkpoint's vega_config.json:
 * log_tau = b + w . phys, tau in [0.2, 20]; probs = softmax(logits / tau).
 *
 * `thetaByRow` overrides the per-type lookup with an explicit theta block per
 * item (the split export calibrates each question with the routed adapter's
 * block).
 *
 * @param {{theta:number[][]}} calibration
 * @param {{logits:ArrayLike<number>, rowWidth:number, qtypes:number[],
 *   phys:ArrayLike<number>, nOpts:number[], thetaByRow?:number[][][]}} spec
 *   phys is [items, 4]
 * @returns {number[][]} probabilities per item, in prompt order
 */
export function vegaThermalProbabilities({
  logits, rowWidth, qtypes, phys, nOpts,
  theta = VEGA_DEFAULT_THETA,
  thetaByRow = null,
}) {
  const rows = Number(logits.length / rowWidth);
  const out = [];
  for (let b = 0; b < rows; b++) {
    const k = nOpts[b];
    const t = thetaByRow ? (thetaByRow[b] ?? theta[0]) : (theta[qtypes[b]] ?? theta[0]);
    let logTau = t[0];
    for (let f = 0; f < 4; f++) logTau += t[1 + f] * phys[b * 4 + f];
    const tau = Math.exp(Math.min(Math.log(20), Math.max(Math.log(0.2), logTau)));
    out.push(softmaxWithTemperature(
      Array.from(logits.slice(b * rowWidth, b * rowWidth + k)), tau));
  }
  return out;
}

/**
 * Per-item thermal features: log1p(kinetic), log1p(nearest), steps/steps_max,
 * log(k) — exactly thermal_features().
 */
export function vegaThermalFeatures({ kinetic, nearest, stepsUsed, nOpts, maxSteps }) {
  const out = [];
  for (let b = 0; b < nOpts.length; b++) {
    out.push([
      Math.log1p(kinetic[b]),
      Math.log1p(nearest[b]),
      stepsUsed[b] / Math.max(1, maxSteps),
      Math.log(Math.max(2, nOpts[b])),
    ]);
  }
  return out;
}

/**
 * Vega answers -> jev-web DecideResult answers (additive vega fields:
 * predictionSet / abstain / unbound / settleSteps / energy, and — when the
 * split export routes per question — adapter / gate).
 *
 * @param {{probs:number[][], qtypes:number[], labels:string[][],
 *   physics:{nearest:ArrayLike<number>, stepsUsed:ArrayLike<number>},
 *   rawLogits:ArrayLike<number>, rowWidth:number,
 *   calibration:object, alpha?:string, risk?:string,
 *   calibrations?:object[], adapterNames?:string[], gates?:number[][],
 *   routes?:number[]}} spec calibrations/gates are per row when routed
 */
export function vegaAnswers({
  probs, qtypes, physics, rawLogits, rowWidth, calibration, questions, labels,
  alpha = VEGA_DEFAULTS.alpha, risk = VEGA_DEFAULTS.risk,
  calibrations = null, adapterNames = null, gates = null, routes = null,
}) {
  const answers = [];
  labels.forEach((optionLabels, r) => {
    const p = probs[r];
    const k = optionLabels.length;
    const qtype = VEGA_QTYPE_NAMES[qtypes[r]];
    const cal = calibrations?.[r] ?? calibration;
    const conf = Math.max(...p);
    const best = argmax(p);
    // LAC prediction set; the top option is always kept so it is never empty.
    const th = cal?.conformal?.[qtype]?.[String(alpha)];
    const predictionSet = th === undefined ? null
      : optionLabels.filter((_, i) => i === best || p[i] >= Math.max(1 - th, 1e-12));
    // energy = -max raw logit over the question's real options
    let bestLogit = -Infinity;
    for (let j = 0; j < k; j++) bestLogit = Math.max(bestLogit, rawLogits[r * rowWidth + j]);
    const a = {
      probabilities: Object.fromEntries(optionLabels.map((l, i) => [l, p[i]])),
      confidence: conf,
      predictionSet,
      abstain: conf < (cal?.abstain_threshold?.[String(risk)] ?? 0),
      unbound: physics.nearest[r] > (cal?.unbound_threshold ?? Infinity),
      settleSteps: physics.stepsUsed[r],
      energy: -bestLogit,
    };
    if (adapterNames) {
      const rt = routes?.[r] ?? -1;
      a.adapter = rt >= 0 ? (adapterNames[rt] ?? null) : null;
      a.gate = gates?.[r]?.length ? Math.max(...gates[r]) : 0;
    }
    if (qtype === "choice") {
      a.type = "choice";
      a.choice = optionLabels[best];
      a.index = best;
    } else if (qtype === "score") {
      a.type = "score";
      a.level = best;
      a.score = p.reduce((acc, pj, j) => acc + pj * j, 0);
    } else {
      a.type = "noul";
      a.noul = p[1];
    }
    answers.push(a);
  });
  return answers;
}

/** Per-question option labels in prompt order (choice keys / score levels / [no, yes]). */
export function vegaOptionLabels(questions) {
  return questions.map((q) => {
    if (q.type === "noul") return [...q.options];
    if (q.type === "score") return q.options.map((_, i) => i);
    return [...q.options];
  });
}

// ---------------------------------------------------------------------------
// Split export: JS-side perception (positions, past state, pooling, routing)
// ---------------------------------------------------------------------------

/**
 * Adapter routing (reference route_adapters): the most open adapter if its
 * gate reaches the threshold, else -1 (the base engine calibrates the answer).
 */
export function vegaRoute(gates, threshold = 0.5) {
  if (!gates?.length) return -1;
  let best = 0;
  for (let i = 1; i < gates.length; i++) if (gates[i] > gates[best]) best = i;
  return gates[best] >= threshold ? best : -1;
}

/** Text M-RoPE positions: 0..L-1 on each of the three rows ([3, 1, L]). */
export function vegaTextPositions(L) {
  const out = new BigInt64Array(3 * L);
  for (let i = 0; i < L; i++) {
    out[i] = BigInt(i);
    out[L + i] = BigInt(i);
    out[2 * L + i] = BigInt(i);
  }
  return out;
}

/**
 * Zero-filled first-read state for every past_* input the cut decoder keeps
 * (conv / recurrent / key-value caches). Shapes come from the graph's own
 * input metadata: batch_size -> 1, any other symbol (past_sequence_length)
 * -> 0.
 */
export function emptyVegaPast(session, Tensor) {
  const feeds = {};
  for (const m of session?.inputMetadata ?? []) {
    if (m?.isTensor === false || !String(m.name).startsWith("past_")) continue;
    const dims = (m.shape ?? []).map(
      (d) => (typeof d === "number" ? d : d === "batch_size" ? 1 : 0));
    feeds[m.name] = new Tensor("float32",
      new Float32Array(dims.reduce((a, b) => a * b, 1)), dims);
  }
  return feeds;
}

/**
 * Perception._pooled_batch for one text item, over the hidden states the split
 * decoder emits: the state/question/final-token span means at every feature
 * layer, the option span means at the deepest option layer, and the state cut
 * into up to C equal chunk means for the memory reader (states shorter than
 * chunkMinTokens get none). Image states get no chunks, as in vega_vision.py.
 *
 * @returns {{feats:Float32Array, opts:Float32Array, chunks:Float32Array,
 *   cmask:Uint8Array}} feats [2, 3, d] (layer, slot, d)
 */
export function poolVegaHidden({
  h13, h19, spans, d,
  chunks: C = VEGA_DEFAULTS.memoryChunks,
  chunkMinTokens = VEGA_DEFAULTS.memoryMinTokens,
  chunkTokens = VEGA_DEFAULTS.memoryChunkTokens,
  withChunks = true,
}) {
  const L = Math.floor(h13.length / d);
  const mean = (h, a, e, out, off) => {
    if (e <= a) return;
    for (let t = a; t < e; t++) {
      for (let j = 0; j < d; j++) out[off + j] += h[t * d + j];
    }
    for (let j = 0; j < d; j++) out[off + j] /= e - a;
  };
  const feats = new Float32Array(2 * 3 * d);
  [h13, h19].forEach((h, l) => {
    mean(h, spans.state[0], spans.state[1], feats, (l * 3 + 0) * d);
    mean(h, spans.question[0], spans.question[1], feats, (l * 3 + 1) * d);
    feats.set(h.subarray((L - 1) * d, L * d), (l * 3 + 2) * d);
  });
  const opts = new Float32Array(spans.options.length * d);
  spans.options.forEach(([a, e], k) => mean(h19, a, e, opts, k * d));
  const chunks = new Float32Array(C * d);
  const cmask = new Uint8Array(C);
  const [s0, s1] = spans.state;
  const m = s1 - s0;
  if (withChunks && m >= Math.max(chunkMinTokens, 2)) {
    const k = Math.max(2, Math.min(C, Math.floor(m / Math.max(1, chunkTokens))));
    // token t goes to chunk floor(t*k/m): chunk c covers [ceil(c*m/k), ceil((c+1)*m/k))
    for (let c = 0; c < k; c++) {
      const a = s0 + Math.ceil((c * m) / k);
      const e = s0 + (c === k - 1 ? m : Math.ceil(((c + 1) * m) / k));
      mean(h19, a, e, chunks, c * d);
    }
    cmask.fill(1, 0, k);
  }
  return { feats, opts, chunks, cmask };
}

// ---------------------------------------------------------------------------
// The decider
// ---------------------------------------------------------------------------

/**
 * Session opener shared by both Vega formats: graph bytes (plus an external
 * data sibling when the graph has one) go through the injected fetch and the
 * Cache API, then ORT creates the session; a WebGPU create failure retries on
 * wasm. Hosts that load weights differently (local files, bundled, offline
 * mirrors) replace the whole step with sessionFactory(url, opts).
 */
function openVegaSessions({
  ortMod, fetchImpl, onProgress, sessionFactory, executionProviders, device,
}) {
  const create = sessionFactory
    ?? (async (url, opts) => {
      const bytes = await fetchCachedBytes(url, {
        fetchImpl, onProgress: (p) => onProgress?.(p), persist: true,
      });
      const external = await fetchOnnxExternalData(url, { fetchImpl });
      const sessionOpts = external ? { ...opts, externalData: [external] } : opts;
      return ortMod.InferenceSession.create(bytes, sessionOpts);
    });
  return async (url) => {
    try {
      return await create(url, { executionProviders, graphOptimizationLevel: "all" });
    } catch (e) {
      if (device !== "webgpu") throw e;
      onProgress?.({ phase: `webgpu unavailable (${e?.message ?? e}) — retrying on wasm` });
      return create(url, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    }
  };
}

/**
 * Create a Vega decider.
 *
 * @param {object} [opts]
 *  variant            — "fp16" (default, this repo's fused export) or "q8"
 *                        (nvkudva/vega-onnx-q8: 8-bit split embed + decoder +
 *                        fp32 engine with adapters, wasm-only). Sets
 *                        model/revision/subfolder/format; explicit
 *                        model/revision/format still win.
 *  format             — "fused" | "split" (normally derived from the variant)
 *  model/revision/subfolder — export repo and, for split exports, the
 *                        version folder that holds the graphs (default "v2")
 *  torsoFile/engineFile — fused tograph paths within the repo
 *  embedFile/decoderFile — split graph names within the subfolder
 *  calibrationModel   — repo to read the config from (default: model;
 *                        point it at the source checkpoint for your own export)
 *  maxLen             — prompt window; null = checkpoint's max_length
 *  alpha / risk       — conformal risk level and abstain risk
 *                        (default: the calibration block's default_risk)
 *  device, wasmPaths, sessionFactory, onProgress — as in the other families
 */
export async function createVegaDecider(opts = {}) {
  const variant = opts.variant ?? null;
  const preset = variant === null ? null : VEGA_VARIANTS[variant];
  if (variant !== null && !preset) {
    throw new TypeError(
      `vega: unknown variant "${variant}" (${Object.keys(VEGA_VARIANTS).join(", ")})`);
  }
  const format = opts.format ?? preset?.format ?? "fused";
  if (format !== "fused" && format !== "split") {
    throw new TypeError(`vega: format must be "fused" or "split" (got ${JSON.stringify(format)})`);
  }
  // Like strands: a named variant replaces the *default* model/revision, so
  // createDecisionRuntime({family: "vega", variant: "q8"}) works even though
  // the registry passes the default target through; a deliberately different
  // model/revision still wins.
  const isDefaultTarget = (opts.model ?? VEGA_DEFAULT_MODEL) === VEGA_DEFAULT_MODEL
    && (opts.revision ?? VEGA_DEFAULT_REVISION) === VEGA_DEFAULT_REVISION;
  const usePreset = preset && isDefaultTarget;
  const resolved = {
    model: usePreset ? preset.model : (opts.model ?? VEGA_DEFAULT_MODEL),
    revision: usePreset ? preset.revision : (opts.revision ?? VEGA_DEFAULT_REVISION),
    subfolder: opts.subfolder ?? preset?.subfolder ?? null,
  };
  return format === "split"
    ? createVegaSplitDecider(opts, resolved)
    : createVegaFusedDecider({ ...opts, ...resolved });
}

/** The fused export (one torso graph + the base engine), the default. */
async function createVegaFusedDecider({
  model = VEGA_DEFAULT_MODEL,
  revision = VEGA_DEFAULT_REVISION,
  torsoFile = VEGA_TORSO_FILE,
  engineFile = VEGA_ENGINE_FILE,
  calibrationModel = null,
  calibrationRevision = null,
  ort = null,
  transformers = null,
  fetchImpl = fetch,
  wasmPaths = null,
  device = "auto",
  sessionFactory = null,
  onProgress = null,
  maxLen = null,
  alpha = VEGA_DEFAULTS.alpha,
  risk = null,
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

  // Calibration + architecture constants come from the checkpoint's
  // vega_config.json (thermal theta, conformal thresholds, chunking, window).
  const configUrl = hfUrl(calibrationModel ?? model,
    calibrationRevision ?? (calibrationModel ? "main" : revision),
    "vega_config.json");
  let config = null;
  let configSource = null;
  try {
    // Cached like every other config; missing file, transport failure or
    // garbage JSON all fall through to the shipped defaults.
    config = await fetchCachedJson(configUrl, { fetchImpl, nullOn404: true });
    if (config != null) configSource = configUrl;
  } catch { /* fall through to the shipped defaults */ }
  const cfg = { ...VEGA_DEFAULTS, ...(config ?? {}) };
  const calibration = config?.calibration ?? {};
  const effRisk = risk ?? calibration.default_risk ?? cfg.default_risk ?? VEGA_DEFAULTS.risk;
  const cfgMaxLen = config?.max_len ?? cfg.maxLen;
  const effMaxLen = Number.isFinite(maxLen) && maxLen > 0 ? maxLen : cfgMaxLen;

  const tok = await tf.AutoTokenizer.from_pretrained(calibrationModel ?? model, {
    revision: calibrationRevision ?? (calibrationModel ? "main" : revision),
    progress_callback: () => {},
  });
  const encode = (text, addSpecial = false) =>
    Array.from(tok(text, { add_special_tokens: addSpecial }).input_ids.data, Number);

  const open = openVegaSessions({
    ortMod, fetchImpl, onProgress, sessionFactory,
    executionProviders, device: resolvedDevice,
  });
  const torso = await open(hfUrl(model, revision, torsoFile));
  const engine = await open(hfUrl(model, revision, engineFile));

  const int64 = (arr, dims) => new ortMod.Tensor("int64", arr, dims);
  const f32 = (arr, dims) => new ortMod.Tensor("float32", arr, dims);
  const bool = (arr, dims) => new ortMod.Tensor("bool", arr, dims);

  const info = {
    family: "vega",
    format: "fused",
    model,
    revision,
    device: resolvedDevice,
    maxLen: effMaxLen,
    alpha: String(alpha),
    risk: String(effRisk),
    calibrationSource: configSource,
    nSteps: cfg.n_steps ?? cfg.nSteps,
    config,
  };

  // One session pair, one `decide()` at a time (a shared session is not safe
  // for interleaved runs).
  const queue = createSerialQueue();

  const decideImpl = async (state, questions) => {
    const t0 = now();
    const normalized = normalizeQuestions(questions);
    const qdefs = normalized.map(vegaQuestionDef);
    const stateText = renderVegaText(state);
    const bosId = tok.bos_token_id ?? null;

    const items = qdefs.map((qdef) => {
      const { ids, spans, truncated } = buildVegaPrompt({
        encode, stateText, qdef, maxLen: effMaxLen,
        bosId,
        queryFirstMaxState: cfg.query_first_max_state ?? cfg.queryFirstMaxState,
        optMaxTokens: cfg.opt_max_tokens ?? cfg.optMaxTokens,
        insMaxTokens: 256,
      });
      return { ids, spans, qtype: VEGA_QTYPE[qdef.t], options: vegaOptionTexts(qdef), truncated };
    });

    const batch = packVegaBatch(items, {
      chunks: cfg.memory_chunks ?? cfg.memoryChunks,
    });
    const chunkMask = packVegaChunks(batch.pool, items, batch.L, {
      chunks: cfg.memory_chunks ?? cfg.memoryChunks,
      chunkMinTokens: cfg.memory_min_tokens ?? cfg.memoryMinTokens,
      chunkTokens: cfg.memory_chunk_tokens ?? cfg.memoryChunkTokens,
      nseg: batch.nseg,
    });

    const tRun = now();
    // Torso: one causal read to the deepest captured layer, all span means
    // (segments + state chunks) pooled in-graph from the one forward.
    const torsoOut = await torso.run({
      input_ids: int64(batch.inputIds, [batch.n, batch.L]),
      attention_mask: int64(batch.attention, [batch.n, batch.L]),
      pool: f32(batch.pool, batch.poolDims),
    });
    const pooled = torsoOut.pooled ?? torsoOut[Object.keys(torsoOut)[0]];
    // pooled: [B, nLayers, nseg + C, d] — nLayers = feature_layers,
    // rows 0..2 = state/question/last, 3..3+K = options, then C chunk rows.
    const pd = pooled.dims;
    const d = pd[3];
    const Lc = pd[1];
    const rowsTotal = pd[2];
    const C = rowsTotal - batch.nseg;
    const pv = pooled.data;
    const feats = new Float32Array(batch.n * Lc * 3 * d);
    const opts = new Float32Array(batch.n * batch.K * d);
    const chunks = new Float32Array(batch.n * C * d);
    for (let b = 0; b < batch.n; b++) {
      for (let l = 0; l < Lc; l++) {
        for (let s = 0; s < 3; s++) {
          const row = s; // pooling rows 0/1/2 are state/question/last
          const src = ((b * Lc + l) * rowsTotal + row) * d;
          feats.set(pv.subarray(src, src + d), ((b * Lc + l) * 3 + s) * d);
        }
      }
      // options from the deepest captured layer only (option_layers)
      for (let k = 0; k < batch.K; k++) {
        const src = ((b * Lc + Lc - 1) * rowsTotal + 3 + k) * d;
        opts.set(pv.subarray(src, src + d), (b * batch.K + k) * d);
      }
      // chunk means: deepest feature layer, rows nseg..nseg + C
      for (let c = 0; c < C; c++) {
        const src = ((b * Lc + Lc - 1) * rowsTotal + batch.nseg + c) * d;
        chunks.set(pv.subarray(src, src + d), (b * C + c) * d);
      }
    }

    // Engine: landscape + integrator -> raw logits and the physics state.
    const engineOut = await engine.run({
      feats: f32(feats, [batch.n, Lc, 3, d]),
      opts: f32(opts, [batch.n, batch.K, d]),
      opt_mask: bool(batch.optMask, [batch.n, batch.K]),
      qtype: int64(batch.qtypes, [batch.n]),
      chunks: f32(chunks, [batch.n, C, d]),
      chunk_mask: bool(chunkMask, [batch.n, C]),
    });
    const logits = engineOut.logits.data;
    const kinetic = engineOut.kinetic.data;
    const nearest = engineOut.nearest.data;
    const stepsUsed = Array.from(engineOut.steps_used.data, Number);

    const nOpts = items.map((it) => it.options.length);
    const phys = vegaThermalFeatures({
      kinetic, nearest, stepsUsed, nOpts, maxSteps: cfg.n_steps ?? cfg.nSteps,
    });
    const probs = vegaThermalProbabilities({
      logits, rowWidth: batch.K, qtypes: batch.qtypes,
      phys: phys.flat(), nOpts,
      theta: calibration.theta ?? VEGA_DEFAULT_THETA,
    });
    const runMs = now() - tRun;
    const labels = vegaOptionLabels(normalized);
    const answers = vegaAnswers({
      probs, qtypes: batch.qtypes, labels,
      physics: { nearest, stepsUsed },
      rawLogits: logits, rowWidth: batch.K,
      calibration, alpha, risk: effRisk,
    });
    const decode = (ids) => {
      try {
        return tok.decode(Array.from(ids), { skip_special_tokens: false });
      } catch {
        return Array.from(ids).join(" ");
      }
    };
    return {
      answers,
      truncated: items.some((it) => it.truncated),
      length: Number(batch.attention.reduce((a, v) => a + v, 0n)),
      prompts: items.map((it) => decode(it.ids)),
      timings: { totalMs: Math.round(now() - t0), runMs: Math.round(runMs) },
    };
  };

  return {
    info,
    decide: (state, questions) => queue(() => decideImpl(state, questions)),
    async dispose() {
      try { await torso.release?.(); } catch { /* optional */ }
      try { await engine.release?.(); } catch { /* optional */ }
    },
  };
}

/**
 * The split export (nvkudva/vega-onnx-q8): an 8-bit embedding graph, a cut
 * decoder that stops after layer 19 and emits the residual stream at layers
 * 13/19 (the LM head and layers 20-23 never run), and the fp32 engine with
 * both gated task adapters. JS pools the hidden states, each question is
 * calibrated with the theta block of the adapter its gate routes it to, and
 * the conformal / abstain / unbound machinery reads that block's thresholds.
 *
 * Text only in v1: the vision encoder (image as the state) is not wired in.
 * The 8-bit graphs are wasm-only (MatMulNBits has no WebGPU kernel), so
 * device "auto" resolves to wasm.
 */
async function createVegaSplitDecider({
  embedFile = VEGA_Q8_EMBED_FILE,
  decoderFile = VEGA_Q8_DECODER_FILE,
  engineFile = VEGA_Q8_ENGINE_FILE,
  calibrationModel = null,
  calibrationRevision = null,
  ort = null,
  transformers = null,
  fetchImpl = fetch,
  wasmPaths = null,
  device = "auto",
  sessionFactory = null,
  onProgress = null,
  maxLen = null,
  alpha = VEGA_DEFAULTS.alpha,
  risk = null,
} = {}, { model, revision, subfolder }) {
  const ortMod = ort ?? (await import("onnxruntime-web"));
  const tf = transformers ?? (await import("@huggingface/transformers"));
  if (wasmPaths) {
    try { ortMod.env.wasm.wasmPaths = wasmPaths; } catch { /* older runtime */ }
  }
  const resolvedDevice = device !== "auto" ? device : "wasm";
  const executionProviders = resolvedDevice === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];

  const basePath = subfolder ? `${subfolder}/` : "";
  const configRepo = calibrationModel ?? model;
  const configRevision = calibrationRevision ?? (calibrationModel ? "main" : revision);
  const configUrl = hfUrl(configRepo, configRevision, `${basePath}vega_web.json`);
  let config = null;
  let configSource = null;
  try {
    // Cached like every other config; missing file, transport failure or
    // garbage JSON all fall through to the shipped defaults.
    config = await fetchCachedJson(configUrl, { fetchImpl, nullOn404: true });
    if (config != null) configSource = configUrl;
  } catch { /* fall through to the shipped defaults */ }
  const cfg = { ...VEGA_DEFAULTS, ...(config ?? {}) };
  const calibration = config?.calibration ?? {};
  const adapters = Array.isArray(config?.adapters) ? config.adapters : [];
  const effRisk = risk ?? calibration.default_risk ?? cfg.default_risk ?? VEGA_DEFAULTS.risk;
  const cfgMaxLen = config?.max_len ?? cfg.maxLen;
  const effMaxLen = Number.isFinite(maxLen) && maxLen > 0 ? maxLen : cfgMaxLen;

  // The tokenizer (Qwen3.5) ships inside the version folder, and
  // transformers.js's AutoTokenizer has no subfolder option here, so build
  // PreTrainedTokenizer from the two files directly -- what the export's own
  // runtime does. Older transformers falls back to AutoTokenizer.
  const fetchJson = async (file) => {
    const url = hfUrl(configRepo, configRevision, `${basePath}${file}`);
    try {
      return await fetchCachedJson(url, { fetchImpl });
    } catch (e) {
      if (e instanceof HttpError) {
        throw new Error(
          `vega: ${url} not found (HTTP ${e.status ?? "no response"})`,
          { cause: e },
        );
      }
      throw e;
    }
  };
  let tok;
  if (typeof tf.PreTrainedTokenizer === "function") {
    const [tokenizerJson, tokenizerConfig] = await Promise.all([
      fetchJson("tokenizer.json"), fetchJson("tokenizer_config.json"),
    ]);
    tok = new tf.PreTrainedTokenizer(tokenizerJson, tokenizerConfig);
  } else {
    tok = await tf.AutoTokenizer.from_pretrained(configRepo, { revision: configRevision });
  }
  const encode = (text, addSpecial = false) => {
    if (typeof tok.encode === "function") {
      const ids = tok.encode(text, { add_special_tokens: addSpecial });
      return Array.isArray(ids) ? ids : Array.from(ids?.input_ids?.data ?? ids, Number);
    }
    return Array.from(tok(text, { add_special_tokens: addSpecial }).input_ids.data, Number);
  };
  const decode = (ids) => {
    try {
      return tok.decode(Array.from(ids), { skip_special_tokens: false });
    } catch {
      return Array.from(ids).join(" ");
    }
  };

  const open = openVegaSessions({
    ortMod, fetchImpl, onProgress, sessionFactory, executionProviders,
    device: resolvedDevice,
  });
  const embed = await open(hfUrl(model, revision, `${basePath}${embedFile}`));
  const decoder = await open(hfUrl(model, revision, `${basePath}${decoderFile}`));
  const engine = await open(hfUrl(model, revision, `${basePath}${engineFile}`));
  const past = emptyVegaPast(decoder, ortMod.Tensor);

  const int64 = (arr, dims) => new ortMod.Tensor("int64", arr, dims);
  const f32 = (arr, dims) => new ortMod.Tensor("float32", arr, dims);
  const bool = (arr, dims) => new ortMod.Tensor("bool", arr, dims);

  const info = {
    family: "vega",
    format: "split",
    model,
    revision,
    subfolder: subfolder ?? null,
    device: resolvedDevice,
    maxLen: effMaxLen,
    alpha: String(alpha),
    risk: String(effRisk),
    calibrationSource: configSource,
    nSteps: cfg.n_steps ?? cfg.nSteps,
    adapters: adapters.map((a) => a.name),
    config,
  };

  // One session set, one `decide()` at a time (a shared session is not safe
  // for interleaved runs).
  const queue = createSerialQueue();

  const decideImpl = async (state, questions) => {
    const t0 = now();
    const normalized = normalizeQuestions(questions);
    const qdefs = normalized.map(vegaQuestionDef);
    const stateText = renderVegaText(state);
    const C = Number(cfg.memory_chunks ?? VEGA_DEFAULTS.memoryChunks);
    const chunkMinTokens = cfg.memory_min_tokens ?? VEGA_DEFAULTS.memoryMinTokens;
    const chunkTokens = cfg.memory_chunk_tokens ?? VEGA_DEFAULTS.memoryChunkTokens;

    const items = qdefs.map((qdef) => {
      const { ids, spans, truncated } = buildVegaPrompt({
        encode, stateText, qdef, maxLen: effMaxLen,
        bosId: null, // the reference adds no BOS for Qwen, and the export never does
        queryFirstMaxState: cfg.query_first_max_state ?? cfg.queryFirstMaxState,
        optMaxTokens: cfg.opt_max_tokens ?? cfg.optMaxTokens,
        insMaxTokens: 256,
        truncateState: false, // the export refuses a state that does not fit
      });
      return { ids, spans, qtype: VEGA_QTYPE[qdef.t], options: vegaOptionTexts(qdef), truncated };
    });

    // One decoder pass per question (the export's own reading), pooling in JS.
    const tRun = now();
    const rows = [];
    const prompts = [];
    for (const it of items) {
      const L = it.ids.length;
      const embedOut = await embed.run({
        input_ids: int64(BigInt64Array.from(it.ids, BigInt), [1, L]),
      });
      const embeds = embedOut.inputs_embeds.data;
      const d = embeds.length / L;
      const decOut = await decoder.run({
        ...past,
        inputs_embeds: f32(embeds, [1, L, d]),
        attention_mask: int64(new BigInt64Array(L).fill(1n), [1, L]),
        position_ids: int64(vegaTextPositions(L), [3, 1, L]),
      });
      const pooled = poolVegaHidden({
        h13: decOut.h13.data, h19: decOut.h19.data, spans: it.spans, d,
        chunks: C, chunkMinTokens, chunkTokens,
      });
      const K = it.options.length;
      const eng = await engine.run({
        feats: f32(pooled.feats, [1, 2, 3, d]),
        opts: f32(pooled.opts, [1, K, 1, d]),
        opt_mask: bool(new Uint8Array(K).fill(1), [1, K]),
        qtype: int64([BigInt(it.qtype)], [1]),
        chunks: f32(pooled.chunks, [1, C, d]),
        chunk_mask: bool(pooled.cmask, [1, C]),
      });
      rows.push({
        qtype: it.qtype,
        K,
        logits: Array.from(eng.logits.data).slice(0, K),
        kinetic: eng.kinetic.data[0],
        nearest: eng.nearest.data[0],
        stepsUsed: Number(eng.steps_used.data[0]),
        gates: Array.from(eng.gates.data),
      });
      prompts.push(decode(it.ids));
    }

    const nOpts = rows.map((r) => r.K);
    const rowWidth = Math.max(...nOpts);
    const logits = new Float32Array(rows.length * rowWidth);
    rows.forEach((r, b) => logits.set(r.logits, b * rowWidth));
    const phys = vegaThermalFeatures({
      kinetic: rows.map((r) => r.kinetic),
      nearest: rows.map((r) => r.nearest),
      stepsUsed: rows.map((r) => r.stepsUsed),
      nOpts, maxSteps: cfg.n_steps ?? cfg.nSteps,
    });
    // Each question is calibrated by the adapter its gate routes it to
    // (route_adapters: the most open gate, if it reaches the threshold).
    const gateThreshold = adapters[0]?.gate_threshold ?? 0.5;
    const gates = rows.map((r) => r.gates);
    const routes = gates.map((g) => vegaRoute(g, gateThreshold));
    const calibrations = routes.map(
      (rt) => (rt >= 0 ? adapters[rt]?.calibration : null) ?? calibration);
    const thetaByRow = calibrations.map(
      (cal, b) => cal?.theta?.[rows[b].qtype] ?? VEGA_DEFAULT_THETA[rows[b].qtype]);
    const probs = vegaThermalProbabilities({
      logits, rowWidth, qtypes: rows.map((r) => r.qtype),
      phys: phys.flat(), nOpts, thetaByRow,
    });
    const runMs = now() - tRun;
    const answers = vegaAnswers({
      probs, qtypes: rows.map((r) => r.qtype), labels: vegaOptionLabels(normalized),
      physics: {
        nearest: rows.map((r) => r.nearest),
        stepsUsed: rows.map((r) => r.stepsUsed),
      },
      rawLogits: logits, rowWidth,
      calibration, calibrations, alpha, risk: effRisk,
      adapterNames: adapters.map((a) => a.name), gates, routes,
    });
    return {
      answers,
      truncated: items.some((it) => it.truncated),
      length: items.reduce((n, it) => n + it.ids.length, 0),
      prompts,
      timings: { totalMs: Math.round(now() - t0), runMs: Math.round(runMs) },
    };
  };

  return {
    info,
    decide: (state, questions) => queue(() => decideImpl(state, questions)),
    async dispose() {
      try { await embed.release?.(); } catch { /* optional */ }
      try { await decoder.release?.(); } catch { /* optional */ }
      try { await engine.release?.(); } catch { /* optional */ }
    },
  };
}
