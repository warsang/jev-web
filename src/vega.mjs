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
 * Two ONNX graphs (one fused torso -> pooled features, one engine -> logits +
 * physics), mirroring the reference runtime exactly; JS owns the protocol, the
 * packing, the thermal calibration and the answer decoding.
 */
import { normalizeQuestions } from "./questions.mjs";
import { softmaxWithTemperature } from "./answers.mjs";
import { fetchCachedBytes, fetchOnnxExternalData } from "./laya.mjs";

export const VEGA_DEFAULT_MODEL = "warsang/vega-08b-web";
// null = the branch default (main) until the browser export repo lands;
// export/vega/05_upload.py prints the sha to pin here.
export const VEGA_DEFAULT_REVISION = null;
// The exact fp16 torso ships by default: q8 on this graph drifts on the
// GatedDeltaNet delta rule (export/vega/04_repo.py measures ~0.46 in the
// pooled features), so torso_quantized.onnx stays an opt-in experiment.
export const VEGA_TORSO_FILE = "onnx/torso_fp16.onnx";
export const VEGA_ENGINE_FILE = "onnx/engine.onnx";

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
  const ins = renderVegaText(question.instructions);
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

function flatText(v) {
  return v === null || v === undefined ? "" : String(v);
}

export function renderVegaText(content) {
  return typeof content === "string" ? content : JSON.stringify(content);
}

/** Option texts in label-index order, exactly as option_texts() renders them. */
export function vegaOptionTexts(qdef) {
  if (qdef.t === "choice") {
    return Object.entries(qdef.crit).map(([k, v]) => {
      const d = v === null || v === undefined ? "" : flatText(v);
      return d ? `${k} - ${d}` : String(k);
    });
  }
  if (qdef.t === "score") {
    return qdef.crit.map((c, i) => `level ${i} - ${flatText(c)}`);
  }
  const crit = qdef.crit ?? {};
  const no = crit.false === null || crit.false === undefined ? "" : flatText(crit.false);
  const yes = crit.true === null || crit.true === undefined ? "" : flatText(crit.true);
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
 * @returns {{ids:number[], spans:{state:[number,number], question:[number,number],
 *   options:[number,number][], last:number}}}
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
}) {
  const options = vegaOptionTexts(qdef);
  const enc = (s) => encode(s, false);
  const bos = bosId === null ? [] : [bosId];
  let head = enc(VEGA_HEAD);
  let qids = enc(VEGA_QUESTION.replace("{t}", qdef.t).replace("{ins}", qdef.ins))
    .slice(0, insMaxTokens);
  let oids = options.map((o) => enc(VEGA_OPTION.replace("{o}", o)).slice(0, optMaxTokens));
  const end = enc(VEGA_END);
  let fixed = bos.length + head.length + qids.length
    + oids.reduce((n, o) => n + o.length, 0) + end.length;
  if (fixed > maxLen - 16) {
    const per = Math.max(3, Math.floor(
      (maxLen - 16 - bos.length - head.length - end.length - Math.min(qids.length, 64))
      / Math.max(1, oids.length)));
    oids = oids.map((o) => o.slice(0, per));
    qids = qids.slice(0, Math.max(
      8, maxLen - 16 - bos.length - head.length - end.length
        - oids.reduce((n, o) => n + o.length, 0)));
    fixed = bos.length + head.length + qids.length
      + oids.reduce((n, o) => n + o.length, 0) + end.length;
  }
  let st = enc(stateText);
  let pre = [];
  if (queryFirstMaxState && st.length < queryFirstMaxState) {
    pre = enc(VEGA_QUERY_FIRST.replace("{t}", qdef.t).replace("{ins}", qdef.ins))
      .slice(0, Math.min(128, insMaxTokens));
    fixed += pre.length;
  }
  const room = Math.max(0, maxLen - fixed);
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
  return { ids, spans };
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
 * @param {{theta:number[][]}} calibration
 * @param {{logits:ArrayLike<number>, rowWidth:number, qtypes:number[],
 *   phys:ArrayLike<number>, nOpts:number[]}} spec phys is [items, 4]
 * @returns {number[][]} probabilities per item, in prompt order
 */
export function vegaThermalProbabilities({
  logits, rowWidth, qtypes, phys, nOpts,
  theta = VEGA_DEFAULT_THETA,
}) {
  const rows = Number(logits.length / rowWidth);
  const out = [];
  for (let b = 0; b < rows; b++) {
    const k = nOpts[b];
    const t = theta[qtypes[b]] ?? theta[0];
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
 * predictionSet / abstain / unbound / settleSteps / energy).
 *
 * @param {{probs:number[][], qtypes:number[], labels:string[][],
 *   physics:{nearest:ArrayLike<number>, stepsUsed:ArrayLike<number>},
 *   rawLogits:ArrayLike<number>, rowWidth:number,
 *   calibration:object, alpha?:string, risk?:string}} spec
 */
export function vegaAnswers({
  probs, qtypes, physics, rawLogits, rowWidth, calibration, questions, labels,
  alpha = VEGA_DEFAULTS.alpha, risk = VEGA_DEFAULTS.risk,
}) {
  const answers = [];
  labels.forEach((optionLabels, r) => {
    const p = probs[r];
    const k = optionLabels.length;
    const qtype = VEGA_QTYPE_NAMES[qtypes[r]];
    const conf = Math.max(...p);
    const best = p.indexOf(Math.max(...p));
    // LAC prediction set; the top option is always kept so it is never empty.
    const th = calibration?.conformal?.[qtype]?.[String(alpha)];
    const predictionSet = th === undefined ? null
      : optionLabels.filter((_, i) => i === best || p[i] >= Math.max(1 - th, 1e-12));
    // energy = -max raw logit over the question's real options
    let bestLogit = -Infinity;
    for (let j = 0; j < k; j++) bestLogit = Math.max(bestLogit, rawLogits[r * rowWidth + j]);
    const a = {
      probabilities: Object.fromEntries(optionLabels.map((l, i) => [l, p[i]])),
      confidence: conf,
      predictionSet,
      abstain: conf < (calibration?.abstain_threshold?.[String(risk)] ?? 0),
      unbound: physics.nearest[r] > (calibration?.unbound_threshold ?? Infinity),
      settleSteps: physics.stepsUsed[r],
      energy: -bestLogit,
    };
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
// The decider
// ---------------------------------------------------------------------------

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

const hfUrl = (model, revision, file) =>
  `https://huggingface.co/${model}/resolve/${revision ? `${revision}/` : ""}${file}`;

/**
 * Create a Vega decider.
 *
 * @param {object} [opts]
 *  model/revision     — browser export repo (torso + engine graphs, tokenizer,
 *                        vega_config.json with the calibration block)
 *  torsoFile/engineFile — graph paths within the repo
 *  calibrationModel   — repo to read vega_config.json from (default: model;
 *                        point it at the source checkpoint for your own export)
 *  maxLen             — prompt window; null = checkpoint's max_length
 *  alpha / risk       — conformal risk level and abstain risk ("0.1" default)
 *  device, wasmPaths, sessionFactory, onProgress — as in the other families
 */
export async function createVegaDecider({
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
  risk = VEGA_DEFAULTS.risk,
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
    const res = await fetchImpl(configUrl);
    if (res?.ok) {
      config = await res.json();
      configSource = configUrl;
    }
  } catch { /* fall through to the shipped defaults */ }
  const cfg = { ...VEGA_DEFAULTS, ...(config ?? {}) };
  const calibration = config?.calibration ?? {};
  const cfgMaxLen = config?.max_len ?? cfg.maxLen;
  const effMaxLen = Number.isFinite(maxLen) && maxLen > 0 ? maxLen : cfgMaxLen;

  const tok = await tf.AutoTokenizer.from_pretrained(calibrationModel ?? model, {
    revision: calibrationRevision ?? (calibrationModel ? "main" : revision),
    progress_callback: () => {},
  });
  const encode = (text, addSpecial = false) =>
    Array.from(tok(text, { add_special_tokens: addSpecial }).input_ids.data, Number);

  const createSession = sessionFactory
    ?? (async (u, opts) => {
      const bytes = await fetchCachedBytes(u, {
        fetchImpl, onProgress: (p) => onProgress?.(p), persist: true,
      });
      const external = await fetchOnnxExternalData(u, { fetchImpl });
      const sessionOpts = external ? { ...opts, externalData: [external] } : opts;
      return ortMod.InferenceSession.create(bytes, sessionOpts);
    });
  const makeSession = async (file) => {
    try {
      return await createSession(hfUrl(model, revision, file), {
        executionProviders, graphOptimizationLevel: "all",
      });
    } catch (e) {
      if (resolvedDevice !== "webgpu") throw e;
      onProgress?.({ phase: `webgpu unavailable (${e?.message ?? e}) — retrying on wasm` });
      return createSession(hfUrl(model, revision, file),
        { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    }
  };
  const torso = await makeSession(torsoFile);
  const engine = await makeSession(engineFile);

  const int64 = (arr, dims) => new ortMod.Tensor("int64", arr, dims);
  const f32 = (arr, dims) => new ortMod.Tensor("float32", arr, dims);
  const bool = (arr, dims) => new ortMod.Tensor("bool", arr, dims);

  return {
    info: {
      family: "vega",
      model,
      revision,
      device: resolvedDevice,
      maxLen: effMaxLen,
      alpha: String(alpha),
      risk: String(risk ?? cfg.default_risk ?? VEGA_DEFAULTS.risk),
      calibrationSource: configSource,
      nSteps: cfg.n_steps ?? cfg.nSteps,
      config,
    },
    async decide(state, questions) {
      const t0 = now();
      const normalized = normalizeQuestions(questions);
      const qdefs = normalized.map(vegaQuestionDef);
      const stateText = renderVegaText(state);
      const bosId = tok.bos_token_id ?? null;

      const items = qdefs.map((qdef) => {
        const { ids, spans } = buildVegaPrompt({
          encode, stateText, qdef, maxLen: effMaxLen,
          bosId,
          queryFirstMaxState: cfg.query_first_max_state ?? cfg.queryFirstMaxState,
          optMaxTokens: cfg.opt_max_tokens ?? cfg.optMaxTokens,
          insMaxTokens: 256,
        });
        return { ids, spans, qtype: VEGA_QTYPE[qdef.t], options: vegaOptionTexts(qdef) };
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
        calibration, alpha, risk,
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
        truncated: false,
        length: Number(batch.attention.reduce((a, v) => a + v, 0n)),
        prompts: items.map((it) => decode(it.ids)),
        timings: { totalMs: Math.round(now() - t0), runMs: Math.round(runMs) },
      };
    },
    async dispose() {
      try { await torso.release?.(); } catch { /* optional */ }
      try { await engine.release?.(); } catch { /* optional */ }
    },
  };
}
