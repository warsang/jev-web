/**
 * clef.mjs — FluidInference clef-text-0.6b family (Qwen3 torso + Cloudflare's
 * joint schema head).
 *
 * clef-text-0.6b is a 0.70B text decision model distilled from Cloudflare's
 * clef-flash 9B: a Qwen3-0.6B torso (LoRA merged) plus Clef's joint schema
 * head — one pass that decides every field of a record jointly. The
 * checkpoint ships Cloudflare's unchanged reference module
 * (`joint_schema_model.py`, Apache-2.0) and speaks the same Jev/SystemOne
 * contract as every other family here: one probability per option of every
 * typed question, one forward pass.
 *
 *   record (built by encodeClefRecord — mirrors encode_record):
 *     <|im_start|>system\nRead the complete state and schema. Decide every field
 *     jointly. Each answer must be exactly one of that field's allowed options.
 *     <|im_end|>\n<|im_start|>user\nSTATE:\n<state>
 *     \n\nSCHEMA FIELDS:\n
 *     \nFIELD 1\nID: <id>\nTYPE: <type>\nINSTRUCTION: <instructions>
 *     \nALLOWED OPTIONS:\nOPTION 1: {"description": …, "option_id": "…"}\n…END FIELD\n
 *     \n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
 *     JOINT SCHEMA DECISIONS:
 *   head: span means of the normalized hidden states (question + option
 *     spans), span means of the tied output embedding (the option lexical
 *     prior), a cross-option evidence pass over the record's options
 *     attending to the whole sequence, per-question routing summaries, and a
 *     transformer-decoder field pass — logits per option of every question.
 *   answers: softmax per question; choice/score/noul with the reference
 *     SystemOne shapes (choice/score confidence is the top probability, score
 *     carries its legend).
 *
 * One fused ONNX graph ships: the head's evidence and field passes attend to
 * the full hidden sequence, so the torso never leaves the graph. Span means
 * travel as pooling matrices (the Vega trick), so no hidden states cross into
 * JS.
 */
import { normalizeQuestions } from "./questions.mjs";
import { softmaxWithTemperature } from "./answers.mjs";
import { now, hfUrl, argmax, createSerialQueue } from "./util.mjs";
import { fetchCachedJson, fetchCachedBytes, fetchOnnxExternalData } from "./fetch.mjs";

export const CLEF_DEFAULT_MODEL = "warsang/clef-text-0.6b-web";
// Pinned to the uploaded browser export (export/clef/04_upload.py).
export const CLEF_DEFAULT_REVISION = "d01e9cfd11e66d593e9c517bb2bc204ffbf67623";
export const CLEF_DEFAULT_FILE = "onnx/clef_text_fp16.onnx";

/** Reference record window (joint_schema_model.encode_record default). */
export const CLEF_DEFAULT_MAX_LEN = 16384;

export const CLEF_SYSTEM_PROMPT =
  "Read the complete state and schema. Decide every field jointly. Each answer "
  + "must be exactly one of that field's allowed options.";

/** Reference QUESTION_TYPES: noul 0, choice 1, score 2. */
export const CLEF_QTYPE = { noul: 0, choice: 1, score: 2 };
export const CLEF_QTYPE_NAMES = ["noul", "choice", "score"];

/** Fixed noul slot descriptions (overridable through question criteria). */
export const CLEF_NOUL_DESCRIPTIONS = {
  true: "The proposition is true or the answer is yes.",
  false: "The proposition is false or the answer is no.",
};

export const CLEF_DEFAULTS = {
  maxLen: CLEF_DEFAULT_MAX_LEN,
  maxStateTokens: null,
};

const CLEF_PREFIX = (system) =>
  `<|im_start|>system\n${system}<|im_end|>\n<|im_start|>user\nSTATE:\n`;
const CLEF_SUFFIX =
  "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:";

// ---------------------------------------------------------------------------
// Protocol: content rendering, question options, record encoding
// ---------------------------------------------------------------------------

/**
 * Reference render(): strings pass through; everything else is compact JSON
 * with sorted keys (json.dumps(ensure_ascii=False, separators=(",", ":"),
 * sort_keys=True)).
 */
export function renderClefContent(content) {
  if (typeof content === "string") return content;
  return JSON.stringify(sortKeysDeep(content));
}

/**
 * Code-point key order — Python's sorted() on str compares code points; the
 * default JS sort compares UTF-16 code units and disagrees past BMP.
 */
function compareCodePoints(a, b) {
  const x = Array.from(a);
  const y = Array.from(b);
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    const cx = x[i].codePointAt(0);
    const cy = y[i].codePointAt(0);
    if (cx !== cy) return cx - cy;
  }
  return x.length - y.length;
}

function sortKeysDeep(value) {
  if (Array.isArray(value)) return value.map(sortKeysDeep);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value)
        .sort(compareCodePoints)
        .map((k) => [k, sortKeysDeep(value[k])]),
    );
  }
  return value;
}

/**
 * Reference question_options(): the option keys and their descriptions, in
 * the order the prompt and the logits use.
 *
 * - noul: fixed keys true/false (default descriptions, criteria overrides
 *   accepted under either {true,false} or {yes,no});
 * - choice: one entry per question option, sorted by key (the reference sorts
 *   the criteria mapping);
 * - score: numeric keys 0..n-1 over the ordered levels.
 *
 * @returns {{keys:string[], descriptions:any[]}}
 */
export function clefQuestionOptions(question) {
  if (question.type === "noul") {
    const c = question.criteria ?? {};
    const trueDesc = c.true ?? c.yes ?? CLEF_NOUL_DESCRIPTIONS.true;
    const falseDesc = c.false ?? c.no ?? CLEF_NOUL_DESCRIPTIONS.false;
    return { keys: ["true", "false"], descriptions: [trueDesc, falseDesc] };
  }
  if (question.type === "choice") {
    const descriptions = question.criteria ?? {};
    const entries = question.options
      .map((label) => [String(label), descriptions[label]]);
    entries.sort((a, b) => compareCodePoints(a[0], b[0]));
    return {
      keys: entries.map(([k]) => k),
      descriptions: entries.map(([, d]) => (d === null || d === undefined ? null : d)),
    };
  }
  if (question.type === "score") {
    return {
      keys: question.options.map((_, i) => String(i)),
      descriptions: question.options,
    };
  }
  throw new TypeError(`clef: unsupported question type ${question.type}`);
}

/**
 * Encode one record exactly like the reference encode_record: every piece is
 * tokenized separately, the state is truncated (head kept) to fit the window,
 * and the question/option token spans are recorded (they become the pooling
 * matrices the graph consumes).
 *
 * @param {{encode:(text:string)=>number[], stateText:string, questions:object[],
 *   maxLength?:number, maxStateTokens?:number|null}} spec
 * @returns {{ids:number[], questions:Array<{id:string, type:number,
 *   questionSpan:[number,number], optionSpans:[number,number][], optionIds:string[]}>,
 *   length:number, truncated:boolean}}
 */
export function encodeClefRecord({
  encode,
  stateText,
  questions,
  maxLength = CLEF_DEFAULT_MAX_LEN,
  maxStateTokens = null,
}) {
  const schema = [];
  const encoded = [];
  questions.forEach((q, i) => {
    const id = q.id ?? String(i);
    const instructions = q.instructions === null || q.instructions === undefined || q.instructions === ""
      ? String(id)
      : q.instructions;
    const { keys, descriptions } = clefQuestionOptions(q);
    schema.push(...encode(`\nFIELD ${i + 1}\nID: ${id}\nTYPE: ${q.type}\nINSTRUCTION: `));
    const qStart = schema.length;
    schema.push(...encode(renderClefContent(instructions)));
    const qEnd = schema.length;
    schema.push(...encode("\nALLOWED OPTIONS:\n"));
    const optionSpans = [];
    keys.forEach((key, j) => {
      schema.push(...encode(`OPTION ${j + 1}: `));
      const oStart = schema.length;
      const semantics = { option_id: key };
      const d = descriptions[j];
      if (d !== null && d !== undefined) semantics.description = d;
      schema.push(...encode(renderClefContent(semantics)));
      optionSpans.push([oStart, schema.length]);
      schema.push(...encode("\n"));
    });
    schema.push(...encode("END FIELD\n"));
    encoded.push({
      id: String(id),
      type: CLEF_QTYPE[q.type],
      instructions,
      optionIds: keys,
      questionSpan: [qStart, qEnd],
      optionSpans,
    });
  });
  if (encoded.length === 0) throw new TypeError("clef: the record has no questions");

  const shortState = "\n\nSCHEMA FIELDS:\n";
  const prefix = encode(CLEF_PREFIX(CLEF_SYSTEM_PROMPT));
  const suffix = encode(CLEF_SUFFIX);
  const schemaHead = encode(shortState);
  const prefixSchema = [...schemaHead, ...schema];

  let stateIds = encode(renderClefContent(stateText ?? ""));
  const fullStateLength = stateIds.length;
  if (maxStateTokens !== null && maxStateTokens !== undefined) {
    stateIds = stateIds.slice(0, maxStateTokens);
  }
  const fixedLength = prefix.length + prefixSchema.length + suffix.length;
  if (fixedLength > maxLength) {
    throw new RangeError(
      `clef: schema requires ${fixedLength} tokens before the state; maximum is ${maxLength}`);
  }
  stateIds = stateIds.slice(0, maxLength - fixedLength);

  const offset = prefix.length + stateIds.length + schemaHead.length;
  const ids = [...prefix, ...stateIds, ...prefixSchema, ...suffix];
  return {
    ids,
    questions: encoded.map((q) => ({
      ...q,
      questionSpan: [q.questionSpan[0] + offset, q.questionSpan[1] + offset],
      optionSpans: q.optionSpans.map(([a, e]) => [a + offset, e + offset]),
    })),
    length: ids.length,
    truncated: stateIds.length < fullStateLength,
  };
}

/**
 * Span means for the graph's pooling matrices.
 *
 * Option slots are question-major: slot = q*K + k. Invalid slots (a question
 * with fewer than K options) keep a zero pooling row and optMask[s] = 0; the
 * graph excludes them from routing and the JS ignores their logits.
 *
 * @returns {{Q:number, K:number, S:number, L:number, poolGlobal:Float32Array,
 *   poolQuestion:Float32Array, poolOption:Float32Array, slotQuestion:number[],
 *   typeIds:number[], optMask:Uint8Array, counts:number[]}}
 */
export function packClefBatch(record) {
  const Q = record.questions.length;
  const K = Math.max(...record.questions.map((q) => q.optionIds.length));
  const S = Q * K;
  const L = record.ids.length;
  const poolGlobal = new Float32Array(L);
  const poolQuestion = new Float32Array(Q * L);
  const poolOption = new Float32Array(S * L);
  const slotQuestion = new Array(S);
  const typeIds = new Array(Q);
  const optMask = new Uint8Array(S);
  const counts = new Array(Q);
  const setSpan = (view, row, [a, e]) => {
    if (e <= a) return;
    const w = 1 / (e - a);
    for (let t = a; t < e; t++) view[row * L + t] = w;
  };
  poolGlobal[L - 1] = 1; // the final token (the record's global vector)
  for (let q = 0; q < Q; q++) {
    const item = record.questions[q];
    typeIds[q] = item.type;
    counts[q] = item.optionIds.length;
    setSpan(poolQuestion, q, item.questionSpan);
    for (let k = 0; k < K; k++) {
      const s = q * K + k;
      slotQuestion[s] = q;
      if (k >= item.optionSpans.length) continue;
      setSpan(poolOption, s, item.optionSpans[k]);
      optMask[s] = 1;
    }
  }
  return {
    Q, K, S, L,
    poolGlobal, poolQuestion, poolOption,
    slotQuestion, typeIds, optMask, counts,
  };
}

/**
 * jev-web answers from per-question option probabilities.
 *
 * @param {{probs:number[][], questions:object[]}} spec probs are in
 *   clefQuestionOptions() key order, sliced to each question's options.
 */
export function clefAnswers({ probs, questions }) {
  return questions.map((q, i) => {
    const p = probs[i];
    if (q.type === "choice") {
      const { keys } = clefQuestionOptions(q);
      const best = argmax(p);
      const choice = keys[best];
      const index = q.options.indexOf(choice);
      return {
        type: "choice",
        choice,
        index: index === -1 ? best : index,
        probabilities: Object.fromEntries(
          q.options.map((label) => [label, p[keys.indexOf(String(label))]])),
        confidence: Math.max(...p),
      };
    }
    if (q.type === "noul") {
      const pTrue = p[0]; // keys are ["true", "false"]
      const pFalse = p[1];
      const [noLabel, yesLabel] = q.options;
      return {
        type: "noul",
        noul: pTrue,
        probabilities: { [noLabel]: pFalse, [yesLabel]: pTrue },
        confidence: Math.max(pTrue, pFalse),
      };
    }
    // score: keys are 0..n-1
    const values = q.options.map((_, j) => j);
    return {
      type: "score",
      score: p.reduce((acc, pj, j) => acc + pj * values[j], 0),
      level: argmax(p),
      legend: Object.fromEntries(q.options.map((level, j) => [String(j), level])),
      probabilities: Object.fromEntries(p.map((pj, j) => [String(j), pj])),
      confidence: Math.max(...p),
    };
  });
}

// ---------------------------------------------------------------------------
// The decider
// ---------------------------------------------------------------------------

/**
 * Create a clef decider.
 *
 * @param {object} [opts]
 *  model/revision     — browser export repo (fused graph, tokenizer,
 *                        clef_config.json)
 *  file               — graph path within the repo
 *  tokenizerModel     — repo to read the tokenizer from (default: model)
 *  configUrl          — override for clef_config.json
 *  maxLen             — record window; null = the checkpoint's max_len
 *  maxStateTokens     — cap on the tokenized state (default: none)
 *  ort, transformers, fetchImpl, wasmPaths, device, sessionFactory,
 *  onProgress, scope — as in the other families
 */
export async function createClefDecider({
  model = CLEF_DEFAULT_MODEL,
  revision = CLEF_DEFAULT_REVISION,
  file = CLEF_DEFAULT_FILE,
  tokenizerModel = null,
  tokenizerRevision = null,
  configUrl = null,
  ort = null,
  transformers = null,
  fetchImpl = fetch,
  wasmPaths = null,
  device = "auto",
  sessionFactory = null,
  onProgress = null,
  maxLen = null,
  maxStateTokens = CLEF_DEFAULTS.maxStateTokens,
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

  // The export repo's clef_config.json carries the window and the head shape
  // (head weights are baked into the graph; the config is informational).
  const url = configUrl ?? hfUrl(model, revision, "clef_config.json");
  let config = null;
  let configSource = null;
  try {
    // Cached like every other config; missing file, transport failure or
    // garbage JSON all fall through to the shipped defaults.
    config = await fetchCachedJson(url, { fetchImpl, nullOn404: true });
    if (config != null) configSource = url;
  } catch { /* fall through to the shipped defaults */ }
  const cfg = { ...CLEF_DEFAULTS, ...(config ?? {}) };
  const effMaxLen = Number.isFinite(maxLen) && maxLen > 0 ? maxLen : cfg.maxLen;

  const tok = await tf.AutoTokenizer.from_pretrained(tokenizerModel ?? model, {
    revision: tokenizerRevision ?? (tokenizerModel ? "main" : revision),
    progress_callback: () => {},
  });
  const encode = (text) =>
    Array.from(tok(text, { add_special_tokens: false }).input_ids.data, Number);

  const createSession = sessionFactory
    ?? (async (u, opts) => {
      const bytes = await fetchCachedBytes(u, {
        fetchImpl, onProgress: (p) => onProgress?.(p), persist: true,
      });
      const external = await fetchOnnxExternalData(u, { fetchImpl });
      const sessionOpts = external ? { ...opts, externalData: [external] } : opts;
      return ortMod.InferenceSession.create(bytes, sessionOpts);
    });
  let session;
  try {
    session = await createSession(hfUrl(model, revision, file), {
      executionProviders, graphOptimizationLevel: "all",
    });
  } catch (e) {
    if (resolvedDevice !== "webgpu") throw e;
    onProgress?.({ phase: `webgpu unavailable (${e?.message ?? e}) — retrying on wasm` });
    session = await createSession(hfUrl(model, revision, file),
      { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  }

  const int64 = (arr, dims) => new ortMod.Tensor("int64", BigInt64Array.from(arr, BigInt), dims);
  const f32 = (arr, dims) => new ortMod.Tensor("float32", arr, dims);
  const bool = (arr, dims) => new ortMod.Tensor("bool", Uint8Array.from(arr), dims);

  const info = {
    family: "clef",
    model,
    revision,
    device: resolvedDevice,
    maxLen: effMaxLen,
    maxStateTokens: maxStateTokens ?? cfg.max_state_tokens ?? null,
    head: config?.head ?? null,
    configSource,
  };

  // One session, one `decide()` at a time (a shared session is not safe for
  // interleaved runs).
  const queue = createSerialQueue();

  const decideImpl = async (state, questions) => {
    const t0 = now();
    const normalized = normalizeQuestions(questions);
    const stateText = renderClefContent(state ?? "");
    const record = encodeClefRecord({
      encode, stateText, questions: normalized,
      maxLength: effMaxLen,
      maxStateTokens: maxStateTokens ?? cfg.max_state_tokens ?? null,
    });
    const batch = packClefBatch(record);

    const tRun = now();
    const out = await session.run({
      input_ids: int64(record.ids, [1, batch.L]),
      attention_mask: int64(record.ids.map(() => 1), [1, batch.L]),
      pool_global: f32(batch.poolGlobal, [1, 1, batch.L]),
      pool_question: f32(batch.poolQuestion, [1, batch.Q, batch.L]),
      pool_option: f32(batch.poolOption, [1, batch.S, batch.L]),
      type_ids: int64(batch.typeIds, [batch.Q]),
      slot_question: int64(batch.slotQuestion, [batch.S]),
      opt_mask: bool(batch.optMask, [batch.S]),
    });
    const runMs = now() - tRun;
    const logits = out.logits ?? out[Object.keys(out)[0]];
    const dims = logits.dims;
    if (dims.length !== 2 || dims[0] !== 1 || dims[1] !== batch.S) {
      throw new Error(
        `clef: invalid logits shape [${dims}] for ${batch.Q} questions with ${batch.S} option slots`);
    }
    const probs = [];
    for (let q = 0; q < batch.Q; q++) {
      const row = Array.from(
        { length: batch.counts[q] },
        (_, k) => Number(logits.data[q * batch.K + k]));
      probs.push(softmaxWithTemperature(row, 1));
    }
    const answers = clefAnswers({ probs, questions: normalized });
    let prompt;
    try {
      prompt = tok.decode(record.ids, { skip_special_tokens: false });
    } catch {
      prompt = record.ids.join(" ");
    }
    return {
      answers,
      truncated: record.truncated,
      length: record.length,
      prompts: [prompt],
      timings: { totalMs: Math.round(now() - t0), runMs: Math.round(runMs) },
    };
  };

  return {
    info,
    decide: (state, questions) => queue(() => decideImpl(state, questions)),
    async dispose() {
      try { await session.release?.(); } catch { /* optional */ }
    },
  };
}
