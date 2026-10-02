/**
 * strands.mjs — Strands Decider 2B family (Qwen3.5-2B torso + pointer head).
 *
 * Mirrors the reference Python runtime (`strands-decider`, Apache-2.0,
 * github.com/strands-labs/strands-decider):
 *
 *   prompt:  <state>\n{state}\n</state>\n
 *            <question type="{kind}">\n{header}\n{instructions}\n<options>\n
 *            1. {label} — {description}\n…\n</options>\n</question>\n<answer>
 *   graph:   input_ids, attention_mask, opt_idx -> logits [B, K]
 *            (torso with LoRA merged; pointer head: LayerNorm -> q/k linears ->
 *             scaled dot of <answer>-position state vs each option's last token)
 *   decode:  per-kind temperature (noul / choice / score) then softmax;
 *            score confidence is ordinal (spread-based), not max-probability.
 *
 * The browser export used here is a single fused fp16 ONNX graph
 * (warsang/strands-decider-2b-web, pinned by revision).
 */

import { normalizeQuestions } from "./questions.mjs";
import { softmaxWithTemperature } from "./answers.mjs";
import { fetchCachedBytes, fetchOnnxExternalData } from "./laya.mjs";

export const STRANDS_DEFAULT_MODEL = "warsang/strands-decider-2b-web";
export const STRANDS_DEFAULT_REVISION = "main"; // pin to the export commit once it lands
export const STRANDS_DEFAULT_FILE = "model.onnx";

/** Reference calibration from the checkpoint's hobson_config.json. */
export const STRANDS_TEMPERATURE = 0.9627721607677362;
export const STRANDS_TEMPERATURE_BY_KIND = {
  noul: 0.9107136998460428,
  choice: 0.734189596436441,
  score: 1.32780942142348,
};
export const STRANDS_ORDINAL_SMOOTHING = 0.1;
export const STRANDS_DEFAULT_MAX_LEN = 4096;
// Largest share of the window the question may claim before the state is squeezed.
export const STRANDS_MAX_QUESTION_FRACTION = 0.75;

export const STRANDS_KIND = { choice: "choice", score: "score", noul: "noul" };

// Rendered on both sides of a noul so the two slots read like any other option list.
const NOUL_DEFAULT_CRITERIA = {
  false: "the statement does not hold for this state",
  true: "the statement holds for this state",
};

const KIND_HEADERS = {
  noul: "Decide whether the statement is true of the state.",
  choice: "Select exactly one option.",
  score: "Rate the state against the ordered levels below (lowest first).",
};

/**
 * Flatten a state or instruction into text, stably (mirrors render_content:
 * dicts/lists become indented JSON so the same state always tokenises
 * identically).
 */
export function renderStrandsContent(content) {
  if (typeof content === "string") return content.trim();
  return JSON.stringify(content, null, 2);
}

function optionBlock(pairs) {
  // Number options from 1 so slot k <-> the line reading `k+1.`.
  // Returns each line's (start, end) character span within the block.
  const lines = [];
  const spans = [];
  let cursor = 0;
  pairs.forEach(([name, desc], i) => {
    const flat = String(desc ?? "").split(/\s+/).join(" ");
    const line = `${i + 1}. ${name}` + (flat ? ` \u2014 ${flat}` : "");
    lines.push(line);
    spans.push([cursor, cursor + line.length]);
    cursor += line.length + 1; // the joining newline
  });
  return { text: lines.join("\n"), spans };
}

/**
 * Render one normalized question to the exact prompt the head was trained on.
 * @returns {{text:string, chunks:string[], slotLabels:string[],
 *   slotDescriptions:string[], kind:string, optionSpans:[number,number][]}}
 *
 * `chunks` is the question text split at pre-tokeniser piece boundaries
 * ([prefix, line_0+"\n", …, line_{k-1}+"\n", suffix]) so tokenising each chunk
 * separately and concatenating gives exactly the full-text tokenisation, with
 * exact option-token positions and no offset mapping needed.
 */
export function renderStrandsQuestion(question) {
  const instructions = renderStrandsContent(question.instructions);
  let pairs;
  let slotLabels;
  let slotDescriptions;
  let kind;
  if (question.type === "noul") {
    const [noLabel, yesLabel] = question.options;
    pairs = [
      [noLabel, NOUL_DEFAULT_CRITERIA.false],
      [yesLabel, NOUL_DEFAULT_CRITERIA.true],
    ];
    slotLabels = [noLabel, yesLabel];
    slotDescriptions = [NOUL_DEFAULT_CRITERIA.false, NOUL_DEFAULT_CRITERIA.true];
    kind = "noul";
  } else if (question.type === "choice") {
    pairs = question.options.map((o) => [o, ""]);
    slotLabels = [...question.options];
    slotDescriptions = question.options.map(() => "");
    kind = "choice";
  } else if (question.type === "score") {
    // Level index is the label; the rubric text is the description.
    pairs = question.options.map((desc, i) => [String(i), desc]);
    slotLabels = question.options.map((_, i) => String(i));
    slotDescriptions = [...question.options];
    kind = "score";
  } else {
    throw new TypeError(`strands: unsupported question type ${question.type}`);
  }

  const header = KIND_HEADERS[kind];
  const block = optionBlock(pairs);
  const prefix = `<question type="${kind}">\n${header}\n${instructions}\n<options>\n`;
  const suffix = `</options>\n</question>\n<answer>`;
  const text = prefix + block.text + "\n" + suffix;
  const base = prefix.length;
  // Every option chunk carries its trailing newline, so no pre-tokeniser piece
  // can span a chunk boundary and concatenating the chunk encodings reproduces
  // the full-text tokenisation exactly.
  const chunks = [prefix, ...block.spans.map(([s, e]) => block.text.slice(s, e) + "\n"), suffix];
  return {
    text,
    chunks,
    slotLabels,
    slotDescriptions,
    kind,
    optionSpans: block.spans.map(([s, e]) => [base + s, base + e]),
  };
}

/** The shared prefix: everything before this point is identical across questions. */
export function renderStrandsState(state) {
  return `<state>\n${renderStrandsContent(state)}\n</state>\n`;
}

/**
 * Scored token index of each option line inside its chunk's tokenisation.
 *
 * chunkIds[i] = encode(line_i + "\n"), lineIds[i] = encode(line_i). The trailing
 * "\n" is either its own token (line tokens = lineIds) or merged with the line's
 * final token (the merged token's span runs past the line, so the reference
 * readout excludes it and scores the token before). Both cases are detected by
 * comparing the two encodings — no offset mapping needed.
 *
 * @returns {number[]} token index (chunk-relative) of each option's scored token
 */
export function strandsOptionTokenIndices(lineIds, chunkIds) {
  if (lineIds.length !== chunkIds.length) {
    throw new TypeError("strandsOptionTokenIndices: lineIds and chunkIds must align");
  }
  return lineIds.map((line, i) => {
    const chunk = chunkIds[i];
    if (line.length === 0 || chunk.length === 0) {
      throw new RangeError(`strands: option ${i} has no tokens left; the prompt was truncated through its option list`);
    }
    if (chunk.length > line.length) {
      // Newline is its own trailing token(s): the line's tokens are intact.
      for (let j = 0; j < line.length; j++) {
        if (chunk[j] !== line[j]) {
          throw new Error(`strands: option ${i} re-tokenised under its trailing newline; cannot place the readout`);
        }
      }
      return line.length - 1;
    }
    // Newline merged with the line's final token(s): that token's span runs past
    // the line, so the readout scores the token before it.
    if (chunk.length < 2) {
      throw new RangeError(`strands: option ${i} has no tokens left; the prompt was truncated through its option list`);
    }
    return chunk.length - 2;
  });
}

/**
 * Tokenise state + questions with the reference `_fit` budget policy: the
 * question gets first claim on the window (up to maxQuestionFraction), the
 * state is truncated from the right into what remains, and a question longer
 * than its reserve is truncated from the front, keeping the tail (options and
 * the trailing `<answer>` marker, which is the pooling position).
 *
 * @param {{encode:(text:string, addSpecial?:boolean)=>number[], stateText:string,
 *   rendered:object[], maxLen?:number, maxQuestionFraction?:number}} spec
 * @returns {{items:{ids:number[], optIdx:number[], kind:string}[], truncated:boolean}}
 */
export function buildStrandsBatch({
  encode,
  stateText,
  rendered,
  maxLen = STRANDS_DEFAULT_MAX_LEN,
  maxQuestionFraction = STRANDS_MAX_QUESTION_FRACTION,
}) {
  // Questions are encoded with add_special_tokens=false throughout; the state
  // alone carries the BOS, exactly like the reference runtime.
  const questionChunks = rendered.map((rq) => ({
    rq,
    chunkIds: rq.chunks.map((c) => encode(c, false)),
  }));
  const longest = Math.max(...questionChunks.map((q) => q.chunkIds.flat().length));
  const reserve = Math.min(longest, Math.max(1, Math.floor(maxLen * maxQuestionFraction)));
  const stateBudget = Math.max(1, maxLen - reserve);
  let stateIds = encode(stateText, true);
  let truncated = false;
  if (stateIds.length > stateBudget) {
    stateIds = stateIds.slice(0, stateBudget);
    truncated = true;
  }

  const items = questionChunks.map(({ rq, chunkIds }) => {
    // Option lines are chunks 1..k (each "line\n"); locate each option's scored
    // token inside its own chunk, where positions are exact by construction.
    const lineChunks = chunkIds.slice(1, 1 + rq.slotLabels.length);
    const lineIds = lineChunks.map((ids, i) =>
      encode(rq.chunks[1 + i].slice(0, -1), false));
    const inChunk = strandsOptionTokenIndices(lineIds, lineChunks);

    let ids = chunkIds.flat();
    let optIdx = inChunk.map((p, i) => {
      let off = 0;
      for (let c = 0; c < 1 + i; c++) off += chunkIds[c].length;
      return off + p;
    });
    if (ids.length > reserve) {
      const cut = ids.length - reserve;
      ids = ids.slice(cut);
      truncated = true;
      optIdx = optIdx.map((p) => p - cut);
      if (optIdx.some((p) => p < 0)) {
        throw new RangeError(
          "strands: the prompt was truncated through its option list; shorten the instructions");
      }
    }
    const full = [...stateIds, ...ids];
    return {
      ids: full,
      optIdx: optIdx.map((p) => p + stateIds.length),
      kind: rq.kind,
    };
  });
  return { items, truncated };
}

/** Pad a batch of Strands items into int64 tensors. opt_idx pads with -1. */
export function collateStrandsItems(items) {
  const n = items.length;
  const L = Math.max(...items.map((it) => it.ids.length));
  const K = Math.max(...items.map((it) => it.optIdx.length));
  const inputIds = new BigInt64Array(n * L);
  const attention = new BigInt64Array(n * L);
  const optIdx = new BigInt64Array(n * K).fill(-1n);
  items.forEach((it, i) => {
    it.ids.forEach((v, j) => { inputIds[i * L + j] = BigInt(v); attention[i * L + j] = 1n; });
    it.optIdx.forEach((p, j) => { optIdx[i * K + j] = BigInt(p); });
  });
  return { n, L, K, inputIds, attention, optIdx };
}

/** Normalised max-probability confidence (mirrors schema.derive_confidence). */
export function strandsChoiceConfidence(probabilities) {
  const n = probabilities.length;
  if (n <= 1) return 1;
  const pMax = Math.max(...probabilities);
  return Math.min(1, Math.max(0, (n * pMax - 1) / (n - 1)));
}

/**
 * Ordinal confidence for score answers (mirrors schema.derive_score_confidence):
 * mass on adjacent levels is agreement ("about 2.5"), not confusion — so this
 * measures spread via the normalised standard deviation, with a floor
 * correction for the ordinal smoothing the training targets carry.
 */
export function strandsScoreConfidence(probabilities, ordinalSmoothing = STRANDS_ORDINAL_SMOOTHING) {
  const n = probabilities.length;
  if (n <= 1) return 1;
  const total = probabilities.reduce((a, b) => a + b, 0);
  if (total <= 0) return 0;
  const p = probabilities.map((x) => x / total);
  const mean = p.reduce((acc, pi, i) => acc + i * pi, 0);
  const variance = p.reduce((acc, pi, i) => acc + pi * (i - mean) ** 2, 0);
  const sigma = Math.sqrt(variance);
  const sigmaMax = (n - 1) / 2;
  const sigmaFloor = ordinalSmoothing > 0 ? Math.sqrt(ordinalSmoothing) : 0;
  if (sigmaMax <= sigmaFloor) return sigma <= sigmaFloor ? 1 : 0;
  return Math.min(1, Math.max(0, (sigmaMax - sigma) / (sigmaMax - sigmaFloor)));
}

/**
 * Decode per-question logits into jev-web answers (mirrors infer._to_answer).
 *
 * `logits` is row-major [n, rowWidth] (one fused forward over the batch);
 * each question reads its first `k` entries.
 * @param {{logits:ArrayLike<number>, rowWidth:number, questions:object[],
 *   rendered:object[], temperaturesByKind?:object, temperature?:number,
 *   ordinalSmoothing?:number}} spec
 */
export function strandsAnswersFromLogits({
  logits,
  rowWidth,
  questions,
  rendered,
  temperaturesByKind = STRANDS_TEMPERATURE_BY_KIND,
  temperature = STRANDS_TEMPERATURE,
  ordinalSmoothing = STRANDS_ORDINAL_SMOOTHING,
}) {
  const flat = Array.from(logits, Number);
  const answers = [];
  questions.forEach((q, i) => {
    const rq = rendered[i];
    const k = rq.slotLabels.length;
    if (!Number.isInteger(rowWidth) || rowWidth < k) {
      throw new TypeError(`strands: rowWidth ${rowWidth} < ${k} options`);
    }
    const t = Number(temperaturesByKind[rq.kind] ?? temperature);
    const probs = softmaxWithTemperature(flat.slice(i * rowWidth, i * rowWidth + k), t);

    if (rq.kind === "choice") {
      const best = probs.indexOf(Math.max(...probs));
      answers.push({
        type: "choice",
        choice: q.options[best],
        index: best,
        probabilities: Object.fromEntries(q.options.map((o, j) => [o, probs[j]])),
        confidence: strandsChoiceConfidence(probs),
      });
    } else if (rq.kind === "score") {
      // Slots may in principle be permuted; re-key by true level before the
      // expectation so a shuffled rendering cannot score backwards.
      const byLevel = Object.fromEntries(rq.slotLabels.map((lv, j) => [lv, probs[j]]));
      const ordered = q.options.map((_, j) => byLevel[String(j)]);
      const expected = ordered.reduce((acc, p, j) => acc + p * j, 0);
      const best = ordered.indexOf(Math.max(...ordered));
      answers.push({
        type: "score",
        score: expected,
        level: best,
        probabilities: Object.fromEntries(q.options.map((o, j) => [o, ordered[j]])),
        confidence: strandsScoreConfidence(ordered, ordinalSmoothing),
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

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

const hfUrl = (model, revision, file) =>
  `https://huggingface.co/${model}/resolve/${revision ? `${revision}/` : ""}${file}`;

/**
 * Create a Strands Decider decider.
 *
 * @param {object} [opts]
 *  model/revision/modelFile — pinned browser ONNX export (or your own mirror)
 *  modelUrl                — override the resolved ONNX URL
 *  ort                     — injected onnxruntime-web (lazy import otherwise)
 *  transformers            — injected @huggingface/transformers (tokenizer only)
 *  tokenizerModel/tokenizerRevision — override the tokenizer source
 *  wasmPaths               — ORT wasm/loader directory (self-hosted builds)
 *  device                  — "auto" | "wasm" | "webgpu"
 *  sessionFactory          — (url, sessionOptions) => Promise<InferenceSession>;
 *                            hosts needing custom loading override this
 */
export async function createStrandsDecider({
  model = STRANDS_DEFAULT_MODEL,
  revision = STRANDS_DEFAULT_REVISION,
  modelFile = STRANDS_DEFAULT_FILE,
  modelUrl = null,
  ort = null,
  transformers = null,
  fetchImpl = fetch,
  tokenizerModel = null,
  tokenizerRevision = null,
  wasmPaths = null,
  device = "auto",
  sessionFactory = null,
  onProgress = null,
  maxLen = STRANDS_DEFAULT_MAX_LEN,
  maxQuestionFraction = STRANDS_MAX_QUESTION_FRACTION,
  temperaturesByKind = STRANDS_TEMPERATURE_BY_KIND,
  temperature = STRANDS_TEMPERATURE,
  ordinalSmoothing = STRANDS_ORDINAL_SMOOTHING,
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

  const tok = await tf.AutoTokenizer.from_pretrained(tokenizerModel ?? model, {
    revision: tokenizerRevision ?? (tokenizerModel ? undefined : revision),
    progress_callback: report("tokenizer"),
  });
  const encode = (text, addSpecial = false) =>
    Array.from(tok(text, { add_special_tokens: addSpecial }).input_ids.data, Number);
  const decode = (ids) => {
    try {
      return tok.decode(Array.from(ids), { skip_special_tokens: false });
    } catch {
      return Array.from(ids).join(" ");
    }
  };

  const url = modelUrl ?? hfUrl(model, revision, modelFile);
  const createSession = sessionFactory
    ?? (async (u, opts) => {
      const modelBytes = await fetchCachedBytes(u, {
        fetchImpl, onProgress: (p) => onProgress?.(p), persist: true,
      });
      const external = await fetchOnnxExternalData(u, {
        fetchImpl, onProgress: (p) => onProgress?.(p),
      });
      const sessionOpts = external ? { ...opts, externalData: [external] } : opts;
      return ortMod.InferenceSession.create(modelBytes, sessionOpts);
    });
  let session;
  let activeDevice = resolvedDevice;
  try {
    session = await createSession(url, {
      executionProviders,
      graphOptimizationLevel: "all",
    });
  } catch (e) {
    if (resolvedDevice === "webgpu") {
      onProgress?.({ phase: `webgpu unavailable (${e?.message ?? e}) — retrying on wasm` });
      session = await createSession(url, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      activeDevice = "wasm";
    } else {
      throw e;
    }
  }

  const int64 = (arr, dims) => new ortMod.Tensor("int64", arr, dims);

  return {
    info: {
      family: "strands-decider",
      model,
      revision,
      device: activeDevice,
      maxLen,
      temperature,
      temperaturesByKind: { ...temperaturesByKind },
      ordinalSmoothing,
    },
    async decide(state, questions) {
      const t0 = now();
      const normalized = normalizeQuestions(questions);
      const rendered = normalized.map(renderStrandsQuestion);
      const stateText = renderStrandsState(state);
      const { items, truncated } = buildStrandsBatch({
        encode, stateText, rendered, maxLen, maxQuestionFraction,
      });
      const b = collateStrandsItems(items);

      const tRun = now();
      const out = await session.run({
        input_ids: int64(b.inputIds, [b.n, b.L]),
        attention_mask: int64(b.attention, [b.n, b.L]),
        opt_idx: int64(b.optIdx, [b.n, b.K]),
      });
      const runMs = now() - tRun;
      const logits = out.logits ?? out[Object.keys(out)[0]];

      const answers = strandsAnswersFromLogits({
        logits: logits.data,
        rowWidth: b.K,
        questions: normalized,
        rendered,
        temperaturesByKind,
        temperature,
        ordinalSmoothing,
      });
      const length = Number(b.attention.reduce((a, v) => a + v, 0n));
      const prompts = items.map((it) => decode(it.ids));
      return {
        answers,
        truncated,
        length,
        prompts,
        timings: { totalMs: Math.round(now() - t0), runMs: Math.round(runMs) },
      };
    },
    async dispose() {
      try { await session.release?.(); } catch { /* optional */ }
    },
  };
}
