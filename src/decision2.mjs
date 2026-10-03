/**
 * decision2.mjs — vLLM Semantic Router Decision 2.0 family.
 *
 * Decision 2.0 is a Qwen3 fine-tune with a candidate head: one prompt per
 * question, built from segments (each tokenized separately so the readout
 * positions are exact):
 *
 *   Context:\n{state}\n\n
 *   Task type: {noul|choice|score}\n
 *   Question:\n{instructions}\n
 *   Options:\n
 *   <option>\n{"description":…,"key":"…"}\n</option>\n   (per option)
 *   \nSelect the single option best supported by the context and instructions.\n
 *   Decision:
 *
 * Graph: input_ids [B,L], attention_mask [B,L], answer_pos [B], option_pos
 * [B,K] -> logits [B,K]. option_pos is the last token of each </option>;
 * answer_pos is the last token of the prompt. Score questions add the
 * per-level score_bias from config.json, then softmax (temperature 1).
 *
 * Reference: onnx-community/Decision-2.0-Kai-0.6B-ONNX (conversion/README).
 */

import { normalizeQuestions } from "./questions.mjs";
import { fetchCachedBytes } from "./laya.mjs";

export const DECISION2_DEFAULT_MODEL = "onnx-community/Decision-2.0-Kai-0.6B-ONNX";
export const DECISION2_DEFAULT_REVISION = "13ab55013ab1";
export const DECISION2_DEFAULT_FILE = "onnx/model_quantized.onnx";
export const DECISION2_MAX_INPUT_TOKENS = 8192;

// Larger siblings (same protocol, same onnx/ layout).
export const DECISION2_EOS_MODEL = "onnx-community/Decision-2.0-Eos-0.8B-ONNX";
export const DECISION2_EOS_REVISION = "6369be38417e";
export const DECISION2_SOL_MODEL = "onnx-community/Decision-2.0-Sol-2B-ONNX";
export const DECISION2_SOL_REVISION = "5b3478e05ce3";

export const DECISION2_NOUL_DESCRIPTIONS = { false: "No", true: "Yes" };
const DECISION_SUFFIX = "Select the single option best supported by the context and instructions.";

/** Flatten a state or instruction into text, stably (object keys sorted). */
export function renderDecision2Content(content) {
  if (typeof content === "string") return content;
  return JSON.stringify(sortKeys(content));
}

/** Recursively sort object keys for deterministic serialization. */
function sortKeys(value) {
  if (Array.isArray(value)) return value.map(sortKeys);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value).sort().map((k) => [k, sortKeys(value[k])]));
  }
  return value;
}

/**
 * Build the option list for one question: [{key, description}]. Descriptions
 * are null when empty (matching the reference encoding).
 */
export function renderDecision2Options(question) {
  if (question.type === "choice") {
    const criteria = question.criteria ?? {};
    return question.options.map((label) => {
      const d = criteria[label];
      return { key: label, description: d == null || String(d).trim() === "" ? null : String(d) };
    });
  }
  if (question.type === "noul") {
    const criteria = question.criteria ?? {};
    const falseDesc = criteria.false ?? criteria.no ?? DECISION2_NOUL_DESCRIPTIONS.false;
    const trueDesc = criteria.true ?? criteria.yes ?? DECISION2_NOUL_DESCRIPTIONS.true;
    return [
      { key: "false", description: String(falseDesc) },
      { key: "true", description: String(trueDesc) },
    ];
  }
  if (question.type === "score") {
    return question.options.map((rubric, i) => ({ key: String(i), description: String(rubric) }));
  }
  throw new TypeError(`decision2: unsupported question type ${question.type}`);
}

/**
 * Build the prompt segments for one question. Each segment is tokenized
 * separately; the returned `optionEnds` marks which segment index ends each
 * option (for option_pos), and the prompt ends with the Decision: segment.
 */
export function buildDecision2Segments(question, stateText) {
  const options = renderDecision2Options(question);
  const segments = [
    `Context:\n${stateText}\n\n`,
    `Task type: ${question.type}\n`,
    `Question:\n${renderDecision2Content(question.instructions)}\n`,
    `Options:`,
  ];
  const optionEnds = [];
  for (const opt of options) {
    // Sorted keys, compact JSON, matching the reference.
    const payload = JSON.stringify({ description: opt.description, key: opt.key });
    segments.push(`\n<option>\n${payload}\n</option>`);
    optionEnds.push(segments.length - 1);
  }
  segments.push(`\n\n${DECISION_SUFFIX}\n`);
  segments.push(`Decision:`);
  return { segments, optionEnds, keys: options.map((o) => o.key) };
}

/**
 * Tokenize segments separately and concatenate, recording the absolute
 * position of each option's last token (option_pos) and the prompt's last
 * token (answer_pos).
 */
export function tokenizeDecision2Segments(segments, optionEnds, encode, maxLength) {
  const ids = [];
  const optionPos = [];
  for (let s = 0; s < segments.length; s++) {
    const segIds = encode(segments[s]);
    const start = ids.length;
    ids.push(...segIds);
    if (optionEnds.includes(s)) {
      optionPos.push(start + segIds.length - 1);
    }
    if (ids.length > maxLength) {
      throw new RangeError(
        `decision2: prompt too long (${ids.length} > ${maxLength} tokens)`);
    }
  }
  return { ids, optionPos, answerPos: ids.length - 1 };
}

/** Softmax with temperature (temperature 1 for Decision 2.0). */
export function decision2Softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(Number(x) - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/**
 * Decode one question's logits into a jev-web answer. Score questions add
 * the per-level bias before the softmax.
 */
export function decision2AnswerFromLogits({ logits, question, keys, scoreBias }) {
  const adjusted = logits.map(Number);
  if (question.type === "score" && scoreBias) {
    const bias = scoreBias[String(keys.length)];
    if (bias) {
      if (bias.length !== adjusted.length) {
        throw new TypeError(
          `decision2: score_bias length ${bias.length} != ${adjusted.length} options`);
      }
      for (let i = 0; i < adjusted.length; i++) adjusted[i] += bias[i];
    }
  }
  const probs = decision2Softmax(adjusted);
  const best = probs.indexOf(Math.max(...probs));

  if (question.type === "choice") {
    const probabilities = Object.fromEntries(keys.map((k, i) => [k, probs[i]]));
    return {
      type: "choice", choice: keys[best], index: best,
      probabilities, confidence: Math.max(...probs),
    };
  }
  if (question.type === "noul") {
    const yesIdx = keys.indexOf("true");
    const yes = probs[yesIdx];
    const [noLabel, yesLabel] = question.options;
    return {
      type: "noul", noul: yes,
      probabilities: { [noLabel]: 1 - yes, [yesLabel]: yes },
      confidence: Math.max(yes, 1 - yes),
    };
  }
  // score
  const values = keys.map((k) => Number(k));
  const score = probs.reduce((acc, p, i) => acc + p * values[i], 0);
  const probabilities = Object.fromEntries(keys.map((k, i) => [k, probs[i]]));
  return {
    type: "score", score, level: best,
    legend: Object.fromEntries(question.options.map((r, i) => [String(i), r])),
    probabilities, confidence: Math.max(...probs),
  };
}

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
const hfUrl = (model, revision, ...parts) =>
  `https://huggingface.co/${model}/resolve/${revision ? `${revision}/` : ""}${parts.join("/")}`;

/**
 * Create a Decision 2.0 decider.
 *
 * @param {object} [opts]
 *  model/revision/file — pinned ONNX export (or your own mirror)
 *  configUrl — override for config.json (carries decision2.score_bias)
 *  ort, transformers, fetchImpl, wasmPaths, device, sessionFactory — as in
 *  the other families
 *  maxLength — token budget (default 8192)
 */
export async function createDecision2Decider({
  model = DECISION2_DEFAULT_MODEL,
  revision = DECISION2_DEFAULT_REVISION,
  file = DECISION2_DEFAULT_FILE,
  configUrl = null,
  ort = null,
  transformers = null,
  fetchImpl = fetch,
  wasmPaths = null,
  device = "auto",
  sessionFactory = null,
  onProgress = null,
  maxLength = DECISION2_MAX_INPUT_TOKENS,
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

  const configRes = await fetchImpl(configUrl ?? hfUrl(model, revision, "config.json"));
  if (!configRes.ok) throw new Error(`decision2: config fetch failed (${configRes.status})`);
  const config = await configRes.json();
  const scoreBias = config?.decision2?.score_bias ?? {};

  const tok = await tf.AutoTokenizer.from_pretrained(model, {
    revision,
    progress_callback: (p) => onProgress?.({ phase: "tokenizer", ...p }),
  });
  const encode = (text) =>
    Array.from(tok(text, { add_special_tokens: false }).input_ids.data, Number);

  const url = hfUrl(model, revision, file);
  const createSession = sessionFactory
    ?? (async (u, opts) => {
      const modelBytes = await fetchCachedBytes(u, {
        fetchImpl, onProgress: (p) => onProgress?.(p), persist: true,
      });
      return ortMod.InferenceSession.create(modelBytes, opts);
    });

  let session;
  let activeDevice = resolvedDevice;
  try {
    session = await createSession(url, { executionProviders, graphOptimizationLevel: "all" });
  } catch (e) {
    if (resolvedDevice === "webgpu") {
      onProgress?.({ phase: `webgpu unavailable (${e?.message ?? e}) — retrying on wasm` });
      session = await createSession(url, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      activeDevice = "wasm";
    } else {
      throw e;
    }
  }

  // Probe external data (the q8 file ships a .onnx_data sidecar).
  try {
    const probe = await fetchImpl(url + "_data", { method: "HEAD" });
    if (!probe.ok) {
      const probe2 = await fetchImpl(url.replace(/\.onnx$/, ".onnx_data"), { method: "HEAD" });
      void probe2;
    }
  } catch { /* optional */ }

  return {
    info: {
      family: "decision2",
      model, revision, device: activeDevice,
      maxLength,
    },
    async decide(state, questions) {
      const t0 = now();
      const normalized = normalizeQuestions(questions);
      const stateText = renderDecision2Content(state);
      const B = normalized.length;
      const K = Math.max(...normalized.map((q) => renderDecision2Options(q).length));

      const allIds = [];
      const answerPos = [];
      const optionPos = [];
      const metas = [];
      let truncated = false;
      let length = 0;

      for (const q of normalized) {
        const { segments, optionEnds, keys } = buildDecision2Segments(q, stateText);
        const { ids, optionPos: oPos, answerPos: aPos } =
          tokenizeDecision2Segments(segments, optionEnds, encode, maxLength);
        allIds.push(ids);
        answerPos.push(aPos);
        // Pad option positions with 0 (matches the strands convention).
        const padded = [...oPos];
        while (padded.length < K) padded.push(0);
        optionPos.push(padded);
        metas.push({ question: q, keys });
        length += ids.length;
      }

      // Pad to a rectangular batch.
      const L = Math.max(...allIds.map((r) => r.length));
      const inputIds = new BigInt64Array(B * L);
      const attentionMask = new BigInt64Array(B * L);
      for (let b = 0; b < B; b++) {
        allIds[b].forEach((id, i) => {
          inputIds[b * L + i] = BigInt(id);
          attentionMask[b * L + i] = 1n;
        });
      }
      const feeds = {
        input_ids: new ortMod.Tensor("int64", inputIds, [B, L]),
        attention_mask: new ortMod.Tensor("int64", attentionMask, [B, L]),
        answer_pos: new ortMod.Tensor("int64", BigInt64Array.from(answerPos.map(BigInt)), [B]),
        option_pos: new ortMod.Tensor(
          "int64",
          BigInt64Array.from(optionPos.flat().map(BigInt)),
          [B, K]),
      };

      const tRun = now();
      const out = await session.run(feeds);
      const runMs = now() - tRun;
      const logits = out.logits ?? out[Object.keys(out)[0]];
      if (logits.dims.length !== 2 || logits.dims[0] !== B || logits.dims[1] !== K) {
        throw new Error(
          `decision2: invalid logits shape [${logits.dims}] for batch ${B}x${K}`);
      }

      const answers = metas.map(({ question, keys }, b) => {
        const row = Array.from(
          { length: keys.length },
          (_, k) => Number(logits.data[b * K + k]));
        return decision2AnswerFromLogits({ logits: row, question, keys, scoreBias });
      });

      return {
        answers, truncated, length,
        prompts: metas.map(({ question, keys }) => ({ question: question.instructions, keys })),
        timings: { totalMs: Math.round(now() - t0), runMs: Math.round(runMs) },
      };
    },
    async dispose() {
      try { await session.release?.(); } catch { /* optional */ }
    },
  };
}
