/**
 * kev.mjs — Kev decision-model family (jaredpalmer/kev, Apache-2.0).
 *
 * Kev is a typed decision model: one document (the *state*) and any number of
 * typed questions are packed into a single sequence with five delimiter
 * tokens. A block-causal mask (derived in-graph from the delimiter ids) lets
 * every question see the state and itself only; a pointer head scores each
 * option's `</opt>` token against its question's `<decide>` token.
 *
 *   graph:  input_ids, attention_mask (right padded) -> logits [B, seq]
 *   decode: read the logit at every option's `</opt>` position,
 *           softmax within each question.
 *
 * Reference: onnx-community/kev-0.6b-ONNX README (ships a JS implementation
 * this mirrors) and config.json `kev` block (delimiters, ids, limits).
 */

import { normalizeQuestions } from "./questions.mjs";
import { answersFromScores } from "./answers.mjs";
import { fetchCachedBytes, fetchOnnxExternalData } from "./laya.mjs";
import { resolveDtype } from "./config.mjs";

const HF_BASE = "https://huggingface.co";
const hfUrl = (model, revision, ...parts) =>
  [HF_BASE, model, "resolve", revision, ...parts.filter(Boolean)].join("/");

export const KEV_DEFAULT_MODEL = "onnx-community/kev-0.6b-ONNX";
export const KEV_DEFAULT_REVISION = "5a827091b3d58fe4fdfc02a5a5b2ce364047d678";
export const KEV_4B_MODEL = "onnx-community/kev-4b-ONNX";
export const KEV_4B_REVISION = "195bdf73096df2bca7150223c9279af7893163c";

/** Named size variants with their pinned revisions and download sizes. */
export const KEV_VARIANTS = {
  "0.6b": {
    model: KEV_DEFAULT_MODEL,
    revision: KEV_DEFAULT_REVISION,
    bytes: 335_000_000,
    description: "kev-0.6b, 335 MB q4f16 browser ONNX",
  },
  "4b": {
    model: KEV_4B_MODEL,
    revision: KEV_4B_REVISION,
    bytes: 1_070_000_000,
    description: "kev-4b, 1.07 GB q4 browser ONNX",
  },
};

// Fallback when config.json has no `kev` block (both shipped repos have one).
const KEV_FALLBACK = {
  delimiters: {
    state: "<|fim_prefix|>",
    question: "<|fim_middle|>",
    option_start: "<|box_start|>",
    option_end: "<|box_end|>",
    decide: "<|fim_suffix|>",
  },
  max_state_tokens: 8192,
  max_branch_tokens: 8192,
  max_options: 255,
};

/** Caller text can never produce a delimiter: <|name|> -> <¦name¦>. */
export function escapeKevText(text) {
  return String(text).replace(/<\|([A-Za-z0-9_]+)\|>/g, "<¦$1¦>");
}

function renderKevContent(content) {
  if (typeof content === "string") return content.trim();
  return JSON.stringify(content);
}

/** Render one option; choice options pick up `criteria` descriptions. */
function renderKevOption(question, label) {
  if (question.type === "choice") {
    const desc = question.criteria?.[label];
    if (desc != null && String(desc).trim() && String(desc).trim() !== label) {
      return `${label}: ${String(desc).trim()}`;
    }
  }
  return label;
}

/**
 * Pack the state and normalized questions into one kev token sequence.
 * Returns { tokens, groups } where groups[i] holds the sequence indices of
 * question i's option `</opt>` tokens.
 */
export function packKevSequence({ stateText, questions, encode, delims, limits }) {
  const { state: STATE, question: Q, option_start: OPT, option_end: END, decide: DECIDE } = delims;
  const stateIds = encode(stateText).slice(0, limits.max_state_tokens);
  const tokens = [STATE, ...stateIds];
  const groups = [];
  let truncated = false;

  for (const q of questions) {
    const instrIds = encode(q.instructions);
    const spans = q.options.map((label) => [OPT, ...encode(renderKevOption(q, label)), END]);
    const branch = [Q, ...instrIds, ...spans.flat(), DECIDE];
    if (branch.length > limits.max_branch_tokens) {
      throw new Error(
        `kev: question branch too long (${branch.length} > ${limits.max_branch_tokens} tokens)`,
      );
    }
    if (tokens.length + branch.length > limits.max_state_tokens) {
      truncated = true;
      break;
    }
    const base = tokens.length;
    let cursor = 1 + instrIds.length;
    const ends = spans.map((s) => {
      cursor += s.length;
      return base + cursor - 1; // index of this option's </opt>
    });
    tokens.push(...branch);
    groups.push(ends);
  }
  return { tokens, groups, truncated };
}

/**
 * Create a Kev decider.
 *
 * @param {object} [opts]
 *  model/revision — pinned ONNX export (or your own mirror)
 *  variant        — "0.6b" | "4b" shortcut for model/revision
 *  modelFile      — override the resolved onnx file (default picks q4f16/q4 by device)
 *  configUrl/modelUrl — override the resolved file URLs
 *  ort            — injected onnxruntime-web (lazy import otherwise)
 *  transformers   — injected @huggingface/transformers (tokenizer only)
 *  fetchImpl      — fetch implementation
 *  wasmPaths      — ORT wasm/loader directory (self-hosted builds)
 *  device         — "auto" | "wasm" | "webgpu"
 *  dtype          — "auto" | "q4" | "q4f16"
 *  sessionFactory — (url, sessionOptions) => Promise<InferenceSession>
 *  onProgress     — progress callback
 */
export async function createKevDecider({
  model = KEV_DEFAULT_MODEL,
  revision = KEV_DEFAULT_REVISION,
  variant = null,
  modelFile = null,
  configUrl = null,
  modelUrl = null,
  ort = null,
  transformers = null,
  fetchImpl = fetch,
  wasmPaths = null,
  device = "auto",
  dtype = "auto",
  sessionFactory = null,
  onProgress = null,
  scope = globalThis,
} = {}) {
  if (variant) {
    const v = KEV_VARIANTS[variant];
    if (!v) throw new TypeError(`kev: unknown variant "${variant}" (0.6b, 4b)`);
    model = v.model;
    revision = v.revision;
  }
  const ortMod = ort ?? (await import("onnxruntime-web"));
  const tf = transformers ?? (await import("@huggingface/transformers"));
  if (wasmPaths) {
    try { ortMod.env.wasm.wasmPaths = wasmPaths; } catch { /* older runtime */ }
  }

  const resolvedDevice = device !== "auto"
    ? device
    : (scope?.navigator?.gpu ? "webgpu" : "wasm");
  const resolvedDtype = resolveDtype(resolvedDevice, dtype);
  const executionProviders = resolvedDevice === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];

  const report = (phase) => (p) =>
    onProgress?.({ phase, file: p?.file ?? null, loaded: p?.loaded ?? null, total: p?.total ?? null });

  // Model config carries the kev protocol block (delimiters, ids, limits).
  const cfgRes = await fetchImpl(configUrl ?? hfUrl(model, revision, "", "config.json"));
  if (!cfgRes.ok) throw new Error(`kev: config fetch failed (${cfgRes.status})`);
  const kevCfg = (await cfgRes.json()).kev ?? {};
  const delimiters = { ...KEV_FALLBACK.delimiters, ...(kevCfg.delimiters ?? {}) };
  const limits = {
    max_state_tokens: kevCfg.max_state_tokens ?? KEV_FALLBACK.max_state_tokens,
    max_branch_tokens: kevCfg.max_branch_tokens ?? KEV_FALLBACK.max_branch_tokens,
    max_options: kevCfg.max_options ?? KEV_FALLBACK.max_options,
  };

  const tok = await tf.AutoTokenizer.from_pretrained(model, {
    revision,
    progress_callback: report("tokenizer"),
  });
  const rawEncode = (text) =>
    Array.from(tok(text, { add_special_tokens: false }).input_ids.data, Number);
  // Delimiter ids are derived from the raw strings; caller text is escaped so
  // it can never produce a delimiter.
  const encode = (text) => rawEncode(escapeKevText(text));
  // Derive delimiter ids through the tokenizer (robust to id changes).
  const delims = Object.fromEntries(
    Object.entries(delimiters).map(([k, t]) => {
      const ids = rawEncode(t);
      if (ids.length !== 1) throw new Error(`kev: delimiter ${t} is not a single token`);
      return [k, ids[0]];
    }),
  );

  const qfile = modelFile ?? (resolvedDtype === "q4f16" ? "model_q4f16.onnx" : "model_q4.onnx");
  const url = modelUrl ?? hfUrl(model, revision, "onnx", qfile);
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

  return {
    info: {
      family: "kev",
      model,
      revision,
      device: activeDevice,
      dtype: resolvedDtype,
      tasks: ["choice", "noul", "score"],
      limits,
    },
    async decide(state, questions) {
      const t0 = Date.now();
      const normalized = normalizeQuestions(questions);
      for (const [i, q] of normalized.entries()) {
        if (q.options.length > limits.max_options) {
          throw new TypeError(`questions[${i}]: kev supports at most ${limits.max_options} options`);
        }
      }
      const stateText = renderKevContent(state);
      const { tokens, groups, truncated } = packKevSequence({
        stateText, questions: normalized, encode, delims, limits,
      });
      if (groups.length !== normalized.length) {
        throw new Error("kev: packed fewer questions than requested");
      }

      const n = tokens.length;
      const inputIds = new ortMod.Tensor("int64", BigInt64Array.from(tokens, BigInt), [1, n]);
      const attentionMask = new ortMod.Tensor("int64", new BigInt64Array(n).fill(1n), [1, n]);
      const tRun = Date.now();
      const out = await session.run({ input_ids: inputIds, attention_mask: attentionMask });
      const runMs = Date.now() - tRun;
      const outLogits = out.logits ?? out[Object.keys(out)[0]];
      if (outLogits.dims.length !== 2 || outLogits.dims[0] !== 1 || outLogits.dims[1] !== n) {
        throw new Error("kev: invalid model logits shape");
      }
      const scores = Array.from(outLogits.data, Number);
      // One flat score list; groups map each question to its option positions.
      const flat = [];
      const flatGroups = groups.map((ends) => ends.map((pos) => {
        flat.push(scores[pos]);
        return flat.length - 1;
      }));
      const answers = answersFromScores(flat, {
        questions: normalized,
        groups: flatGroups,
      });

      return {
        answers,
        truncated,
        length: n,
        prompts: normalized.map((q) => ({
          instructions: q.instructions,
          options: q.options,
        })),
        timings: { totalMs: Date.now() - t0, runMs },
      };
    },
    async dispose() {
      try { await session.release?.(); } catch { /* optional */ }
    },
  };
}