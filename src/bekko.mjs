/**
 * bekko.mjs — Bekko System One v0 family (Ettin reranker cross-encoder).
 *
 * Mirrors the reference browser runtime (hotchpotch/bekko-system-one, MIT,
 * browser/src/core.js + browser/src/decision.js):
 *
 *   query:      [cls] Instruction: {instruction}\nState: {state} [sep]
 *               (instruction and state split the query budget 50/50, with
 *               unused capacity spilling over to the other side)
 *   candidates: "Candidate: {id}: {description}" (+ [sep]), one per option
 *   graph:      prefix_ids, prefix_mask, doc_ids, doc_mask, owners
 *               -> logits [N, T] (one column per task: choice, noul, score)
 *   decode:     softmax over the task column's candidate logits.
 *
 * The browser export used here is hotchpotch/bekko-system-one-v0-17m
 * (onnx_browser/, pinned by revision) — 29 MB, the smallest of the family.
 * Larger siblings: v0-68m and v0-400m.
 */

import { normalizeQuestions } from "./questions.mjs";
import { fetchCachedBytes } from "./laya.mjs";

export const BEKKO_DEFAULT_MODEL = "hotchpotch/bekko-system-one-v0-17m";
export const BEKKO_DEFAULT_REVISION = "b886a1f9b91f4e8368d7830080d52d955e2c8dfa";
export const BEKKO_DEFAULT_SUBDIR = "onnx_browser";
export const BEKKO_68M_MODEL = "hotchpotch/bekko-system-one-v0-68m";
export const BEKKO_68M_REVISION = "ab7685f23e5edbc1acb12ced4f2c4e12591efa69";
export const BEKKO_400M_MODEL = "hotchpotch/bekko-system-one-v0-400m";
export const BEKKO_400M_REVISION = "1960df5602bd93cc8d336fdebd9fb68a30926e13";

/** Named size variants with their pinned revisions and download sizes. */
export const BEKKO_VARIANTS = {
  "17m": {
    model: BEKKO_DEFAULT_MODEL,
    revision: BEKKO_DEFAULT_REVISION,
    bytes: 29_000_000,
    description: "17m, 29 MB browser ONNX",
  },
  "68m": {
    model: BEKKO_68M_MODEL,
    revision: BEKKO_68M_REVISION,
    bytes: 196_342_909,
    description: "68m, 196 MB browser ONNX",
  },
  "400m": {
    model: BEKKO_400M_MODEL,
    revision: BEKKO_400M_REVISION,
    bytes: 1_426_199_724,
    description: "400m, 1.43 GB browser ONNX",
  },
};

// Candidate batching heuristic from the reference runtime: bound the
// candidate-attention growth per forward; not a total RAM limit.
export const BEKKO_ATTENTION_BUDGET = 1_048_576;

export const BEKKO_DEFAULT_YES = "Yes, the condition in the question holds.";
export const BEKKO_DEFAULT_NO = "No, the condition in the question does not hold.";
// Rendered on both sides of a noul so the two candidates read like the
// reference runtime's defaults.
export const BEKKO_NOUL_DESCRIPTIONS = { true: BEKKO_DEFAULT_YES, false: BEKKO_DEFAULT_NO };

/** Flatten a state or instruction into text, stably. */
export function renderBekkoContent(content) {
  if (typeof content === "string") return content.trim();
  return JSON.stringify(content);
}

/**
 * Render one normalized question to a Bekko request (mirrors decision.js
 * renderDecision): the task, the instruction, the JSON-encoded state and the
 * candidate list. Also returns the metadata jev-web needs to map the answer
 * back: `labels` (answer keys in candidate order) and `kind`.
 */
export function renderBekkoRequest(question, stateText) {
  const instruction = renderBekkoContent(question.instructions);
  const kind = question.type;
  if (kind === "choice") {
    const criteria = question.criteria ?? {};
    const candidates = question.options.map((label) => {
      const desc = String(criteria[label] ?? label);
      return { id: label, text: `Candidate: ${label}: ${desc}` };
    });
    return {
      task: "choice",
      instruction,
      state: JSON.stringify(stateText),
      layout: "instruction_state",
      candidates,
      labels: [...question.options],
      kind,
    };
  }
  if (kind === "noul") {
    const criteria = question.criteria ?? {};
    const yesDesc = String(criteria.true ?? criteria.yes ?? BEKKO_NOUL_DESCRIPTIONS.true);
    const noDesc = String(criteria.false ?? criteria.no ?? BEKKO_NOUL_DESCRIPTIONS.false);
    return {
      task: "noul",
      instruction,
      state: JSON.stringify({ noul: { yes: yesDesc, no: noDesc }, state: stateText }),
      layout: "instruction_state",
      candidates: [
        { id: "true", text: `Candidate: true: ${yesDesc}` },
        { id: "false", text: `Candidate: false: ${noDesc}` },
      ],
      // answer keys parallel to candidates: candidate 0 ("true") is the
      // question's "yes" option, candidate 1 ("false") is the "no" option.
      labels: [question.options[1], question.options[0]],
      kind,
    };
  }
  if (kind === "score") {
    const candidates = question.options.map((rubric, i) => ({
      id: String(i),
      text: `Candidate: ${i}: ${rubric}`,
      value: i,
    }));
    return {
      task: "score",
      instruction,
      state: JSON.stringify(stateText),
      layout: "instruction_state",
      candidates,
      labels: question.options.map((_, i) => String(i)),
      kind,
    };
  }
  throw new TypeError(`bekko: unsupported question type ${question.type}`);
}

/**
 * Tokenise a Bekko request (mirrors core.js tokenize): the shared query
 * prefix plus one tokenised candidate per option. Returns the raw id arrays
 * and whether the instruction/state were truncated to the query budget.
 */
export function tokenizeBekkoRequest(request, encode, manifest, system = "") {
  const max = manifest.query_length;
  const systemIds = encode(system.trim() ? `${system}\n\n` : "").slice(0, max);
  const instructionIds = encode(request.instruction).slice(0, max);
  const stateIds = encode(request.state).slice(0, max);
  const im = encode("Instruction: ");
  const sm = encode("State: ");
  const sep = encode("\n");
  const budget = max - 2 - systemIds.length - im.length - sm.length - sep.length;
  if (budget < 2) {
    throw new RangeError("bekko: system prompt leaves fewer than two content tokens");
  }
  let ni = Math.min(instructionIds.length, Math.ceil(budget / 2));
  let ns = Math.min(stateIds.length, Math.floor(budget / 2));
  ni += Math.min(instructionIds.length - ni, budget - ni - ns);
  ns += Math.min(stateIds.length - ns, budget - ni - ns);
  const truncated = ni < instructionIds.length || ns < stateIds.length;
  const ins = [...im, ...instructionIds.slice(0, ni)];
  const ctx = [...sm, ...stateIds.slice(0, ns)];
  const body = request.layout === "state_instruction"
    ? [...ctx, ...sep, ...ins]
    : [...ins, ...sep, ...ctx];
  return {
    prefixIds: [manifest.cls_token_id, ...systemIds, ...body, manifest.sep_token_id],
    docIds: request.candidates.map(
      (c) => [...encode(c.text).slice(0, manifest.document_length - 1), manifest.sep_token_id],
    ),
    truncated,
  };
}

/**
 * Build the ONNX feeds for one question (mirrors core.js feedsFor): the
 * shared prefix plus a batch of candidate documents.
 */
export function collateBekkoDocs(prefixIds, docIds, manifest, ort) {
  const pad = BigInt(manifest.pad_token_id);
  const width = Math.max(...docIds.map((row) => row.length));
  const ids = new BigInt64Array(docIds.length * width).fill(pad);
  const mask = new Uint8Array(ids.length);
  docIds.forEach((row, i) => row.forEach((id, j) => {
    ids[i * width + j] = BigInt(id);
    mask[i * width + j] = 1;
  }));
  const pIds = new BigInt64Array(prefixIds.length);
  const pMask = new Uint8Array(prefixIds.length);
  prefixIds.forEach((id, j) => { pIds[j] = BigInt(id); pMask[j] = 1; });
  return {
    prefix_ids: new ort.Tensor("int64", pIds, [1, prefixIds.length]),
    prefix_mask: new ort.Tensor("bool", pMask, [1, prefixIds.length]),
    doc_ids: new ort.Tensor("int64", ids, [docIds.length, width]),
    doc_mask: new ort.Tensor("bool", mask, [docIds.length, width]),
    owners: new ort.Tensor("int64", new BigInt64Array(docIds.length), [docIds.length]),
  };
}

/** Softmax over candidate logits (mirrors core.js interpret). */
export function bekkoSoftmax(logits) {
  if (!Array.isArray(logits) || logits.length === 0) {
    throw new TypeError("bekkoSoftmax: logits must be a non-empty array");
  }
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(Number(x) - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/**
 * Decode one question's task-column logits into a jev-web answer
 * (mirrors core.js interpret, mapped onto the shared answer shapes).
 *
 * @param {{logits:ArrayLike<number>, task:string, request:object}} spec
 */
export function bekkoAnswerFromLogits({ logits, task, request }) {
  const probs = bekkoSoftmax(Array.from(logits, Number));
  if (probs.length !== request.candidates.length) {
    throw new TypeError(
      `bekko: got ${probs.length} logits for ${request.candidates.length} candidates`);
  }
  const probabilities = Object.fromEntries(request.labels.map((label, i) => [label, probs[i]]));
  const best = probs.indexOf(Math.max(...probs));

  if (task === "choice") {
    return {
      type: "choice",
      choice: request.labels[best],
      index: best,
      probabilities,
      confidence: Math.max(...probs),
    };
  }
  if (task === "noul") {
    const yes = probs[request.candidates.findIndex((c) => c.id === "true")];
    // labels are [yesOption, noOption] (candidate order); re-key in the
    // question's option order for the shared answer contract.
    const [yesLabel, noLabel] = request.labels;
    return {
      type: "noul",
      noul: yes,
      probabilities: { [noLabel]: 1 - yes, [yesLabel]: yes },
      confidence: Math.max(yes, 1 - yes),
    };
  }
  // score: expected value over the numeric candidate values.
  const values = request.candidates.map((c) => c.value);
  const score = probs.reduce((acc, p, i) => acc + p * values[i], 0);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  return {
    type: "score",
    score,
    level: best,
    normalizedScore: hi === lo ? 0 : (score - lo) / (hi - lo),
    probabilities,
    confidence: Math.max(...probs),
  };
}

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

const hfUrl = (model, revision, ...parts) =>
  `https://huggingface.co/${model}/resolve/${revision ? `${revision}/` : ""}${parts.join("/")}`;

/**
 * Create a Bekko System One decider.
 *
 * @param {object} [opts]
 *  model/revision/subdir — pinned browser ONNX export (or your own mirror)
 *  manifestUrl/modelUrl  — override the resolved file URLs
 *  ort                   — injected onnxruntime-web (lazy import otherwise)
 *  transformers          — injected @huggingface/transformers (tokenizer only)
 *  tokenizerModel/tokenizerRevision — override the tokenizer source
 *  wasmPaths             — ORT wasm/loader directory (self-hosted builds)
 *  device                — "auto" | "wasm" | "webgpu"
 *  sessionFactory        — (url, sessionOptions) => Promise<InferenceSession>
 *  attentionBudget       — candidate batching budget (reference: 1M pairs)
 */
export async function createBekkoDecider({
  model = BEKKO_DEFAULT_MODEL,
  revision = BEKKO_DEFAULT_REVISION,
  variant = null,
  subdir = BEKKO_DEFAULT_SUBDIR,
  manifestUrl = null,
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
  system = "",
  attentionBudget = BEKKO_ATTENTION_BUDGET,
  scope = globalThis,
} = {}) {
  // Named size shortcut: {variant: "400m"} sets model/revision.
  if (variant) {
    const v = BEKKO_VARIANTS[variant];
    if (!v) throw new TypeError(`bekko: unknown variant "${variant}" (17m, 68m, 400m)`);
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
  const executionProviders = resolvedDevice === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];

  const report = (phase) => (p) =>
    onProgress?.({ phase, file: p?.file ?? null, loaded: p?.loaded ?? null, total: p?.total ?? null });

  const manifestRes = await fetchImpl(manifestUrl ?? hfUrl(model, revision, subdir, "manifest.json"));
  if (!manifestRes.ok) {
    throw new Error(`bekko: manifest fetch failed (${manifestRes.status})`);
  }
  const manifest = await manifestRes.json();
  const taskColumns = manifest.tasks;

  /**
   * Load the tokenizer. The bekko repos keep tokenizer.json under
   * onnx_browser/, not the repo root, so AutoTokenizer.from_pretrained(repo)
   * finds nothing there. Load the subdir's tokenizer.json directly and
   * construct the base tokenizer (special tokens come from the manifest).
   * An explicit tokenizerModel still goes through from_pretrained.
   */
  async function loadBekkoTokenizer() {
    if (tokenizerModel) {
      return tf.AutoTokenizer.from_pretrained(tokenizerModel, {
        revision: tokenizerRevision,
        progress_callback: report("tokenizer"),
      });
    }
    const tokRes = await fetchImpl(hfUrl(model, revision, subdir, "tokenizer.json"));
    if (!tokRes.ok) {
      throw new Error(`bekko: tokenizer fetch failed (${tokRes.status})`);
    }
    report("tokenizer")({ file: "tokenizer.json" });
    return new tf.PreTrainedTokenizer(await tokRes.json(), {});
  }

  const tok = await loadBekkoTokenizer();
  const encode = (text) =>
    Array.from(tok(text, { add_special_tokens: false }).input_ids.data, Number);

  const url = modelUrl ?? hfUrl(model, revision, subdir, manifest.model_file ?? "model.onnx");
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
      family: "bekko",
      model,
      revision,
      device: activeDevice,
      tasks: [...taskColumns],
      queryLength: manifest.query_length,
      documentLength: manifest.document_length,
    },
    async decide(state, questions) {
      const t0 = now();
      const normalized = normalizeQuestions(questions);
      const stateText = renderBekkoContent(state);
      const answers = [];
      const prompts = [];
      let truncated = false;
      let length = 0;
      let runMs = 0;

      for (const q of normalized) {
        const request = renderBekkoRequest(q, stateText);
        const column = taskColumns.indexOf(request.task);
        if (column < 0) throw new Error(`bekko: task ${request.task} absent from this model`);
        const tokens = tokenizeBekkoRequest(request, encode, manifest, system);
        truncated = truncated || tokens.truncated;
        length += tokens.prefixIds.length + tokens.docIds.reduce((a, d) => a + d.length, 0);
        prompts.push({
          query: request.instruction,
          candidates: request.candidates.map((c) => c.text),
        });

        // Batch candidates by the reference attention budget.
        const docLen = Math.max(...tokens.docIds.map((d) => d.length));
        const attentionPairs = docLen * (tokens.prefixIds.length + docLen);
        const batchSize = Math.max(1, Math.floor(attentionBudget / attentionPairs));
        const logits = [];
        for (let offset = 0; offset < tokens.docIds.length; offset += batchSize) {
          const docs = tokens.docIds.slice(offset, offset + batchSize);
          const feeds = collateBekkoDocs(tokens.prefixIds, docs, manifest, ortMod);
          const tRun = now();
          const out = await session.run(feeds);
          runMs += now() - tRun;
          const outLogits = out.logits ?? out[Object.keys(out)[0]];
          const T = taskColumns.length;
          if (outLogits.dims.length !== 2 || outLogits.dims[0] !== docs.length || outLogits.dims[1] !== T) {
            throw new Error("bekko: invalid model logits shape");
          }
          for (let r = 0; r < docs.length; r++) {
            logits.push(Number(outLogits.data[r * T + column]));
          }
        }
        answers.push(bekkoAnswerFromLogits({ logits, task: request.task, request }));
      }

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