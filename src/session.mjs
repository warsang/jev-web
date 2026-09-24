// jev-web — decider session: tokenizer + model + decide().
//
// transformers.js is an optional peer dependency, imported lazily so the
// package works anywhere and hosts can inject their own copy (tests, custom
// builds). Everything heavy stays out of module scope.

import { normalizeQuestions } from "./questions.mjs";

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
import { buildDecisionInput, typedMarkerIds } from "./encode.mjs";
import { answersFromScores } from "./answers.mjs";
import {
  DEFAULT_MODEL,
  DEFAULT_REVISION,
  loadModelConfig,
  resolveDevice,
  resolveDtype,
  detectWebGPU,
} from "./config.mjs";

export async function createDecider({
  model = DEFAULT_MODEL,
  revision = DEFAULT_REVISION,
  dtype = "auto",
  device = "auto",
  temperature = null,
  maxStateTokens = null,
  onProgress = null,
  transformers = null,
  fetchImpl = fetch,
  scope = globalThis,
  // Directory (or CDN base) holding the onnxruntime-web wasm/loader files.
  // Bundlers usually cannot ship the 26 MB asyncify wasm under static-host
  // file-size limits; hosts that self-host a compressed copy set this to
  // e.g. "/assets/". Defaults to the runtime's own resolution.
  wasmPaths = null,
} = {}) {
  const tf = transformers ?? (await import("@huggingface/transformers"));
  if (wasmPaths) {
    try {
      tf.env.backends.onnx.wasm.wasmPaths = wasmPaths;
    } catch { /* older runtime: ignore */ }
  }
  const remote = await loadModelConfig({ model, revision, fetchImpl });
  const resolvedDevice = resolveDevice({ device, webgpu: await detectWebGPU(scope) });
  const resolvedDtype = resolveDtype(resolvedDevice, dtype);

  const report = (phase) => (p) =>
    onProgress?.({
      phase,
      status: p?.status ?? null,
      file: p?.file ?? null,
      loaded: p?.loaded ?? null,
      total: p?.total ?? null,
      progress: p?.progress ?? null,
    });

  const tokenizer = await tf.AutoTokenizer.from_pretrained(model, {
    revision,
    progress_callback: report("tokenizer"),
  });
  const net = await tf.AutoModel.from_pretrained(model, {
    revision,
    dtype: resolvedDtype,
    device: resolvedDevice,
    progress_callback: report("model"),
  });

  const encode = (text) => Array.from(tokenizer(text, { add_special_tokens: false }).input_ids.data, Number);
  const markers = typedMarkerIds(encode, { markers: remote.markers });

  const effective = {
    temperature: Number.isFinite(temperature) ? temperature : remote.temperature,
    maxStateTokens: Number.isFinite(maxStateTokens) ? maxStateTokens : remote.maxStateTokens,
    maxLen: remote.maxLen,
  };

  const i64 = (values, dims) => new tf.Tensor("int64", BigInt64Array.from(values, BigInt), dims);

  return {
    info: {
      family: "open-jev",
      model,
      revision,
      device: resolvedDevice,
      dtype: resolvedDtype,
      temperature: effective.temperature,
      maxStateTokens: effective.maxStateTokens,
      maxLen: effective.maxLen,
      pool: remote.pool,
      configSource: remote.source,
    },
    async decide(state, questions) {
      const t0 = now();
      const normalized = normalizeQuestions(questions);
      const tokenized = normalized.map((q) => ({
        type: q.type,
        instructionIds: encode(q.instructions),
        optionIds: q.options.map((option) => encode(option)),
      }));
      const stateIds = encode(String(state ?? ""));
      const input = buildDecisionInput({
        stateIds,
        markers,
        questions: tokenized,
        maxStateTokens: effective.maxStateTokens,
        maxLen: effective.maxLen,
      });
      const feeds = {
        input_ids: i64(input.inputIds, [1, input.length]),
        attention_mask: i64(input.inputIds.map(() => 1), [1, input.length]),
        seg: i64(input.seg, [1, input.length]),
        pair_q: i64(input.pairQ, [1, input.pairQ.length]),
        pair_opt: i64(input.pairOpt, [1, input.pairOpt.length]),
      };
      const tInfer = now();
      const out = await net(feeds);
      const inferMs = now() - tInfer;
      const logits = out?.logits;
      if (!logits) throw new Error("model output has no `logits` — not a typed-decision export?");
      const scores = Array.from(logits.to("float32").data, Number);
      const answers = answersFromScores(scores, {
        questions: normalized,
        groups: input.groups,
        temperature: effective.temperature,
      });
      // Exact view of the single fused sequence the graph consumed.
      let prompt;
      try {
        prompt = tokenizer.decode(input.inputIds, { skip_special_tokens: false });
      } catch {
        prompt = input.inputIds.join(" ");
      }
      return {
        answers,
        truncated: input.truncated,
        length: input.length,
        prompts: [prompt],
        timings: { totalMs: Math.round(now() - t0), inferMs: Math.round(inferMs) },
      };
    },
    async decideMany(cases) {
      const out = [];
      for (const c of cases ?? []) out.push(await this.decide(c.state, c.questions));
      return out;
    },
    dispose() {
      net?.dispose?.();
    },
  };
}
