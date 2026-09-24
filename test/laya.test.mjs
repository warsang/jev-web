/**
 * Laya family: sequence construction, calibration buckets, answer decoding
 * and the end-to-end decider wiring (with injected ORT/tokenizer fakes).
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  buildLayaSequence,
  collateLayaItems,
  layaAnswersFromLogits,
  renderLayaOptions,
  layaTempBucket,
  clampTemperature,
  createLayaDecider,
  loadLayaTokenizer,
  LAYA_QTYPE,
} from "../src/laya.mjs";

const fakeEncode = (text) => [...String(text)].map((c) => (c.charCodeAt(0) % 90) + 10);
const ids = { cls: 1, sep: 2, mask: 3 };

const choice6 = {
  type: "choice",
  instructions: "What is this?",
  options: ["a", "b", "c", "d", "e", "f"],
};
const noul = { type: "noul", instructions: "Is it malicious?", options: ["no", "yes"] };
const score3 = { type: "score", instructions: "How bad?", options: ["low", "mid", "high"] };

test("renderLayaOptions mirrors the reference runtime", () => {
  assert.deepEqual(renderLayaOptions(choice6), ["a", "b", "c", "d", "e", "f"]);
  assert.deepEqual(renderLayaOptions(score3), ["level 0: low", "level 1: mid", "level 2: high"]);
  assert.deepEqual(renderLayaOptions(noul), [
    "false: no, the statement does not hold",
    "true: yes, the statement holds",
  ]);
});

test("buildLayaSequence lays out [CLS] head [SEP] options [SEP] state [SEP]", () => {
  const { ids: seq, markers, qtype } = buildLayaSequence({
    encode: fakeEncode, ids, state: "STATE", question: choice6,
  });
  assert.equal(qtype, LAYA_QTYPE.choice);
  assert.equal(seq[0], ids.cls);
  assert.equal(seq.at(-1), ids.sep);
  assert.equal(markers.length, 6);
  for (const m of markers) assert.equal(seq[m], ids.mask, `marker ${m} is [MASK]`);
  for (let i = 1; i < markers.length; i++) assert.ok(markers[i] > markers[i - 1]);
  // state tokens sit between the options' [SEP] and the trailing [SEP]
  const stateStart = seq.length - fakeEncode("STATE").length - 1;
  assert.deepEqual(seq.slice(stateStart, stateStart + fakeEncode("STATE").length), fakeEncode("STATE"));
});

test("buildLayaSequence enforces maxPrefixes and truncates to maxLen", () => {
  assert.throws(() => buildLayaSequence({
    encode: fakeEncode, ids, state: "s",
    question: { type: "choice", instructions: "x", options: ["1", "2", "3", "4", "5", "6", "7"] },
  }), /max_prefixes/);

  const { ids: seq, markers } = buildLayaSequence({
    encode: fakeEncode, ids, state: "A".repeat(200), question: noul, maxLen: 64,
  });
  assert.equal(seq.length, 64);
  assert.ok(markers.every((m) => m < 64));
});

test("temperature buckets and clamping match the Python runtime", () => {
  assert.equal(layaTempBucket(0, 6), "choice:6-10");
  assert.equal(layaTempBucket(2, 2), "noul:2");
  assert.equal(layaTempBucket(1, 3), "score:3-5");
  assert.equal(layaTempBucket(0, 2), "choice:2");
  assert.equal(clampTemperature(0.1006), 0.5);
  assert.equal(clampTemperature(9), 5);
  assert.equal(clampTemperature(Number.NaN), 1);
  assert.equal(clampTemperature(1.5), 1.5);
});

test("layaAnswersFromLogits decodes choice/noul with per-bucket temperature", () => {
  // row 0: choice6 logits -> option index 4 wins
  // row 1: noul logits [0, 2]
  const answers = layaAnswersFromLogits({
    logits: [0, 0, 0, 0, 5, 0, 0, 2],
    questions: [choice6, noul],
    markerCounts: [6, 2],
  });
  assert.equal(answers[0].type, "choice");
  assert.equal(answers[0].choice, "e");
  assert.ok(answers[0].confidence > 0.85);
  const yes = 1 / (1 + Math.exp(-2 / 1.983399510383606));
  assert.ok(Math.abs(answers[1].noul - yes) < 1e-9);
  assert.equal(answers[1].type, "noul");
  // equal logits -> 0.5 regardless of temperature
  const flat = layaAnswersFromLogits({ logits: [1, 1], questions: [noul], markerCounts: [2] });
  assert.equal(flat[0].noul, 0.5);
});

test("collateLayaItems pads to max length/markers and sets qtype", () => {
  const a = buildLayaSequence({ encode: fakeEncode, ids, state: "s1", question: choice6 });
  const b = buildLayaSequence({ encode: fakeEncode, ids, state: "s2", question: noul });
  const c = collateLayaItems([a, b], 0);
  assert.equal(c.n, 2);
  assert.equal(c.L, Math.max(a.ids.length, b.ids.length));
  assert.equal(c.M, 6);
  assert.equal(c.qtype[0], 0n);
  assert.equal(c.qtype[1], 2n);
  assert.equal(c.markerMask[1 * c.M + 2], 0, "unused marker slots masked off");
});

// --------------------------------------------------------------------------
// End-to-end with injected fakes
// --------------------------------------------------------------------------

function fakeFetch() {
  const files = {
    "tokenizer.json": { model: { type: "WordPiece", vocab: {} } },
    "tokenizer_config.json": { tokenizer_class: "PreTrainedTokenizerFast" },
  };
  return async (url) => {
    if (!String(url).includes("/v1/")) return { ok: false, status: 404, json: async () => ({}) };
    const name = String(url).split("/").pop();
    const body = files[name];
    if (!body) return { ok: false, status: 404, json: async () => ({}) };
    return {
      ok: true,
      headers: { get: () => null },
      body: null,
      json: async () => body,
    };
  };
}

class FakePreTrainedTokenizer {
  constructor(json, cfg) {
    // transformers.js tokenizers are callable objects; mimic that.
    const fn = (text) => ({ input_ids: { data: BigInt64Array.from(fakeEncode(text), BigInt) } });
    Object.assign(fn, {
      json, cfg,
      cls_token_id: 1, sep_token_id: 2, mask_token_id: 3, pad_token_id: 0,
      decode: (ids) => `decoded:${Array.from(ids).join(" ")}`,
    });
    return fn;
  }
}

function fakeTokenizer() {
  const tok = (text) => ({
    input_ids: { data: BigInt64Array.from(fakeEncode(text), BigInt) },
  });
  tok.cls_token_id = 1;
  tok.sep_token_id = 2;
  tok.mask_token_id = 3;
  tok.pad_token_id = 0;
  tok.decode = (ids) => `decoded:${Array.from(ids).join(" ")}`;
  return tok;
}

function fakeOrt() {
  class Tensor {
    constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
  }
  const calls = { encoder: 0, head: 0, heads: [] };
  const encoder = {
    run: async (inputs) => {
      calls.encoder++;
      const n = inputs.input_ids.dims[0];
      const L = inputs.input_ids.dims[1];
      return { hidden: new Tensor("float32", new Float32Array(n * L * 4), [n, L, 4]) };
    },
    release: async () => {},
  };
  const head = {
    run: async (inputs) => {
      calls.head++;
      calls.heads.push(inputs);
      const n = inputs.marker_mask.dims[0];
      const M = inputs.marker_mask.dims[1];
      const data = new Float32Array(n * M).fill(-100);
      for (let i = 0; i < n; i++) {
        let k = 0;
        for (let j = 0; j < M; j++) if (inputs.marker_mask.data[i * M + j]) k++;
        // deterministic pattern: last option wins for choice rows, true for noul
        if (inputs.qtype.data[i] === 0n) data[i * M + (k - 1)] = 4;
        else { data[i * M + 0] = 0; data[i * M + 1] = 5; }
      }
      return {
        logits: new Tensor("float32", data, [n, M]),
        act_logits: new Tensor("float32", new Float32Array(n * 2), [n, 2]),
      };
    },
    release: async () => {},
  };
  return {
    env: { wasm: {} },
    Tensor,
    InferenceSession: {
      create: async (url) => (String(url).includes("encoder") ? encoder : head),
    },
    __calls: calls,
  };
}

test("loadLayaTokenizer fetches the v1 JSONs and instantiates the named class", async () => {
  const tok = await loadLayaTokenizer({
    urlBase: "https://example.test/v1/",
    fetchImpl: fakeFetch(),
    transformers: { PreTrainedTokenizer: FakePreTrainedTokenizer },
  });
  assert.equal(typeof tok, "function", "tokenizer is callable");
  assert.equal(tok.cfg.tokenizer_class, "PreTrainedTokenizerFast");

  await assert.rejects(
    loadLayaTokenizer({ urlBase: "https://example.test/missing/", fetchImpl: fakeFetch() }),
    /HTTP 404/,
  );
});

test("createLayaDecider wires tokenizer + encoder + head and decodes answers", async () => {
  const ort = fakeOrt();
  const transformers = {
    AutoTokenizer: { from_pretrained: async () => fakeTokenizer() },
    PreTrainedTokenizer: FakePreTrainedTokenizer,
  };
  // Fake ORT: the decider passes model bytes to InferenceSession.create (the
  // default factory now fetches + caches weights itself), so inject a factory.
  const decider = await createLayaDecider({
    ort, transformers, device: "wasm", fetchImpl: fakeFetch(),
    sessionFactory: async (url, opts) => ort.InferenceSession.create(url, opts),
  });
  assert.equal(decider.info.family, "laya");
  assert.equal(decider.info.device, "wasm");

  const { answers, length, prompts } = await decider.decide("driver state text", [choice6, noul]);
  assert.equal(ort.__calls.encoder, 1);
  assert.equal(ort.__calls.head, 2, "batch-1 head runs once per question");
  assert.equal(answers.length, 2);
  assert.equal(answers[0].choice, "f");
  assert.ok(answers[1].noul > 0.9);
  assert.ok(length > 0);
  assert.equal(prompts.length, 2, "one decoded prompt per question");
  assert.ok(prompts[0].startsWith("decoded:"));

  // the encoder ran batched; each head call got its own row's tensors
  const [h0, h1] = ort.__calls.heads;
  assert.deepEqual(h0.marker_pos.dims, [1, 6]);
  assert.deepEqual(h1.marker_pos.dims, [1, 2]);
  assert.deepEqual(h0.hidden.dims, [1, h0.attention_mask.dims[1], 4]);
  assert.equal(h0.qtype.data[0], 0n);
  assert.equal(h1.qtype.data[0], 2n);
});

// --- external data (browser weights split) ---
import { fetchOnnxExternalData } from "../src/laya.mjs";

test("fetchOnnxExternalData streams the .data sibling with progress", async () => {
  const bytes = new Uint8Array([1, 2, 3, 4, 5]);
  const progress = [];
  const fakeFetch = async (url) => {
    assert.match(url, /encoder_q8\.onnx\.data$/);
    return {
      ok: true,
      headers: { get: (h) => (h === "content-length" ? String(bytes.length) : null) },
      body: {
        getReader() {
          let sent = false;
          return {
            async read() {
              if (sent) return { done: true, value: undefined };
              sent = true;
              return { done: false, value: bytes };
            },
          };
        },
      },
    };
  };
  const ext = await fetchOnnxExternalData("https://hf.test/v1/encoder_q8.onnx", {
    fetchImpl: fakeFetch,
    onProgress: (p) => progress.push(p),
  });
  assert.equal(ext.path, "encoder_q8.onnx.data");
  assert.deepEqual([...ext.data], [1, 2, 3, 4, 5]);
  assert.ok(progress.some((p) => p.loaded === 5 && p.total === 5));
});

test("fetchOnnxExternalData returns null when there is no .data file", async () => {
  const ext = await fetchOnnxExternalData("https://hf.test/v1/model.onnx", {
    fetchImpl: async () => ({ ok: false, headers: { get: () => null } }),
  });
  assert.equal(ext, null);
});

test("fetchCachedBytes serves the second request from the Cache API", async () => {
  const { fetchCachedBytes } = await import("../src/laya.mjs");
  const store = new Map();
  const fakeCaches = {
    async match(url) {
      const hit = store.get(url);
      return hit ? new Response(hit, { headers: { "content-length": String(hit.length) } }) : undefined;
    },
    async put(url, response) {
      store.set(url, new Uint8Array(await response.arrayBuffer()));
    },
  };
  const prevCaches = globalThis.caches;
  globalThis.caches = fakeCaches;
  try {
    let fetches = 0;
    const phases = [];
    const bytes = new Uint8Array([1, 2, 3, 4]);
    const fakeFetch = async () => {
      fetches++;
      return {
        ok: true,
        headers: { get: (h) => (h === "content-length" ? String(bytes.length) : null) },
        body: {
          getReader() {
            let sent = false;
            return { async read() { if (sent) return { done: true }; sent = true; return { done: false, value: bytes }; } };
          },
        },
      };
    };
    const url = "https://hf.test/v1/encoder_q8.onnx.data";
    const first = await fetchCachedBytes(url, { fetchImpl: fakeFetch, onProgress: (p) => phases.push(p.phase) });
    assert.deepEqual([...first], [1, 2, 3, 4]);
    assert.equal(fetches, 1);
    const second = await fetchCachedBytes(url, { fetchImpl: fakeFetch, onProgress: (p) => phases.push(p.phase) });
    assert.deepEqual([...second], [1, 2, 3, 4]);
    assert.equal(fetches, 1, "second call must not hit the network");
    assert.ok(phases.includes("download"));
    assert.ok(phases.includes("cache"));
    await fetchCachedBytes(url, { fetchImpl: fakeFetch, force: true });
    assert.equal(fetches, 2, "force bypasses the cache");
  } finally {
    if (prevCaches === undefined) delete globalThis.caches;
    else globalThis.caches = prevCaches;
  }
});
