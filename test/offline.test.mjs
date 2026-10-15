/**
 * Offline reload — the package's core promise: "no network calls after the
 * weights are cached", config files included. A review asked for exactly this:
 * "Load each family, go offline, reload, and confirm it still answers. Also
 * cache those config files, so a non-cached request doesn't show up in your
 * own request counter."
 *
 * Node has no CacheStorage, so this file installs a Map-backed stand-in, then
 * drives every family twice:
 *
 *   1. warm-up through a fake fetch that serves the repo's JSON and lets the
 *      Cache API layer store it;
 *   2. recreation with a fetch implementation that throws on ANY call — the
 *      reload must create, decide, and make zero requests.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { createDecider } from "../src/session.mjs";
import { createLayaDecider } from "../src/laya.mjs";
import { createStrandsDecider, loadStrandsCalibration } from "../src/strands.mjs";
import { createBekkoDecider } from "../src/bekko.mjs";
import { createDecision2Decider } from "../src/decision2.mjs";
import { createKevDecider } from "../src/kev.mjs";
import { createVegaDecider } from "../src/vega.mjs";
import { createClefDecider } from "../src/clef.mjs";
import {
  createDecision3Decider,
  renderDecision3Chat,
  DECISION3_MAX_OPTIONS,
} from "../src/decision3.mjs";
import { fetchCachedJson, fetchOnnxExternalData } from "../src/fetch.mjs";

const fixture = (name) => JSON.parse(readFileSync(
  fileURLToPath(new URL(`./fixtures/${name}`, import.meta.url)), "utf8"));
const vegaFixture = fixture("vega-protocol.json");
const clefFixture = fixture("clef-protocol.json");
const d3Fixture = fixture("d3-protocol.json");

// --------------------------------------------------------------------------
// CacheStorage stand-in (Node has none)
// --------------------------------------------------------------------------

const cacheStores = new Map();

globalThis.caches = {
  async open(name) {
    if (!cacheStores.has(name)) cacheStores.set(name, new Map());
    const entries = cacheStores.get(name);
    return {
      async match(url) {
        const bytes = entries.get(String(url));
        return bytes
          ? new Response(bytes.slice(), { headers: { "content-type": "application/octet-stream" } })
          : undefined;
      },
      async put(url, response) {
        entries.set(String(url), new Uint8Array(await response.arrayBuffer()));
      },
    };
  },
};

/**
 * A fake fetch that serves `files` by URL suffix (body `null` = a definite
 * 404), records every call in `calls`, and — with `offline` — refuses to
 * serve anything.
 */
function serve(files, calls, { offline = false } = {}) {
  return async (url) => {
    const key = String(url);
    calls.push(key);
    if (offline) throw new Error(`offline: unexpected fetch ${key}`);
    for (const [suffix, body] of Object.entries(files)) {
      if (!key.endsWith(suffix)) continue;
      if (body === null) return { ok: false, status: 404 };
      return { ok: true, json: async () => body };
    }
    throw new Error(`unexpected fetch: ${key}`);
  };
}

// --------------------------------------------------------------------------
// Shared fakes
// --------------------------------------------------------------------------

class Tensor {
  constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
}

// char-level stand-in tokenizer encoding, one id per character
const charIds = (text) => [...String(text)].map((c) => (c.charCodeAt(0) % 90) + 10);
const rawTokenizer = () => {
  const tok = (text) => ({
    input_ids: { data: BigInt64Array.from(charIds(text), BigInt) },
  });
  tok.decode = (ids) => `decoded:${Array.from(ids).join(" ")}`;
  return tok;
};
const withMeta = (tok, meta = {}) => Object.assign(tok, {
  cls_token_id: 1, sep_token_id: 2, mask_token_id: 3, pad_token_id: 0, ...meta,
});
const release = async () => {};

const STATE = "I was charged twice for the same order.";
const QUESTIONS = [
  { type: "choice", instructions: "Which team handles this?", options: ["billing", "sales"] },
  { type: "noul", instructions: "Is the customer asking for a refund?" },
];

// --------------------------------------------------------------------------
// Per-family recipes: `files` servers the repo's JSON; `make(fetchImpl)`
// builds the decider with the fakes each family's runtime needs.
// --------------------------------------------------------------------------

const RECIPES = [
  {
    id: "open-jev",
    files: {
      "open_jev_config.json": {
        temperature: 1.05,
        markers: ["[STATE]", "[Q]", "[OPT]"],
        max_state_tokens: 256,
        max_len: 512,
      },
    },
    make(fetchImpl) {
      const MARKERS = { "[STATE]": 1, "[Q]": 2, "[OPT]": 3, "[CLS]": 4, "[SEP]": 5 };
      const tok = (text) => ({
        input_ids: {
          data: BigInt64Array.from(
            MARKERS[text] != null ? [MARKERS[text]] : charIds(text), BigInt),
        },
      });
      tok.decode = (ids) => `decoded:${Array.from(ids).join(" ")}`;
      const net = async (feeds) => {
        const pairs = feeds.pair_opt.dims[1];
        const data = new Float32Array(pairs);
        if (pairs) data[0] = 5; // first option of the first question wins
        return { logits: { to: () => ({ data }) } };
      };
      return createDecider({
        transformers: {
          Tensor,
          AutoTokenizer: { from_pretrained: async () => tok },
          AutoModel: { from_pretrained: async () => net },
        },
        fetchImpl,
        scope: {}, // no navigator.gpu -> wasm
      });
    },
  },
  {
    id: "laya",
    files: {
      "v1/tokenizer.json": { model: { type: "WordPiece", vocab: {} } },
      "v1/tokenizer_config.json": { tokenizer_class: "PreTrainedTokenizerFast" },
    },
    make(fetchImpl) {
      const tok = withMeta(rawTokenizer());
      class FakePreTrainedTokenizer { constructor() { return tok; } }
      const encoder = {
        run: async (inputs) => {
          const n = inputs.input_ids.dims[0];
          const L = inputs.input_ids.dims[1];
          return { hidden: new Tensor("float32", new Float32Array(n * L * 4), [n, L, 4]) };
        },
        release,
      };
      const head = {
        run: async (inputs) => {
          const n = inputs.marker_mask.dims[0];
          const M = inputs.marker_mask.dims[1];
          const data = new Float32Array(n * M).fill(-100);
          for (let i = 0; i < n; i++) {
            let k = 0;
            for (let j = 0; j < M; j++) if (inputs.marker_mask.data[i * M + j]) k++;
            if (inputs.qtype.data[i] === 0n) data[i * M + (k - 1)] = 4;
            else { data[i * M] = 0; data[i * M + 1] = 5; }
          }
          return {
            logits: new Tensor("float32", data, [n, M]),
            act_logits: new Tensor("float32", new Float32Array(n * 2), [n, 2]),
          };
        },
        release,
      };
      const ort = { env: { wasm: {} }, Tensor, InferenceSession: {} };
      return createLayaDecider({
        ort,
        transformers: { PreTrainedTokenizer: FakePreTrainedTokenizer },
        device: "wasm",
        fetchImpl,
        sessionFactory: async (url) =>
          (String(url).includes("encoder") ? encoder : head),
      });
    },
  },
  {
    id: "strands-decider",
    files: {
      // The newer name is absent (a definite 404), the legacy one answers —
      // the same shape as the real checkpoint repo, and it exercises the
      // negative 404 memo on reload.
      "strands_decider_config.json": null,
      "hobson_config.json": {
        head_type: "pointer",
        temperature: 1.05,
        temperature_by_kind: { noul: 1.07, choice: 0.56, score: 1.47 },
        ordinal_smoothing: 0.3,
        max_length: 4096,
      },
    },
    make(fetchImpl) {
      const tok = (text, { add_special_tokens = false } = {}) => ({
        input_ids: {
          data: BigInt64Array.from(
            charIds(add_special_tokens ? `<s>${text}` : text), BigInt),
        },
      });
      tok.decode = (ids) => `decoded:${Array.from(ids).join(" ")}`;
      const session = {
        run: async (inputs) => {
          const n = inputs.input_ids.dims[0];
          const K = inputs.option_pos.dims[1];
          const data = new Float32Array(n * K).fill(-100);
          for (let i = 0; i < n; i++) {
            let k = 0;
            for (let j = 0; j < K; j++) if (inputs.option_pos.data[i * K + j] > 0n) k++;
            data[i * K + (k - 1)] = 5; // last option wins every row
          }
          return { logits: new Tensor("float32", data, [n, K]) };
        },
        release,
      };
      return createStrandsDecider({
        ort: { env: { wasm: {} }, Tensor, InferenceSession: {} },
        transformers: { AutoTokenizer: { from_pretrained: async () => tok } },
        device: "wasm",
        fetchImpl,
        sessionFactory: async () => session,
      });
    },
  },
  {
    id: "bekko",
    files: {
      "manifest.json": {
        query_length: 4096,
        document_length: 96,
        cls_token_id: 50281,
        sep_token_id: 50282,
        pad_token_id: 50283,
        tasks: ["choice", "noul", "score"],
        model_file: "model.onnx",
      },
      "tokenizer.json": {},
    },
    make(fetchImpl) {
      const tok = rawTokenizer();
      const session = {
        run: async (inputs) => {
          const n = inputs.doc_ids.dims[0];
          const data = new Float32Array(n * 3);
          data[0] = 5; // first candidate of the first task column
          return { logits: new Tensor("float32", data, [n, 3]) };
        },
        release,
      };
      return createBekkoDecider({
        ort: { env: { wasm: {} }, Tensor, InferenceSession: {} },
        transformers: { PreTrainedTokenizer: function () { return tok; } },
        device: "wasm",
        fetchImpl,
        sessionFactory: async () => session,
      });
    },
  },
  {
    id: "decision2",
    files: { "config.json": { decision2: { score_bias: {} } } },
    make(fetchImpl) {
      const session = {
        run: async (inputs) => {
          const B = inputs.input_ids.dims[0];
          const K = inputs.option_pos.dims[1];
          const data = new Float32Array(B * K);
          for (let b = 0; b < B; b++) data[b * K] = 5; // candidate 0 wins
          return { logits: new Tensor("float32", data, [B, K]) };
        },
        release,
      };
      return createDecision2Decider({
        ort: { env: { wasm: {} }, Tensor, InferenceSession: {} },
        transformers: { AutoTokenizer: { from_pretrained: async () => rawTokenizer() } },
        device: "wasm",
        fetchImpl,
        sessionFactory: async () => session,
      });
    },
  },
  {
    id: "kev",
    files: {
      "config.json": {
        kev: {
          delimiters: {
            state: "<|fim_prefix|>",
            question: "<|fim_middle|>",
            option_start: "<|box_start|>",
            option_end: "<|box_end|>",
            decide: "<|fim_suffix|>",
          },
          delimiter_ids: {
            "<|fim_prefix|>": 101,
            "<|fim_middle|>": 102,
            "<|box_start|>": 103,
            "<|box_end|>": 104,
            "<|fim_suffix|>": 105,
          },
          max_state_tokens: 8192,
          max_branch_tokens: 8192,
          max_options: 255,
        },
      },
    },
    make(fetchImpl) {
      const DELIM_IDS = {
        "<|fim_prefix|>": 101, "<|fim_middle|>": 102, "<|box_start|>": 103,
        "<|box_end|>": 104, "<|fim_suffix|>": 105,
      };
      const tok = (text) => ({
        input_ids: {
          data: BigInt64Array.from(
            DELIM_IDS[String(text)] != null ? [DELIM_IDS[String(text)]] : charIds(text), BigInt),
        },
      });
      const session = {
        run: async (inputs) => ({
          logits: new Tensor("float32", new Float32Array(inputs.input_ids.dims[1]),
            [1, inputs.input_ids.dims[1]]),
        }),
        release,
      };
      return createKevDecider({
        ort: { env: { wasm: {} }, Tensor, InferenceSession: {} },
        transformers: { AutoTokenizer: { from_pretrained: async () => tok } },
        device: "wasm",
        fetchImpl,
        sessionFactory: async () => session,
      });
    },
  },
  {
    id: "vega",
    files: {
      "vega_config.json": {
        ...vegaFixture.config,
        // The char-level stand-in tokenizes ~1 id per character, so the
        // reference 32-token option cap would clip the noul defaults.
        opt_max_tokens: 64,
        calibration: vegaFixture.calibration,
        theta: vegaFixture.thermal.theta,
      },
    },
    make(fetchImpl) {
      const tok = withMeta(rawTokenizer(), { bos_token_id: null });
      const torso = {
        run: async (inputs) => {
          const b = inputs.pool.dims[0];
          const rows = inputs.pool.dims[1];
          const d = 1024;
          const pooled = new Float32Array(b * 2 * rows * d).fill(0.01);
          return { pooled: new Tensor("float32", pooled, [b, 2, rows, d]) };
        },
        release,
      };
      const engine = {
        run: async (inputs) => {
          const b = inputs.qtype.dims[0];
          const K = inputs.opts.dims[1];
          const data = new Float32Array(b * K);
          for (let i = 0; i < b; i++) { data[i * K] = 5; for (let j = 1; j < K; j++) data[i * K + j] = -1; }
          return {
            logits: new Tensor("float32", data, [b, K]),
            kinetic: new Tensor("float32", [0.02, 0.01].slice(0, b), [b]),
            nearest: new Tensor("float32", [0.3, 0.2].slice(0, b), [b]),
            steps_used: new Tensor("int64", BigInt64Array.from([12, 5].slice(0, b), BigInt), [b]),
            grad_norm: new Tensor("float32", [0.1, 0.1].slice(0, b), [b]),
          };
        },
        release,
      };
      return createVegaDecider({
        ort: { env: { wasm: {} }, Tensor, InferenceSession: {} },
        transformers: { AutoTokenizer: { from_pretrained: async () => tok } },
        device: "wasm",
        fetchImpl,
        sessionFactory: async (url) =>
          (String(url).includes("torso") ? torso : engine),
      });
    },
  },
  {
    id: "clef",
    files: {
      "clef_config.json": { max_len: clefFixture.maxLength, head: clefFixture.headConfig },
    },
    make(fetchImpl) {
      const session = {
        run: async (inputs) => {
          const Q = inputs.type_ids.dims[0];
          const S = inputs.opt_mask.dims[0];
          const K = S / Q;
          const data = new Float32Array(S);
          for (let q = 0; q < Q; q++) {
            data[q * K] = 4;
            for (let k = 1; k < K; k++) data[q * K + k] = -1;
          }
          return { logits: new Tensor("float32", data, [1, S]) };
        },
        release,
      };
      return createClefDecider({
        ort: { Tensor },
        transformers: { AutoTokenizer: { from_pretrained: async () => rawTokenizer() } },
        device: "wasm",
        fetchImpl,
        sessionFactory: async () => session,
      });
    },
  },
  {
    id: "decision3",
    files: {
      "decision_config.json": {
        format_id: d3Fixture.formatId,
        codes: d3Fixture.codes,
        temperature: d3Fixture.config.temperature,
        max_length: d3Fixture.config.max_length,
      },
    },
    make(fetchImpl) {
      const tok = rawTokenizer();
      tok.apply_chat_template = (messages) =>
        renderDecision3Chat(messages[0].content, messages[1].content);
      tok.pad_token_id = 0;
      const torso = {
        run: async (inputs) => {
          const B = inputs.input_ids.dims[0];
          return { last: new Tensor("float32", new Float32Array(B * 4).fill(0.5), [B, 4]) };
        },
        release,
      };
      const readout = {
        run: async (inputs) => {
          const B = inputs.last.dims[0];
          const width = DECISION3_MAX_OPTIONS;
          const data = new Float32Array(B * width);
          for (let b = 0; b < B; b++) {
            data[b * width] = 4; // favour code A
            for (let k = 1; k < width; k++) data[b * width + k] = -4;
            for (let k = Number(inputs.counts.data[b]); k < width; k++) data[b * width + k] = -Infinity;
          }
          return { logits: new Tensor("float32", data, [B, width]) };
        },
        release,
      };
      return createDecision3Decider({
        ort: { Tensor },
        transformers: { AutoTokenizer: { from_pretrained: async () => tok } },
        device: "wasm",
        fetchImpl,
        sessionFactory: async (url) =>
          (String(url).includes("readout") ? readout : torso),
      });
    },
  },
];

// --------------------------------------------------------------------------
// The reload cycle, once per family
// --------------------------------------------------------------------------

for (const recipe of RECIPES) {
  test(`offline reload: ${recipe.id} recreates and answers with zero requests`, async () => {
    const warm = [];
    const first = await recipe.make(serve(recipe.files, warm));
    assert.ok(warm.length > 0, "warm-up fetched the repo's files");
    const warmResult = await first.decide(STATE, QUESTIONS);
    assert.equal(warmResult.answers.length, QUESTIONS.length);
    await first.dispose();

    const offline = [];
    const second = await recipe.make(serve(recipe.files, offline, { offline: true }));
    assert.deepEqual(offline, [], "the reload must not touch the network");
    const reloaded = await second.decide(STATE, QUESTIONS);
    assert.equal(reloaded.answers.length, warmResult.answers.length);
    for (const answer of reloaded.answers) {
      assert.equal(typeof answer.confidence, "number");
    }
    await second.dispose();
  });
}

// --------------------------------------------------------------------------
// The cache layer itself
// --------------------------------------------------------------------------

test("fetchCachedJson parses once and serves the reload from the cache", async () => {
  const url = "https://example.test/repo/resolve/rev/config.json";
  const body = { temperature: 1.05 };
  const calls = [];
  const value = await fetchCachedJson(url, { fetchImpl: serve({ "config.json": body }, calls) });
  assert.deepEqual(value, body);
  assert.equal(calls.length, 1);

  const again = await fetchCachedJson(url, {
    fetchImpl: serve({}, [], { offline: true }),
  });
  assert.deepEqual(again, body, "the second read never hit the network");

  const forced = await fetchCachedJson(url, {
    force: true,
    fetchImpl: serve({ "config.json": { temperature: 2 } }, calls),
  });
  assert.deepEqual(forced, { temperature: 2 }, "force re-fetches");
});

test("a definite 404 for an optional file is remembered (no re-probe)", async () => {
  const model = "org/optional-config-repo";
  const revision = "rev1";
  const warm = [];
  const first = await loadStrandsCalibration({
    model, revision,
    fetchImpl: serve({
      "strands_decider_config.json": null,
      "hobson_config.json": { head_type: "pointer", max_length: 4096 },
    }, warm),
  });
  assert.equal(warm.length, 2, "the absent name 404s, the legacy one answers");
  assert.match(first.source, /hobson_config\.json$/);

  const second = await loadStrandsCalibration({
    model, revision,
    fetchImpl: serve({}, [], { offline: true }),
  });
  assert.equal(second.calibration.maxLen, 4096);
  assert.equal(second.source, first.source, "answered from the cache, offline");
});

test("fetchOnnxExternalData remembers 'no external data' so reloads don't re-probe", async () => {
  const url = "https://example.test/repo/resolve/rev/model.onnx";
  const probes = [];
  const fetchImpl = async (u, init) => {
    probes.push(`${init?.method ?? "GET"} ${u}`);
    return { ok: false, status: 404 };
  };
  assert.equal(await fetchOnnxExternalData(url, { fetchImpl }), null);
  assert.deepEqual(probes, [
    `HEAD ${url}.data`,
    `HEAD ${url}_data`,
  ]);

  const again = [];
  assert.equal(await fetchOnnxExternalData(url, {
    fetchImpl: async (u) => { again.push(u); throw new Error("offline"); },
  }), null);
  assert.deepEqual(again, [], "the negative marker answers the probe");
});
