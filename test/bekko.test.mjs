/**
 * Bekko System One v0 family: request rendering, token budgeting, feed
 * collation, answer decoding, plus end-to-end decider wiring (with injected
 * ORT/tokenizer/manifest fakes). Mirrors the reference browser runtime
 * (hotchpotch/bekko-system-one, browser/src/core.js + decision.js).
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  renderBekkoContent,
  renderBekkoRequest,
  tokenizeBekkoRequest,
  collateBekkoDocs,
  bekkoSoftmax,
  bekkoAnswerFromLogits,
  createBekkoDecider,
  BEKKO_DEFAULT_MODEL,
  BEKKO_DEFAULT_REVISION,
  BEKKO_400M_MODEL,
  BEKKO_VARIANTS,
  BEKKO_NOUL_DESCRIPTIONS,
} from "../src/bekko.mjs";

import { normalizeQuestions } from "../src/questions.mjs";

const MANIFEST = {
  query_length: 4096,
  document_length: 16,
  cls_token_id: 50281,
  sep_token_id: 50282,
  pad_token_id: 50283,
  tasks: ["choice", "noul", "score"],
  model_file: "model.onnx",
};

// char-level stand-in: every character is one token.
const fakeEncode = (text) => [...String(text)].map((c) => (c.charCodeAt(0) % 90) + 10);

const choiceQ = {
  type: "choice",
  instructions: "Pick a team.",
  options: ["billing", "sales"],
  criteria: { billing: "money stuff", sales: "selling stuff" },
};
const noulQ = normalizeQuestions([
  { type: "noul", instructions: "Is it urgent?" },
])[0];
const scoreQ = { type: "score", instructions: "How bad?", options: ["low", "high"] };

test("renderBekkoContent flattens strings and JSON-encodes objects", () => {
  assert.equal(renderBekkoContent("  hi  "), "hi");
  assert.equal(renderBekkoContent({ a: 1 }), JSON.stringify({ a: 1 }));
});

test("renderBekkoRequest builds choice candidates with descriptions", () => {
  const r = renderBekkoRequest(choiceQ, "some state");
  assert.equal(r.task, "choice");
  assert.equal(r.instruction, "Pick a team.");
  assert.equal(r.state, JSON.stringify("some state"));
  assert.deepEqual(r.candidates.map((c) => c.id), ["billing", "sales"]);
  assert.equal(r.candidates[0].text, "Candidate: billing: money stuff");
  assert.deepEqual(r.labels, ["billing", "sales"]);
});

test("renderBekkoRequest builds noul candidates and embeds meanings in state", () => {
  const r = renderBekkoRequest(noulQ, "some state");
  assert.equal(r.task, "noul");
  assert.deepEqual(r.candidates.map((c) => c.id), ["true", "false"]);
  assert.ok(r.candidates[0].text.startsWith("Candidate: true: "));
  const state = JSON.parse(r.state);
  assert.equal(state.noul.yes, BEKKO_NOUL_DESCRIPTIONS.true);
  assert.equal(state.noul.no, BEKKO_NOUL_DESCRIPTIONS.false);
  assert.equal(state.state, "some state");
});

test("renderBekkoRequest builds score candidates with numeric values", () => {
  const r = renderBekkoRequest(scoreQ, "s");
  assert.equal(r.task, "score");
  assert.deepEqual(r.candidates.map((c) => c.value), [0, 1]);
  assert.equal(r.candidates[1].text, "Candidate: 1: high");
  assert.deepEqual(r.labels, ["0", "1"]);
});

test("tokenizeBekkoRequest splits the query budget and marks truncation", () => {
  const r = renderBekkoRequest(choiceQ, "state");
  const t = tokenizeBekkoRequest(r, fakeEncode, MANIFEST);
  assert.equal(t.prefixIds[0], MANIFEST.cls_token_id);
  assert.equal(t.prefixIds[t.prefixIds.length - 1], MANIFEST.sep_token_id);
  assert.equal(t.docIds.length, 2);
  assert.ok(t.docIds.every((d) => d[d.length - 1] === MANIFEST.sep_token_id));
  assert.ok(t.docIds.every((d) => d.length <= MANIFEST.document_length));
  assert.equal(t.truncated, false);

  const long = renderBekkoRequest({ ...choiceQ, instructions: "x".repeat(5000) }, "y".repeat(5000));
  const t2 = tokenizeBekkoRequest(long, fakeEncode, MANIFEST);
  assert.equal(t2.truncated, true, "over-budget instruction/state truncate");
  assert.ok(t2.prefixIds.length <= MANIFEST.query_length);
});

test("collateBekkoDocs builds the five model feeds", () => {
  const r = renderBekkoRequest(choiceQ, "state");
  const t = tokenizeBekkoRequest(r, fakeEncode, MANIFEST);
  class Tensor {
    constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
  }
  const feeds = collateBekkoDocs(t.prefixIds, t.docIds, MANIFEST, { Tensor });
  assert.deepEqual(feeds.prefix_ids.dims, [1, t.prefixIds.length]);
  assert.equal(feeds.prefix_ids.type, "int64");
  assert.equal(feeds.prefix_mask.type, "bool");
  assert.deepEqual(feeds.doc_ids.dims[0], 2);
  assert.equal(feeds.doc_ids.type, "int64");
  assert.equal(feeds.owners.dims[0], 2);
  assert.ok([...feeds.owners.data].every((v) => v === 0n), "every doc owned by prefix 0");
  // padding uses the manifest pad token where the mask is 0
  const w = feeds.doc_ids.dims[1];
  feeds.doc_ids.data.forEach((v, idx) => {
    if (feeds.doc_mask.data[idx] === 0) assert.equal(v, BigInt(MANIFEST.pad_token_id));
  });
});

test("bekkoSoftmax normalises candidate logits", () => {
  const p = bekkoSoftmax([2, 1, 0]);
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.ok(p[0] > p[1] && p[1] > p[2]);
});

test("bekkoAnswerFromLogits decodes choice, noul and score", () => {
  const rc = renderBekkoRequest(choiceQ, "s");
  const a = bekkoAnswerFromLogits({ logits: [3, 1], task: "choice", request: rc });
  assert.equal(a.type, "choice");
  assert.equal(a.choice, "billing");
  assert.equal(a.index, 0);
  assert.ok(Math.abs(a.probabilities.billing + a.probabilities.sales - 1) < 1e-12);

  const rn = renderBekkoRequest(noulQ, "s");
  const n = bekkoAnswerFromLogits({ logits: [1.8, 0.2], task: "noul", request: rn });
  assert.equal(n.type, "noul");
  assert.ok(n.noul > 0.8, "true wins");
  assert.deepEqual(Object.keys(n.probabilities), ["no", "yes"]);

  const rs = renderBekkoRequest(scoreQ, "s");
  const s = bekkoAnswerFromLogits({ logits: [0, 4], task: "score", request: rs });
  assert.equal(s.type, "score");
  assert.ok(s.score > 0.9 && s.score <= 1);
  assert.equal(s.level, 1);
  assert.ok(s.normalizedScore >= 0 && s.normalizedScore <= 1);
});

function fakeOrt() {
  class Tensor {
    constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
  }
  const calls = [];
  const session = {
    run: async (inputs) => {
      calls.push(inputs);
      const n = inputs.doc_ids.dims[0];
      const T = 3;
      // rigged: candidate 0 wins every question, task column = offset
      const data = new Float32Array(n * T);
      // rigged: candidate 0 wins every question; the hot column follows the
      // call order (choice=0, noul=1, score=2).
      data[(calls.length - 1) % T] = 5;
      return { logits: new Tensor("float32", data, [n, T]) };
    },
    release: async () => {},
  };
  return {
    env: { wasm: {} },
    Tensor,
    InferenceSession: { create: async () => session },
    __calls: calls,
  };
}

function fakeTransformers() {
  const tok = (text) => ({
    input_ids: { data: fakeEncode(text) },
  });
  return { AutoTokenizer: { from_pretrained: async () => tok } };
}

function fakeFetch(manifest) {
  return async (url) => {
    if (String(url).endsWith("manifest.json")) {
      return { ok: true, json: async () => manifest };
    }
    throw new Error(`unexpected fetch ${url}`);
  };
}

test("createBekkoDecider wires manifest + session and decodes all three kinds", async () => {
  const ort = fakeOrt();
  const decider = await createBekkoDecider({
    ort,
    transformers: fakeTransformers(),
    fetchImpl: fakeFetch(MANIFEST),
    device: "wasm",
    sessionFactory: async (url, opts) => ort.InferenceSession.create(url, opts),
  });
  assert.equal(decider.info.family, "bekko");
  assert.equal(decider.info.model, BEKKO_DEFAULT_MODEL);
  assert.equal(decider.info.revision, BEKKO_DEFAULT_REVISION);
  assert.deepEqual(decider.info.tasks, ["choice", "noul", "score"]);

  const { answers, truncated, prompts } = await decider.decide("driver state", [choiceQ, noulQ, scoreQ]);
  assert.equal(answers.length, 3);
  assert.equal(truncated, false);
  assert.equal(prompts.length, 3);

  const inputs = ort.__calls[0];
  assert.deepEqual(Object.keys(inputs).sort(),
    ["doc_ids", "doc_mask", "owners", "prefix_ids", "prefix_mask"]);
  assert.equal(inputs.prefix_ids.dims[0], 1, "one shared prefix");

  assert.equal(answers[0].type, "choice");
  assert.equal(answers[1].type, "noul");
  assert.ok(answers[1].noul > 0.99, "true wins the rigged logits");
  assert.equal(answers[2].type, "score");
  assert.ok(answers[2].score < 0.1, "level 0 wins the rigged logits");

  await decider.dispose();
});

test("createBekkoDecider variant shortcut selects 68m/400m", async () => {
  const decider = await createBekkoDecider({
    variant: "400m",
    ort: fakeOrt(),
    transformers: fakeTransformers(),
    fetchImpl: fakeFetch(MANIFEST),
    device: "wasm",
    sessionFactory: async (url, opts) => fakeOrt().InferenceSession.create(url, opts),
  });
  assert.equal(decider.info.model, BEKKO_400M_MODEL);
  assert.equal(decider.info.revision, BEKKO_VARIANTS["400m"].revision);
  await decider.dispose();

  await assert.rejects(
    createBekkoDecider({ variant: "nope", ort: fakeOrt() }),
    /unknown variant/,
  );
});

test("createBekkoDecider rejects bad questions before touching the network", async () => {
  const decider = await createBekkoDecider({
    ort: fakeOrt(),
    transformers: fakeTransformers(),
    fetchImpl: fakeFetch(MANIFEST),
    device: "wasm",
    sessionFactory: async (url, opts) => fakeOrt().InferenceSession.create(url, opts),
  });
  await assert.rejects(
    decider.decide("s", [{ type: "choice", instructions: "x", options: ["only"] }]),
    /at least 2 options/,
  );
  await decider.dispose();
});