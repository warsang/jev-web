/**
 * Kev decision-model family: delimiter escaping, sequence packing,
 * plus end-to-end decider wiring (with injected ORT/tokenizer/config fakes).
 * Mirrors the reference JS implementation in onnx-community/kev-0.6b-ONNX.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  escapeKevText,
  packKevSequence,
  createKevDecider,
  KEV_DEFAULT_MODEL,
  KEV_DEFAULT_REVISION,
  KEV_4B_MODEL,
  KEV_VARIANTS,
} from "../src/kev.mjs";

import { normalizeQuestions } from "../src/questions.mjs";

const DELIMS = {
  state: "<|fim_prefix|>",
  question: "<|fim_middle|>",
  option_start: "<|box_start|>",
  option_end: "<|box_end|>",
  decide: "<|fim_suffix|>",
};
const DELIM_IDS = {
  "<|fim_prefix|>": 101,
  "<|fim_middle|>": 102,
  "<|box_start|>": 103,
  "<|box_end|>": 104,
  "<|fim_suffix|>": 105,
};
const CONFIG = {
  kev: {
    delimiters: DELIMS,
    delimiter_ids: DELIM_IDS,
    max_state_tokens: 8192,
    max_branch_tokens: 8192,
    max_options: 255,
  },
};

// char-level stand-in, except whole delimiter strings map to one id.
const fakeEncode = (text) => {
  const t = String(text);
  if (DELIM_IDS[t] != null) return [DELIM_IDS[t]];
  return [...t].map((c) => (c.charCodeAt(0) % 90) + 10);
};
// escapeKevText is applied by the real encode path; replicate it here.
const fakeEncodeEscaped = (text) =>
  fakeEncode(String(text).replace(/<\|([A-Za-z0-9_]+)\|>/g, "<¦$1¦>"));

const choiceQ = {
  type: "choice",
  instructions: "Pick a team.",
  options: ["billing", "sales"],
  criteria: { billing: "money stuff" },
};
const noulQ = normalizeQuestions([{ type: "noul", instructions: "Refund?" }])[0];
const scoreQ = { type: "score", instructions: "How bad?", options: ["low", "high"] };

test("escapeKevText neutralizes delimiter-like spans", () => {
  assert.equal(escapeKevText("use <|fim_prefix|> here"), "use <¦fim_prefix¦> here");
  assert.equal(escapeKevText("plain text"), "plain text");
});

test("packKevSequence lays out state + branches and tracks </opt> positions", () => {
  const questions = normalizeQuestions([choiceQ, noulQ]);
  const { tokens, groups, truncated } = packKevSequence({
    stateText: "some state",
    questions,
    encode: fakeEncodeEscaped,
    delims: {
      state: DELIM_IDS["<|fim_prefix|>"],
      question: DELIM_IDS["<|fim_middle|>"],
      option_start: DELIM_IDS["<|box_start|>"],
      option_end: DELIM_IDS["<|box_end|>"],
      decide: DELIM_IDS["<|fim_suffix|>"],
    },
    limits: { max_state_tokens: 8192, max_branch_tokens: 8192, max_options: 255 },
  });
  assert.equal(truncated, false);
  assert.equal(tokens[0], DELIM_IDS["<|fim_prefix|>"], "state delimiter first");
  assert.equal(groups.length, 2);
  assert.equal(groups[0].length, 2, "choice has 2 options");
  assert.equal(groups[1].length, 2, "noul has 2 options");
  for (const [gi, ends] of groups.entries()) {
    for (const pos of ends) {
      assert.equal(tokens[pos], DELIM_IDS["<|box_end|>"], `group ${gi} </opt> marker`);
    }
  }
  // criteria description is baked into the choice option text
  const text = tokens.map(String).join(",");
  assert.ok(tokens.includes(DELIM_IDS["<|fim_suffix|>"]), "each branch ends with <decide>");
});

test("packKevSequence truncates when branches exceed the budget", () => {
  const questions = normalizeQuestions([choiceQ, choiceQ, choiceQ]);
  const { groups, truncated } = packKevSequence({
    stateText: "s",
    questions,
    encode: fakeEncodeEscaped,
    delims: {
      state: DELIM_IDS["<|fim_prefix|>"],
      question: DELIM_IDS["<|fim_middle|>"],
      option_start: DELIM_IDS["<|box_start|>"],
      option_end: DELIM_IDS["<|box_end|>"],
      decide: DELIM_IDS["<|fim_suffix|>"],
    },
    limits: { max_state_tokens: 40, max_branch_tokens: 8192, max_options: 255 },
  });
  assert.equal(truncated, true);
  assert.ok(groups.length < 3, "later questions dropped");
});

function fakeTransformers() {
  const tok = (text) => ({ input_ids: { data: fakeEncode(text) } });
  return { AutoTokenizer: { from_pretrained: async () => tok } };
}

function fakeFetch() {
  const seen = [];
  const fetchImpl = async (url) => {
    seen.push(String(url));
    if (String(url).endsWith("config.json")) {
      return { ok: true, json: async () => CONFIG };
    }
    throw new Error(`unexpected fetch ${url}`);
  };
  fetchImpl.seen = seen;
  return fetchImpl;
}

function fakeOrt(rig) {
  class Tensor {
    constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
  }
  const calls = [];
  const session = {
    run: async (inputs) => {
      calls.push(inputs);
      const n = inputs.input_ids.dims[1];
      const data = new Float32Array(n);
      for (const p of rig.winners) data[p] = 5; // rigged winners
      return { logits: new Tensor("float32", data, [1, n]) };
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

test("createKevDecider packs, runs once, and decodes all three kinds", async () => {
  // Pre-compute the packing so the fake session can rig the winners.
  const questions = normalizeQuestions([choiceQ, noulQ, scoreQ]);
  const packed = packKevSequence({
    stateText: "driver state",
    questions,
    encode: fakeEncodeEscaped,
    delims: {
      state: DELIM_IDS["<|fim_prefix|>"],
      question: DELIM_IDS["<|fim_middle|>"],
      option_start: DELIM_IDS["<|box_start|>"],
      option_end: DELIM_IDS["<|box_end|>"],
      decide: DELIM_IDS["<|fim_suffix|>"],
    },
    limits: { max_state_tokens: 8192, max_branch_tokens: 8192, max_options: 255 },
  });
  const rig = { winners: packed.groups.map((g) => g[0]) };

  const ort = fakeOrt(rig);
  const fetchImpl = fakeFetch();
  const decider = await createKevDecider({
    ort,
    transformers: fakeTransformers(),
    fetchImpl,
    device: "wasm",
    sessionFactory: async (url, opts) => ort.InferenceSession.create(url, opts),
  });
  assert.equal(decider.info.family, "kev");
  assert.equal(decider.info.model, KEV_DEFAULT_MODEL);
  assert.equal(decider.info.revision, KEV_DEFAULT_REVISION);
  assert.deepEqual(decider.info.tasks, ["choice", "noul", "score"]);
  assert.ok(fetchImpl.seen.some((u) => u.endsWith("config.json")), "config fetched");

  const { answers, truncated, prompts } = await decider.decide("driver state", [choiceQ, noulQ, scoreQ]);
  assert.equal(answers.length, 3);
  assert.equal(truncated, false);
  assert.equal(prompts.length, 3);

  // Exactly one forward pass for all questions.
  assert.equal(ort.__calls.length, 1);
  const inputs = ort.__calls[0];
  assert.deepEqual(Object.keys(inputs).sort(), ["attention_mask", "input_ids"]);
  assert.equal(inputs.input_ids.type, "int64");
  assert.deepEqual(inputs.input_ids.dims, [1, packed.tokens.length]);
  assert.ok(inputs.attention_mask.data.every((v) => v === 1n), "mask all ones");

  assert.equal(answers[0].type, "choice");
  assert.equal(answers[0].choice, "billing", "rigged winner");
  assert.equal(answers[1].type, "noul");
  assert.ok(answers[1].noul < 0.1, "no wins the rigged logits");
  assert.equal(answers[2].type, "score");
  assert.ok(answers[2].score < 0.5, "level 0 wins the rigged logits");
  assert.equal(answers[2].level, 0);

  await decider.dispose();
});

test("createKevDecider variant shortcut selects 4b", async () => {
  const decider = await createKevDecider({
    variant: "4b",
    ort: fakeOrt({ winners: [] }),
    transformers: fakeTransformers(),
    fetchImpl: fakeFetch(),
    device: "wasm",
    sessionFactory: async (url, opts) => fakeOrt({ winners: [] }).InferenceSession.create(url, opts),
  });
  assert.equal(decider.info.model, KEV_4B_MODEL);
  assert.equal(decider.info.revision, KEV_VARIANTS["4b"].revision);
  await decider.dispose();

  await assert.rejects(
    createKevDecider({ variant: "nope", ort: fakeOrt({ winners: [] }) }),
    /unknown variant/,
  );
});

test("createKevDecider rejects bad questions before running", async () => {
  const decider = await createKevDecider({
    ort: fakeOrt({ winners: [] }),
    transformers: fakeTransformers(),
    fetchImpl: fakeFetch(),
    device: "wasm",
    sessionFactory: async (url, opts) => fakeOrt({ winners: [] }).InferenceSession.create(url, opts),
  });
  await assert.rejects(decider.decide("s", []), /non-empty array/);
  await assert.rejects(
    decider.decide("s", [{ type: "choice", instructions: "x", options: ["only-one"] }]),
    /at least 2 options/,
  );
  await decider.dispose();
});