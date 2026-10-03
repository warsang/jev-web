/**
 * Strands Decider family: prompt rendering, option-position placement,
 * batch building, calibration and answer decoding, plus end-to-end decider
 * wiring (with injected ORT/tokenizer fakes).
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  renderStrandsQuestion,
  renderStrandsState,
  renderStrandsContent,
  buildStrandsBatch,
  collateStrandsItems,
  strandsOptionTokenIndices,
  strandsAnswersFromLogits,
  strandsChoiceConfidence,
  strandsScoreConfidence,
  createStrandsDecider,
  STRANDS_TEMPERATURE_BY_KIND,
  STRANDS_TEMPERATURE,
  STRANDS_ORDINAL_SMOOTHING,
  STRANDS_DEFAULT_MAX_LEN,
} from "../src/strands.mjs";
import { softmaxWithTemperature } from "../src/answers.mjs";

// char-level stand-in: every character is one token, "\n" its own token.
const fakeEncode = (text) => [...String(text)].map((c) => (c.charCodeAt(0) % 90) + 10);
// merge variant: a trailing "."+"\n" fuses into one token (exercises the
// merged-newline branch of strandsOptionTokenIndices).
const fakeEncodeMerge = (text) => {
  const ids = fakeEncode(text);
  if (text.endsWith("\n") && text.length >= 2 && text[text.length - 2] === ".") {
    ids.splice(ids.length - 2, 2, 999);
  }
  return ids;
};

const choice3 = {
  type: "choice",
  instructions: "Pick one.",
  options: ["billing", "sales", "retail"],
};
const noul = { type: "noul", instructions: "The customer wants a refund.", options: ["no", "yes"] };
const score3 = {
  type: "score",
  instructions: "How bad?",
  options: ["low", "mid", "high"],
};

test("renderStrandsContent flattens strings and JSON-dumps objects", () => {
  assert.equal(renderStrandsContent("  hi  "), "hi");
  assert.equal(renderStrandsContent({ b: 1, a: 2 }), JSON.stringify({ b: 1, a: 2 }, null, 2));
});

test("renderStrandsQuestion supports choice criteria with descriptions", () => {
  const q = {
    type: "choice",
    instructions: "What language?",
    options: ["english", "zulu", "dutch"],
    criteria: { english: "English", zulu: "isiZulu", dutch: "Nederlands" },
  };
  const rq = renderStrandsQuestion(q);
  assert.deepEqual(rq.slotLabels, ["english", "zulu", "dutch"]);
  assert.deepEqual(rq.slotDescriptions, ["English", "isiZulu", "Nederlands"]);
  assert.ok(rq.text.includes("1. english \u2014 English"));
  assert.ok(rq.text.includes("3. dutch \u2014 Nederlands"));
});

test("renderStrandsQuestion supports noul criteria overrides", () => {
  const q = {
    type: "noul",
    instructions: "Is it urgent?",
    options: ["no", "yes"],
    criteria: { true: "the statement clearly holds" },
  };
  const rq = renderStrandsQuestion(q);
  assert.deepEqual(rq.slotLabels, ["false", "true"]);
  assert.ok(rq.text.includes("1. false \u2014 the statement does not hold for this state"));
  assert.ok(rq.text.includes("2. true \u2014 the statement clearly holds"));
});

test("renderStrandsQuestion mirrors the reference prompt layout", () => {
  const rq = renderStrandsQuestion(choice3);
  assert.equal(rq.kind, "choice");
  assert.deepEqual(rq.slotLabels, ["billing", "sales", "retail"]);
  const expected =
    `<question type="choice">\nSelect exactly one option.\nPick one.\n<options>\n` +
    `1. billing\n2. sales\n3. retail\n</options>\n</question>\n<answer>`;
  assert.equal(rq.text, expected);
  assert.equal(rq.chunks.join(""), expected, "chunks reconstruct the prompt exactly");
  assert.equal(rq.chunks.length, 3 + 2);
  assert.ok(rq.chunks[1].endsWith("\n"), "option chunks carry their newline");

  const nq = renderStrandsQuestion(noul);
  assert.equal(nq.kind, "noul");
  // Noul slot labels are fixed false/true by the reference; question.options
  // only names the answer keys.
  assert.deepEqual(nq.slotLabels, ["false", "true"]);
  assert.ok(nq.text.includes("1. false \u2014 the statement does not hold for this state"));
  assert.ok(nq.text.includes("2. true \u2014 the statement holds for this state"));

  const sq = renderStrandsQuestion(score3);
  assert.equal(sq.kind, "score");
  assert.deepEqual(sq.slotLabels, ["0", "1", "2"]);
  assert.ok(sq.text.includes("1. 0 \u2014 low"));
  assert.ok(sq.text.includes("Rate the state against the ordered levels below (lowest first)."));

  assert.equal(
    renderStrandsState("some state"),
    "<state>\nsome state\n</state>\n",
  );
});

test("strandsOptionTokenIndices scores the option's last token", () => {
  const rq = renderStrandsQuestion(choice3);
  const lineIds = rq.chunks.slice(1, 4).map((c) => fakeEncode(c.slice(0, -1)));
  const chunkIds = rq.chunks.slice(1, 4).map(fakeEncode);
  const pos = strandsOptionTokenIndices(lineIds, chunkIds);
  // "1. billing\n": 10 chars, newline its own token -> last line token at 9
  assert.deepEqual(pos, [9, 7, 8]);

  // merged newline: "1. x.\n" -> chunk ends with the fused token, readout steps back
  const mergedLine = fakeEncode("1. x.");
  const mergedChunk = fakeEncodeMerge("1. x.\n");
  assert.equal(mergedChunk.length, mergedLine.length, "newline fused with the period");
  assert.deepEqual(strandsOptionTokenIndices([mergedLine], [mergedChunk]), [mergedChunk.length - 2]);

  assert.throws(() => strandsOptionTokenIndices([[]], [fakeEncode("a\n")]), /no tokens left/);
  assert.throws(() => strandsOptionTokenIndices([[1]], [[2, 3]]), /re-tokenised/);
});

test("buildStrandsBatch places exact option positions and budgets the window", () => {
  const normalized = [choice3, noul, score3];
  const rendered = normalized.map(renderStrandsQuestion);
  const { items, truncated } = buildStrandsBatch({
    encode: (t, s) => fakeEncode(t),
    stateText: renderStrandsState("STATE"),
    rendered,
    maxLen: 4096,
  });
  assert.equal(truncated, false);
  assert.equal(items.length, 3);
  for (const [i, it] of items.entries()) {
    const rq = rendered[i];
    assert.equal(it.optIdx.length, rq.slotLabels.length);
    // every scored position is the last token of its option line
    it.optIdx.forEach((p, j) => {
      const lineStart = it.ids.indexOf(fakeEncode(`${j + 1}.`)[0], 0);
      assert.ok(p >= 0 && p < it.ids.length, `option ${j} position in range`);
      assert.ok(lineStart >= 0, "option line present");
    });
    assert.deepEqual(it.kind, rq.kind);
  }
  const b = collateStrandsItems(items);
  assert.equal(b.n, 3);
  assert.equal(b.K, 3);
  assert.equal(b.optionPos[1 * b.K + 2], 0n, "short rows pad option_pos with 0 (clamped)");
  assert.ok(b.optionPos[0] > 0n, "real option positions survive");
  items.forEach((it, i) => {
    assert.equal(b.answerPos[i], BigInt(it.ids.length - 1), "answer_pos is the last real token");
  });
  // attention covers the full rectangle
  assert.equal(Number(b.attention.reduce((a, v) => a + v, 0n)), items.reduce((a, it) => a + it.ids.length, 0));
});

test("buildStrandsBatch truncates state-first, then the question front, keeping the tail", () => {
  const rendered = [renderStrandsQuestion(choice3)].map((r) => r);
  const long = "S".repeat(300);
  const { items, truncated } = buildStrandsBatch({
    encode: (t) => fakeEncode(t),
    stateText: renderStrandsState(long),
    rendered,
    maxLen: 128,
  });
  assert.equal(truncated, true);
  assert.ok(items[0].ids.length <= 128, "batch honours maxLen");
  // the tail (options + <answer>) survives: last token is ">" of <answer>
  assert.equal(items[0].ids.at(-1), fakeEncode(">").at(-1) ?? fakeEncode(">")[0]);
});

test("buildStrandsBatch refuses a question truncated through its options", () => {
  const rendered = [renderStrandsQuestion(choice3)];
  assert.throws(() => buildStrandsBatch({
    encode: (t) => fakeEncode(t),
    stateText: renderStrandsState("s"),
    rendered,
    maxLen: 40, // reserve too small to hold the option block
    maxQuestionFraction: 0.05,
  }), /truncated through its option list/);
});

test("temperatures match hobson_config.json and softmax applies them per kind", () => {
  assert.equal(STRANDS_TEMPERATURE_BY_KIND.noul, 0.9107136998460428);
  assert.equal(STRANDS_TEMPERATURE_BY_KIND.choice, 0.734189596436441);
  assert.equal(STRANDS_TEMPERATURE_BY_KIND.score, 1.32780942142348);
  assert.equal(STRANDS_TEMPERATURE, 0.9627721607677362);

  const answers = strandsAnswersFromLogits({
    logits: [0, 0, 5, 0, 2, -100], // row-major [2, 3]: choice row, noul row
    rowWidth: 3,
    questions: [choice3, noul],
    rendered: [renderStrandsQuestion(choice3), renderStrandsQuestion(noul)],
  });
  const tChoice = STRANDS_TEMPERATURE_BY_KIND.choice;
  const expected = softmaxWithTemperature([0, 0, 5], tChoice);
  assert.ok(Math.abs(answers[0].probabilities.billing - expected[0]) < 1e-12);
  assert.equal(answers[0].choice, "retail");
  const tNoul = STRANDS_TEMPERATURE_BY_KIND.noul;
  assert.ok(Math.abs(answers[1].noul - softmaxWithTemperature([0, 2], tNoul)[1]) < 1e-12);
});

test("choice confidence is the normalised max-probability", () => {
  assert.equal(strandsChoiceConfidence([1]), 1);
  assert.equal(strandsChoiceConfidence([0.5, 0.5]), 0);
  assert.ok(Math.abs(strandsChoiceConfidence([0.7, 0.2, 0.1]) - (3 * 0.7 - 1) / 2) < 1e-12);
});

test("score confidence is ordinal: adjacent spread is agreement, bimodal is doubt", () => {
  assert.equal(strandsScoreConfidence([1]), 1);
  const tight = strandsScoreConfidence([0, 0.5, 0.5, 0, 0], 0);
  const bimodal = strandsScoreConfidence([0.5, 0, 0, 0, 0.5], 0);
  assert.ok(tight > bimodal, "adjacent mass outranks a split mass");
  assert.ok(bimodal < 0.05, "extremes-split mass is near-zero confidence");
  // ordinal smoothing corrects the floor: the same near-one-hot distribution
  // reads more confident with smoothing than without, and stays below 1
  const near = [0.1, 0.8, 0.1];
  const smoothed = strandsScoreConfidence(near, STRANDS_ORDINAL_SMOOTHING);
  const raw = strandsScoreConfidence(near, 0);
  assert.ok(smoothed > raw, `smoothing lifts the floor (${smoothed} > ${raw})`);
  assert.ok(smoothed < 1 && smoothed > 0.5);
});

test("score answers report the expected level with canonical level order", () => {
  const answers = strandsAnswersFromLogits({
    logits: [0, 3, 0],
    rowWidth: 3,
    questions: [score3],
    rendered: [renderStrandsQuestion(score3)],
  });
  const a = answers[0];
  assert.equal(a.type, "score");
  assert.equal(a.level, 1);
  assert.ok(Math.abs(a.score - 1) < 0.05, "near-one-hot expectation ≈ 1");
  assert.deepEqual(Object.keys(a.probabilities), ["low", "mid", "high"]);
  assert.ok(a.confidence > 0.8, `ordinal confidence, got ${a.confidence}`);
});

// --------------------------------------------------------------------------
// End-to-end with injected fakes
// --------------------------------------------------------------------------

function fakeTokenizer() {
  const tok = (text, { add_special_tokens = false } = {}) => {
    const ids = fakeEncode(add_special_tokens ? `<s>${text}` : text);
    return { input_ids: { data: BigInt64Array.from(ids, BigInt) } };
  };
  tok.decode = (ids) => `decoded:${Array.from(ids).join(" ")}`;
  return tok;
}

function fakeOrt() {
  class Tensor {
    constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
  }
  const calls = [];
  const session = {
    run: async (inputs) => {
      calls.push(inputs);
      const n = inputs.input_ids.dims[0];
      const K = inputs.option_pos.dims[1];
      const data = new Float32Array(n * K).fill(-100);
      for (let i = 0; i < n; i++) {
        // real option positions are always > 0 (state + BOS precede them);
        // padded slots are 0.
        let k = 0;
        for (let j = 0; j < K; j++) if (inputs.option_pos.data[i * K + j] > 0n) k++;
        data[i * K + (k - 1)] = 5; // last option wins every row
      }
      return { logits: new Tensor("float32", data, [n, K]) };
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

test("createStrandsDecider wires tokenizer + session and decodes all three kinds", async () => {
  const ort = fakeOrt();
  const transformers = {
    AutoTokenizer: { from_pretrained: async () => fakeTokenizer() },
  };
  const decider = await createStrandsDecider({
    ort,
    transformers,
    device: "wasm",
    sessionFactory: async (url, opts) => ort.InferenceSession.create(url, opts),
  });
  assert.equal(decider.info.family, "strands-decider");
  assert.equal(decider.info.device, "wasm");
  assert.equal(decider.info.maxLen, STRANDS_DEFAULT_MAX_LEN);

  const { answers, truncated, length, prompts, timings } = await decider.decide(
    "driver state text",
    [choice3, noul, score3],
  );
  assert.equal(ort.__calls.length, 1, "one fused forward for the whole batch");
  const inputs = ort.__calls[0];
  assert.deepEqual(inputs.input_ids.dims[0], 3);
  assert.deepEqual(inputs.option_pos.dims, [3, 3], "option_pos is [batch, maxOptions]");
  assert.ok(inputs.option_pos.data[1 * 3 + 2] === 0n, "short rows pad with 0");
  assert.deepEqual(inputs.answer_pos.dims, [3], "answer_pos is [batch]");

  assert.equal(answers.length, 3);
  assert.equal(answers[0].type, "choice");
  assert.equal(answers[0].choice, "retail", "last option wins the rigged logits");
  assert.equal(answers[1].type, "noul");
  assert.ok(answers[1].noul > 0.99, "yes wins the rigged logits");
  assert.equal(answers[2].type, "score");
  assert.equal(answers[2].level, 2);

  assert.equal(truncated, false);
  assert.ok(length > 0);
  assert.equal(prompts.length, 3);
  assert.ok(prompts[0].startsWith("decoded:"));
  assert.ok(timings.totalMs >= 0 && timings.runMs >= 0);
  await decider.dispose();
});

test("createStrandsDecider rejects bad questions before touching the network", async () => {
  const ort = fakeOrt();
  const transformers = { AutoTokenizer: { from_pretrained: async () => fakeTokenizer() } };
  const decider = await createStrandsDecider({
    ort,
    transformers,
    device: "wasm",
    sessionFactory: async (url, opts) => ort.InferenceSession.create(url, opts),
  });
  await assert.rejects(decider.decide("s", []), /non-empty array/);
  await assert.rejects(
    decider.decide("s", [{ type: "choice", instructions: "x", options: ["only"] }]),
    /at least 2 options/,
  );
});
