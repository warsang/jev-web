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
  parseStrandsCalibration,
  loadStrandsCalibration,
  STRANDS_TEMPERATURE_BY_KIND,
  STRANDS_TEMPERATURE,
  STRANDS_ORDINAL_SMOOTHING,
  STRANDS_DEFAULT_MAX_LEN,
  STRANDS_CONFIG_FILES,
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

// --------------------------------------------------------------------------
// Checkpoint calibration config
// --------------------------------------------------------------------------

// StrandsAgents/strands-decider-E2B-gemma4-v1-2610's real config values.
const E2B_CONFIG = {
  base_model: "google/gemma-4-E2B-it",
  head_type: "pointer",
  max_length: 4096,
  temperature: 0.7997762858861774,
  temperature_by_kind: { noul: 1.0704394404346953, choice: 0.5608428296613732, score: 1.4788446944896205 },
  ordinal_smoothing: 0.1,
};

test("parseStrandsCalibration overrides every field the checkpoint declares", () => {
  const calibration = parseStrandsCalibration(E2B_CONFIG, {
    temperature: STRANDS_TEMPERATURE,
    temperaturesByKind: STRANDS_TEMPERATURE_BY_KIND,
    ordinalSmoothing: STRANDS_ORDINAL_SMOOTHING,
    maxLen: STRANDS_DEFAULT_MAX_LEN,
  });
  assert.equal(calibration.temperature, 0.7997762858861774);
  assert.deepEqual(calibration.temperaturesByKind, E2B_CONFIG.temperature_by_kind);
  assert.equal(calibration.ordinalSmoothing, 0.1);
  assert.equal(calibration.maxLen, 4096);
});

test("parseStrandsCalibration falls back field-by-field on a partial config", () => {
  const calibration = parseStrandsCalibration(
    { temperature: 1.05 },
    {
      temperature: STRANDS_TEMPERATURE,
      temperaturesByKind: { ...STRANDS_TEMPERATURE_BY_KIND },
      ordinalSmoothing: 0.4,
      maxLen: 2048,
    },
  );
  assert.equal(calibration.temperature, 1.05, "declared field wins");
  assert.deepEqual(calibration.temperaturesByKind, STRANDS_TEMPERATURE_BY_KIND, "undeclared kept");
  assert.equal(calibration.ordinalSmoothing, 0.4);
  assert.equal(calibration.maxLen, 2048);
});

test("parseStrandsCalibration rejects a non-pointer head and junk numbers", () => {
  assert.throws(
    () => parseStrandsCalibration({ head_type: "linear" }),
    /head_type "linear".*pointer head/,
  );
  const calibration = parseStrandsCalibration(
    { temperature: "hot", temperature_by_kind: { choice: -1, noul: "1.5" }, max_length: 7 },
    { temperature: 2, temperaturesByKind: { choice: 3, noul: 3, score: 3 }, maxLen: 512 },
  );
  assert.equal(calibration.temperature, 2, "non-numeric temperature ignored");
  assert.equal(calibration.temperaturesByKind.choice, 3, "non-positive temperature ignored");
  assert.equal(calibration.temperaturesByKind.noul, 1.5, "numeric string accepted");
  assert.equal(calibration.maxLen, 512, "absurd max_length ignored");
});

function configServer(files) {
  const calls = [];
  return {
    calls,
    fetchImpl: async (url) => {
      calls.push(url);
      const file = Object.keys(files).find((f) => String(url).endsWith(`/${f}`));
      if (file === undefined) return { ok: false, status: 404, json: async () => ({}) };
      const body = files[file];
      return {
        ok: true,
        status: 200,
        json: async () => (typeof body === "string" ? JSON.parse(body) : body),
      };
    },
  };
}

test("loadStrandsCalibration reads strands_decider_config.json before hobson_config.json", async () => {
  const server = configServer({
    "strands_decider_config.json": E2B_CONFIG,
    "hobson_config.json": { temperature: 0.1 },
  });
  const { calibration, source } = await loadStrandsCalibration({
    model: "StrandsAgents/strands-decider-E2B-gemma4-v1-2610",
    revision: "abc123",
    fetchImpl: server.fetchImpl,
  });
  assert.equal(calibration.temperature, 0.7997762858861774);
  assert.equal(source, "https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v1-2610/resolve/abc123/strands_decider_config.json");
  assert.equal(server.calls.length, 1, "first hit wins");
});

test("loadStrandsCalibration falls back to the legacy filename then to pinned defaults", async () => {
  const legacy = configServer({
    "hobson_config.json": { temperature: 0.9627721607677362 },
  });
  const { calibration, source } = await loadStrandsCalibration({
    model: "StrandsAgents/strands-decider-2B-hobson-v19",
    fetchImpl: legacy.fetchImpl,
  });
  assert.equal(calibration.temperature, STRANDS_TEMPERATURE, "legacy file still parses");
  assert.match(source, /hobson_config\.json$/);
  assert.equal(legacy.calls.length, 2, "new name 404s first");

  const none = configServer({});
  const { calibration: pinned, source: noSource } = await loadStrandsCalibration({
    model: "onnx-community/strands-decider-2B-hobson-v19-ONNX",
    fetchImpl: none.fetchImpl,
  });
  assert.equal(pinned.temperature, undefined, "nothing declared anywhere");
  assert.equal(noSource, null);
});

test("createStrandsDecider takes calibration from the checkpoint config", async () => {
  const ort = fakeOrt();
  const transformers = { AutoTokenizer: { from_pretrained: async () => fakeTokenizer() } };
  const server = configServer({ "strands_decider_config.json": E2B_CONFIG });
  const decider = await createStrandsDecider({
    ort,
    transformers,
    device: "wasm",
    fetchImpl: server.fetchImpl,
    sessionFactory: async (url, opts) => ort.InferenceSession.create(url, opts),
  });
  assert.equal(decider.info.temperature, 0.7997762858861774, "checkpoint temperature wins");
  assert.equal(decider.info.temperaturesByKind.choice, 0.5608428296613732);
  assert.equal(decider.info.ordinalSmoothing, 0.1);
  assert.equal(decider.info.maxLen, 4096);
  assert.match(decider.info.configSource, /strands_decider_config\.json$/);
  await decider.dispose();
});

test("createStrandsDecider explicit calibration beats the checkpoint config", async () => {
  const ort = fakeOrt();
  const transformers = { AutoTokenizer: { from_pretrained: async () => fakeTokenizer() } };
  const server = configServer({ "strands_decider_config.json": E2B_CONFIG });
  const decider = await createStrandsDecider({
    ort,
    transformers,
    device: "wasm",
    fetchImpl: server.fetchImpl,
    sessionFactory: async (url, opts) => ort.InferenceSession.create(url, opts),
    temperature: 2.5,
    maxLen: 2048,
  });
  assert.equal(decider.info.temperature, 2.5);
  assert.equal(decider.info.maxLen, 2048);
  assert.equal(decider.info.temperaturesByKind.noul, 1.0704394404346953, "kinds still come from the config");
  await decider.dispose();
});

test("createStrandsDecider keeps the pinned defaults when the repo ships no config", async () => {
  const ort = fakeOrt();
  const transformers = { AutoTokenizer: { from_pretrained: async () => fakeTokenizer() } };
  const server = configServer({});
  const decider = await createStrandsDecider({
    ort,
    transformers,
    device: "wasm",
    fetchImpl: server.fetchImpl,
    sessionFactory: async (url, opts) => ort.InferenceSession.create(url, opts),
  });
  assert.equal(decider.info.temperature, STRANDS_TEMPERATURE);
  assert.deepEqual(decider.info.temperaturesByKind, STRANDS_TEMPERATURE_BY_KIND);
  assert.equal(decider.info.ordinalSmoothing, STRANDS_ORDINAL_SMOOTHING);
  assert.equal(decider.info.maxLen, STRANDS_DEFAULT_MAX_LEN);
  assert.equal(decider.info.configSource, null);
  await decider.dispose();
});

test("createStrandsDecider reads calibration from calibrationModel when told", async () => {
  const ort = fakeOrt();
  const transformers = { AutoTokenizer: { from_pretrained: async () => fakeTokenizer() } };
  const server = configServer({});
  const decider = await createStrandsDecider({
    ort,
    transformers,
    device: "wasm",
    fetchImpl: server.fetchImpl,
    sessionFactory: async (url, opts) => ort.InferenceSession.create(url, opts),
    model: "warsang/strands-decider-e2b-web",
    revision: "deadbeef",
    calibrationModel: "StrandsAgents/strands-decider-E2B-gemma4-v1-2610",
  });
  assert.ok(
    server.calls.some((u) => u === "https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v1-2610/resolve/main/strands_decider_config.json"),
    "config read from the source checkpoint, not the ONNX repo",
  );
  assert.equal(decider.info.model, "warsang/strands-decider-e2b-web");
  await decider.dispose();
});

test("loadStrandsCalibration surfaces a corrupt config rather than guessing", async () => {
  const server = configServer({ "strands_decider_config.json": "{ not json" });
  await assert.rejects(
    () => loadStrandsCalibration({
      model: "some-org/some-export",
      fetchImpl: server.fetchImpl,
    }),
    /not valid JSON/,
  );
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
