/**
 * clef family: record encoding, packing and answer decoding — validated
 * against fixtures generated from Cloudflare's reference joint_schema_model.py
 * on the real tokenizer (export/clef/make_js_fixtures.py runs
 * encode_record/systemone_answer and writes test/fixtures/clef-protocol.json).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import {
  CLEF_DEFAULTS,
  CLEF_QTYPE,
  CLEF_SYSTEM_PROMPT,
  clefAnswers,
  clefQuestionOptions,
  encodeClefRecord,
  packClefBatch,
  renderClefContent,
} from "../src/clef.mjs";
import { normalizeQuestions } from "../src/questions.mjs";

const fixture = JSON.parse(readFileSync(
  fileURLToPath(new URL("./fixtures/clef-protocol.json", import.meta.url)), "utf8"));

/** encode() replaying the reference tokenizer's piece encodings for a case. */
function fixtureEncoder() {
  const map = new Map(fixture.pieces.map(([text, ids]) => [text, ids]));
  return (text) => {
    const ids = map.get(text);
    if (ids) return ids;
    throw new Error(`unexpected encode() input: ${JSON.stringify(text.slice(0, 80))}`);
  };
}

test("fixture sanity: the shipped constants match the reference module", () => {
  assert.equal(fixture.systemPrompt, CLEF_SYSTEM_PROMPT);
  assert.equal(fixture.headConfig.hidden_size, 1024);
});

test("encodeClefRecord reproduces the reference record exactly", () => {
  for (const c of fixture.cases) {
    const encode = fixtureEncoder();
    const record = encodeClefRecord({
      encode,
      stateText: renderClefContent(c.state),
      questions: normalizeQuestions(c.questionDefs),
      maxLength: fixture.maxLength,
      maxStateTokens: c.maxStateTokens ?? null,
    });
    assert.deepEqual(record.ids, c.ids, `${c.name}: ids`);
    assert.equal(record.length, c.ids.length, `${c.name}: length`);
    assert.equal(record.questions.length, c.questions.length, `${c.name}: questions`);
    record.questions.forEach((q, i) => {
      const ref = c.questions[i];
      assert.equal(q.id, ref.questionId, `${c.name}[${i}]: id`);
      assert.equal(q.type, ref.questionType, `${c.name}[${i}]: qtype`);
      assert.deepEqual(q.questionSpan, ref.questionSpan, `${c.name}[${i}]: question span`);
      assert.deepEqual(q.optionSpans, ref.optionSpans, `${c.name}[${i}]: option spans`);
      assert.deepEqual(q.optionIds, ref.optionIds, `${c.name}[${i}]: option ids`);
    });
    if (c.maxStateTokens) {
      assert.equal(record.truncated, true, `${c.name}: state was truncated`);
    } else {
      assert.equal(record.truncated, false, `${c.name}: state fits`);
    }
  }
});

test("encodeClefRecord fails loudly when the schema exceeds the window", () => {
  const encode = () => [1];
  assert.throws(
    () => encodeClefRecord({
      encode, stateText: "x", maxLength: 10,
      questions: normalizeQuestions([
        { type: "noul", instructions: "Is it true?" },
      ]),
    }),
    /schema requires \d+ tokens before the state/);
});

test("clefQuestionOptions renders like the reference question_options", () => {
  // choice: sorted by key, null descriptions stay null
  assert.deepEqual(
    clefQuestionOptions({
      type: "choice", instructions: "x",
      options: ["sales", "billing"], criteria: { billing: "pay", sales: null },
    }),
    { keys: ["billing", "sales"], descriptions: ["pay", null] });
  // noul: fixed true/false slots with default descriptions
  assert.deepEqual(
    clefQuestionOptions({ type: "noul", instructions: "x", options: ["no", "yes"] }),
    { keys: ["true", "false"], descriptions: [
      "The proposition is true or the answer is yes.",
      "The proposition is false or the answer is no.",
    ] });
  // noul criteria accepted under either naming
  assert.deepEqual(
    clefQuestionOptions({
      type: "noul", instructions: "x", options: ["no", "yes"],
      criteria: { yes: "it is", no: "it is not" },
    }),
    { keys: ["true", "false"], descriptions: ["it is", "it is not"] });
  // score: numeric keys over the ordered levels
  assert.deepEqual(
    clefQuestionOptions({ type: "score", instructions: "x", options: ["lo", "hi"] }),
    { keys: ["0", "1"], descriptions: ["lo", "hi"] });
});

test("renderClefContent mirrors the reference render()", () => {
  assert.equal(renderClefContent("as is"), "as is");
  assert.equal(
    renderClefContent({ b: 1, a: "ünïcode ✓" }),
    '{"a":"ünïcode ✓","b":1}');
  assert.equal(renderClefContent(["x", null, true]), '["x",null,true]');
});

test("packClefBatch lays out the pooling rows the graph consumes", () => {
  const c = fixture.cases[0]; // mixed: choice(3) + noul(2) + score(3)
  const record = encodeClefRecord({
    encode: fixtureEncoder(),
    stateText: renderClefContent(c.state),
    questions: normalizeQuestions(c.questionDefs),
    maxLength: fixture.maxLength,
  });
  const batch = packClefBatch(record);
  assert.equal(batch.Q, 3);
  assert.equal(batch.K, 3);
  assert.equal(batch.S, 9);
  assert.equal(batch.L, record.length);
  assert.deepEqual(batch.typeIds, [CLEF_QTYPE.choice, CLEF_QTYPE.noul, CLEF_QTYPE.score]);
  assert.deepEqual(batch.counts, [3, 2, 3]);
  assert.deepEqual([...batch.optMask],
    [1, 1, 1, 1, 1, 0, 1, 1, 1], "short questions leave their extra slots unmasked");
  assert.deepEqual(batch.slotQuestion, [0, 0, 0, 1, 1, 1, 2, 2, 2]);

  const rowSum = (view, row, width) => {
    let s = 0;
    for (let t = 0; t < width; t++) s += view[row * width + t];
    return Number(s.toFixed(6));
  };
  assert.equal(rowSum(batch.poolGlobal, 0, batch.L), 1, "global row is a one-hot");
  for (let q = 0; q < batch.Q; q++) {
    assert.equal(rowSum(batch.poolQuestion, q, batch.L), 1, `question row ${q}`);
  }
  for (let s = 0; s < batch.S; s++) {
    assert.equal(rowSum(batch.poolOption, s, batch.L), batch.optMask[s],
      `option row ${s} (masked slots stay zero)`);
  }
});

test("clefAnswers decodes the reference SystemOne answers", () => {
  for (const c of fixture.cases) {
    const questions = normalizeQuestions(c.questionDefs);
    const probs = c.questions.map((q) => q.probsIn);
    const answers = clefAnswers({ probs, questions });
    answers.forEach((a, i) => {
      const ref = c.questions[i].expected;
      assert.equal(a.type, ref.type, `${c.name}[${i}]: type`);
      if (ref.type === "choice") {
        assert.equal(a.choice, ref.choice, `${c.name}[${i}]: choice`);
        for (const [k, v] of Object.entries(ref.probabilities)) {
          assert.ok(Math.abs(a.probabilities[k] - v) <= 1e-4, `${c.name}[${i}]: p(${k})`);
        }
        assert.ok(Math.abs(a.confidence - ref.confidence) <= 1e-4, `${c.name}[${i}]: confidence`);
      } else if (ref.type === "noul") {
        assert.ok(Math.abs(a.noul - ref.noul) <= 1e-4, `${c.name}[${i}]: noul`);
      } else {
        assert.ok(Math.abs(a.score - ref.score) <= 1e-4, `${c.name}[${i}]: score`);
        assert.ok(Math.abs(a.confidence - ref.confidence) <= 1e-4, `${c.name}[${i}]: confidence`);
        assert.deepEqual(a.legend, ref.legend, `${c.name}[${i}]: legend`);
      }
    });
  }
});

// ---------------------------------------------------------------------------
// createClefDecider with injected fakes (no network, no real graph)
// ---------------------------------------------------------------------------

class Tensor {
  constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
}

function fakeClefSessions({ logits } = {}) {
  const calls = [];
  const session = {
    run: async (inputs) => {
      calls.push(inputs);
      const Q = inputs.type_ids.dims[0];
      const S = inputs.opt_mask.dims[0];
      const K = S / Q;
      const data = logits ?? new Float32Array(S);
      if (!logits) {
        // favour option 0 of every question, so decoded answers are predictable
        for (let q = 0; q < Q; q++) {
          data[q * K] = 4;
          for (let k = 1; k < K; k++) data[q * K + k] = -1;
        }
      }
      return { logits: new Tensor("float32", data, [1, S]) };
    },
    release: async () => {},
  };
  return { session, calls };
}

/** char-level tokenizer stand-in that replays the fixture piece encoding. */
function fakeClefTokenizer(encoder) {
  const tok = (text) => ({
    input_ids: { data: BigInt64Array.from(encoder(text), BigInt) },
  });
  tok.decode = (ids) => `decoded:${Array.from(ids).join(" ")}`;
  return tok;
}

test("createClefDecider wires the graph and decodes all three kinds", async () => {
  const { createClefDecider } = await import("../src/clef.mjs");
  const c = fixture.cases[0];
  const { session, calls } = fakeClefSessions();
  const ort = { Tensor };
  const decider = await createClefDecider({
    ort,
    transformers: { AutoTokenizer: { from_pretrained: async () => fakeClefTokenizer(fixtureEncoder()) } },
    device: "wasm",
    sessionFactory: async () => session,
    fetchImpl: async () => ({ ok: true, json: async () => ({ max_len: fixture.maxLength, head: fixture.headConfig }) }),
  });
  assert.equal(decider.info.family, "clef");
  assert.equal(decider.info.maxLen, fixture.maxLength);
  assert.deepEqual(decider.info.head, fixture.headConfig);
  assert.match(decider.info.configSource, /clef_config\.json$/);

  const { answers, truncated, length, prompts, timings } = await decider.decide(
    c.state, c.questionDefs);
  assert.equal(calls.length, 1, "one forward pass for the whole record");
  assert.equal(truncated, false);
  assert.equal(length, c.ids.length);
  assert.equal(prompts.length, 1);
  assert.ok(timings.totalMs >= 0 && timings.runMs >= 0);
  assert.deepEqual(calls[0].pool_question.dims, [1, 3, c.ids.length]);
  assert.deepEqual(calls[0].pool_option.dims, [1, 9, c.ids.length]);
  assert.deepEqual([...calls[0].opt_mask.data], [1, 1, 1, 1, 1, 0, 1, 1, 1]);

  assert.equal(answers[0].type, "choice");
  assert.equal(answers[0].choice, "billing"); // the fake favours option 0 of the sorted keys
  assert.equal(answers[0].index, 0);
  assert.ok(answers[0].confidence > 0.5);
  assert.equal(answers[1].type, "noul");
  assert.ok(answers[1].noul > 0.5);
  assert.equal(answers[2].type, "score");
  assert.equal(answers[2].level, 0);
  assert.ok(answers[2].score >= 0 && answers[2].score <= 2);
  await decider.dispose();
});

test("createClefDecider keeps pinned defaults when the config is absent", async () => {
  const { createClefDecider } = await import("../src/clef.mjs");
  const c = fixture.cases[1];
  const { session } = fakeClefSessions();
  const decider = await createClefDecider({
    ort: { Tensor },
    transformers: { AutoTokenizer: { from_pretrained: async () => fakeClefTokenizer(fixtureEncoder()) } },
    device: "wasm",
    sessionFactory: async () => session,
    fetchImpl: async () => ({ ok: false, json: async () => ({}) }),
    maxLen: 512,
  });
  assert.equal(decider.info.maxLen, 512, "explicit maxLen wins over the checkpoint window");
  assert.equal(decider.info.configSource, null);
  assert.equal(decider.info.maxLen, 512);
  const { answers } = await decider.decide(c.state, c.questionDefs);
  assert.equal(answers.length, 1);
  assert.equal(answers[0].type, "choice");
  await decider.dispose();
});

test("clef defaults are the reference window and no extra state cap", () => {
  assert.equal(CLEF_DEFAULTS.maxLen, 16384);
  assert.equal(CLEF_DEFAULTS.maxStateTokens, null);
});
