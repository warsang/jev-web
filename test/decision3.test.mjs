/**
 * decision3 (d3) family: prompt construction, answer decoding and batching —
 * validated against fixtures generated from the reference d3 runtime
 * (export/d3/make_js_fixtures.py runs d3_format/d3_runtime on the real
 * tokenizer and writes test/fixtures/d3-protocol.json).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import {
  DECISION3_DEFAULT_CODES,
  DECISION3_MAX_OPTIONS,
  DECISION3_SYSTEM_PROMPT,
  buildDecision3Prompt,
  decision3Answers,
  decision3Confidence,
  decision3OptionTexts,
  decision3ProbsFromLogits,
  normalizeDecision3Question,
  packDecision3Batch,
  renderDecision3Chat,
} from "../src/decision3.mjs";
import { normalizeQuestions } from "../src/questions.mjs";

const fixture = JSON.parse(readFileSync(
  fileURLToPath(new URL("./fixtures/d3-protocol.json", import.meta.url)), "utf8"));

test("fixture sanity: the code table is the checkpoint's own", () => {
  assert.equal(fixture.systemPrompt, DECISION3_SYSTEM_PROMPT);
  assert.equal(fixture.formatId, "d3-code-readout-v1");
  assert.equal(fixture.configMatch, true,
    "the shipped decision_config table matches the tokenizer");
  assert.equal(fixture.codes.length, DECISION3_MAX_OPTIONS);
  assert.equal(DECISION3_DEFAULT_CODES.length, DECISION3_MAX_OPTIONS);
  assert.equal(DECISION3_DEFAULT_CODES[0], "A");
});

test("buildDecision3Prompt reproduces the reference user prompt", () => {
  for (const c of fixture.cases) {
    c.questions.forEach((q, i) => {
      const question = normalizeQuestions([q.questionDef])[0];
      const meta = normalizeDecision3Question(question);
      const user = buildDecision3Prompt(c.state, meta.rendered, fixture.codes);
      assert.equal(user, q.userPrompt, `${c.name}[${i}] ${q.id}: user prompt`);
    });
  }
});

test("renderDecision3Chat matches the checkpoint's chat template byte-for-byte", () => {
  for (const c of fixture.cases) {
    c.questions.forEach((q, i) => {
      const prompt = renderDecision3Chat(DECISION3_SYSTEM_PROMPT, q.userPrompt);
      assert.equal(prompt, q.prompt, `${c.name}[${i}] ${q.id}: chat prompt`);
    });
  }
});

test("decision3OptionTexts renders like the reference options()", () => {
  assert.deepEqual(
    decision3OptionTexts({ type: "choice", criteria: { billing: "pay", other: null } }),
    ["billing: pay", "other"]);
  assert.deepEqual(
    decision3OptionTexts({ type: "noul" }),
    ["No / false", "Yes / true"]);
  assert.deepEqual(
    decision3OptionTexts({ type: "noul", criteria: { false: "it is not", true: "it is" } }),
    ["it is not", "it is"]);
});

test("normalizeDecision3Question shapes each kind like the reference", () => {
  const choice = normalizeDecision3Question(
    normalizeQuestions([{
      type: "choice", instructions: "Pick.", options: ["b", "a"],
      criteria: { a: "desc", b: null },
    }])[0]);
  assert.deepEqual(choice, {
    kind: "choice",
    rendered: { type: "choice", instructions: "Pick.", criteria: { b: null, a: "desc" } },
    keys: ["b", "a"],
    descriptions: [null, "desc"],
  });
  const noul = normalizeDecision3Question(
    normalizeQuestions([{ type: "noul", instructions: "Yes?" }])[0]);
  assert.deepEqual(noul, {
    kind: "noul", rendered: { type: "noul", instructions: "Yes?" },
    keys: ["false", "true"], descriptions: [],
  });
  const score = normalizeDecision3Question(
    normalizeQuestions([{
      type: "score", instructions: "How bad?", options: ["lo", "hi"],
    }])[0]);
  assert.equal(score.kind, "score");
  assert.deepEqual(score.keys, ["0", "1"]);
  assert.deepEqual(score.rendered.criteria, { lo: null, hi: null });
  assert.deepEqual(score.descriptions, ["lo", "hi"]);
  assert.throws(
    () => normalizeDecision3Question({
      type: "score", instructions: "x", options: ["same", "same"],
    }),
    /distinct and nonempty/);
});

test("decision3ProbsFromLogits masks, tempers and slices like the reference", () => {
  const logits = [2, 0, 1, 5, 4, 99];
  const probs = decision3ProbsFromLogits({ logits, rowWidth: 3, counts: [3, 2], temperature: 1 });
  const softmax = (xs) => {
    const m = Math.max(...xs);
    const e = xs.map((x) => Math.exp(x - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / s);
  };
  assert.deepEqual(probs[0], softmax([2, 0, 1]));
  assert.deepEqual(probs[1], softmax([5, 4]));
  const hot = decision3ProbsFromLogits({ logits: [1, 0], rowWidth: 2, counts: [2], temperature: 0.5 });
  assert.deepEqual(hot[0], softmax([2, 0]), "temperature divides the logits");
  const masked = decision3ProbsFromLogits({
    logits: [1, -Infinity], rowWidth: 2, counts: [2], temperature: 1,
  });
  assert.equal(masked[0][1], 0, "-inf logits from the masked readout stay zero");
});

test("decision3Answers decodes the reference product_answer expectations", () => {
  for (const c of fixture.cases) {
    c.questions.forEach((q, i) => {
      const question = normalizeQuestions([q.questionDef])[0];
      const [answer] = decision3Answers({ probs: [q.probsIn], questions: [question] });
      const ref = q.expected;
      assert.equal(answer.type, ref.type, `${c.name}[${i}] ${q.id}: type`);
      if (ref.type === "choice") {
        assert.equal(answer.choice, ref.choice, `${c.name}[${i}]: choice`);
        for (const [k, v] of Object.entries(ref.probabilities)) {
          assert.ok(Math.abs(answer.probabilities[k] - v) < 1e-9, `${c.name}[${i}]: p(${k})`);
        }
        assert.ok(Math.abs(answer.confidence - ref.confidence) < 1e-9, `${c.name}[${i}]: confidence`);
      } else if (ref.type === "noul") {
        assert.ok(Math.abs(answer.noul - ref.noul) < 1e-9, `${c.name}[${i}]: noul`);
      } else {
        assert.ok(Math.abs(answer.score - ref.score) < 1e-9, `${c.name}[${i}]: score`);
        assert.deepEqual(answer.legend, ref.legend, `${c.name}[${i}]: legend`);
        assert.ok(Math.abs(answer.confidence - ref.confidence) < 1e-9, `${c.name}[${i}]: confidence`);
      }
    });
  }
  assert.equal(decision3Confidence([1, 0]), 1);
  assert.equal(decision3Confidence([0.5, 0.5]), 0);
});

test("packDecision3Batch left-pads exactly like the reference logits()", () => {
  const batch = packDecision3Batch({
    sequences: [[1, 2, 3], [4, 5]],
    counts: [3, 2],
    padId: 99,
  });
  assert.equal(batch.B, 2);
  assert.equal(batch.width, 3);
  assert.deepEqual(Array.from(batch.inputIds, Number), [1, 2, 3, 99, 4, 5]);
  assert.deepEqual(Array.from(batch.attention, Number), [1, 1, 1, 0, 1, 1]);
  assert.deepEqual(batch.counts, [3, 2]);
});

// ---------------------------------------------------------------------------
// createDecision3Decider with injected fakes (no network, no real graphs)
// ---------------------------------------------------------------------------

class Tensor {
  constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
}

function fakeD3Sessions() {
  const torsoCalls = [];
  const readoutCalls = [];
  const torso = {
    run: async (inputs) => {
      torsoCalls.push(inputs);
      const B = inputs.input_ids.dims[0];
      const data = new Float32Array(B * 4).fill(0.5);
      return { last: new Tensor("float32", data, [B, 4]) };
    },
    release: async () => {},
  };
  const readout = {
    run: async (inputs) => {
      readoutCalls.push(inputs);
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
    release: async () => {},
  };
  return { torso, readout, torsoCalls, readoutCalls };
}

/** char-level tokenizer stand-in; apply_chat_template replays the Qwen3.5 rendering. */
function fakeD3Tokenizer({ withTemplate = true } = {}) {
  const tok = (text) => ({
    input_ids: {
      data: BigInt64Array.from([...String(text)].map((c) => (c.charCodeAt(0) % 90) + 10), BigInt),
    },
  });
  if (withTemplate) {
    tok.apply_chat_template = (messages) =>
      renderDecision3Chat(messages[0].content, messages[1].content);
  }
  tok.pad_token_id = 0;
  tok.decode = (ids) => `decoded:${Array.from(ids).join(" ")}`;
  return tok;
}

async function makeDecider(overrides = {}) {
  const { createDecision3Decider } = await import("../src/decision3.mjs");
  const sessions = fakeD3Sessions();
  const decider = await createDecision3Decider({
    ort: { Tensor },
    transformers: {
      AutoTokenizer: {
        from_pretrained: async () => fakeD3Tokenizer(overrides.tokenizer ?? {}),
      },
    },
    device: "wasm",
    sessionFactory: async (url) =>
      (String(url).includes("readout") ? sessions.readout : sessions.torso),
    fetchImpl: async () => ({
      ok: true,
      json: async () => ({
        format_id: fixture.formatId,
        codes: fixture.codes,
        temperature: fixture.config.temperature,
        max_length: fixture.config.max_length,
      }),
    }),
    ...overrides.options,
  });
  return { decider, sessions };
}

test("createDecision3Decider wires both graphs and batches questions", async () => {
  const c = fixture.cases[0]; // 4 questions
  const { decider, sessions } = await makeDecider({ options: { batchSize: 2 } });
  assert.equal(decider.info.family, "decision3");
  assert.equal(decider.info.codes, fixture.codes.length);
  assert.equal(decider.info.temperature, 1);
  assert.match(decider.info.configSource, /decision_config\.json$/);

  const { answers, truncated, length, prompts, timings } = await decider.decide(
    c.state, c.questions.map((q) => q.questionDef));
  assert.equal(sessions.torsoCalls.length, 2, "4 questions in batches of 2");
  assert.equal(sessions.readoutCalls.length, 2);
  assert.equal(truncated, false);
  assert.ok(length > 0);
  assert.equal(prompts.length, 4);
  assert.ok(timings.totalMs >= 0);

  assert.equal(answers[0].type, "choice");
  assert.equal(answers[0].choice, "billing");
  assert.equal(answers[0].index, 0);
  assert.equal(answers[1].type, "noul");
  assert.ok(answers[1].noul < 0.5, "the fake favours code A -> p(false) is the high one");
  assert.equal(answers[2].type, "noul");
  assert.equal(answers[3].type, "score");
  assert.equal(answers[3].level, 0);
  assert.equal(answers[3].legend["0"], "Routine");
  await decider.dispose();
});

test("createDecision3Decider works without apply_chat_template (fallback rendering)", async () => {
  const c = fixture.cases[0];
  const { decider } = await makeDecider({
    tokenizer: { withTemplate: false },
    options: { batchSize: 8 },
  });
  const { answers, prompts } = await decider.decide(
    c.state, c.questions.map((q) => q.questionDef));
  assert.equal(answers.length, 4);
  assert.equal(prompts.length, 4);
  // the prompts are the char-encoded fallback rendering of the same messages
  assert.ok(prompts[0].startsWith("decoded:"));
  await decider.dispose();
});

test("createDecision3Decider refuses prompts over the window (nothing is truncated)", async () => {
  const c = fixture.cases[2]; // the long-state case
  const { decider } = await makeDecider({ options: { maxLen: 10 } });
  await assert.rejects(
    decider.decide(c.state, c.questions.map((q) => q.questionDef)),
    /over the maximum context length of 10 tokens/);
  await decider.dispose();
});

test("createDecision3Decider honours an injected code table and temperature", async () => {
  const { createDecision3Decider } = await import("../src/decision3.mjs");
  const sessions = fakeD3Sessions();
  const codes = ["A", "B"];
  const decider = await createDecision3Decider({
    ort: { Tensor },
    transformers: { AutoTokenizer: { from_pretrained: async () => fakeD3Tokenizer() } },
    device: "wasm",
    sessionFactory: async (url) =>
      (String(url).includes("readout") ? sessions.readout : sessions.torso),
    fetchImpl: async () => ({ ok: false, json: async () => ({}) }),
    codes,
    temperature: 2,
  });
  assert.equal(decider.info.codes, 2);
  assert.equal(decider.info.temperature, 2);
  const { answers } = await decider.decide(
    "state", [{ type: "noul", instructions: "Is it true?" }]);
  assert.equal(answers[0].type, "noul");
  await decider.dispose();
});
