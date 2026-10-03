/**
 * vLLM Decision 2.0 family: segment rendering, token positions, score bias,
 * and end-to-end decider wiring (with injected fakes).
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  renderDecision2Content,
  renderDecision2Options,
  buildDecision2Segments,
  tokenizeDecision2Segments,
  decision2Softmax,
  decision2AnswerFromLogits,
  createDecision2Decider,
  DECISION2_DEFAULT_MODEL,
  DECISION2_DEFAULT_REVISION,
} from "../src/decision2.mjs";
import { normalizeQuestions } from "../src/questions.mjs";

// char-level stand-in tokenizer
const fakeEncode = (text) => [...String(text)].map((c) => (c.charCodeAt(0) % 90) + 10);

const choiceQ = normalizeQuestions([{
  type: "choice", instructions: "Pick a team.",
  options: ["billing", "sales"], criteria: { billing: "money", sales: "" },
}])[0];
const noulQ = normalizeQuestions([{ type: "noul", instructions: "Urgent?" }])[0];
const scoreQ = normalizeQuestions([{
  type: "score", instructions: "How bad?", options: ["low", "high"],
}])[0];

test("renderDecision2Options builds keys and nulls empty descriptions", () => {
  assert.deepEqual(renderDecision2Options(choiceQ), [
    { key: "billing", description: "money" },
    { key: "sales", description: null },
  ]);
  assert.deepEqual(renderDecision2Options(noulQ), [
    { key: "false", description: "No" },
    { key: "true", description: "Yes" },
  ]);
  assert.deepEqual(renderDecision2Options(scoreQ), [
    { key: "0", description: "low" },
    { key: "1", description: "high" },
  ]);
});

test("buildDecision2Segments matches the reference prompt layout", () => {
  const { segments, optionEnds, keys } = buildDecision2Segments(choiceQ, "some state");
  assert.deepEqual(keys, ["billing", "sales"]);
  assert.ok(segments[0].startsWith("Context:\nsome state\n\n"));
  assert.ok(segments[1] === "Task type: choice\n");
  assert.ok(segments[2].startsWith("Question:\nPick a team.\n"));
  assert.ok(segments[3] === "Options:");
  // option segments hold sorted-key compact JSON, starting with a newline
  assert.ok(segments[4].startsWith("\n<option>\n"));
  assert.ok(segments[4].includes('{"description":"money","key":"billing"}'));
  assert.ok(segments[5].includes('{"description":null,"key":"sales"}'));
  assert.deepEqual(optionEnds, [4, 5]);
  assert.ok(segments[6].includes("Select the single option best supported"));
  assert.ok(segments[7] === "Decision:");
});

test("tokenizeDecision2Segments records exact option and answer positions", () => {
  const { segments, optionEnds } = buildDecision2Segments(noulQ, "s");
  const { ids, optionPos, answerPos } = tokenizeDecision2Segments(
    segments, optionEnds, fakeEncode, 8192);
  assert.equal(optionPos.length, 2);
  assert.equal(answerPos, ids.length - 1);
  // option_pos is the last token of each </option> segment
  let offset = 0;
  for (let s = 0; s < segments.length; s++) {
    const len = fakeEncode(segments[s]).length;
    if (optionEnds.includes(s)) {
      assert.ok(optionPos.includes(offset + len - 1));
    }
    offset += len;
  }
  assert.equal(offset, ids.length);
});

test("decision2AnswerFromLogits applies score bias before softmax", () => {
  const scoreQ3 = normalizeQuestions([{
    type: "score", instructions: "How bad?", options: ["low", "mid", "high"],
  }])[0];
  const biased = decision2AnswerFromLogits({
    logits: [0, 0, 0], question: scoreQ3, keys: ["0", "1", "2"],
    scoreBias: { 3: [0, 10, 0] },
  });
  assert.equal(biased.level, 1);
  assert.ok(biased.score > 0.9 && biased.score < 1.1);
  assert.deepEqual(Object.keys(biased.legend), ["0", "1", "2"]);

  const choice = decision2AnswerFromLogits({
    logits: [2, 0], question: choiceQ, keys: ["billing", "sales"], scoreBias: {},
  });
  assert.equal(choice.choice, "billing");
  assert.ok(choice.probabilities.billing > 0.8);

  const nq = normalizeQuestions([{ type: "noul", instructions: "x" }])[0];
  const noul = decision2AnswerFromLogits({
    logits: [0, 3], question: nq, keys: ["false", "true"], scoreBias: {},
  });
  assert.ok(noul.noul > 0.9);
  assert.deepEqual(Object.keys(noul.probabilities), ["no", "yes"]);
});

function fakeOrt() {
  class Tensor {
    constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
  }
  const calls = [];
  return {
    Tensor,
    InferenceSession: {
      create: async () => ({
        run: async (inputs) => {
          calls.push(inputs);
          const B = inputs.input_ids.dims[0], K = inputs.option_pos.dims[1];
          const data = new Float32Array(B * K);
          // candidate 0 wins every row
          for (let b = 0; b < B; b++) data[b * K] = 5;
          return { logits: new Tensor("float32", data, [B, K]) };
        },
        release: async () => {},
      }),
    },
    __calls: calls,
  };
}

function fakeTransformers() {
  const tok = (text) => ({ input_ids: { data: fakeEncode(text) } });
  return { AutoTokenizer: { from_pretrained: async () => tok } };
}

function fakeFetch(config) {
  return async (url, opts) => {
    if (String(url).endsWith("config.json")) {
      return { ok: true, json: async () => config };
    }
    if (opts?.method === "HEAD") return { ok: false };
    throw new Error(`unexpected fetch ${url}`);
  };
}

const CONFIG = { decision2: { score_bias: { 5: [0, 0, 0, 0, 0] } } };

test("createDecision2Decider batches questions and decodes answers", async () => {
  const ort = fakeOrt();
  const decider = await createDecision2Decider({
    ort,
    transformers: fakeTransformers(),
    fetchImpl: fakeFetch(CONFIG),
    device: "wasm",
    sessionFactory: async (url, opts) => ort.InferenceSession.create(url, opts),
  });
  assert.equal(decider.info.family, "decision2");
  assert.equal(decider.info.model, DECISION2_DEFAULT_MODEL);
  assert.equal(decider.info.revision, DECISION2_DEFAULT_REVISION);

  const { answers } = await decider.decide("state", [choiceQ, noulQ, scoreQ]);
  assert.equal(answers.length, 3);
  assert.equal(answers[0].choice, "billing");
  // candidate 0 ("false") wins the rigged logits, so P(true) is tiny
  assert.ok(answers[1].noul < 0.1);
  assert.equal(answers[2].level, 0);

  // one forward pass for the whole batch, with the 4 graph inputs
  assert.equal(ort.__calls.length, 1);
  const inputs = ort.__calls[0];
  assert.deepEqual(Object.keys(inputs).sort(),
    ["answer_pos", "attention_mask", "input_ids", "option_pos"]);
  assert.deepEqual(inputs.input_ids.dims[0], 3);
  assert.deepEqual(inputs.option_pos.dims, [3, 2]);

  await decider.dispose();
});
