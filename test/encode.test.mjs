import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { buildDecisionInput, typedMarkerIds, DecisionInputTooLongError } from "../src/encode.mjs";
import { normalizeQuestions, NOUL_OPTIONS } from "../src/questions.mjs";

const MARKERS = { cls: 1, sep: 2, state: 3, question: 4, option: 5 };

// One choice (2 options) + one noul (default 2 options).
const SIMPLE = [
  { type: "choice", instructionIds: [20], optionIds: [[30], [31]] },
  { type: "noul", instructionIds: [21], optionIds: [[40], [41]] },
];

describe("buildDecisionInput", () => {
  it("lays out tokens, span slots and pair groups", () => {
    const input = buildDecisionInput({ stateIds: [10, 11], markers: MARKERS, questions: SIMPLE });
    assert.deepEqual(input.inputIds, [
      1, 3, 10, 11, 4, 20, 5, 30, 5, 31, 4, 21, 5, 40, 5, 41, 2,
    ]);
    // -1 outside spans; option tokens get their pair index; question tokens
    // get totalPairs + question index.
    assert.deepEqual(input.seg, [-1, -1, -1, -1, -1, 4, -1, 0, -1, 1, -1, 5, -1, 2, -1, 3, -1]);
    assert.deepEqual(input.pairQ, [4, 4, 5, 5]);
    assert.deepEqual(input.pairOpt, [0, 1, 2, 3]);
    assert.deepEqual(input.groups, [[0, 1], [2, 3]]);
    assert.equal(input.totalPairs, 4);
    assert.equal(input.length, 17);
    assert.equal(input.truncated, false);
  });

  it("truncates long states to maxStateTokens and flags it", () => {
    const stateIds = Array.from({ length: 300 }, (_, i) => 100 + i);
    const input = buildDecisionInput({ stateIds, markers: MARKERS, questions: SIMPLE, maxStateTokens: 256 });
    assert.equal(input.truncated, true);
    assert.equal(input.inputIds[2], 100);
    assert.equal(input.inputIds[2 + 255], 355);
    assert.equal(input.inputIds[2 + 256], MARKERS.question); // state is exactly 256 tokens
  });

  it("rejects inputs over the sequence limit", () => {
    const stateIds = Array.from({ length: 256 }, () => 7);
    const heavy = Array.from({ length: 40 }, () => ({
      type: "choice",
      instructionIds: Array.from({ length: 5 }, () => 8),
      optionIds: [[9], [9, 9, 9, 9, 9]],
    }));
    assert.throws(
      () => buildDecisionInput({ stateIds, markers: MARKERS, questions: heavy }),
      DecisionInputTooLongError,
    );
  });

  it("validates markers and questions", () => {
    assert.throws(() => buildDecisionInput({ stateIds: [1], markers: {}, questions: SIMPLE }), /markers\.cls/);
    assert.throws(() => buildDecisionInput({ stateIds: [1], markers: MARKERS, questions: [] }), /non-empty/);
  });
});

describe("typedMarkerIds", () => {
  it("resolves single-token markers", () => {
    const table = { "[CLS]": 1, "[SEP]": 2, "[STATE]": 128001, "[Q]": 128002, "[OPT]": 128003 };
    const markers = typedMarkerIds((t) => [table[t]]);
    assert.deepEqual(markers, { cls: 1, sep: 2, state: 128001, question: 128002, option: 128003 });
  });

  it("fails loudly when markers are missing from the tokenizer", () => {
    assert.throws(() => typedMarkerIds((t) => (t === "[CLS]" ? [1] : [0, 1, 2])), /exactly one token/);
  });
});

describe("normalizeQuestions", () => {
  it("defaults noul options to no/yes", () => {
    const [q] = normalizeQuestions([{ type: "noul", instructions: "Is this positive?" }]);
    assert.deepEqual(q.options, NOUL_OPTIONS);
  });

  it("trims instructions and coerces option labels", () => {
    const [q] = normalizeQuestions([{ type: "choice", instructions: "  Which?  ", options: [1, 2] }]);
    assert.equal(q.instructions, "Which?");
    assert.deepEqual(q.options, ["1", "2"]);
  });

  it("enforces per-type limits", () => {
    assert.throws(() => normalizeQuestions([{ type: "pick" }]), /type must be/);
    assert.throws(() => normalizeQuestions([{ type: "choice", instructions: "x", options: ["only-one"] }]), /at least 2/);
    assert.throws(
      () => normalizeQuestions([{ type: "choice", instructions: "x", options: Array.from({ length: 256 }, (_, i) => `o${i}`) }]),
      /at most 255/,
    );
    assert.throws(
      () => normalizeQuestions([{ type: "score", instructions: "x", options: Array.from({ length: 11 }, (_, i) => `l${i}`) }]),
      /score needs 2-10/,
    );
    assert.throws(() => normalizeQuestions([{ type: "noul", instructions: "x", options: ["a", "b", "c"] }]), /exactly 2/);
    assert.throws(() => normalizeQuestions([]), /non-empty/);
  });
});
