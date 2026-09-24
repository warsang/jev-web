import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { softmaxWithTemperature, answersFromScores } from "../src/answers.mjs";

const sum = (xs) => xs.reduce((a, b) => a + b, 0);

describe("softmaxWithTemperature", () => {
  it("returns a normalized distribution", () => {
    const p = softmaxWithTemperature([1, 2, 3]);
    assert.equal(p.length, 3);
    assert.ok(Math.abs(sum(p) - 1) < 1e-12);
    assert.ok(p[2] > p[1] && p[1] > p[0]);
  });

  it("temperature < 1 sharpens, > 1 flattens", () => {
    const sharp = softmaxWithTemperature([2, 0], 0.5);
    const flat = softmaxWithTemperature([2, 0], 2);
    assert.ok(sharp[0] > flat[0]);
    assert.ok(Math.abs(sum(sharp) - 1) < 1e-12);
  });

  it("is shift-invariant and guards bad temperatures", () => {
    const a = softmaxWithTemperature([5, 1]);
    const b = softmaxWithTemperature([105, 101]);
    assert.ok(Math.abs(a[0] - b[0]) < 1e-12);
    const t = softmaxWithTemperature([5, 1], -3); // falls back to 1
    assert.ok(Math.abs(sum(t) - 1) < 1e-12);
  });
});

describe("answersFromScores", () => {
  const questions = [
    { type: "choice", instructions: "Which?", options: ["a", "b", "c"] },
    { type: "score", instructions: "How much?", options: ["none", "some", "lots"] },
    { type: "noul", instructions: "Is it b?", options: ["no", "yes"] },
  ];
  const groups = [[0, 1, 2], [3, 4, 5], [6, 7]];
  // choice -> c wins; score -> leans level 1..2; noul -> p(yes)=0.75
  const scores = [0, 1, 3, 0, 2, 1, Math.log(0.25), Math.log(0.75)];

  it("decodes choice, score and noul answers", () => {
    const answers = answersFromScores(scores, { questions, groups, temperature: 1 });
    const [choice, score, noul] = answers;

    assert.equal(choice.type, "choice");
    assert.equal(choice.choice, "c");
    assert.equal(choice.index, 2);
    assert.ok(Math.abs(sum(Object.values(choice.probabilities)) - 1) < 1e-9);

    assert.equal(score.type, "score");
    // softmax([0,2,1]) -> expected level between 1 and 2
    assert.ok(score.score > 1 && score.score < 2);
    assert.equal(score.level, 1);

    assert.equal(noul.type, "noul");
    assert.ok(Math.abs(noul.noul - 0.75) < 1e-9);
    assert.ok(Math.abs(noul.confidence - 0.75) < 1e-9);
  });

  it("probability keys follow the option labels", () => {
    const [choice, , noul] = answersFromScores(scores, { questions, groups, temperature: 1 });
    assert.deepEqual(Object.keys(choice.probabilities), ["a", "b", "c"]);
    assert.deepEqual(Object.keys(noul.probabilities), ["no", "yes"]);
  });

  it("noul confidence is max(p, 1-p)", () => {
    const [, , noul] = answersFromScores(
      [1, 0, 0, 1, 1, 0, Math.log(0.9), Math.log(0.1)],
      { questions, groups, temperature: 1 },
    );
    assert.ok(Math.abs(noul.noul - 0.1) < 1e-9);
    assert.ok(Math.abs(noul.confidence - 0.9) < 1e-9);
  });
});
