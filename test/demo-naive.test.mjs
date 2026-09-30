import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { parseAll, parseOne } from "../demo/naive.mjs";
import { PRESETS } from "../demo/presets.mjs";

// The demo's keyword baseline is shipped on GitHub Pages and screenshotted into
// the README, so a crash in it is a user-visible bug even though demo/ is not in
// the published tarball. These tests exist because a wrong LEXICON key took out
// the `score` branch at runtime and only on presets that contained sentiment
// words — exactly the case a happy-path check misses.

describe("demo/naive.mjs (raw-text-parsing baseline)", () => {
  it("answers every preset without throwing", () => {
    for (const p of PRESETS) {
      const r = parseAll(p.state, p.questions);
      assert.equal(r.passes, p.questions.length, `${p.id}: one rule set per question`);
      assert.equal(r.answers.length, p.questions.length, `${p.id}: one answer per question`);
      for (const a of r.answers) {
        assert.equal(typeof a.rule, "string");
        assert.equal(typeof a.ruled, "boolean");
        assert.equal(a.confidence, null, "a rule parser can never return a confidence");
      }
    }
  });

  it("exercises the score branch, which needs sentiment keywords to be reached", () => {
    // Regression guard: this path was unreachable from PRESETS[0], so the whole
    // `score` arm shipped untested.
    const score = {
      type: "score",
      instructions: "How positive is this?",
      options: ["very negative", "negative", "neutral", "positive", "very positive"],
    };
    const a = parseOne("The sound is fantastic and the design is beautiful.", score);
    assert.equal(a.type, "score");
    assert.equal(a.ruled, true);
    assert.ok(a.value >= 0 && a.value < score.options.length);
  });

  it("returns no answer rather than a wrong one when nothing matches", () => {
    const a = parseOne("Quantum chromodynamics seminar next Tuesday.", {
      type: "choice",
      instructions: "Which area?",
      options: ["billing", "refunds"],
    });
    assert.equal(a.ruled, false);
    assert.equal(a.value, null);
  });

  it("honours negation on a yes/no question", () => {
    const q = { type: "noul", instructions: "The customer is asking for a refund." };
    assert.equal(parseOne("I would like a refund please.", q).value, 1);
    assert.equal(parseOne("I am not asking for a refund.", q).value, 0);
  });

  it("only reports passes/matched, never probabilities", () => {
    const r = parseAll(PRESETS[0].state, PRESETS[0].questions);
    assert.equal(r.passes, PRESETS[0].questions.length);
    assert.ok(r.matched >= 0 && r.matched <= r.answers.length);
    for (const a of r.answers) {
      assert.equal(a.probabilities, undefined, "the baseline must not fake a distribution");
    }
  });
});
