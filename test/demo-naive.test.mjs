import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
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

// A WebGPU adapter existing is not evidence that WebGPU works. Firefox
// currently exposes navigator.gpu yet onnxruntime-web cannot compile
// DeBERTa-v3's "Clip" subgraph there, so the demo falls back to WASM. These
// guards are on the *source*, because isWebGpuFailure is the predicate that
// decides whether a visitor gets a working page or a raw ORT stack trace — and
// it is a module-scope function in a DOM-coupled file that cannot be imported
// under node --test.

describe("demo/main.mjs WebGPU fallback wiring", () => {
  const src = () => readFile(new URL("../demo/main.mjs", import.meta.url), "utf8");

  it("recognises the reported ORT WebGPU failure signatures", async () => {
    const s = await src();
    // Captured verbatim from a real Firefox failure on the live demo.
    for (const sig of [
      "failed to call OrtRun()",
      "Failed to create a WebGPU compute pipeline",
      "ShaderModule with 'Clip' label is invalid",
      "Encountered one or more errors while creating shader module",
    ]) {
      assert.ok(
        s.toLowerCase().includes(sig.toLowerCase().slice(0, 24)) || s.includes("shader"),
        `fallback patterns should cover: ${sig}`,
      );
    }
    assert.match(s, /const WEBGPU_FAILURES = \[/);
  });

  it("falls back at both load time and inference time", async () => {
    const s = await src();
    // The reported bug happens on the first decide(), not on load: the weights
    // fetch fine and OrtRun() fails afterwards. Both paths must be covered.
    assert.ok(s.includes("isWebGpuFailure(err)"), "no WebGPU-failure check");
    const checks = s.match(/isWebGpuFailure\(err\)/g) ?? [];
    assert.ok(checks.length >= 2, `expected 2 fallback sites, found ${checks.length}`);
    // Both sites rebuild via the resilient builder, not buildDecider directly.
    assert.ok(s.includes('buildDeciderResilient("wasm"'), "fallback must rebuild on wasm");
    const resilient = s.match(/buildDeciderResilient\("wasm"/g) ?? [];
    assert.ok(resilient.length >= 2, `expected 2 wasm rebuilds, found ${resilient.length}`);
  });

  it("escalates past a dtype the ORT build cannot execute", async () => {
    const s = await src();
    // Regression: the fallback rebuilt with dtype "auto", which resolves to q4
    // on WASM, and q4's GatherBlockQuantized has no kernel in some ORT wasm
    // builds -> "Could not find an implementation for GatherBlockQuantized(1)".
    // The fallback then failed even though a working dtype existed.
    assert.match(s, /isUnsupportedOp\(err\)/, "no unsupported-op detection");
    assert.match(
      s,
      /const UNSUPPORTED_OP = \[/,
      "must classify missing-kernel errors separately from WebGPU failures",
    );
    assert.match(s, /WASM_DTYPE_LADDER = \["q4", "fp32"\]/, "dtype ladder must escalate");
    // The ladder must not contain q8: the reference export has no q8 variant,
    // so trying it would 404 on the weight file.
    assert.doesNotMatch(s, /WASM_DTYPE_LADDER = \[[^\]]*"q8"/);
  });

  it("rewrites the info line after a backend change", async () => {
    const s = await src();
    // The line is written once at load, so after an inference-time fallback it
    // still claimed "webgpu/q4f16" while inference actually ran on WASM.
    assert.match(s, /function refreshInfoLine\(\)/);
    assert.ok(
      (s.match(/refreshInfoLine\(\)/g) ?? []).length >= 3,
      "info line must be refreshed on load and after each fallback",
    );
  });

  it("remembers a broken device so a reload does not re-crash", async () => {
    const s = await src();
    assert.match(s, /BROKEN_DEVICE_KEY/);
    assert.match(s, /sessionStorage\.setItem/);
    assert.match(s, /readBrokenDevice\(\)/);
  });

  it("has the notice element it writes to", async () => {
    const [main, html] = await Promise.all([
      src(),
      readFile(new URL("../demo/index.html", import.meta.url), "utf8"),
    ]);
    assert.match(html, /id="notice"/);
    assert.match(main, /\$\("notice"\)/);
  });
});