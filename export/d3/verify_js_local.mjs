// Local end-to-end verification of the d3 browser export through the REAL
// jev-web runtime: the exported torso + readout graphs and the checkpoint
// tokenizer, compared against reference probabilities produced by the Python
// reference runtime.
//
//   node export/d3/verify_js_local.mjs
//
// Env (defaults shown):
//   D3_VERIFY_DIR   ~/workspace/d3edge-export/final
//   D3_VERIFY_REF   <repo>/test/fixtures/d3-e2e-reference.json
//
// The reference dump is written by:
//   D3_SRC=<checkpoint> python export/d3/make_js_fixtures.py --e2e
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { AutoTokenizer } from "@huggingface/transformers";
import ort from "onnxruntime-node";
import { createDecision3Decider } from "../../src/decision3.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const FINAL = process.env.D3_VERIFY_DIR
  ?? join(homedir(), "workspace", "d3edge-export", "final");
const REF_PATH = process.env.D3_VERIFY_REF
  ?? resolve(HERE, "..", "..", "test", "fixtures", "d3-e2e-reference.json");
const FIXTURE_PATH = resolve(HERE, "..", "..", "test", "fixtures", "d3-protocol.json");

const reference = JSON.parse(readFileSync(REF_PATH, "utf8"));
const fixture = JSON.parse(readFileSync(FIXTURE_PATH, "utf8"));

const decider = await createDecision3Decider({
  ort,
  transformers: { AutoTokenizer },
  device: "cpu",
  tokenizerModel: FINAL,
  sessionFactory: async (url) =>
    String(url).includes("readout")
      ? ort.InferenceSession.create(join(FINAL, "onnx", "readout.onnx"), { executionProviders: ["cpu"] })
      : ort.InferenceSession.create(join(FINAL, "onnx", "torso_fp16.onnx"), { executionProviders: ["cpu"] }),
  fetchImpl: async (url) => {
    if (String(url).endsWith("decision_config.json")) {
      return { ok: true, json: async () => JSON.parse(readFileSync(join(FINAL, "decision_config.json"))) };
    }
    throw new Error(`unexpected fetch: ${url}`);
  },
});
console.log(`decider: ${decider.info.family}, ${decider.info.codes} codes, temperature ${decider.info.temperature}`);

let worst = 0;
let promptMismatches = 0;
for (const ref of reference) {
  const questions = Object.entries(ref.questions).map(([id, q]) => ({ id, ...q }));
  const { answers, prompts, timings } = await decider.decide(ref.state, questions);
  console.log(`\n${ref.name}: ${timings.runMs} ms`);

  const fixtureCase = fixture.cases.find((c) => c.name === ref.name);
  const firstId = Object.keys(ref.questions)[0];
  const fixtureQuestion = fixtureCase.questions.find((q) => q.id === firstId);
  if (prompts[0] !== fixtureQuestion.prompt) {
    promptMismatches += 1;
    console.log(`  prompt mismatch for ${firstId}`);
  }

  answers.forEach((answer, i) => {
    const key = Object.keys(ref.questions)[i];
    const expected = ref.perQuestion[key];
    const pairs = [];
    if (answer.type === "noul") {
      const [noLabel, yesLabel] = Object.keys(answer.probabilities);
      pairs.push(["false", answer.probabilities[noLabel]]);
      pairs.push(["true", answer.probabilities[yesLabel]]);
    } else {
      for (const [k, p] of Object.entries(answer.probabilities)) pairs.push([k, p]);
    }
    const diffs = pairs.map(([k, p]) => {
      const target = expected.probs[expected.keys.indexOf(k)];
      worst = Math.max(worst, Math.abs(p - target));
      return `${k}:${Math.abs(p - target).toExponential(1)}`;
    });
    console.log(`  ${key} (${answer.type}) ${diffs.join(" ")}`);
  });
}
console.log(`\nprompt rendering matches the reference for all cases: ${promptMismatches === 0}`);
console.log(`worst probability diff: ${worst.toExponential(2)}`);
if (promptMismatches > 0 || worst > 5e-2) {
  console.log("D3 E2E FAILED");
  process.exitCode = 1;
} else {
  console.log("D3 E2E OK: the JS runtime reproduces the reference within fp16 tolerance");
}
await decider.dispose();
