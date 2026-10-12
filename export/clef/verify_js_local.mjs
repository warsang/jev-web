// Local end-to-end verification of the clef browser export through the REAL
// jev-web runtime: the exported ONNX graph + the checkpoint tokenizer,
// compared against a reference dump produced by the Python reference
// implementation.
//
//   node export/clef/verify_js_local.mjs
//
// Env (defaults shown):
//   CLEF_VERIFY_GRAPH   ~/workspace/clef-export/final/onnx/clef_text_fp16.onnx
//   CLEF_VERIFY_CKPT    ~/workspace/clef-export/ckpt
//   CLEF_VERIFY_REF     <repo>/test/fixtures/clef-e2e-reference.json
//
// The reference dump is written by:
//   python export/clef/make_js_fixtures.py --e2e   (or any script that runs
//   joint_schema_model + softmax and writes {state, questions, perQuestion});
// see export/clef/RESUME.md.
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { AutoTokenizer } from "@huggingface/transformers";
import ort from "onnxruntime-node";
import { createClefDecider } from "../../src/clef.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const WORKSPACE = join(homedir(), "workspace");
const GRAPH = process.env.CLEF_VERIFY_GRAPH
  ?? join(WORKSPACE, "clef-export", "final", "onnx", "clef_text_fp16.onnx");
const CKPT = process.env.CLEF_VERIFY_CKPT
  ?? join(WORKSPACE, "clef-export", "ckpt");
const REF_PATH = process.env.CLEF_VERIFY_REF
  ?? resolve(HERE, "..", "..", "test", "fixtures", "clef-e2e-reference.json");
const FINAL = dirname(dirname(GRAPH));

const reference = JSON.parse(readFileSync(REF_PATH, "utf8"));
const questions = Object.entries(reference.questions).map(([id, q]) => {
  if (q.type === "choice") {
    return { id, type: "choice", instructions: q.instructions,
             options: Object.keys(q.criteria), criteria: q.criteria };
  }
  if (q.type === "noul") return { id, type: "noul", instructions: q.instructions };
  return { id, type: "score", instructions: q.instructions, options: q.criteria };
});

const decider = await createClefDecider({
  ort,
  transformers: { AutoTokenizer },
  device: "cpu",
  tokenizerModel: CKPT,
  sessionFactory: async () => ort.InferenceSession.create(GRAPH, { executionProviders: ["cpu"] }),
  fetchImpl: async (url) => {
    if (String(url).endsWith("clef_config.json")) {
      return { ok: true, json: async () => JSON.parse(readFileSync(join(FINAL, "clef_config.json"))) };
    }
    throw new Error(`unexpected fetch: ${url}`);
  },
});

const { answers, length, timings } = await decider.decide(reference.state, questions);
console.log(`graph: ${GRAPH}`);
console.log(`decide(): ${length} tokens, ${timings.runMs} ms\n`);

let worst = 0;
answers.forEach((answer, i) => {
  const ref = reference.perQuestion[i];
  console.log(`${ref.questionId} (${answer.type})`);
  const pairs = []; // [label, probability, referenceIndex]
  if (answer.type === "noul") {
    const [noLabel, yesLabel] = Object.keys(answer.probabilities);
    pairs.push([noLabel, answer.probabilities[noLabel], ref.optionIds.indexOf("false")]);
    pairs.push([yesLabel, answer.probabilities[yesLabel], ref.optionIds.indexOf("true")]);
  } else {
    for (const [key, p] of Object.entries(answer.probabilities)) {
      pairs.push([key, p, ref.optionIds.indexOf(key)]);
    }
  }
  for (const [key, p, refIndex] of pairs) {
    const expected = ref.probs[refIndex];
    const diff = Math.abs(p - expected);
    worst = Math.max(worst, diff);
    console.log(`  ${key}: ${p.toFixed(4)} (ref ${expected.toFixed(4)}, d ${diff.toExponential(1)})`);
  }
  const argmax = pairs.reduce((a, b) => (b[1] > a[1] ? b : a));
  const refArgmaxIndex = ref.probs.indexOf(Math.max(...ref.probs));
  const ok = argmax[2] === refArgmaxIndex;
  console.log(`  argmax: ${argmax[0]} (ref ${ref.optionIds[refArgmaxIndex]})${ok ? "" : "  <-- MISMATCH"}`);
  if (!ok) process.exitCode = 1;
});
console.log(`\nworst probability diff: ${worst.toExponential(2)}`);
if (worst > 2e-2) {
  console.log("E2E FAILED: probabilities drift past 2e-2");
  process.exitCode = 1;
} else {
  console.log("E2E OK: the JS runtime reproduces the reference");
}
await decider.dispose();
