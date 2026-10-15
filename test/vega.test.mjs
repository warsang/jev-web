/**
 * Vega family: protocol (prompt / option rendering), packing, thermal
 * calibration and answer decoding — validated against fixtures generated from
 * the reference implementation (export/vega/make_js_fixtures.py runs the
 * author's vega_common.py on the real tokenizer and writes
 * test/fixtures/vega-protocol.json).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import {
  VEGA_DEFAULTS,
  buildVegaPrompt,
  packVegaBatch,
  packVegaChunks,
  vegaAnswers,
  vegaOptionTexts,
  vegaQuestionDef,
  vegaThermalFeatures,
  vegaThermalProbabilities,
} from "../src/vega.mjs";

const fixture = JSON.parse(readFileSync(
  fileURLToPath(new URL("./fixtures/vega-protocol.json", import.meta.url)), "utf8"));
const cfg = fixture.config;

/** encode() replaying the reference tokenizer's piece encodings for a case. */
function fixtureEncoder(c) {
  const map = new Map();
  map.set("Observation:\n", c.pieceIds.head[0]);
  map.set(c.questionText, c.pieceIds.question[0]);
  map.set("\nOutcome:", c.pieceIds.end[0]);
  c.optionTexts.forEach((text, i) => map.set(text, c.pieceIds.options[i]));
  map.set(c.state, c.pieceIds.state[0]);
  if (c.pieceIds.query_first?.[0]) {
    map.set(`Question to keep in mind while reading (${c.name === "score" ? "score" : c.name === "boolean" ? "boolean" : "choice"}): ${fixtureQuestionIns(c)}\n\n`,
      c.pieceIds.query_first[0]);
  }
  return (text) => {
    const ids = map.get(text);
    if (ids) return ids;
    throw new Error(`unexpected encode() input: ${JSON.stringify(text.slice(0, 60))}`);
  };
}

function fixtureQuestionIns(c) {
  return {
    "choice-short-state": "Which team handles this?",
    score: "How severe is this?",
    boolean: "Does this convey urgency?",
  }[c.name];
}

test("buildVegaPrompt reproduces the reference prompt exactly", () => {
  for (const c of fixture.cases) {
    const encode = fixtureEncoder(c);
    const qtype = c.name === "score" ? "score" : c.name === "boolean" ? "boolean" : "choice";
    const { ids, spans } = buildVegaPrompt({
      encode,
      stateText: c.state,
      qdef: { t: qtype, ins: fixtureQuestionIns(c), crit: fixtureCrit(c) },
      maxLen: cfg.max_len,
      bosId: null,
      queryFirstMaxState: cfg.query_first_max_state,
      optMaxTokens: cfg.opt_max_tokens,
      insMaxTokens: 256,
    });
    assert.deepEqual(ids, c.ids, `${c.name}: ids`);
    assert.deepEqual(spans.state, c.spans.state, `${c.name}: state span`);
    assert.deepEqual(spans.question, c.spans.question, `${c.name}: question span`);
    assert.deepEqual(spans.options, c.spans.options, `${c.name}: option spans`);
    assert.equal(spans.last, c.spans.last, `${c.name}: last token`);
  }
});

test("buildVegaPrompt reports truncated only when user text was shortened", () => {
  // char-level stand-in: one id per character
  const encode = (text) => Array.from({ length: String(text).length }, () => 7);
  const qdef = { t: "choice", ins: "Instructions.", crit: { a: null, b: null } };
  assert.equal(
    buildVegaPrompt({ encode, stateText: "state", qdef, maxLen: 4096 }).truncated,
    false, "everything fits");
  assert.equal(
    buildVegaPrompt({ encode, stateText: "state", qdef, maxLen: 4096, optMaxTokens: 3 }).truncated,
    true, "an option over its cap");
  assert.equal(
    buildVegaPrompt({ encode, stateText: "state", qdef, maxLen: 4096, insMaxTokens: 5 }).truncated,
    true, "instructions over their cap");
});

function fixtureCrit(c) {
  if (c.name === "score") return ["cosmetic", "annoying", "severe", "outage"];
  if (c.name === "boolean") return {};
  return {
    "billing": "payments and refunds",
    "tech": "bugs",
    "sales": "pricing",
  };
}

test("packVegaBatch pools each span to the reference row sums", () => {
  for (const c of fixture.cases) {
    const encode = fixtureEncoder(c);
    const qtype = c.name === "score" ? "score" : c.name === "boolean" ? "boolean" : "choice";
    const { ids, spans } = buildVegaPrompt({
      encode, stateText: c.state,
      qdef: { t: qtype, ins: fixtureQuestionIns(c), crit: fixtureCrit(c) },
      maxLen: cfg.max_len, bosId: null,
      queryFirstMaxState: cfg.query_first_max_state,
      optMaxTokens: cfg.opt_max_tokens, insMaxTokens: 256,
    });
    const batch = packVegaBatch([{ ids, spans, qtype: 0, options: c.optionTexts }], {
      chunks: cfg.memory_chunks,
    });
    assert.equal(batch.L, ids.length, `${c.name}: padded length`);
    const rows = batch.poolDims[1];
    assert.equal(rows, c.nseg + cfg.memory_chunks, `${c.name}: rows = segments + chunks`);
    const rowSums = [];
    for (let r = 0; r < c.nseg; r++) {
      let s = 0;
      for (let t = 0; t < batch.L; t++) s += batch.pool[r * batch.L + t];
      rowSums.push(Number(s.toFixed(6)));
    }
    assert.deepEqual(rowSums, c.poolRowSums, `${c.name}: pool row sums`);
    // the final-token row is a one-hot; padded option rows never leak in
    assert.equal(rowSums[2], 1);
  }
});

test("packVegaChunks cuts the state into equal pieces like _chunk_pool", () => {
  const token = (i) => (i % 90) + 10;
  const stateLen = 600; // >= memory_min_tokens (128) -> k = min(C, 600/128) = 4
  const ids = Array.from({ length: stateLen + 20 }, (_, i) => token(i));
  const spans = { state: [2, 2 + stateLen], question: [2 + stateLen, 2 + stateLen + 8], options: [[2 + stateLen + 8, 2 + stateLen + 12]], last: ids.length - 1 };
  const batch = packVegaBatch([{ ids, spans, qtype: 0, options: ["a - b"] }], {
    chunks: cfg.memory_chunks,
  });
  const chunkMask = packVegaChunks(batch.pool, [{ ids, spans }], batch.L, {
    chunks: cfg.memory_chunks,
    chunkMinTokens: cfg.memory_min_tokens,
    chunkTokens: cfg.memory_chunk_tokens,
    nseg: batch.nseg,
  });
  const active = [...chunkMask].filter(Boolean).length;
  assert.equal(active, 4, "600-token state -> 4 chunks");
  const nseg = batch.nseg;
  const L = batch.L;
  const sums = [];
  for (let c = 0; c < cfg.memory_chunks; c++) {
    let s = 0;
    for (let t = 0; t < L; t++) s += batch.pool[(0 * (nseg + cfg.memory_chunks) + nseg + c) * L + t];
    sums.push(Number(s.toFixed(6)));
  }
  assert.deepEqual(sums, [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
});

test("packVegaChunks leaves short states unchunked", () => {
  const ids = Array.from({ length: 40 }, (_, i) => i + 5);
  const spans = { state: [2, 20], question: [20, 30], options: [[30, 34]], last: 39 };
  const batch = packVegaBatch([{ ids, spans, qtype: 0, options: ["a - b"] }], { chunks: 16 });
  const mask = packVegaChunks(batch.pool, [{ ids, spans }], batch.L, {
    chunks: 16, chunkMinTokens: cfg.memory_min_tokens,
    chunkTokens: cfg.memory_chunk_tokens, nseg: batch.nseg,
  });
  assert.equal([...mask].some(Boolean), false, "state under memory_min_tokens gets no chunks");
});

test("vegaOptionTexts renders like the reference option_texts", () => {
  assert.deepEqual(
    vegaOptionTexts({ t: "choice", ins: "x", crit: { billing: "pay", tech: "bugs" } }),
    ["billing - pay", "tech - bugs"]);
  assert.deepEqual(
    vegaOptionTexts({ t: "choice", ins: "x", crit: { a: null, b: "" } }),
    ["a", "b"]);
  assert.deepEqual(
    vegaOptionTexts({ t: "score", ins: "x", crit: ["calm", "urgent"] }),
    ["level 0 - calm", "level 1 - urgent"]);
  assert.deepEqual(
    vegaOptionTexts({ t: "boolean", ins: "x", crit: {} }),
    ["no - the statement does not hold", "yes - the statement holds"]);
  assert.deepEqual(
    vegaOptionTexts({ t: "boolean", ins: "x", crit: { false: "it is false", true: "it is true" } }),
    ["no - it is false", "yes - it is true"]);
});

test("vegaQuestionDef maps jev-web questions onto vega defs", () => {
  assert.deepEqual(
    vegaQuestionDef({ type: "choice", instructions: "Team?", options: ["a", "b"], criteria: { a: "desc", b: "" } }),
    { t: "choice", ins: "Team?", crit: { a: "desc", b: "" } });
  assert.deepEqual(
    vegaQuestionDef({ type: "choice", instructions: "Team?", options: ["a", "b"] }),
    { t: "choice", ins: "Team?", crit: { a: null, b: null } });
  assert.deepEqual(
    vegaQuestionDef({ type: "score", instructions: "How?", options: ["lo", "hi"] }),
    { t: "score", ins: "How?", crit: ["lo", "hi"] });
  assert.deepEqual(
    vegaQuestionDef({ type: "noul", instructions: "Urgent?", options: ["no", "yes"] }),
    { t: "boolean", ins: "Urgent?", crit: {} });
});

test("thermal features and probabilities match the reference fit", () => {
  const t = fixture.thermal;
  const phys = vegaThermalFeatures({
    kinetic: [0.01, 0.4, 0.02],
    nearest: [0.5, 3.2, 0.1],
    stepsUsed: [12, 5, 12],
    nOpts: t.nOpts,
    maxSteps: cfg.n_steps,
  });
  for (let b = 0; b < 3; b++) {
    for (let f = 0; f < 4; f++) {
      assert.ok(Math.abs(phys[b][f] - t.phys[b][f]) < 1e-9, `phys[${b}][${f}]`);
    }
  }
  const probs = vegaThermalProbabilities({
    logits: t.logits.flat(),
    rowWidth: 4,
    qtypes: t.qtypes,
    phys: phys.flat(),
    nOpts: t.nOpts,
    theta: t.theta,
  });
  probs.forEach((row, b) => {
    row.forEach((p, j) => {
      assert.ok(Math.abs(p - t.expectedProbs[b][j]) < 1e-6,
        `probs[${b}][${j}]: ${p} vs ${t.expectedProbs[b][j]}`);
    });
    const sum = row.reduce((a, v) => a + v, 0);
    assert.ok(Math.abs(sum - 1) < 1e-6, "probabilities sum to 1");
  });
});

// ---------------------------------------------------------------------------
// createVegaDecider with injected fakes (no network, no real graphs)
// ---------------------------------------------------------------------------

function fakeVegaSessions({ logits = [1.5, -0.4, 0.2, 0.9] } = {}) {
  class Tensor {
    constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
  }
  const torsoCalls = [];
  const engineCalls = [];
  const torso = {
    run: async (inputs) => {
      torsoCalls.push(inputs);
      const [b, rows] = inputs.pool.dims[0] === undefined ? [1, 1] : [inputs.pool.dims[0], inputs.pool.dims[1]];
      const d = 1024;
      // [B, Lcap=2, rows, d]: pretend every segment is a constant vector.
      const pooled = new Float32Array(b * 2 * rows * d);
      for (let i = 0; i < pooled.length; i++) pooled[i] = 0.01 * ((i % 17) - 8);
      return { pooled: new Tensor("float32", pooled, [b, 2, rows, d]) };
    },
    release: async () => {},
  };
  const engine = {
    run: async (inputs) => {
      engineCalls.push(inputs);
      const b = inputs.qtype.dims[0];
      const K = inputs.opts.dims[1];
      // favour option 0 in every row, so decoded answers are predictable
      const data = new Float32Array(b * K);
      for (let i = 0; i < b; i++) { data[i * K] = 5; for (let j = 1; j < K; j++) data[i * K + j] = -1; }
      return {
        logits: new Tensor("float32", data, [b, K]),
        kinetic: new Tensor("float32", [0.02, 0.01].slice(0, b), [b]),
        nearest: new Tensor("float32", [0.3, 0.2].slice(0, b), [b]),
        steps_used: new Tensor("int64", BigInt64Array.from([12, 5].slice(0, b), BigInt), [b]),
        grad_norm: new Tensor("float32", [0.1, 0.1].slice(0, b), [b]),
      };
    },
    release: async () => {},
  };
  return { torso, engine, torsoCalls, engineCalls, Tensor };
}

test("createVegaDecider wires both graphs and decodes all three kinds", async () => {
  const { createVegaDecider } = await import("../src/vega.mjs");
  const { torso, engine, torsoCalls, engineCalls, Tensor } = fakeVegaSessions();
  const ort = { env: { wasm: {} }, Tensor, InferenceSession: { create: async () => torso } };
  const transformers = { AutoTokenizer: { from_pretrained: async () => fakeVegaTokenizer() } };
  const decider = await createVegaDecider({
    ort,
    transformers,
    device: "wasm",
    sessionFactory: await fakeVegaSessionFactory({ torso, engine }),
    fetchImpl: async () => ({
      ok: true,
      json: async () => ({
        ...cfg,
        // The char-level stand-in tokenizes ~1 id per character, so the
        // reference 32-token option cap would clip the noul defaults (35 ids)
        // even though the real tokenizer packs them well under it.
        opt_max_tokens: 64,
        calibration: fixture.calibration,
        theta: fixture.thermal.theta,
      }),
    }),
  });
  assert.equal(decider.info.family, "vega");
  assert.equal(decider.info.maxLen, cfg.max_len);
  assert.match(decider.info.calibrationSource, /vega_config\.json$/);
  assert.equal(decider.info.nSteps, cfg.n_steps);

  const choiceQ = { type: "choice", instructions: "Which team?", options: ["billing", "tech", "sales"] };
  const noulQ = { type: "noul", instructions: "Urgent?", options: ["no", "yes"] };
  const scoreQ = { type: "score", instructions: "How bad?", options: ["lo", "mid", "hi"] };
  const { answers, truncated, length, prompts, timings } = await decider.decide(
    "I was charged twice for the same order.", [choiceQ, noulQ, scoreQ]);

  assert.equal(torsoCalls.length, 1, "one causal read for the whole batch");
  assert.equal(engineCalls.length, 1, "one engine forward for the whole batch");
  assert.equal(truncated, false);
  assert.ok(length > 0);
  assert.equal(prompts.length, 3);
  assert.ok(timings.totalMs >= 0 && timings.runMs >= 0);

  const [a0, a1, a2] = answers;
  assert.equal(a0.type, "choice");
  // the fake logits favour option 0 for every row
  assert.equal(a0.choice, "billing");
  assert.ok(a0.confidence > 0.5);
  assert.ok("predictionSet" in a0 && "abstain" in a0 && "unbound" in a0);
  assert.equal("settleSteps" in a0 && "energy" in a0, true);
  assert.equal(a1.type, "noul");
  assert.ok(a1.noul > 0 && a1.noul < 1);
  assert.equal(a2.type, "score");
  assert.equal(a2.level, 0);
  assert.ok(a2.score >= 0 && a2.score <= 2);
  await decider.dispose();
});

test("createVegaDecider keeps pinned defaults without a config and honours maxLen", async () => {
  const { createVegaDecider } = await import("../src/vega.mjs");
  const { torso, engine, Tensor } = fakeVegaSessions();
  const ort = { env: { wasm: {} }, Tensor, InferenceSession: { create: async () => torso } };
  const decider = await createVegaDecider({
    ort,
    transformers: { AutoTokenizer: { from_pretrained: async () => fakeVegaTokenizer() } },
    device: "wasm",
    sessionFactory: await fakeVegaSessionFactory({ torso, engine }),
    fetchImpl: async () => ({ ok: false, json: async () => ({}) }),
    maxLen: 512,
  });
  assert.equal(decider.info.maxLen, 512, "explicit maxLen wins over the checkpoint window");
  assert.equal(decider.info.calibrationSource, null);
  assert.equal(decider.info.nSteps, VEGA_DEFAULTS.nSteps);
  const { answers } = await decider.decide("s", [{ type: "choice", instructions: "?", options: ["a", "b"] }]);
  assert.equal(answers.length, 1);
  await decider.dispose();
});

async function fakeVegaSessionFactory(sessions) {
  // (url, opts) => the right fake session by file path
  return async (url) => {
    const path = String(url);
    if (path.includes("torso")) return sessions.torso;
    if (path.includes("engine")) return sessions.engine;
    throw new Error(`unexpected model URL: ${path}`);
  };
}

/** char-level tokenizer stand-in. */
function fakeVegaTokenizer() {
  const tok = (text) => ({
    input_ids: {
      data: BigInt64Array.from([...String(text)].map((c) => (c.charCodeAt(0) % 90) + 10), BigInt),
    },
  });
  tok.bos_token_id = null;
  tok.decode = (ids) => `decoded:${Array.from(ids).join(" ")}`;
  return tok;
}

test("vegaAnswers decodes the typed answers with conformal + abstain flags", () => {  const t = fixture.thermal;
  const phys = vegaThermalFeatures({
    kinetic: [0.01, 0.4, 0.02], nearest: [0.5, 3.2, 0.1],
    stepsUsed: [12, 5, 12], nOpts: t.nOpts, maxSteps: cfg.n_steps,
  });
  const probs = vegaThermalProbabilities({
    logits: t.logits.flat(), rowWidth: 4, qtypes: t.qtypes,
    phys: phys.flat(), nOpts: t.nOpts, theta: t.theta,
  });
  const labels = [["billing", "tech", "sales"], [0, 1, 2, 3], ["no", "yes"]];
  const answers = vegaAnswers({
    probs, qtypes: t.qtypes, labels,
    physics: { nearest: [0.5, 3.2, 0.1], stepsUsed: [12, 5, 12] },
    rawLogits: t.logits.flat(), rowWidth: 4,
    calibration: fixture.calibration, alpha: "0.1", risk: "0.1",
  });
  assert.equal(answers[0].type, "choice");
  assert.equal(answers[0].choice, "billing");
  assert.equal(answers[0].index, 0);
  assert.ok(answers[0].confidence > 0.9);
  assert.ok(answers[0].predictionSet.includes("billing"));
  assert.equal(answers[0].abstain, false);
  assert.equal(answers[0].unbound, false);
  assert.equal(answers[0].settleSteps, 12);
  assert.ok(answers[0].energy < 0);

  assert.equal(answers[1].type, "score");
  assert.equal(answers[1].level, 1); // 0.498 vs 0.491 -> level 1
  assert.ok(Math.abs(answers[1].score - (1 * 0.498343 + 2 * 0.491095 + 3 * 0.000094)) < 0.02);

  assert.equal(answers[2].type, "noul");
  assert.ok(Math.abs(answers[2].noul - 0.201653) < 1e-5);

  // abstain flips when the confidence drops below the risk threshold
  const flat = vegaAnswers({
    probs: [[0.5, 0.5]], qtypes: [0], labels: [["a", "b"]],
    physics: { nearest: [0.5], stepsUsed: [12] },
    rawLogits: [0, 0], rowWidth: 2,
    calibration: fixture.calibration, alpha: "0.1", risk: "0.1",
  });
  assert.equal(flat[0].abstain, true, "conf 0.5 < abstain(0.1) 0.5959");
  assert.deepEqual(flat[0].predictionSet.sort(), ["a", "b"]);
  // unbound flips past the energy threshold
  const ood = vegaAnswers({
    probs: [[0.9, 0.1]], qtypes: [0], labels: [["a", "b"]],
    physics: { nearest: [9.9], stepsUsed: [12] },
    rawLogits: [1, 0], rowWidth: 2,
    calibration: fixture.calibration, alpha: "0.1", risk: "0.1",
  });
  assert.equal(ood[0].unbound, true);
  assert.deepEqual(ood[0].predictionSet, ["a"], "0.1 < 1 - 0.7476, so only the top option");
});
