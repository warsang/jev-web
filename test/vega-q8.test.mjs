/**
 * Split Vega (nvkudva/vega-onnx-q8): JS-side perception (text M-RoPE positions,
 * zeroed past state, hidden-state pooling), adapter routing/calibration, and
 * the variant wiring.
 *
 * Fixtures come from the reference implementation and the real graphs:
 * export/vega/make_q8_js_fixtures.py runs vega_common.build_prompt (real
 * tokenizer), Perception._pooled_batch (fixed hidden states), thermal_tau /
 * softmax_np / route_adapters (the q8 export's vega_web.json blocks), and
 * records the shipped graphs' input names/shapes. The real-graph end-to-end
 * check lives in export/vega/06_verify_q8_export.py.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import {
  buildVegaPrompt,
  createVegaDecider,
  emptyVegaPast,
  poolVegaHidden,
  renderVegaText,
  vegaAnswers,
  vegaJsonDumps,
  vegaRoute,
  vegaTextPositions,
  vegaThermalFeatures,
  vegaThermalProbabilities,
  VEGA_DEFAULT_MODEL,
  VEGA_Q8_DEFAULT_MODEL,
  VEGA_Q8_DEFAULT_REVISION,
  VEGA_Q8_DEFAULT_SUBFOLDER,
  VEGA_VARIANTS,
} from "../src/vega.mjs";

const fixture = JSON.parse(readFileSync(
  fileURLToPath(new URL("./fixtures/vega-q8-protocol.json", import.meta.url)), "utf8"));
const cfg = fixture.prompt.config;

const QUESTION_INS = {
  "choice-short-state": "Which team handles this?",
  score: "How severe is this?",
  boolean: "Does this convey urgency?",
};
const QUESTION_CRIT = {
  "choice-short-state": { billing: "payments and refunds", tech: "bugs", sales: "pricing" },
  score: ["cosmetic", "annoying", "severe", "outage"],
  boolean: {},
};

function qtypeOf(name) {
  return name === "score" ? "score" : name === "boolean" ? "boolean" : "choice";
}

/** encode() replaying the reference tokenizer's piece encodings for a case. */
function fixtureEncoder(c) {
  const map = new Map();
  map.set("Observation:\n", c.pieceIds.head[0]);
  map.set(c.questionText, c.pieceIds.question[0]);
  map.set("\nOutcome:", c.pieceIds.end[0]);
  c.optionTexts.forEach((text, i) => map.set(text, c.pieceIds.options[i]));
  map.set(c.state, c.pieceIds.state[0]);
  if (c.pieceIds.query_first?.[0]) {
    map.set(`Question to keep in mind while reading (${qtypeOf(c.name)}): ${QUESTION_INS[c.name]}\n\n`,
      c.pieceIds.query_first[0]);
  }
  return (text) => {
    const ids = map.get(text);
    if (!ids) throw new Error(`unexpected encode() input: ${JSON.stringify(text.slice(0, 60))}`);
    return ids;
  };
}

test("vegaJsonDumps matches Python json.dumps (separators, escapes, ensure_ascii)", () => {
  assert.equal(vegaJsonDumps({ a: 1, b: [1, "x", true, null] }),
    '{"a": 1, "b": [1, "x", true, null]}');
  assert.equal(vegaJsonDumps("é", true), '"\\u00e9"');
  assert.equal(vegaJsonDumps("é", false), '"é"');
  assert.equal(vegaJsonDumps("😀", true), '"\\ud83d\\ude00"');
  assert.equal(vegaJsonDumps("a\nb\t\u0001"), '"a\\nb\\t\\u0001"');
  assert.equal(vegaJsonDumps({ 'k"x': "v\\" }), '{"k\\"x": "v\\\\"}');
  assert.equal(vegaJsonDumps(null), "null");
  assert.equal(vegaJsonDumps(3.5), "3.5");
  assert.equal(vegaJsonDumps([{ a: "b" }]), '[{"a": "b"}]');
  // the reference render_text: strings verbatim, anything else json.dumps
  assert.equal(renderVegaText("plain state"), "plain state");
  assert.equal(renderVegaText({ order: 12, tags: ["a", "b"] }),
    '{"order": 12, "tags": ["a", "b"]}');
});

test("buildVegaPrompt reproduces the reference prompt for the q8 config", () => {
  for (const c of fixture.prompt.cases) {
    const { ids, spans } = buildVegaPrompt({
      encode: fixtureEncoder(c),
      stateText: c.state,
      qdef: { t: qtypeOf(c.name), ins: QUESTION_INS[c.name], crit: QUESTION_CRIT[c.name] },
      maxLen: cfg.max_len,
      bosId: null,
      queryFirstMaxState: cfg.query_first_max_state,
      optMaxTokens: cfg.opt_max_tokens,
      insMaxTokens: 256,
      truncateState: false,
    });
    assert.deepEqual(ids, c.ids, `${c.name}: ids`);
    assert.deepEqual(spans.state, c.spans.state, `${c.name}: state span`);
    assert.deepEqual(spans.question, c.spans.question, `${c.name}: question span`);
    assert.deepEqual(spans.options, c.spans.options, `${c.name}: option spans`);
    assert.equal(spans.last, c.spans.last, `${c.name}: last token`);
  }
});

test("buildVegaPrompt refuses an over-long state when truncation is off", () => {
  const encode = (text) => (text === "STATE" ? new Array(4000).fill(1) : [1]);
  const qdef = { t: "choice", ins: "?", crit: { a: null, b: null } };
  assert.throws(() => buildVegaPrompt({
    encode, stateText: "STATE", qdef, maxLen: 512, truncateState: false,
  }), RangeError);
  const cut = buildVegaPrompt({ encode, stateText: "STATE", qdef, maxLen: 512 });
  assert.ok(cut.ids.length <= 512, "the default path still truncates");
  assert.equal(cut.truncated, true, "and reports it");
});

test("poolVegaHidden reproduces Perception._pooled_batch on the same hidden states", () => {
  const p = fixture.pool;
  const out = poolVegaHidden({
    h13: Float32Array.from(p.h13.flat()),
    h19: Float32Array.from(p.h19.flat()),
    spans: p.spans,
    d: p.d,
    chunks: cfg.memory_chunks,
    chunkMinTokens: cfg.memory_min_tokens,
    chunkTokens: cfg.memory_chunk_tokens,
  });
  const close = (a, b, label) => {
    assert.equal(a.length, b.length, `${label}: length`);
    a.forEach((v, i) => {
      assert.ok(Math.abs(v - b[i]) < 1e-6, `${label}[${i}]: ${v} vs ${b[i]}`);
    });
  };
  // feats are [2, 3, d]
  for (let l = 0; l < 2; l++) {
    for (let s = 0; s < 3; s++) {
      close(out.feats.subarray((l * 3 + s) * p.d, (l * 3 + s + 1) * p.d),
        p.expected.feats[l][s], `feats[${l}][${s}]`);
    }
  }
  p.expected.opts.forEach((row, k) => close(out.opts.subarray(k * p.d, (k + 1) * p.d), row, `opts[${k}]`));
  p.expected.chunks.forEach((row, c) => close(out.chunks.subarray(c * p.d, (c + 1) * p.d), row, `chunks[${c}]`));
  assert.deepEqual([...out.cmask], p.expected.cmask);
  assert.equal(p.expected.cmask.reduce((a, b) => a + b, 0), 3, "384-token state -> 3 chunks");
});

test("poolVegaHidden leaves short states unchunked", () => {
  const d = 2;
  const L = 40;
  const h = new Float32Array(L * d).fill(0.25);
  const spans = { state: [2, 20], question: [20, 30], options: [[30, 34]] };
  const out = poolVegaHidden({
    h13: h, h19: h, spans, d, chunks: 16,
    chunkMinTokens: cfg.memory_min_tokens, chunkTokens: cfg.memory_chunk_tokens,
  });
  assert.equal([...out.cmask].some(Boolean), false);
});

test("vegaRoute picks the most open adapter above the threshold, else base", () => {
  assert.equal(vegaRoute([0.6, 0.2], 0.5), 0);
  assert.equal(vegaRoute([0.2, 0.9], 0.5), 1);
  assert.equal(vegaRoute([0.4, 0.2], 0.5), -1);
  assert.equal(vegaRoute([0.5, 0.5], 0.5), 0, "a tie opens the first adapter");
  assert.equal(vegaRoute([], 0.5), -1);
  assert.equal(vegaRoute(null, 0.5), -1);
});

test("vegaTextPositions lays 0..L-1 on all three M-RoPE rows", () => {
  assert.deepEqual([...vegaTextPositions(3)], [0n, 1n, 2n, 0n, 1n, 2n, 0n, 1n, 2n]);
});

test("emptyVegaPast zeroes every past input the shipped decoder declares", () => {
  class Tensor {
    constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
  }
  const session = {
    inputMetadata: fixture.graphs.decoder.inputs.map(
      (i) => ({ name: i.name, isTensor: true, shape: i.shape, type: "float32" })),
  };
  const past = emptyVegaPast(session, Tensor);
  const names = Object.keys(past);
  assert.equal(names.length, 40);
  assert.deepEqual(past["past_conv.0"].dims, [1, 6144, 4]);
  assert.deepEqual(past["past_recurrent.18"].dims, [1, 16, 128, 128]);
  assert.deepEqual(past["past_key_values.19.value"].dims, [1, 2, 0, 256]);
  for (const t of Object.values(past)) {
    assert.equal(t.type, "float32");
    assert.ok(t.data.every((v) => v === 0), "zero-filled first read");
  }
});

test("adapter routing + thermal calibration match the reference", () => {
  const r = fixture.routing;
  const rows = r.rows;
  const nOpts = rows.map((x) => x.logits.length);
  const rowWidth = Math.max(...nOpts);
  const logits = new Float32Array(rows.length * rowWidth);
  rows.forEach((x, b) => logits.set(x.logits, b * rowWidth));
  const qtypes = rows.map((x) => x.qtype);
  const phys = vegaThermalFeatures({
    kinetic: rows.map((x) => x.kinetic),
    nearest: rows.map((x) => x.nearest),
    stepsUsed: rows.map((x) => x.steps),
    nOpts,
    maxSteps: cfg.n_steps,
  });
  const adapters = fixture.config.adapters;
  const gates = rows.map((x) => x.gates);
  const routes = gates.map((g) => vegaRoute(g, r.gateThreshold));
  assert.deepEqual(routes, r.expected.map((e) => e.route));
  const calibrations = routes.map(
    (rt) => (rt >= 0 ? adapters[rt]?.calibration : null) ?? fixture.config.calibration);
  const thetaByRow = calibrations.map((cal, b) => cal.theta[qtypes[b]]);
  const probs = vegaThermalProbabilities({
    logits, rowWidth, qtypes, phys: phys.flat(), nOpts, thetaByRow,
  });
  const answers = vegaAnswers({
    probs, qtypes,
    labels: rows.map((x) => Array.from({ length: x.logits.length }, (_, i) => i)),
    physics: {
      nearest: rows.map((x) => x.nearest),
      stepsUsed: rows.map((x) => x.steps),
    },
    rawLogits: logits, rowWidth,
    calibration: fixture.config.calibration,
    calibrations, alpha: r.alpha, risk: r.risk,
    adapterNames: adapters.map((a) => a.name), gates, routes,
  });
  answers.forEach((a, i) => {
    const e = r.expected[i];
    a.probabilities && Object.values(a.probabilities).forEach((p, j) => {
      assert.ok(Math.abs(p - e.probabilities[j]) < 1e-6, `row ${i} p[${j}]`);
    });
    assert.ok(Math.abs(a.confidence - e.confidence) < 1e-6, `row ${i} confidence`);
    assert.deepEqual(a.predictionSet, e.predictionSet, `row ${i} predictionSet`);
    assert.equal(a.abstain, e.abstain, `row ${i} abstain`);
    assert.equal(a.unbound, e.unbound, `row ${i} unbound`);
    assert.equal(a.settleSteps, e.settleSteps, `row ${i} settleSteps`);
    assert.ok(Math.abs(a.energy - e.energy) < 1e-6, `row ${i} energy`);
    assert.equal(a.adapter, e.adapter, `row ${i} adapter`);
    assert.ok(Math.abs(a.gate - e.gate) < 1e-6, `row ${i} gate`);
  });
  assert.equal(r.expected[1].predictionSet, null, "public_intents ships no boolean conformal block");
});

// ---------------------------------------------------------------------------
// createVegaDecider({variant: "q8"}) with injected fakes (no network, no graphs)
// ---------------------------------------------------------------------------

class FakeTensor {
  constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
}

const D_FAKE = 8;

function fakeSplitWorld() {
  const calls = { urls: [], fetches: [], embed: [], decoder: [], engine: [] };
  const embed = {
    run: async (feeds) => {
      calls.embed.push(feeds);
      const L = Number(feeds.input_ids.dims[1]);
      return {
        inputs_embeds: new FakeTensor("float32",
          new Float32Array(L * D_FAKE).fill(0.5), [1, L, D_FAKE]),
      };
    },
    release: async () => {},
  };
  const decoder = {
    inputMetadata: fixture.graphs.decoder.inputs.map(
      (i) => ({ name: i.name, isTensor: true, shape: i.shape, type: "float32" })),
    run: async (feeds) => {
      calls.decoder.push(feeds);
      const L = Number(feeds.inputs_embeds.dims[1]);
      const h = new Float32Array(L * D_FAKE);
      for (let i = 0; i < h.length; i++) h[i] = 0.01 * ((i % 13) - 6);
      return {
        h13: new FakeTensor("float32", h, [1, L, D_FAKE]),
        h19: new FakeTensor("float32", Float32Array.from(h, (v) => v * 2), [1, L, D_FAKE]),
      };
    },
    release: async () => {},
  };
  const engine = {
    run: async (feeds) => {
      calls.engine.push(feeds);
      const K = Number(feeds.opts.dims[1]);
      const logits = new Float32Array(K).fill(-1);
      logits[0] = 2;
      return {
        logits: new FakeTensor("float32", logits, [1, K]),
        kinetic: new FakeTensor("float32", [0.05], [1]),
        nearest: new FakeTensor("float32", [0.3], [1]),
        steps_used: new FakeTensor("int64", BigInt64Array.from([12n]), [1]),
        gates: new FakeTensor("float32", [0.9, 0.1], [1, 2]),
      };
    },
    release: async () => {},
  };
  return { calls, embed, decoder, engine };
}

function fakeTokenizerFactory() {
  return class FakePreTrainedTokenizer {
    encode(text) {
      return [...String(text)].map((c) => (c.charCodeAt(0) % 90) + 10);
    }

    decode(ids) {
      return `decoded:${Array.from(ids).join(" ")}`;
    }
  };
}

test("createVegaDecider({variant: \"q8\"}) wires the split graphs and routes answers", async () => {
  const world = fakeSplitWorld();
  const ort = {
    env: { wasm: {} },
    Tensor: FakeTensor,
    InferenceSession: { create: async () => world.embed },
  };
  const sessionFactory = async (url) => {
    world.calls.urls.push(String(url));
    if (String(url).includes("embed_q8")) return world.embed;
    if (String(url).includes("decoder_q8")) return world.decoder;
    if (String(url).includes("engine.onnx")) return world.engine;
    throw new Error(`unexpected model URL: ${url}`);
  };
  const fetchImpl = async (url) => {
    const u = String(url);
    world.calls.fetches.push(u);
    if (u.endsWith("vega_web.json")) {
      // opt_max_tokens is raised for the char-level fake tokenizer: the
      // reference 32-token option cap would otherwise clip the noul defaults
      // (35 ids) even though the real tokenizer packs them well under it.
      return { ok: true, json: async () => ({ ...fixture.config, opt_max_tokens: 64 }) };
    }
    if (u.endsWith("tokenizer.json") || u.endsWith("tokenizer_config.json")) {
      return { ok: true, json: async () => ({}) };
    }
    return { ok: false, status: 404, json: async () => ({}) };
  };
  const transformers = { PreTrainedTokenizer: fakeTokenizerFactory() };
  const decider = await createVegaDecider({
    variant: "q8", ort, transformers, fetchImpl, sessionFactory,
  });

  assert.equal(decider.info.family, "vega");
  assert.equal(decider.info.format, "split");
  assert.equal(decider.info.model, VEGA_Q8_DEFAULT_MODEL);
  assert.equal(decider.info.revision, VEGA_Q8_DEFAULT_REVISION);
  assert.equal(decider.info.subfolder, VEGA_Q8_DEFAULT_SUBFOLDER);
  assert.equal(decider.info.device, "wasm", "8-bit MatMulNBits is wasm-only");
  assert.equal(decider.info.risk, fixture.config.calibration.default_risk);
  assert.deepEqual(decider.info.adapters, ["public_intents", "typed_decisions"]);
  assert.match(decider.info.calibrationSource, /vega_web\.json$/);
  assert.ok(world.calls.urls.every((u) =>
    u.includes(`${VEGA_Q8_DEFAULT_REVISION}/v2/`)), "graphs load from the pinned v2 folder");
  assert.ok(world.calls.urls.some((u) => u.endsWith("embed_q8.onnx")));
  assert.ok(world.calls.urls.some((u) => u.endsWith("decoder_q8.onnx")));
  assert.ok(world.calls.urls.some((u) => u.endsWith("engine.onnx")));

  const { answers, truncated, length, prompts, timings } = await decider.decide(
    "I was charged twice.", [
      { type: "choice", instructions: "Which team?", options: ["billing", "tech", "sales"] },
      { type: "noul", instructions: "Urgent?" },
    ]);
  assert.equal(world.calls.decoder.length, 2, "one decoder pass per question");
  assert.equal(world.calls.engine.length, 2);
  assert.equal(truncated, false);
  assert.ok(length > 0);
  assert.equal(prompts.length, 2);
  assert.ok(timings.totalMs >= 0 && timings.runMs >= 0);

  // decoder feeds: text M-RoPE positions, causal mask, zeroed past state
  const dec = world.calls.decoder[0];
  const L0 = Number(dec.inputs_embeds.dims[1]);
  assert.deepEqual([...dec.position_ids.dims], [3, 1, L0]);
  for (let row = 0; row < 3; row++) {
    for (let t = 0; t < L0; t++) {
      assert.equal(Number(dec.position_ids.data[row * L0 + t]), t, "position 0..L-1 on every row");
    }
  }
  assert.ok([...dec.attention_mask.data].every((v) => v === 1n));
  assert.deepEqual([...dec["past_conv.0"].dims], [1, 6144, 4]);
  assert.ok([...dec["past_key_values.3.key"].data].every((v) => v === 0));

  // engine feeds: the split engine's own layout
  const eng = world.calls.engine[0];
  assert.deepEqual([...eng.feats.dims], [1, 2, 3, D_FAKE]);
  assert.deepEqual([...eng.opts.dims], [1, 3, 1, D_FAKE]);
  assert.deepEqual([...eng.opt_mask.dims], [1, 3]);
  assert.ok([...eng.opt_mask.data].every((v) => v === 1));
  assert.deepEqual([...eng.qtype.dims], [1]);
  assert.deepEqual([...eng.chunks.dims], [1, cfg.memory_chunks, D_FAKE]);
  assert.deepEqual([...eng.chunk_mask.dims], [1, cfg.memory_chunks]);

  const [a0, a1] = answers;
  assert.equal(a0.type, "choice");
  assert.equal(a0.choice, "billing");
  assert.equal(a0.adapter, "public_intents", "gate 0.9 routes to the first adapter");
  assert.equal(a1.type, "noul");
  assert.equal(a1.adapter, "public_intents");
  assert.equal(a1.predictionSet, null, "public_intents has no boolean conformal block");
  await decider.dispose();
});

test("explicit model/subfolder still win over the q8 variant preset", async () => {
  const world = fakeSplitWorld();
  const decider = await createVegaDecider({
    variant: "q8",
    model: "org/mirror",
    subfolder: "v9",
    ort: { env: { wasm: {} }, Tensor: FakeTensor },
    transformers: { PreTrainedTokenizer: fakeTokenizerFactory() },
    fetchImpl: async () => ({ ok: true, json: async () => fixture.config }),
    sessionFactory: async (url) => {
      world.calls.urls.push(String(url));
      if (String(url).includes("embed")) return world.embed;
      if (String(url).includes("decoder")) return world.decoder;
      return world.engine;
    },
  });
  assert.equal(decider.info.model, "org/mirror");
  assert.equal(decider.info.subfolder, "v9");
  assert.ok(world.calls.urls.every((u) => u.includes("/org/mirror/resolve/") && u.includes("/v9/")));
  await decider.dispose();
});

test("a registry-style default target does not shadow the q8 variant", async () => {
  const world = fakeSplitWorld();
  const decider = await createVegaDecider({
    variant: "q8",
    // what createDecisionRuntime injects from the registry defaults
    model: VEGA_DEFAULT_MODEL,
    revision: VEGA_VARIANTS.fp16.revision,
    ort: { env: { wasm: {} }, Tensor: FakeTensor },
    transformers: { PreTrainedTokenizer: fakeTokenizerFactory() },
    fetchImpl: async () => ({ ok: true, json: async () => fixture.config }),
    sessionFactory: async (url) => {
      world.calls.urls.push(String(url));
      if (String(url).includes("embed")) return world.embed;
      if (String(url).includes("decoder")) return world.decoder;
      return world.engine;
    },
  });
  assert.equal(decider.info.format, "split");
  assert.equal(decider.info.model, VEGA_Q8_DEFAULT_MODEL);
  await decider.dispose();
});

test("unknown variants and formats are rejected", async () => {
  await assert.rejects(() => createVegaDecider({ variant: "nope" }), /unknown variant/);
  await assert.rejects(() => createVegaDecider({ format: "wat" }), /format must be/);
});

test("the default variant still targets this repo's fused export", () => {
  assert.deepEqual(Object.keys(VEGA_VARIANTS).sort(), ["fp16", "q8"]);
  assert.equal(VEGA_VARIANTS.fp16.format, "fused");
  assert.equal(VEGA_VARIANTS.q8.format, "split");
  assert.equal(VEGA_VARIANTS.fp16.model, VEGA_DEFAULT_MODEL);
  assert.equal(VEGA_VARIANTS.q8.model, VEGA_Q8_DEFAULT_MODEL);
});
