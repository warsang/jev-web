/**
 * registry.mjs — multi-family runtime: registration, defaults merging,
 * ordered fallback and the shared DecideResult contract.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  registerDecisionFamily,
  createDecisionRuntime,
  listDecisionFamilies,
  getDecisionFamily,
} from "../src/registry.mjs";

function makeRuntime(id, opts) {
  return {
    info: { family: id, model: opts.model, echo: opts.echo },
    decide: async () => ({ answers: [], truncated: false, length: 0 }),
  };
}

test("built-in families are registered", () => {
  const ids = listDecisionFamilies();
  assert.ok(ids.includes("open-jev"));
  assert.ok(ids.includes("laya"));
  assert.ok(ids.includes("strands-decider"));
  assert.ok(ids.includes("bekko"));
  assert.equal(getDecisionFamily("open-jev").create.name, "createDecider");
  assert.equal(getDecisionFamily("laya").create.name, "createLayaDecider");
  assert.equal(getDecisionFamily("strands-decider").create.name, "createStrandsDecider");
  assert.equal(getDecisionFamily("bekko").create.name, "createBekkoDecider");
  assert.ok(ids.includes("decision2"));
  assert.equal(getDecisionFamily("decision2").create.name, "createDecision2Decider");
});

test("registerDecisionFamily validates descriptors", () => {
  assert.throws(() => registerDecisionFamily("", { create() {} }), /id is required/);
  assert.throws(() => registerDecisionFamily("x", {}), /descriptor\.create is required/);
});

test("createDecisionRuntime merges family defaults with caller opts", async () => {
  let seen = null;
  registerDecisionFamily("echo", {
    defaults: { model: "default-model", echo: "default" },
    create: async (opts) => { seen = opts; return makeRuntime("echo", opts); },
  });
  const runtime = await createDecisionRuntime({ family: "echo", echo: "caller" });
  assert.equal(seen.model, "default-model", "default kept");
  assert.equal(seen.echo, "caller", "caller overrides");
  assert.equal(runtime.info.family, "echo");
  assert.equal(typeof runtime.decide, "function");
});

test("createDecisionRuntime falls back in order and reports each failure", async () => {
  const failures = [];
  registerDecisionFamily("broken", {
    create: async () => { throw new Error("boom"); },
  });
  registerDecisionFamily("second", {
    create: async () => { throw new Error("also broken"); },
  });
  const runtime = await createDecisionRuntime({
    family: "broken",
    fallback: ["second", "echo"],
    onFallback: (e) => failures.push(`${e.family}:${e.error.message}`),
  });
  assert.equal(runtime.info.family, "echo");
  assert.deepEqual(failures, ["broken:boom", "second:also broken"]);
});

test("createDecisionRuntime throws when nothing can be created", async () => {
  await assert.rejects(
    createDecisionRuntime({ family: "does-not-exist", fallback: [] }),
    /unknown decision family/,
  );
});

test("DecideResult contract is family-agnostic", async () => {
  registerDecisionFamily("contract", {
    create: async () => ({
      info: { family: "contract" },
      async decide() {
        return { answers: [{ type: "noul", noul: 0.5 }], truncated: false, length: 3, prompts: ["x"], timings: { totalMs: 1 } };
      },
      async dispose() {},
    }),
  });
  const runtime = await createDecisionRuntime({ family: "contract" });
  const result = await runtime.decide("state", [{ type: "noul", instructions: "?" }]);
  assert.deepEqual(Object.keys(result).sort(), ["answers", "length", "prompts", "timings", "truncated"]);
  await runtime.dispose();
});
