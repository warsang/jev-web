/**
 * registry.mjs — multi-family runtime for JEV-style typed-decision models.
 *
 * jev-web ships three families today (open-jev's fused DeBERTa graph, Laya's
 * ModernBERT encoder + typed head, Strands Decider 2B's Qwen torso + pointer
 * head, and Bekko System One v0's cross-encoder) and any third party can add
 * its own without touching core:
 *
 *   registerDecisionFamily("my-model", {
 *     defaults: { model: "org/my-typed-decision-ONNX", revision: "main" },
 *     create: async (opts) => ({ info, decide(state, questions), dispose? }),
 *   });
 *
 *   const runtime = await createDecisionRuntime({ family: "my-model" });
 *
 * `createDecisionRuntime` tries the requested family, then any `fallback`
 * families in order, reporting each failure through `onFallback`. The returned
 * object always exposes the shared contract:
 *   { info: { family, ... }, decide(state, questions) -> DecideResult, dispose? }
 * where DecideResult is `{ answers, truncated, length, prompts?, timings? }`.
 */

import { createDecider } from "./session.mjs";
import { DEFAULT_MODEL, DEFAULT_REVISION } from "./config.mjs";
import {
  createLayaDecider,
  LAYA_DEFAULT_MODEL,
  LAYA_DEFAULT_REVISION,
  LAYA_DEFAULT_SUBFOLDER,
} from "./laya.mjs";
import {
  createStrandsDecider,
  STRANDS_DEFAULT_MODEL,
  STRANDS_DEFAULT_REVISION,
} from "./strands.mjs";
import {
  createBekkoDecider,
  BEKKO_DEFAULT_MODEL,
  BEKKO_DEFAULT_REVISION,
} from "./bekko.mjs";

const families = new Map();

/**
 * Register (or replace) a decision family.
 * @param {string} id short stable identifier, e.g. "open-jev"
 * @param {{create:(opts:object)=>Promise<object>, defaults?:object}} descriptor
 */
export function registerDecisionFamily(id, descriptor) {
  if (!id || typeof id !== "string") throw new TypeError("registerDecisionFamily: id is required");
  if (!descriptor || typeof descriptor.create !== "function") {
    throw new TypeError(`registerDecisionFamily(${id}): descriptor.create is required`);
  }
  families.set(id, { defaults: {}, ...descriptor });
  return id;
}

/** @returns {string[]} registered family ids in registration order. */
export function listDecisionFamilies() {
  return [...families.keys()];
}

export function getDecisionFamily(id) {
  return families.get(id) ?? null;
}

registerDecisionFamily("open-jev", {
  defaults: { model: DEFAULT_MODEL, revision: DEFAULT_REVISION },
  create: createDecider,
});

registerDecisionFamily("laya", {
  defaults: {
    model: LAYA_DEFAULT_MODEL,
    revision: LAYA_DEFAULT_REVISION,
    subfolder: LAYA_DEFAULT_SUBFOLDER,
  },
  create: createLayaDecider,
});

registerDecisionFamily("strands-decider", {
  defaults: {
    model: STRANDS_DEFAULT_MODEL,
    revision: STRANDS_DEFAULT_REVISION,
  },
  create: createStrandsDecider,
});

registerDecisionFamily("bekko", {
  defaults: {
    model: BEKKO_DEFAULT_MODEL,
    revision: BEKKO_DEFAULT_REVISION,
  },
  create: createBekkoDecider,
});

/**
 * Create a decision runtime, preferring `family` and falling back in order.
 * @param {{family?:string, fallback?:string[], onFallback?:Function}} [opts]
 * @returns {Promise<{info:object, decide:Function, dispose?:Function}>}
 */
export async function createDecisionRuntime({
  family = "laya",
  fallback = [],
  onFallback = null,
  ...opts
} = {}) {
  const order = [family, ...(Array.isArray(fallback) ? fallback : [])].filter(Boolean);
  let lastError = null;
  for (const id of order) {
    const descriptor = families.get(id);
    if (!descriptor) {
      lastError = new Error(`unknown decision family "${id}"`);
      onFallback?.({ family: id, error: lastError });
      continue;
    }
    try {
      const runtime = await descriptor.create({ ...descriptor.defaults, ...opts });
      return {
        ...runtime,
        info: { family: id, ...(runtime?.info ?? {}) },
      };
    } catch (error) {
      lastError = error;
      onFallback?.({ family: id, error });
    }
  }
  throw lastError ?? new Error("createDecisionRuntime: no family could be created");
}
