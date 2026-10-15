// jev-web — model defaults, backend resolution, and model-side config.
//
// Defaults point at the reference ONNX export of com-kotobalabs/open-jev-
// deberta-v3-large (Apache-2.0), pinned by revision for reproducibility. Any
// repo with the same typed-decision graph layout works: pass `model` and
// optionally `revision`.

import { hfUrl } from "./util.mjs";
import { fetchCachedJson } from "./fetch.mjs";

export const DEFAULT_MODEL = "onnx-community/open-jev-deberta-v3-large-ONNX";
export const DEFAULT_REVISION = "7c79f25b5ac496089f448a969c801872ad59d31c";
export const DEFAULT_TEMPERATURE = 1.05;
export const DEFAULT_MARKERS = ["[STATE]", "[Q]", "[OPT]"];
export const DEFAULT_MAX_STATE_TOKENS = 256;
export const DEFAULT_MAX_LEN = 512;

export function resolveDevice({ device = "auto", webgpu = false } = {}) {
  if (device && device !== "auto") return device;
  return webgpu ? "webgpu" : "wasm";
}

// Quantized WebGPU kernels are f16-friendly (q4f16); WASM wants the
// non-f16 q4 build. fp32/fp16/q8 stay explicit opt-ins.
export function resolveDtype(device, dtype = "auto") {
  if (dtype && dtype !== "auto") return dtype;
  return device === "webgpu" ? "q4f16" : "q4";
}

export async function detectWebGPU(scope = globalThis) {
  try {
    if (!scope?.navigator?.gpu?.requestAdapter) return false;
    return !!(await scope.navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

// Read the model's own calibration/marker config when the repo ships one
// (open_jev_config.json), else fall back to the published reference values.
// The read goes through the Cache API, so an offline reload keeps working.
export async function loadModelConfig({ model = DEFAULT_MODEL, revision = DEFAULT_REVISION, fetchImpl = fetch } = {}) {
  const fallback = {
    temperature: DEFAULT_TEMPERATURE,
    markers: [...DEFAULT_MARKERS],
    maxStateTokens: DEFAULT_MAX_STATE_TOKENS,
    maxLen: DEFAULT_MAX_LEN,
    pool: "span",
    source: "defaults",
  };
  const url = hfUrl(model, revision, "open_jev_config.json");
  try {
    const cfg = await fetchCachedJson(url, { fetchImpl, nullOn404: true });
    if (!cfg) return fallback;
    return {
      temperature: Number.isFinite(Number(cfg.temperature)) ? Number(cfg.temperature) : fallback.temperature,
      markers: Array.isArray(cfg.markers) && cfg.markers.length === 3 ? cfg.markers.map(String) : fallback.markers,
      maxStateTokens: Number.isFinite(Number(cfg.max_state_tokens)) ? Number(cfg.max_state_tokens) : fallback.maxStateTokens,
      maxLen: Number.isFinite(Number(cfg.max_len)) ? Number(cfg.max_len) : fallback.maxLen,
      pool: typeof cfg.pool === "string" ? cfg.pool : fallback.pool,
      source: "model",
    };
  } catch {
    return fallback;
  }
}
