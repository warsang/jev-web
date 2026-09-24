import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  resolveDevice,
  resolveDtype,
  loadModelConfig,
  detectWebGPU,
  DEFAULT_MODEL,
  DEFAULT_REVISION,
  DEFAULT_TEMPERATURE,
} from "../src/config.mjs";

describe("resolveDevice", () => {
  it("auto picks webgpu only when an adapter exists", () => {
    assert.equal(resolveDevice({ device: "auto", webgpu: true }), "webgpu");
    assert.equal(resolveDevice({ device: "auto", webgpu: false }), "wasm");
  });

  it("explicit device wins", () => {
    assert.equal(resolveDevice({ device: "wasm", webgpu: true }), "wasm");
    assert.equal(resolveDevice({ device: "webgpu", webgpu: false }), "webgpu");
  });
});

describe("resolveDtype", () => {
  it("auto matches the backend's quantization", () => {
    assert.equal(resolveDtype("webgpu", "auto"), "q4f16");
    assert.equal(resolveDtype("wasm", "auto"), "q4");
  });

  it("explicit dtype passes through", () => {
    assert.equal(resolveDtype("wasm", "fp32"), "fp32");
  });
});

describe("loadModelConfig", () => {
  it("reads calibration values from the model repo", async () => {
    const fetchImpl = async (url) => {
      assert.match(url, new RegExp(`^https://huggingface.co/${DEFAULT_MODEL}/resolve/${DEFAULT_REVISION}/`));
      return {
        ok: true,
        json: async () => ({
          pool: "span",
          temperature: 1.2,
          markers: ["[STATE]", "[Q]", "[OPT]"],
          max_state_tokens: 128,
          max_len: 512,
        }),
      };
    };
    const cfg = await loadModelConfig({ fetchImpl });
    assert.equal(cfg.temperature, 1.2);
    assert.equal(cfg.maxStateTokens, 128);
    assert.equal(cfg.maxLen, 512);
    assert.equal(cfg.source, "model");
  });

  it("falls back to published defaults on any failure", async () => {
    const missing = await loadModelConfig({ fetchImpl: async () => ({ ok: false, status: 404 }) });
    assert.equal(missing.temperature, DEFAULT_TEMPERATURE);
    assert.equal(missing.maxStateTokens, 256);
    assert.equal(missing.source, "defaults");

    const broken = await loadModelConfig({ fetchImpl: async () => { throw new Error("offline"); } });
    assert.equal(broken.temperature, DEFAULT_TEMPERATURE);
  });
});

describe("detectWebGPU", () => {
  it("reports false without navigator.gpu and true with an adapter", async () => {
    assert.equal(await detectWebGPU({}), false);
    assert.equal(
      await detectWebGPU({ navigator: { gpu: { requestAdapter: async () => ({}) } } }),
      true,
    );
    assert.equal(
      await detectWebGPU({ navigator: { gpu: { requestAdapter: async () => null } } }),
      false,
    );
  });

  it("swallows adapter errors", async () => {
    const scope = { navigator: { gpu: { requestAdapter: async () => { throw new Error("denied"); } } } };
    assert.equal(await detectWebGPU(scope), false);
  });
});
