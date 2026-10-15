/**
 * Shared helpers (src/util.mjs): Hugging Face URL building, argmax, and the
 * serial queue every runtime funnels decide() through.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { now, hfUrl, argmax, createSerialQueue } from "../src/util.mjs";

test("now() returns a number", () => {
  assert.equal(typeof now(), "number");
});

test("hfUrl builds pinned resolve URLs and drops empty path parts", () => {
  assert.equal(
    hfUrl("org/model", "rev1", "config.json"),
    "https://huggingface.co/org/model/resolve/rev1/config.json");
  assert.equal(
    hfUrl("org/model", "rev1", "sub", "", "model.onnx"),
    "https://huggingface.co/org/model/resolve/rev1/sub/model.onnx");
  assert.equal(
    hfUrl("org/model", null, "config.json"),
    "https://huggingface.co/org/model/resolve/config.json",
    "an unpinned revision omits the segment");
  assert.equal(
    hfUrl("org/model", "rev1", "v1/"),
    "https://huggingface.co/org/model/resolve/rev1/v1/",
    "trailing slashes survive (laya appends file names)");
});

test("argmax keeps the first index on ties and handles empty input", () => {
  assert.equal(argmax([1, 3, 3, 2]), 1);
  assert.equal(argmax([3, 1]), 0);
  assert.equal(argmax([]), -1);
  assert.equal(argmax(Float32Array.of(-2, -1, -3)), 1);
});

test("argmax survives inputs Math.max(...) would choke on", () => {
  const million = Array.from({ length: 1_000_000 }, (_, i) => i);
  assert.equal(argmax(million), 999_999);
  assert.throws(() => Math.max(...million), RangeError, "the spread version overflows the stack");
});

test("createSerialQueue runs tasks one at a time, in submission order", async () => {
  const queue = createSerialQueue();
  const order = [];
  let release;
  const gate = new Promise((resolve) => { release = resolve; });

  const first = queue(async () => {
    order.push("first:start");
    await gate;
    order.push("first:end");
    return 1;
  });
  const second = queue(async () => {
    order.push("second");
    return 2;
  });

  // Give the second task every chance to start early.
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.deepEqual(order, ["first:start"], "the second task waits for the first");
  release();
  assert.equal(await first, 1);
  assert.equal(await second, 2);
  assert.deepEqual(order, ["first:start", "first:end", "second"]);
});

test("a rejected task does not wedge the queue", async () => {
  const queue = createSerialQueue();
  const failed = queue(async () => { throw new Error("boom"); });
  const next = queue(async () => "ok");
  await assert.rejects(failed, /boom/);
  assert.equal(await next, "ok");
});

test("idle() resolves once every queued task (and its failure) settled", async () => {
  const queue = createSerialQueue();
  const seen = [];
  queue(async () => { seen.push(1); }).catch(() => {});
  queue(async () => { throw new Error("boom"); }).catch(() => {});
  queue(async () => { seen.push(2); });
  await queue.idle();
  assert.deepEqual(seen, [1, 2]);
});
