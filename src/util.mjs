// jev-web — tiny shared helpers (timing, Hugging Face URLs, argmax, queues).
//
// Every family used to carry its own copy of `now()` / `hfUrl()`; the copies
// had drifted (some filtered empty path parts, some did not). They live here
// now, together with the serial queue every runtime uses to make `decide()`
// safe under concurrent callers.

/** Monotonic-ish clock: performance.now() where available, else Date.now(). */
export const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

/**
 * Build a Hugging Face `resolve` URL.
 *
 * @param {string} model repo id, e.g. "onnx-community/open-jev-deberta-v3-large-ONNX"
 * @param {string|null|undefined} revision commit sha / branch / tag; omitted when falsy
 * @param {...(string|null|undefined|false)} parts path parts, joined with "/";
 *   empty parts are dropped (so `hfUrl(m, r, "", "config.json")` works)
 * @returns {string}
 */
export function hfUrl(model, revision, ...parts) {
  const path = parts.filter(Boolean).join("/");
  return `https://huggingface.co/${model}/resolve/${revision ? `${revision}/` : ""}${path}`;
}

/**
 * Index of the largest value. Unlike `values.indexOf(Math.max(...values))`
 * this never spreads a large array onto the stack; ties keep the first index.
 *
 * @param {ArrayLike<number>} values
 * @returns {number} -1 for an empty input
 */
export function argmax(values) {
  let best = -1;
  let bestValue = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = Number(values[i]);
    if (best === -1 || v > bestValue) { best = i; bestValue = v; }
  }
  return best;
}

/**
 * A promise queue that runs tasks one at a time, in submission order.
 *
 * Every runtime's `decide()` runs through one of these: a model session is a
 * single shared resource, so two overlapping `decide()` calls would interleave
 * their feeds and corrupt each other's result.
 *
 * @returns {((task:()=>Promise<any>)=>Promise<any>) & {idle:()=>Promise<void>}}
 */
export function createSerialQueue() {
  let tail = Promise.resolve();
  const enqueue = (task) => {
    const run = tail.then(() => task());
    // Keep failures from wedging the chain for later callers.
    tail = run.then(() => {}, () => {});
    return run;
  };
  enqueue.idle = () => tail;
  return enqueue;
}
