# jev-web

Run **open-jev-shaped typed-decision models** in the browser. One *state* text
plus any number of typed *questions* (`choice` / `score` / `noul`) go in; a
calibrated probability distribution per question comes back from a single
forward pass. No server, no API keys — weights and inference stay on-device.

This is a generic runtime, not a classifier for any particular domain. You
bring the state text and the questions; the model answers by choosing among
the options you give it.

## Install

```bash
npm i jev-web @huggingface/transformers
```

`@huggingface/transformers` is an optional peer dependency (it is imported
lazily). Hosts that already ship their own copy can inject it via
`createDecider({ transformers })`.

## Use

```js
import { createDecider } from "jev-web";

const decider = await createDecider({
  // defaults to the reference open-jev DeBERTa-v3-large ONNX export,
  // pinned by revision; any repo with the same typed-decision graph works
  onProgress: (p) => console.log(p.phase, p.file, p.progress),
});

const { answers } = await decider.decide(
  "I was charged twice for the same order and nobody answers my emails.",
  [
    {
      type: "choice",
      instructions: "Which product area is the message about?",
      options: ["fees & charges", "refund & dispute", "card", "other"],
    },
    { type: "noul", instructions: "The customer is asking for a refund." },
    {
      type: "score",
      instructions: "How negative is the message?",
      options: ["very negative", "negative", "neutral", "positive", "very positive"],
    },
  ],
);

// [
//   { type: "choice", choice: "refund & dispute", index: 1, probabilities: {...}, confidence: 0.61 },
//   { type: "noul",  noul: 0.93, probabilities: { no: 0.07, yes: 0.93 }, confidence: 0.93 },
//   { type: "score", score: 0.4,  level: 1, probabilities: {...}, confidence: 0.55 },
// ]
```

## Question types

| type | options | answer |
|---|---|---|
| `choice` | 2–255 labels | `choice` (argmax), `probabilities`, `confidence` |
| `score` | 2–10 ordered levels | `score` (expected level index, may fall between levels) |
| `noul` | exactly 2 (defaults `["no","yes"]`) | `noul` (p of the second option = p(yes)), `confidence` |

Every answer carries the full distribution. `decide()` also returns
`{ truncated, length }` — the state is capped (256 tokens by default) and the
flag tells you when it was cut.

## Backends and quantization

`device` / `dtype` default to `auto`: WebGPU → `q4f16`, otherwise WASM → `q4`.
Override explicitly (`fp16`, `fp32`, `q8`, …) when accuracy matters more than
download size. First load downloads the weights once; transformers.js caches
them (Cache API / IndexedDB) so later visits are offline-fast.

`decider.info` reports the resolved `{ model, revision, device, dtype,
temperature, maxStateTokens, maxLen, pool, configSource }`.

## Model config

Defaults come from the model repo's `open_jev_config.json` when present
(temperature, marker tokens, state cap), else the published reference values.
To use a different typed-decision export:

```js
createDecider({ model: "your-org/your-typed-decision-ONNX", revision: "main" });
```

The export must add the `[STATE]`, `[Q]`, `[OPT]` markers to its tokenizer,
take `input_ids, attention_mask, seg, pair_q, pair_opt`, and return per-pair
`logits`. `typedMarkerIds()` fails loudly when a repo is missing them.


## Multiple families (registry)

`jev-web` is a multi-family runtime: `createDecisionRuntime({ family, fallback })`
tries the requested family and then any fallbacks, returning the shared contract
`{ info, decide(state, questions), dispose? }`:

```js
import { createDecisionRuntime, listDecisionFamilies } from "jev-web";

listDecisionFamilies(); // ["open-jev", "laya"]
const runtime = await createDecisionRuntime({
  family: "laya",
  fallback: ["open-jev"],
  onFallback: ({ family, error }) => console.warn(family, error),
});
const { answers, truncated, length, prompts, timings } = await runtime.decide(state, questions);
```

Third-party families plug in without touching core:

```js
registerDecisionFamily("my-model", {
  defaults: { model: "org/my-typed-decision-ONNX", revision: "main" },
  create: async (opts) => ({ info, decide, dispose }),
});
```

`decide()` returns `prompts` (decoded input sequence(s), for audit/debug) and
`timings` (per-stage ms) alongside the answers.

## Laya family (ModernBERT encoder + typed head)

`createLayaDecider(options)` runs the [Laya](https://huggingface.co/convaiinnovations/laya)
System-1 decision models, whose browser export is a **two-file q8 ONNX split**
(encoder + head) rather than one fused graph:

```js
import { createLayaDecider } from "jev-web";

const decider = await createLayaDecider({
  // defaults to alfred361/laya-typed-decisions-web-q8 @ pinned revision
  wasmPaths: "/assets/",            // self-hosted ORT wasm (optional)
  onProgress: (p) => console.log(p.phase, p.loaded, p.total),
});
const { answers } = await decider.decide(stateText, questions);
```

Protocol implemented here (mirrors the reference `laya` Python runtime):

- sequence `[CLS] <type> question: <instructions> [SEP] [MASK]opt0 … [SEP] <state> [SEP]`,
  `marker_pos` at each option's `[MASK]`, `qtype` 0 choice / 1 score / 2 noul;
- encoder `input_ids, attention_mask → hidden`; head
  `hidden, attention_mask, marker_pos, marker_mask, qtype → logits`;
- per-question calibration from `rl_agent_config.json`
  (`temperature` + `temperature_by_options` buckets, clamped to [0.5, 5]).

Notes:
- the q8 web head is **batch-1 only**, so `decide()` batches the encoder and
  runs the head once per question;
- `choice` supports at most `max_prefixes` options (6 for the typed-decisions
  export);
- weights download once (~520 MB q8) and are browser-cached.

Lower-level helpers are exported for custom runtimes:
`buildLayaSequence`, `collateLayaItems`, `layaAnswersFromLogits`,
`layaTokenizerAdapter`, `renderLayaOptions`, `layaTempBucket`, `clampTemperature`.

## Worker / thread

`jev-web` does not bundle a worker, so hosts can create one the way their
bundler prefers:

```js
// host code
const worker = new Worker(new URL("./my-worker.mjs", import.meta.url), { type: "module" });
// my-worker.mjs
import { createDecider } from "jev-web";
```

## API

- `createDecider(options)` → open-jev family: `{ info, decide, decideMany, dispose }`
- `createLayaDecider(options)` → Laya family: `{ info, decide, dispose }` (same `decide` contract)
- `normalizeQuestions`, `buildDecisionInput`, `typedMarkerIds`, `answersFromScores`,
  `softmaxWithTemperature`, `resolveDevice`, `resolveDtype`, `detectWebGPU`,
  `loadModelConfig`, defaults (model/revision/temperature/caps).

## Playground

```bash
npm run dev   # packages/jev-web → http://localhost:5188
```

Generic presets plus a paste-your-own state/questions box, download progress,
latency, and a model-cache reset.

## Limits

- Typed-decision models answer **only** by picking among your options; they do
  not explain or generate text.
- Accuracy is domain-bound: questions and states unlike the model's training
  domain are out of distribution. Measure before relying on any answer.
- English, 512-token sequence (state capped), one state per forward pass.

## License

MIT (runtime code). Model weights keep their own license (the reference export
is Apache-2.0).
