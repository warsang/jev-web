# clef-text-0.6b — browser export: flow notes (2026-10-12)

Status: **exported, parity-passed, JS-verified end to end; uploaded and
pinned.** The fused fp16 graph (Qwen3-0.6B torso + joint schema head)
reproduces Cloudflare's reference `joint_schema_model.py` through both the
Python parity gate and the real jev-web runtime:

| check | result |
|---|---|
| in-process, fp32 wrapper vs `ClefModel` | max logit diff **9.5e-07** |
| in-process, fp16 wrapper vs `ClefModel` | max logit diff **4.2e-03** |
| ONNX fp16 vs reference (3 records, 5 questions) | logits ≤ **4.2e-03**, probs ≤ **2.5e-04**, argmax **5/5** |
| real JS runtime vs reference (fp32 graph, one mixed record) | worst probability diff **1.25e-07**, argmax 3/3 |

`final/` is 1.42 GB (graph 1.35 GB + tokenizer). Uploaded to
`warsang/clef-text-0.6b-web` at
`d01e9cfd11e66d593e9c517bb2bc204ffbf67623` (pinned in `src/clef.mjs`).

## What ships

| checkpoint | base | head | browser? |
|---|---|---|---|
| `FluidInference/clef-text-0.6b` (0.70B) | Qwen3-0.6B (LoRA merged) | 102M-param joint schema head | ✅ ~1.35 GB (fused fp16 graph) |

The checkpoint's `joint_head.safetensors` is 205.8 MB fp32; the shipped graph
is one fused file at 1.35 GB (torso + head in fp16).
Text-only v1: the multimodal clef checkpoints' image/video path (processor +
vision tower) is not part of the browser export.

## Layout

```
~/workspace/clef-export/
  ckpt/      FluidInference/clef-text-0.6b snapshot
             (model.safetensors, joint_head.safetensors, joint_schema_model.py,
              joint_head_config.json, tokenizer files)
  export/    clef_text_fp32.onnx   fp32 graph (exact; parity reference)
             clef_text_fp16.onnx   the shipped graph (2.67 GB fp32 / 1.35 GB fp16)
             parity-*.json         parity summaries
  final/     the browser repo contents (graph + clef_config.json + tokenizer
             + MANIFEST.json)
```

## Flow

```bash
python -m venv ~/workspace/.venvs/sdx && . ~/workspace/.venvs/sdx/bin/activate
pip install torch transformers safetensors onnx onnxruntime numpy

CLEF_OUT=~/workspace/clef-export python export/clef/01_export.py       # both graphs, fused
CLEF_OUT=~/workspace/clef-export python export/clef/02_parity.py       # fp32 gate
CLEF_OUT=~/workspace/clef-export python export/clef/03_repo.py         # fp16 gate + final/
CLEF_SRC=~/workspace/clef-export/ckpt python export/clef/make_js_fixtures.py
CLEF_SRC=~/workspace/clef-export/ckpt python export/clef/make_js_fixtures.py --e2e
node export/clef/verify_js_local.mjs                                   # real JS runtime
hf auth login && CLEF_OUT=~/workspace/clef-export python export/clef/04_upload.py
```

## Graph contract (must match src/clef.mjs)

one fused graph (`onnx/clef_text_fp16.onnx`), batch 1 — one record, Q
questions, K options per question slot, S = Q*K:

- in `input_ids[i64 1xL]`, `attention_mask[i64 1xL]`,
  `pool_global[f32 1x1xL]` (one-hot at the final token),
  `pool_question[f32 1xQxL]` (question-span means),
  `pool_option[f32 1xSxL]` (option-span means, question-major, masked slots
  zero), `type_ids[i64 Q]` (noul 0 / choice 1 / score 2),
  `slot_question[i64 S]`, `opt_mask[bool S]`
- out `logits[f32 1xS]` — per option slot; JS slices each question's first
  `counts[q]` slots and softmaxes.

## Notes / gotchas

- **fp16 IO stays fp32**: the wrapper casts the pooling matrices to the model
  dtype at entry and the logits back at exit, so JS feeds fp32 exactly like
  the Vega family. The fp32 graph is the exact reference parity artifact.
- **onnxruntime version skew**: the fp16 graph loads and runs on ORT 1.31
  (Python parity gate) and rejects at load on `onnxruntime-node` 1.30 with a
  precision-cast type error. transformers.js 4.3.0 pins ORT-web
  `1.31.0-dev.20260914` in the browser, so the browser path is on the 1.31
  generation; if a host pins 1.30, use the fp32 graph (`CLEF_DEFAULT_FILE`
  option) or upgrade.
- **Local JS verification**: `node export/clef/verify_js_local.mjs` runs the
  real `createClefDecider` against a local graph (env `CLEF_VERIFY_GRAPH`,
  default `final/onnx/clef_text_fp16.onnx`) and compares to
  `test/fixtures/clef-e2e-reference.json` (regenerate with
  `make_js_fixtures.py --e2e`). Measured on the fp32 graph: worst probability
  diff 1.25e-07, all argmaxes agree.
- `torch.onnx.export(..., dynamo=True)` prints a ✅ and dies on a cp1252
  console — run every stage with `PYTHONUTF8=1` (the Windows gotcha from the
  Vega export).
- The head's three sequence-level reads (evidence keys/values, the decoder
  field pass, the lexical prior) all need the full hidden sequence, so the
  torso cannot be a separate graph; span means travel as pooling matrices and
  the whole thing is one graph.
- `onnxconverter_common.convert_float_to_float16` leaves mixed fp16/fp32
  nodes on this graph (a `Mul` bound to both types) — the fp16 graph is
  therefore exported directly from `model.half()` in 01_export.py.
- The in-process check (wrapper vs `ClefModel`, fp32) prints
  `max |ref - wrapper|`; it must stay < 1e-4 or the export is aborted.
- `choice` option keys are sorted for the prompt, but probabilities are
  reported back under the caller's labels (the JS maps them).
