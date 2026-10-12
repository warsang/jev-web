# d3-edge — browser export: flow notes (2026-10-12)

Status: **exported, parity-passed, JS-verified end to end; uploaded.** Three
browser repos are live (`warsang/d3-edge-web`, `warsang/d3-lite-web`,
`warsang/d3-nano-web`) and pinned in `src/decision3.mjs`. The fp16 torso
(Qwen3.5, noncausal full attention, GatedDeltaNet patch) plus the fp32 code
readout reproduce the reference `d3_runtime.D3` (BF16) — both in Python and
through the real jev-web runtime:

| check | result |
|---|---|
| in-process, wrapper vs the runtime's noncausal path | max diff **0.00e+00** (edge + lite) |
| in-process, fp16 wrapper vs fp32 wrapper | max last-hidden diff **4.7e-03** (edge), **9.8e-03** (lite) |
| ONNX fp16 torso+readout vs reference (6 questions) | probs ≤ **1.18e-02** (edge), **4.85e-03** (lite), argmax **6/6** |
| real JS runtime vs reference (fixtures, fp16 graphs) | probs ≤ **1.18e-02**, prompt bytes **identical** |

`final/` sizes: edge 1.06 GB (torso 995 MB), lite 1.68 GB (torso 1.59 GB),
nano ~4.4 GB (torso, fp16).

Two runtime generations ship in the collection: d3-edge/d3-lite install a mask
hook (`enable_noncausal_full_attention`), while d3-nano's runtime runs its own
`PromptEncoder` over the layers — the same mask contract, and
`01_export_torso.py` auto-detects which one the checkpoint ships and checks
the export wrapper against it.

## What ships

| checkpoint | base | reads | browser? |
|---|---|---|---|
| `vllm-sr/d3-edge` (0.59B, +0.10B vision) | Qwen3.5-0.8B | text | ✅ ~1.0 GB fp16 torso + 1 MB readout |
| `vllm-sr/d3-lite` (0.85B) | Qwen3.5-0.8B | text | ✅ same pipeline, `D3_OUT=~/workspace/d3lite-export` |
| `vllm-sr/d3-nano` (2.21B) | Qwen3.5-2B | text | ⚠️ heavy (~4.4 GB fp16) |

Multimodal v2: images/videos need the checkpoint's vision tower and processor
(1.6 MP per image, ≤32 video frames at 2 fps) — a separate graph on top of the
same code readout.

## Flow

```bash
python -m venv ~/workspace/.venvs/sdx && . ~/workspace/.venvs/sdx/bin/activate
pip install torch transformers safetensors onnx onnxruntime numpy

D3_OUT=~/workspace/d3edge-export python export/d3/01_export_torso.py    # fp32 torso
D3_OUT=~/workspace/d3edge-export python export/d3/02_export_readout.py  # 255-way readout
D3_OUT=~/workspace/d3edge-export python export/d3/03_parity.py          # fp32 gate
D3_OUT=~/workspace/d3edge-export python export/d3/04_repo.py            # fp16 + final/
D3_OUT=~/workspace/d3edge-export python export/d3/make_js_fixtures.py   # test fixtures
hf auth login && D3_OUT=~/workspace/d3edge-export python export/d3/05_upload.py
```

## Graph contracts (must match src/decision3.mjs)

torso (`onnx/torso_fp16.onnx`):
- in `input_ids[i64 BxL]`, `attention_mask[i64 BxL]` (left padding)
- out `last[f32 BxD]` — the final-norm hidden at the last position. The
  reference's noncausal mask contract is baked in: full-attention layers get
  the 2D padding mask broadcast non-causally; GatedDeltaNet layers keep the
  causal recurrence with the 2D mask so pads never leak into the state.

readout (`onnx/readout.onnx`):
- in `last[f32 BxD]`, `counts[i64 B]`
- out `logits[f32 Bx255]`, entries beyond `counts` masked to `-inf`; the JS
  softmaxes with the checkpoint temperature.

## Notes / gotchas

- The Gated DeltaNet chunk scan is patched for dynamic-seq export
  (`export_patch.py`, the vega/strands patch); the scan accumulator covers
  `D3_MAX_CHUNKS` chunks (env, default 128 = 8192 tokens) — that (and the
  browser budget) is the effective window. 03_parity.py runs the *unpatched*
  reference, so a patch regression fails loudly.
- 01 loads the stock `Qwen3_5Model` like the runtime does and reproduces
  `enable_noncausal_full_attention`'s mask dict in the wrapper, then checks
  the two against each other in-process before exporting.
- The checkpoint's `decision_config.json` ships the answer-code table; the
  browser decider loads it at runtime (falling back to the shipped A…Z/AA…ZZ
  list). `make_js_fixtures.py` asserts the checkpoint's table against its own
  tokenizer.
- fp16 is exported the same way as clef (a half-precision wrapper with fp32
  IO): onnxconverter_common leaves mixed fp16/fp32 nodes on these graphs.
- Run every stage with `PYTHONUTF8=1` (the dynamo exporter's ✅ on a cp1252
  console).
- `permutation_average` (the reference's option-order averaging) is **not**
  part of the browser export: the npm runtime scores each question once.
