# Vega 0.8B — browser export: flow notes (2026-10-10)

Status: **exported, parity-passed, ready to upload.** fp16 torso (exact) +
fp32 engine, verified against the reference on the author's own eval items:
worst feature diff 2.5e-3, logits 3.8e-3, probabilities 1.3e-3, argmax
agreement 100% across choice/score/boolean.

## What ships

| checkpoint | base | engine | browser? |
|---|---|---|---|
| `nandakishorm/vega-08b-public-intents` (root, 0.8B) | Qwen3.5-0.8B | 14.3M params (57 MB) | ✅ ~1.7 GB repo (1.5 GB torso + 57 MB engine) |
| same repo's `4b/` variant | Qwen3.5-4B | 30.9M params | ❌ over the browser budget |

## Layout

```
~/workspace/vega-export/
  base/     Qwen/Qwen3.5-0.8B (single shard renamed to model.safetensors)
  ckpt/     nandakishorm/vega-08b-public-intents @ dd3600b
            (engine.safetensors, adapters/, vega_config.json, vega_common.py,
             vega_api.py, eval/typed_decisions_items.jsonl)
  export/   01_export_torso.py, 02_export_engine.py, 03_parity.py,
            04_repo.py, 05_upload.py, make_js_fixtures.py, export_patch.py
  final/    the browser repo contents (~1.7 GB), manifest rebuilt to exclude
            the opt-in q8 graph
```

## Flow (all steps ran on this box, 2026-10-10)

```bash
python -m venv ~/workspace/.venvs/sdx && . ~/workspace/.venvs/sdx/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=5.0" peft safetensors onnx onnxruntime onnxscript numpy
PYTHONUTF8=1 python export/01_export_torso.py   # ~25 min dynamo trace -> torso_fp16.onnx
PYTHONUTF8=1 python export/02_export_engine.py  # -> engine.onnx
PARITY_ITEMS=24 PYTHONUTF8=1 python export/03_parity.py
PYTHONUTF8=1 python export/04_repo.py           # final/ + MANIFEST.json
python export/make_js_fixtures.py               # test/fixtures/vega-protocol.json
hf auth login && python export/05_upload.py     # prints the sha to pin
```

Windows gotchas that cost time: the dynamo exporter prints a ✅ and dies on
the cp1252 console (`PYTHONUTF8=1`), and `wsl --shutdown` happened twice
while downloads ran — run everything from the Windows side.

## Graph contracts (must match src/vega.mjs)

torso (`onnx/torso_fp16.onnx`):
- in `input_ids[i64 BxL]`, `attention_mask[i64 BxL]`, `pool[f32 B x rows x L]`
  (rows = 3 + K + C per item: state, question, options..., last, then C chunk
  rows; each row is 1/len over its span)
- out `pooled[f32 B x Lcap x rows x d]` — Lcap = 2 (layers 13, 19),
  d = 1024; the JS reads slots 0..2 per layer, options + chunk rows from the
  deepest layer.

engine (`onnx/engine.onnx`):
- in `feats[f32 Bx2x3xd]`, `opts[f32 BxKxd]`, `opt_mask[bool BxK]`,
  `qtype[i64 B]`, `chunks[f32 BxCxd]`, `chunk_mask[bool BxC]`
- out `logits[f32 BxK]`, `kinetic[f32 B]`, `nearest[f32 B]`,
  `steps_used[i64 B]`, `grad_norm[f32 B]` (raw; the JS applies the thermal
  calibration from `vega_config.json`).

## Notes / gotchas

- The **tokenizer ships with the backbone**, not the checkpoint: Perception
  loads `AutoTokenizer.from_pretrained(backbone)`. The export and the JS
  `calibrationModel`/tokenizer both point at the base repo.
- Qwen3.5's GatedDeltaNet chunked scan is patched before the export
  (`export_patch.py`, the strands one; validated bit-identical there). The
  parity stage compares against the **unpatched** reference, so a patch
  regression fails loudly.
- The 01 script exported **fp32 as well** (`torso_fp32.onnx`) before the
  fp16 ship path was chosen; it is not part of the upload.
- The backbone runs all 24 layers even though only 13/19 are read
  (`output_hidden_states=True` — ~26% wasted torso compute). Exporting a
  truncated stack needs manual decoder-layer plumbing; parity is what guards
  correctness, not the flop count.
- The engine graph is exported with the **legacy tracer** and a mixed demo
  batch (choice + score + boolean), so the score-rail branch and the settle
  loop are baked in; verified on a choice/boolean-only batch to still match
  the reference to 1e-5.
- **q8 does not ship**: dynamic q8 (per-channel, pool MatMuls + Convs
  excluded, reduced range) drifts up to 7.3 raw / 0.46 after excluding Convs
  in the pooled features and flips demo argmaxes — the linear-attention delta
  rule rejects activation quantization. `onnx/torso_quantized.onnx` is
  uploaded out-of-band as an experiment only. The default is the exact fp16
  torso (~1.5 GB, the strands 2B q8 default's weight class).
- adapters (`ckpt/adapters/*`) are **not** in v1: the runtime uses the base
  engine, which is the source repo's default (`adapters="off"` parity).
- The 73,728-token trained window stays in the config; hosts can pass
  `maxLen` to cap it.

## Split 8-bit export (nvkudva/vega-onnx-q8) — supported 2026-10-12

The runtime side lives in `src/vega.mjs` (`createVegaDecider({variant: "q8"})`),
fixtures in `test/fixtures/vega-q8-protocol.json`
(`make_q8_js_fixtures.py`) and the real-graph checks in
`06_verify_q8_export.py` (graphs cached in `~/workspace/vega-q8`, 840 MB,
wasm-only). Graph contracts it verified against the shipped files:

- `embed_q8`: `input_ids -> inputs_embeds` (i64 [B,L] -> f32 [B,L,1024]);
- `decoder_q8`: `inputs_embeds + attention_mask + position_ids[3,B,L]` (text:
  `0..L-1` on all three rows) + 40 zeroed `past_*` inputs -> `h13`, `h19`
  (the cut residual stream; the LM head/layers 20-23 never run);
- `engine`: `feats[B,2,3,1024] + opts[B,K,1,1024] + opt_mask + qtype +
  chunks[B,16,1024] + chunk_mask -> logits/kinetic/nearest/steps_used/gates[2]`
  (fp32, both adapters in-graph as gates).

Measured: JS pooling reproduces the reference `Perception._pooled_batch`
exactly (0.0); the q8 engine graph matches the PyTorch engine 1.4e-6; final
probabilities 1.9e-7. Against the fp16 backbone reference on the smoke prompts:
argmax 3/3, max |Δp| 2.4e-3, KL <= 1.7e-5 — consistent with the export's own
card (argmax preserved, mean KL 4.0e-04 on their 30-question set). The vision
encoder is not wired in (text states only).

The whole JS runtime was also run end-to-end in Node against the real graphs
(wasm onnxruntime-web, all three sessions from the local cache): the printed
prompt is byte-identical to the reference layout, and the choice smoke prompt
returns p=[0.4532, 0.3956, 0.1512] vs the Python reference's
[0.453, 0.396, 0.151] (argmax billing, gate ~1.5e-7 -> base calibration, the
same route the reference takes).
