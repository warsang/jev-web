# Vega 0.8B — browser export: flow notes (2026-10-10)

Status: **engine exported + parity-exact; torso export in progress.** The
pipeline mirrors the strands one but is smaller (0.8B backbone) and uses the
author's own `vega_common.py` as the reference for both graphs.

## What ships

| checkpoint | base | engine | browser? |
|---|---|---|---|
| `nandakishorm/vega-08b-public-intents` (root, 0.8B) | Qwen3.5-0.8B | 14.3M params (57 MB) | ✅ ~0.9 GB q8 torso + 57 MB engine |
| same repo's `4b/` variant | Qwen3.5-4B | 30.9M params | ❌ ~4 GB q8 — over the browser budget |

The 0.8B is the one jev-web ships; the 4B stays server-side.

## Layout

```
~/workspace/vega-export/
  base/     Qwen/Qwen3.5-0.8B (model.safetensors renamed from the single shard)
  ckpt/     nandakishorm/vega-08b-public-intents @ dd3600b
            (engine.safetensors, adapters/, vega_config.json, vega_common.py,
             vega_api.py, eval/typed_decisions_items.jsonl)
  export/   01_export_torso.py, 02_export_engine.py, 03_parity.py,
            04_repo.py, make_js_fixtures.py, export_patch.py
  final/    the browser repo contents (see 04_repo.py)
```

## Flow

```bash
python -m venv ~/workspace/.venvs/sdx && . ~/workspace/.venvs/sdx/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=5.0" peft safetensors onnx onnxruntime onnxscript numpy
# base + checkpoint are already downloaded under base/ and ckpt/
# (see 00 download in the commit history; snapshot_download with
#  BASE_REV/CKPT_REV pins)

python export/01_export_torso.py   # frozen torso + span pooling -> torso_fp16.onnx
python export/02_export_engine.py  # VegaEngine (halt=True) -> engine.onnx
python export/03_parity.py         # vs the reference Vega on the author's eval items
python export/04_repo.py           # q8 torso + final/ layout + MANIFEST.json
python export/make_js_fixtures.py  # test/fixtures/vega-protocol.json (JS tests)
```

## Graph contracts (must match src/vega.mjs)

torso:
- in `input_ids[i64 BxL]`, `attention_mask[i64 BxL]`, `pool[f32 B x rows x L]`
  (rows = 3 + K + C per item: state, question, options..., last, then C chunk
  rows; each row is 1/len over its span)
- out `pooled[f32 B x Lcap x rows x d]` — Lcap = 2 (layers 13, 19),
  d = 1024; the JS reads slots 0..2 per layer, options + chunk rows from the
  deepest layer.

engine:
- in `feats[f32 Bx2x3xd]`, `opts[f32 BxKxd]`, `opt_mask[bool BxK]`,
  `qtype[i64 B]`, `chunks[f32 BxCxd]`, `chunk_mask[bool BxC]`
- out `logits[f32 BxK]`, `kinetic[f32 B]`, `nearest[f32 B]`,
  `steps_used[i64 B]`, `grad_norm[f32 B]` (raw; the JS applies the thermal
  calibration from `vega_config.json`).

## Notes / gotchas

- The **tokenizer ships with the backbone**, not the checkpoint: Perception
  loads `AutoTokenizer.from_pretrained(backbone)`. Point the export and the
  JS `calibrationModel`/tokenizer at the right repo.
- Qwen3.5's GatedDeltaNet chunked scan is patched before the export
  (`export_patch.py`, the strands one; validated bit-identical there). The
  parity stage below compares against the **unpatched** reference, so a patch
  regression fails loudly.
- The backbone runs all 24 layers even though only 13/19 are read
  (`output_hidden_states=True` — ~26% wasted torso compute). Exporting a
  truncated stack was rejected: it needs manual decoder-layer plumbing, and
  the parity gate is what protects correctness, not the flop count.
- The engine graph is exported with the **legacy tracer** and a mixed demo
  batch (choice + score + boolean), so the score-rail branch and the settle
  loop are baked in; verified on a choice/boolean-only batch to still match
  the reference to 1e-5.
- adapters (`ckpt/adapters/*`) are **not** in v1: the runtime uses the base
  engine, which is the source repo's default (`adapters="off"` parity).
- The 73,728-token trained window stays in the config; hosts can pass
  `maxLen` to cap it.
