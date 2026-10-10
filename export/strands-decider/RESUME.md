# Strands Decider — browser export: resume guide

Status as of 2026-10-10: the pipeline is **torso-agnostic and
calibration-shipping**. The original hobson-v19 run was blocked on RAM for the
final ONNX save (build box had 7 GB; the save needs 16 GB+); the *published*
`onnx-community/strands-decider-2B-hobson-v19-ONNX` pin that jev-web ships was
produced by the onnx-community `conversion/` scripts (fuse.py etc.), NOT by the
scripts below. Those scripts remain the repo-local path for a *new* checkpoint
and now handle every torso family + the browser size gate.

## Which checkpoints may be exported to the browser

The fused graph holds the whole torso, so the download is the full parameter
count at the chosen quant (`torso_support.assert_browser_feasible` enforces):

| checkpoint                                | torso       | ~params | verdict |
| ----------------------------------------- | ----------- | ------- | ------- |
| strands-decider-2B-hobson-v19 (shipped)   | qwen3_5     | 2.0B    | shipped (1.8 GB q8) |
| strands-decider-2B-qwen3.5-v1-2610        | qwen3_5     | 2.1B    | export candidate |
| strands-decider-E2B-gemma4-v1-2610        | gemma4      | 2.1B    | export candidate (see caveat) |
| strands-decider-E4B-gemma4-v1-2610        | gemma4      | 5.7B    | opt-in heavy (~5.7 GB q8 / ~3.2 GB q4f16) |
| strands-decider-12B-gemma4-v1-2610        | gemma4      | ~13.3B  | REFUSED (> MAX_DENSE_PARAMS_B=6) |
| strands-decider-26B-A4B-gemma4-v1-2610    | gemma4 MoE  | ~26B    | REFUSED (128 experts — routing needs all resident; ~13 GB even q4f16) |

`EXPORT_ALLOW_BIG=1` overrides the size gate (never the MoE gate) for a
desktop / fat-VRAM WebGPU experiment.

**E2B caveat (upstream, model card):** on requests with several questions,
E2B's yes/no answers went through a faulty GPU kernel path before the reference
repo's E2B attention fix; its Decision Index was measured pre-fix. jev-web
batches *all* questions into one forward pass — exactly the affected shape.
Prefer E4B / qwen3.5-v1, or run the multi-question parity set in 03_parity.py
against the reference runtime before pinning E2B.

## Graph contract (as of 2026-10-10 — matches the shipped pin AND the JS)

inputs `input_ids[i64 BxL]`, `attention_mask[i64 BxL]`,
`answer_pos[i64 B]` (the `<answer>` token = row's last real token),
`option_pos[i64 BxK]` (-1 = padded, clamped to 0 in-graph) →
output `logits[f32 BxK]`, raw (no temperature/softmax; the JS applies the
per-kind temperatures read from `strands_decider_config.json`).

This pipeline previously exported the older 3-input `opt_idx` form, which the
JS runtime cannot feed — 02/03/04 now emit the 4-input contract.

## Resume steps on the new box (any checkpoint)

```bash
# 1. work dir (scripts hardcode this layout)
mkdir -p ~/workspace/strands-export
cp <repo>/export/strands-decider/*.py <repo>/export/strands-decider/*.sh \
   <repo>/export/strands-decider/*.json ~/workspace/strands-export/
cd ~/workspace/strands-export

# 2. env (python 3.12)
python3 -m venv ~/workspace/.venvs/sdx
~/workspace/.venvs/sdx/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/workspace/.venvs/sdx/bin/pip install "transformers>=5.0" peft safetensors onnx onnxruntime huggingface_hub numpy

# 3. pin the checkpoint + torso family. Examples:
#    qwen3.5 v1 (2B):
TORSO_FAMILY=qwen3_5 \
BASE_MODEL=Qwen/Qwen3.5-2B-Base BASE_REV=<sha> \
CKPT_MODEL=StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 CKPT_REV=<sha> \
  bash download.sh
#    gemma4 E2B / E4B (the family switch is all it takes):
TORSO_FAMILY=gemma4 \
BASE_MODEL=google/gemma-4-E2B-it BASE_REV=<sha> \
CKPT_MODEL=StrandsAgents/strands-decider-E2B-gemma4-v1-2610 CKPT_REV=<sha> \
  bash download.sh

# 4. merge LoRA -> merged/torso_fp16.safetensors + merged/head_fp32.safetensors
#    (refuses MoE / >6B torsos; records torso_family + the calibration config
#    into merged/hidden_size.json)
~/workspace/.venvs/sdx/bin/python 01_merge.py
~/workspace/.venvs/sdx/bin/python 01b_cast.py

# 5. export -> export/model_fp16.onnx (+ .data)  — 16GB+ RAM to SAVE
~/workspace/.venvs/sdx/bin/python 02_export.py export

# 6. parity: torch fp16 vs ONNX on real prompts — require 100% argmax agreement
~/workspace/.venvs/sdx/bin/python 03_parity.py torch
~/workspace/.venvs/sdx/bin/python 03_parity.py onnx

# 7. quantize -> final/model.onnx + final/model_fp16.onnx
~/workspace/.venvs/sdx/bin/python 04_quantize.py

# 8. manifest — copies the checkpoint's calibration file into final/ as
#    strands_decider_config.json so the browser export serves it
~/workspace/.venvs/sdx/bin/python 05_manifest.py
```

Then: upload `final/` + tokenizer to your ONNX repo on Hugging Face, pin the
immutable revision into `src/strands.mjs` (`STRANDS_DEFAULT_MODEL` /
`STRANDS_DEFAULT_REVISION`), run the full test suite + a real-weights smoke
test, and cut the release. **No JS change is needed for a new checkpoint**:
`createStrandsDecider` reads `strands_decider_config.json` from the model repo
at load time (falls back to the pinned constants when absent, `.info.configSource`
reports which). Point `calibrationModel:` at the source checkpoint when your
ONNX export predates the config-shipping step.

## Notes

- Weights (`*.safetensors`, `*.onnx`, `*.data`, `base/`, `ckpt/`, `merged/`) are
  **never committed** — they are GBs and GitHub rejects files >100MB. Re-download
  and re-merge on the new box; it is deterministic.
- `export_patch.py` (the Qwen3.5 GatedDeltaNet while_loop) is applied
  automatically only for `TORSO_FAMILY=qwen3_5`; gemma4 is plain attention.
- Optional shortcut: the old box already streamed the full while_loop weight set to
  `export/model_fp16.onnx.data` (3.77GB) with the layout in `offsets.json`. If you
  scp that file over, `save_reuse_data.py` / `save_ir_stream.py` show how it was
  built — but re-running the export is cleaner.
- `02b_tiny_ort.py` / `02c_tiny_torch.py`: tiny-shape smoke tests used during
  development. `save_graph_first.py`, `save_streaming.py`: failed save attempts,
  kept for reference only.
- The 114k-node unrolled export (Python loop unrolled 64x) saves fine but **OOMs
  in ONNX Runtime at load** — do not use it; the while_loop export is the good one.
