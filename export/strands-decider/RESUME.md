# Strands Decider 2B — browser export: resume guide

Status as of 2026-10-02: **blocked on RAM, not on correctness.** Everything below is
proven except the final ONNX save, which needs >7GB RAM (build box had 7GB; 9 save
attempts OOM'd). Resume on any box with **16GB+ RAM** (CPU-only is fine, no GPU needed).

## What is already done and verified

- **LoRA merge**: Qwen3.5-2B-Base (`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`) +
  StrandsAgents/strands-decider-2B-hobson-v19 (`bb282d786bc251fd4e3068de3ada9ddbb38127cd`)
  merged and bit-verified.
- **Export graph correctness**: `export_patch.py` replaces the Python
  `for i in range(num_chunks)` loop in the Gated DeltaNet layers with
  `torch._higher_order_ops.while_loop`, verified **bit-identical** (max|d|=0.00e+00)
  to the reference. The while_loop model translates to ONNX cleanly
  (90,106 nodes, dynamic batch/seq/n_options).
- **Target contract** (must match `src/strands.mjs`): inputs
  `input_ids[i64 BxL], attention_mask[i64 BxL], opt_idx[i64 BxK]` (-1 = padded) →
  output `logits[f32 BxK]`, raw (no temperature/softmax; JS applies per-kind temps).

## Resume steps on the new box

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

# 3. weights (public HF, no auth)
bash download.sh

# 4. merge LoRA -> merged/torso_fp16.safetensors + merged/head_fp32.safetensors (~1h)
~/workspace/.venvs/sdx/bin/python 01_merge.py
~/workspace/.venvs/sdx/bin/python 01b_cast.py

# 5. export -> export/model_fp16.onnx (+ .data) (~1-2h, needs 16GB+ RAM to SAVE)
~/workspace/.venvs/sdx/bin/python 02_export.py export

# 6. parity: torch fp16 vs ONNX on real prompts — require 100% argmax agreement
~/workspace/.venvs/sdx/bin/python 03_parity.py torch
~/workspace/.venvs/sdx/bin/python 03_parity.py onnx

# 7. quantize -> final/model.onnx (q4 if parity holds, else q8) + final/model_fp16.onnx
~/workspace/.venvs/sdx/bin/python 04_quantize.py

# 8. manifest
~/workspace/.venvs/sdx/bin/python 05_manifest.py
```

Then: upload `final/` + tokenizer to `warsang/strands-decider-2b-web` on Hugging Face,
pin the immutable revision into `src/strands.mjs` (`STRANDS_DEFAULT_REVISION`),
run the full test suite + a real-weights smoke test, and cut the release.

## Notes

- Weights (`*.safetensors`, `*.onnx`, `*.data`, `base/`, `ckpt/`, `merged/`) are
  **never committed** — they are GBs and GitHub rejects files >100MB. Re-download
  and re-merge on the new box; it is deterministic.
- Optional shortcut: the old box already streamed the full while_loop weight set to
  `export/model_fp16.onnx.data` (3.77GB) with the layout in `offsets.json`. If you
  scp that file over, `save_reuse_data.py` / `save_ir_stream.py` show how it was
  built — but re-running the export is cleaner.
- `02b_tiny_ort.py` / `02c_tiny_torch.py`: tiny-shape smoke tests used during
  development. `save_graph_first.py`, `save_streaming.py`: failed save attempts,
  kept for reference only.
- The 114k-node unrolled export (Python loop unrolled 64x) saves fine but **OOMs
  in ONNX Runtime at load** — do not use it; the while_loop export is the good one.
