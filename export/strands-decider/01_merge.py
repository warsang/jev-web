"""Stage 1: merge LoRA into the torso, save fp16 merged weights.

Mirrors modeling.py::_load_torso for the configured TORSO_FAMILY:
  qwen3_5  Qwen3_5ForCausalLM.from_pretrained(base, config=text_cfg).model
  gemma4   Gemma4ForConditionalGeneration.from_pretrained(base, config=cfg)
             .model.language_model
then PeftModel.from_pretrained(torso, lora_dir) + merge_and_unload().

MoE and >6B dense torsos are refused up front (see torso_support.py): the fused
ONNX would be a tens-of-GB browser download.

Outputs:
  merged/torso_fp16.safetensors   (fp16, GBs)
  merged/head_fp32.safetensors    (validated copy of the pointer head)
  merged/hidden_size.json         (+ torso_family for the later stages)
"""
import json, os, sys, torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torso_support import (
    TorsoFamilyError, assert_browser_feasible, read_ckpt_config,
    resolve_torso_class, text_config_of, torso_family,
)

import transformers

OUT = os.path.expanduser("~/workspace/strands-export")
BASE = f"{OUT}/base"
CKPT = f"{OUT}/ckpt"
MERGED = f"{OUT}/merged"
BASE_REV = os.environ.get("BASE_REV", "b1485b2fa6dfa1287294f269f5fb618e03d52d7c")
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-2B-Base")

FAMILY = torso_family()
os.makedirs(MERGED, exist_ok=True)

print(f"torso family: {FAMILY}", flush=True)
text_cfg, full_cfg = text_config_of(BASE)
print("model_type:", full_cfg.model_type, flush=True)
print("hidden_size:", text_cfg.hidden_size, flush=True)
params_b = assert_browser_feasible(text_cfg, FAMILY)
print(f"dense estimate: ~{params_b:.1f}B params (browser-feasible)", flush=True)

TorsoCls, accessor = resolve_torso_class(transformers, FAMILY)
print(f"loading {TorsoCls.__name__} (text tower only)...", flush=True)
lm = TorsoCls.from_pretrained(
    BASE, config=full_cfg if FAMILY == "gemma4" else text_cfg,
    dtype=torch.bfloat16, trust_remote_code=False,
)
torso = lm
for part in accessor.split("."):
    torso = getattr(torso, part)
del lm
print("torso params:",
      f"{sum(p.numel() for p in torso.parameters())/1e9:.2f}B", flush=True)

print("attaching LoRA adapter...", flush=True)
from peft import PeftModel
torso = PeftModel.from_pretrained(torso, f"{CKPT}/lora", is_trainable=False)

print("merging adapter into base (this takes a few minutes)...", flush=True)
with torch.no_grad():
    torso = torso.merge_and_unload()
print("merged.", flush=True)

print("saving merged torso (bf16, resumable checkpoint)...", flush=True)
state = {k: v.contiguous() for k, v in torso.state_dict().items()}
save_file(state, f"{MERGED}/torso_bf16.safetensors")
del state
print("bf16 checkpoint saved.", flush=True)

print("casting torso to fp16...", flush=True)
with torch.no_grad():
    torso = torso.to(torch.float16)
# release the merged bf16 copies
import gc; gc.collect()

print("validating pointer head state dict...", flush=True)
head_state = load_file(f"{CKPT}/head.safetensors")
print("head keys:", sorted(head_state.keys()), flush=True)
expected = {"norm.weight", "norm.bias", "q.weight", "q.bias", "k.weight", "k.bias"}
missing = expected - set(head_state.keys())
extra = set(head_state.keys()) - expected
assert not missing, f"missing head tensors: {missing}"
if extra:
    print("note: extra head tensors (kept):", extra, flush=True)
for k, v in head_state.items():
    print(f"  {k}: {tuple(v.shape)} {v.dtype}", flush=True)

print("saving merged torso (fp16 safetensors)...", flush=True)
state = {k: v.contiguous() for k, v in torso.state_dict().items()}
save_file(state, f"{MERGED}/torso_fp16.safetensors")
del state, torso
gc.collect()

print("saving head (fp32)...", flush=True)
head_fp32 = {k: v.to(torch.float32).contiguous() for k, v in head_state.items()}
save_file(head_fp32, f"{MERGED}/head_fp32.safetensors")

with open(f"{MERGED}/hidden_size.json", "w") as fh:
    json.dump({"hidden_size": text_cfg.hidden_size,
               "base_revision": BASE_REV,
               "torso_family": FAMILY,
               "ckpt_config": read_ckpt_config(CKPT)}, fh, indent=2)
print("STAGE 1 DONE", flush=True)
