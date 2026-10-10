"""Stage 1b: stream-cast merged/torso_bf16.safetensors -> torso_fp16.safetensors.

Avoids Module.to() on the live multi-B-param model (killed twice, likely memory).
Streams one tensor at a time: peak RAM ~1GB. bf16->fp16 keeps element size,
so data_offsets are unchanged; only the dtype tag flips BF16->F16.
"""
import json
import os
import struct
import torch
from safetensors import safe_open

OUT = os.environ.get("STRANDS_OUT", os.path.expanduser("~/workspace/strands-export"))
SRC = f"{OUT}/merged/torso_bf16.safetensors"
DST = f"{OUT}/merged/torso_fp16.safetensors"

with open(SRC, "rb") as fin:
    n = struct.unpack("<Q", fin.read(8))[0]
    header = json.loads(fin.read(n))

items = list(header.items())
new_items = []
for k, v in items:
    if k == "__metadata__":
        new_items.append((k, v))
        continue
    assert v["dtype"] == "BF16", f"unexpected dtype in {k}: {v['dtype']}"
    new_items.append((k, {"dtype": "F16", "shape": v["shape"],
                          "data_offsets": v["data_offsets"]}))
new_header_json = json.dumps(dict(new_items)).encode()

order = sorted((v["data_offsets"][0], k) for k, v in new_items
               if k != "__metadata__")
print(f"{len(order)} tensors, header {len(new_header_json)} bytes", flush=True)

with open(DST, "wb") as fout:
    fout.write(struct.pack("<Q", len(new_header_json)))
    fout.write(new_header_json)
    with safe_open(SRC, framework="pt", device="cpu") as f:
        for i, (_, k) in enumerate(order):
            t = f.get_tensor(k).to(torch.float16).contiguous()
            fout.write(t.numpy().tobytes())
            del t
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(order)}", flush=True)

print("wrote", DST, flush=True)

# verify: reload + spot-check against bf16 originals
import numpy as np
with safe_open(SRC, framework="pt", device="cpu") as fb, \
     safe_open(DST, framework="pt", device="cpu") as ff:
    assert set(fb.keys()) == set(ff.keys()), "key mismatch"
    worst = 0.0
    for k in ["model.embed_tokens.weight",
              "model.layers.0.self_attn.q_proj.weight",
              "model.layers.23.mlp.down_proj.weight",
              "model.norm.weight"]:
        if k not in fb.keys():
            continue
        a = fb.get_tensor(k).float()
        b = ff.get_tensor(k).float()
        d = (a - b).abs().max().item()
        worst = max(worst, d)
        print(f"  {k}: max|bf16-fp16|={d:.2e}", flush=True)
    print(f"spot-check worst diff: {worst:.2e}", flush=True)
    assert worst < 1e-2, "cast drift too large"
print("STAGE 1b DONE", flush=True)
