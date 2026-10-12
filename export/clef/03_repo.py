"""Stage 3: assemble the browser repo around the shipped fp16 graph.

Outputs (final/):
  onnx/clef_text_fp16.onnx (+ external data)  the shipped graph (01_export.py)
  clef_config.json                            window + head shape + source
  tokenizer.json, tokenizer_config.json       the checkpoint tokenizer
  config.json                                 backbone config
  MANIFEST.json                               pins + sizes + parity summary

The fp16 graph is exported directly from a half-precision copy of the wrapper
(01_export.py) — onnxconverter_common leaves mixed fp16/fp32 nodes on this
graph, and the direct export keeps the fp32 IO contract (casts in-graph). The
head's numerics feed the reported probabilities directly, so fp16 is the
quantization floor here (no q8). This stage runs the fp16 parity gate.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

OUT = os.environ.get("CLEF_OUT", os.path.expanduser("~/workspace/clef-export"))
CKPT = os.environ.get("CLEF_CKPT", f"{OUT}/ckpt")
EXPORT = f"{OUT}/export"
FINAL = f"{OUT}/final"
DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{FINAL}/onnx", exist_ok=True)


def main():
    fp16 = f"{EXPORT}/clef_text_fp16.onnx"
    if not os.path.exists(fp16):
        raise SystemExit("clef_text_fp16.onnx missing — run 01_export.py first")

    # Re-check the shipped graph against the reference (gate: 02_parity).
    parity = f"{EXPORT}/parity-clef_text_fp16.onnx.json"
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "02_parity.py"), "clef_text_fp16.onnx"],
                   check=True)
    summary = json.load(open(parity))

    for name in ("clef_text_fp16.onnx", "clef_text_fp16.onnx.data"):
        if os.path.exists(f"{EXPORT}/{name}"):
            shutil.copy(f"{EXPORT}/{name}", f"{FINAL}/onnx/{name}")

    head_config = json.load(open(os.path.join(CKPT, "joint_head_config.json")))
    with open(f"{FINAL}/clef_config.json", "w") as fh:
        json.dump({
            "max_len": int(os.environ.get("CLEF_MAX_LEN", 16384)),
            "max_state_tokens": None,
            "head": head_config,
            "source": {
                "checkpoint": os.environ.get("CKPT_MODEL", "FluidInference/clef-text-0.6b"),
                "checkpoint_revision": os.environ.get("CKPT_REV"),
                "base_model": os.environ.get("BASE_MODEL", "Qwen/Qwen3-0.6B"),
            },
        }, fh, indent=2)

    for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        path = f"{CKPT}/{name}"
        if os.path.exists(path):
            shutil.copy(path, f"{FINAL}/{name}")

    def sha256(path, n=1 << 20):
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(n), b""):
                h.update(chunk)
        return h.hexdigest()

    files = {}
    for root, _, fs in os.walk(FINAL):
        for f in fs:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, FINAL).replace("\\", "/")
            if rel == "MANIFEST.json":
                continue
            files[rel] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}

    with open(f"{FINAL}/MANIFEST.json", "w") as fh:
        json.dump({
            "model": "clef-text-0.6b-web",
            "source": {
                "checkpoint": os.environ.get("CKPT_MODEL", "FluidInference/clef-text-0.6b"),
                "checkpoint_revision": os.environ.get("CKPT_REV"),
                "base_model": os.environ.get("BASE_MODEL", "Qwen/Qwen3-0.6B"),
                "reference": "joint_schema_model.py (Cloudflare, Apache-2.0)",
            },
            "graphs": {
                "decider": "onnx/clef_text_fp16.onnx (fused Qwen3-0.6B torso + joint schema head)",
            },
            "config": "clef_config.json (window + head shape)",
            "parity": summary,
            "files": files,
        }, fh, indent=2)
    print("manifest written; final size:",
          round(sum(v["bytes"] for v in files.values()) / 1e9, 2), "GB", flush=True)


if __name__ == "__main__":
    main()
