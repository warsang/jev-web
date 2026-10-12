"""Stage 4: assemble the browser repo around the shipped fp16 torso.

Outputs (final/):
  onnx/torso_fp16.onnx (+ external data)  the shipped torso (01_export_torso.py)
  onnx/readout.onnx                       fp32 255-way code readout (1 MB)
  decision_config.json                    codes + temperature + window
  tokenizer.json, tokenizer_config.json   the checkpoint tokenizer
  config.json                             backbone config
  MANIFEST.json                           pins + sizes + parity summary

fp16 is exported directly from a half-precision wrapper (same approach as the
clef export; onnxconverter_common leaves mixed fp16/fp32 nodes on this graph).
Runs 03_parity.py on the shipped torso as the gate.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

OUT = os.environ.get("D3_OUT", os.path.expanduser("~/workspace/d3edge-export"))
CKPT = os.environ.get("D3_CKPT", f"{OUT}/ckpt")
EXPORT = f"{OUT}/export"
FINAL = f"{OUT}/final"
DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{FINAL}/onnx", exist_ok=True)


def main():
    if not os.path.exists(f"{EXPORT}/torso_fp16.onnx"):
        raise SystemExit("torso_fp16.onnx missing — run 01_export_torso.py first")

    if not os.environ.get("D3_SKIP_PARITY"):
        # Skip only when 03_parity.py has already run on this exact graph
        # (used for the heavy d3-nano export to avoid a second reference load).
        subprocess.run([sys.executable, os.path.join(DIR, "03_parity.py"), "torso_fp16.onnx"],
                       check=True)
    summary = json.load(open(f"{EXPORT}/parity-torso_fp16.onnx.json"))

    for name in ("torso_fp16.onnx", "torso_fp16.onnx.data", "readout.onnx"):
        if os.path.exists(f"{EXPORT}/{name}"):
            shutil.copy(f"{EXPORT}/{name}", f"{FINAL}/onnx/{name}")
    for name in ("decision_config.json", "tokenizer.json", "tokenizer_config.json", "config.json"):
        path = f"{CKPT}/{name}"
        if os.path.exists(path):
            shutil.copy(path, f"{FINAL}/{name}")
    os.makedirs(f"{FINAL}/assets", exist_ok=True)
    for name in ("example-receipt.png", "example-video.mp4"):
        path = f"{CKPT}/assets/{name}"
        if os.path.exists(path):
            shutil.copy(path, f"{FINAL}/assets/{name}")

    def sha256(path, n=1 << 20):
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
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
            "model": "d3-edge-web",
            "source": {
                "checkpoint": os.environ.get("CKPT_MODEL", "vllm-sr/d3-edge"),
                "checkpoint_revision": os.environ.get("CKPT_REV"),
                "base_model": os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-0.8B"),
            },
            "graphs": {
                "torso": "onnx/torso_fp16.onnx (Qwen3.5, noncausal full attention, last-position hidden)",
                "readout": "onnx/readout.onnx (fp32 255 x d code readout)",
            },
            "scope": "text-only v1; images/videos are not part of the browser export yet",
            "parity": summary,
            "files": files,
        }, fh, indent=2)
    print("manifest written; final size:",
          round(sum(v["bytes"] for v in files.values()) / 1e9, 2), "GB", flush=True)


if __name__ == "__main__":
    main()
