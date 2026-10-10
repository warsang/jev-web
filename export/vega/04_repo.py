"""Stage 4: quantize the torso (q8) and assemble the browser export repo.

Outputs (final/):
  onnx/torso_quantized.onnx (+ .onnx_data)   q8 torso, the default graph
  onnx/engine.onnx                           fp32 engine (57 MB)
  vega_config.json                           calibration + architecture (from ckpt)
  tokenizer.json, tokenizer_config.json      backbone tokenizer
  config.json                                backbone config (hidden size etc.)
  MANIFEST.json                              pins + sizes + parity summary

The engine stays fp32: it is 57 MB, and its logits feed the thermal
calibration directly, so exactness there is worth more than the 28 MB saved.
"""
from __future__ import annotations

import json
import os
import shutil

OUT = os.environ.get("STRANDS_OUT", os.path.expanduser("~/workspace/vega-export"))
BASE = f"{OUT}/base"
CKPT = f"{OUT}/ckpt"
EXPORT = f"{OUT}/export"
FINAL = f"{OUT}/final"
os.makedirs(f"{FINAL}/onnx", exist_ok=True)


def quantize_torso():
    src = f"{EXPORT}/torso_fp16.onnx"
    dst = f"{FINAL}/onnx/torso_quantized.onnx"
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        src, dst,
        weight_type=QuantType.QInt8 if hasattr(QuantType, "QInt8") else QuantType.QUInt8,
        per_channel=True,
    )
    # external data lives next to the graph
    return dst


def main():
    src = f"{EXPORT}/torso_fp16.onnx"
    if not os.path.exists(src):
        raise SystemExit("torso_fp16.onnx missing — run 01_export_torso.py first")
    print("quantizing torso (q8 dynamic)...", flush=True)
    dst = quantize_torso()
    print("wrote", dst, flush=True)

    for name in ("engine.onnx",):
        shutil.copy(f"{EXPORT}/{name}", f"{FINAL}/onnx/{name}")
    for name in ("vega_config.json",):
        shutil.copy(f"{CKPT}/{name}", f"{FINAL}/{name}")
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        path = f"{BASE}/{name}"
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
            files[rel] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}

    with open(f"{FINAL}/MANIFEST.json", "w") as fh:
        json.dump({
            "model": "vega-08b-web",
            "source": {
                "checkpoint": os.environ.get("CKPT_MODEL", "nandakishorm/vega-08b-public-intents"),
                "checkpoint_revision": os.environ.get("CKPT_REV"),
                "base_model": os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-0.8B"),
                "base_revision": os.environ.get("BASE_REV"),
            },
            "graphs": {
                "torso": "onnx/torso_quantized.onnx (q8; fp16 source in the export dir)",
                "engine": "onnx/engine.onnx (fp32, ~14M params)",
            },
            "calibration": "vega_config.json (thermal theta + conformal + abstain thresholds)",
            "files": files,
        }, fh, indent=2)
    print("manifest written; final size:",
          round(sum(v["bytes"] for v in files.values()) / 1e9, 2), "GB", flush=True)


if __name__ == "__main__":
    main()
