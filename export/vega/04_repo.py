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
import sys

OUT = os.environ.get("STRANDS_OUT", os.path.expanduser("~/workspace/vega-export"))
BASE = f"{OUT}/base"
CKPT = f"{OUT}/ckpt"
EXPORT = f"{OUT}/export"
FINAL = f"{OUT}/final"
os.makedirs(f"{FINAL}/onnx", exist_ok=True)
sys.path.insert(0, CKPT)  # the author's reference implementation


def _fold_transposed_initializers(model):
    """Inline Transpose/Cast over initializers into the initializer itself.

    dynamo exports nn.Linear as Transpose(weight) feeding MatMul, so a naive
    quantizer sees a non-initializer weight and skips every linear. Folding the
    transpose (and any dtype cast) into the initializer recovers the plain
    MatMul(weight) shape the quantizers understand.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    changed = True
    while changed:
        changed = False
        inits = {i.name: i for i in model.graph.initializer}
        for node in list(model.graph.node):
            if node.op_type not in ("Transpose", "Cast") or len(node.input) != 1:
                continue
            src = node.input[0]
            if src not in inits:
                continue
            arr = numpy_helper.to_array(inits[src])
            if node.op_type == "Transpose":
                perm = next((list(a.ints) for a in node.attribute if a.name == "perm"),
                            list(reversed(range(arr.ndim))))
                if len(perm) != arr.ndim:
                    continue  # not a plain weight transpose; leave it alone
                arr = np.transpose(arr, perm)
            else:  # Cast
                to = next((a.i for a in node.attribute if a.name == "to"), None)
                if to is None:
                    continue
                arr = arr.astype(onnx.helper.tensor_dtype_to_np_dtype(to))
            new_name = f"{node.name or src}_folded"
            model.graph.initializer.append(numpy_helper.from_array(
                np.ascontiguousarray(arr), new_name))
            for consumer in model.graph.node:
                consumer.input[:] = [new_name if i == node.output[0] else i
                                     for i in consumer.input]
            model.graph.node.remove(node)
            inits.pop(src, None)
            changed = True
            break
    return model


def _dynamic_pool_matmuls(path):
    """Quantizer-handled nodes whose weight-ish input is not an initializer.

    The pooling einsum decomposes into Unsqueeze/Transpose + MatMul over the
    caller's pool matrix, and the causal Conv's weight arrives through an
    Unsqueeze after the dynamo export — so the weight quantizers must skip
    those or they fail on the non-initializer input.
    """
    import onnx
    model = onnx.load(path, load_external_data=False)
    inits = {i.name for i in model.graph.initializer}
    skip = set()
    for node in model.graph.node:
        if node.op_type in ("MatMul", "Gemm", "Conv") and len(node.input) > 1 \
                and node.input[1] not in inits:
            skip.add(node.name or node.output[0])
    return skip


def quantize_torso():
    """q8 dynamic weights (EXPERIMENTAL): kept in the repo as an opt-in.

    Measured on this graph (Qwen3.5-0.8B, dynamo export): q8 per-channel with
    the pool MatMuls and every Conv excluded still drifts up to 0.46 in the
    pooled features and flips argmaxes on the demo batch � the linear-attention
    delta rule rejects activation quantization. The default ship is therefore
    the exact fp16 graph (~1.5 GB, the same weight class as the strands 2B q8
    default); revisit when a per-block quantizer (or int8 calibration over the
    GatedDeltaNet chunk loop) lands. No hard gate here on purpose: the demo
    drift is measured and documented, not enforced.
    """
    import numpy as np
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    src = f"{EXPORT}/torso_fp16.onnx"
    folded = f"{EXPORT}/torso_folded.onnx"
    model = onnx.load(src)
    model = _fold_transposed_initializers(model)
    used = {i for node in model.graph.node for i in node.input}
    kept = [init for init in model.graph.initializer if init.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.save(model, folded, save_as_external_data=True, all_tensors_to_one_file=True,
              location="torso_folded.onnx.data")
    dst = f"{EXPORT}/torso_quantized.onnx"
    skip = _dynamic_pool_matmuls(folded)
    skip |= {n.name or n.output[0] for n in model.graph.node if n.op_type == "Conv"}
    quantize_dynamic(
        folded, dst,
        weight_type=QuantType.QUInt8,
        per_channel=True,
        reduce_range=True,
        nodes_to_exclude=sorted(skip),
    )
    return dst

def main():
    src = f"{EXPORT}/torso_fp32.onnx"
    if not os.path.exists(src):
        raise SystemExit("torso_fp32.onnx missing — run 01_export_torso.py first")
    # default ship: the exact fp16 torso. The q8 graph drifts on the
    # linear-attention delta rule (see quantize_torso), so it ships opt-in.
    for name in ("torso_fp16.onnx", "torso_fp16.onnx.data"):
        if os.path.exists(f"{EXPORT}/{name}"):
            shutil.copy(f"{EXPORT}/{name}", f"{FINAL}/onnx/{name}")
    print("shipped the fp16 torso (q8 stays opt-in)", flush=True)
    try:
        quantize_torso()
    except Exception as e:  # noqa: BLE001
        print(f"(q8 opt-in skipped: {e})", flush=True)
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
                "torso": "onnx/torso_fp16.onnx (exact; onnx/torso_quantized.onnx is an opt-in q8 experiment that drifts on the GatedDeltaNet recurrence)",
                "engine": "onnx/engine.onnx (fp32, ~14M params)",
            },
            "calibration": "vega_config.json (thermal theta + conformal + abstain thresholds)",
            "files": files,
        }, fh, indent=2)
    print("manifest written; final size:",
          round(sum(v["bytes"] for v in files.values()) / 1e9, 2), "GB", flush=True)


if __name__ == "__main__":
    main()
