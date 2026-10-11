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


def _cast_fp16_to_fp32(model):
    """fp16 -> fp32 across initializers and value types.

    The dynamic quantizer inserts DynamicQuantizeLinear on activations, which
    accepts float only — an fp16 torso graph is invalid after quantize_dynamic.
    The weights become int8 anyway; the trade is ~0.3 GB of fp32 embeddings.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, numpy_helper

    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            new = numpy_helper.from_array(np.ascontiguousarray(arr), init.name)
            init.CopyFrom(new)
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT
    return model


def quantize_torso():
    """Fold the export's transpose/cast wrappers, then q8 the weights.

    01_export_torso exports fp32 on purpose: quantize_dynamic needs float
    activations (DynamicQuantizeLinear rejects fp16), and fp16 surgery on the
    exported graph is not reliable. The dynamic pooling MatMuls and the causal
    Convs (weights arriving through Unsqueeze) are excluded from weight
    quantization; everything else becomes int8 weights.
    """
    import vega_common
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType

    src = f"{EXPORT}/torso_fp32.onnx"
    folded = f"{EXPORT}/torso_folded.onnx"
    model = onnx.load(src)
    model = _fold_transposed_initializers(model)
    # drop initializers no node consumes any more (the fold duplicates them)
    used = {i for node in model.graph.node for i in node.input}
    kept = [init for init in model.graph.initializer if init.name in used]
    removed = len(model.graph.initializer) - len(kept)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.save(model, folded, save_as_external_data=True, all_tensors_to_one_file=True,
              location="torso_folded.onnx.data")
    print(f"transpose/cast initializers folded ({removed} orphaned removed)", flush=True)

    dst = f"{EXPORT}/torso_quantized.onnx"
    skip = _dynamic_pool_matmuls(folded)
    print(f"excluding {len(skip)} dynamic-input matmuls from quantization", flush=True)
    quantize_dynamic(
        folded, dst,
        weight_type=QuantType.QInt8 if hasattr(QuantType, "QInt8") else QuantType.QUInt8,
        per_channel=True,
        nodes_to_exclude=sorted(skip),
    )

    # demo batch (same construction as 01_export_torso)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    cfg = json.load(open(f"{CKPT}/vega_config.json"))
    state = "".join(f"Support log line {i} with some filler tokens to pad the state.\n" for i in range(40))
    questions = [
        {"t": "choice", "ins": "Which team?", "crit": {"billing": "payments", "tech": "bugs"}},
        {"t": "score", "ins": "How urgent?", "crit": ["calm", "urgent"]},
        {"t": "boolean", "ins": "Does this convey urgency?", "crit": {}},
    ]
    items = []
    for q in questions:
        ids, spans = vega_common.build_prompt(tok, state, q, 1024,
                                              opt_max_tokens=cfg["opt_max_tokens"])
        items.append((ids, spans))
    n = len(items)
    L = max(len(ids) for ids, _ in items)
    C = int(cfg.get("memory_chunks", 16) or 0)
    K = max(len(sp["options"]) for _, sp in items)
    rows = 3 + K + C
    input_ids = np.full((n, L), tok.pad_token_id or 0, np.int64)
    attention = np.zeros((n, L), np.int64)
    pool = np.zeros((n, rows, L), np.float32)
    for b, (ids, sp) in enumerate(items):
        m = len(ids)
        input_ids[b, :m] = ids
        attention[b, :m] = 1
        def fill(row, a, e):
            if e > a:
                pool[b, row, a:e] = 1.0 / (e - a)
        fill(0, *sp["state"]); fill(1, *sp["question"])
        for k, (a, e) in enumerate(sp["options"]):
            fill(3 + k, a, e)
        pool[b, 2, m - 1] = 1.0

    # fp16 reference vs q4 quantized on the same feeds
    ref = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(
        None, {"input_ids": input_ids, "attention_mask": attention, "pool": pool})[0]
    q4 = ort.InferenceSession(dst, providers=["CPUExecutionProvider"]).run(
        None, {"input_ids": input_ids, "attention_mask": attention, "pool": pool})[0]
    diff = float(np.abs(q4 - ref).max())
    agree = bool((q4.argmax(-1) == ref.argmax(-1)).all())
    print(f"q4 vs fp16 on demo batch: max|d|={diff:.2e} argmax_agree={agree}", flush=True)
    if not agree or diff > 0.5:
        raise SystemExit("quantized torso drifted too far — ship the fp16 graph")
    np.savez(f"{EXPORT}/torso_demo.npz", input_ids=input_ids, attention=attention,
             pool=pool, pooled=ref)
    return dst


def main():
    src = f"{EXPORT}/torso_fp32.onnx"
    if not os.path.exists(src):
        raise SystemExit("torso_fp32.onnx missing — run 01_export_torso.py first")
    print("quantizing torso (q8 dynamic)...", flush=True)
    dst = quantize_torso()
    shutil.copy(dst, f"{FINAL}/onnx/torso_quantized.onnx")
    for ext in (".onnx_data", ".onnx_data_1"):
        if os.path.exists(f"{EXPORT}/torso_quantized.onnx{ext}"):
            shutil.copy(f"{EXPORT}/torso_quantized.onnx{ext}", f"{FINAL}/onnx/torso_quantized.onnx{ext}")
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
                "torso": "onnx/torso_quantized.onnx (q8; fp32 source in the export dir)",
                "engine": "onnx/engine.onnx (fp32, ~14M params)",
            },
            "calibration": "vega_config.json (thermal theta + conformal + abstain thresholds)",
            "files": files,
        }, fh, indent=2)
    print("manifest written; final size:",
          round(sum(v["bytes"] for v in files.values()) / 1e9, 2), "GB", flush=True)


if __name__ == "__main__":
    main()
