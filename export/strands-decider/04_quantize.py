"""Stage 4: quantize + pick the shipping graph.

- model_fp16.onnx : full-precision reference (kept).
- model_q4.onnx    : onnxruntime weight-only 4-bit (MatMul4BitsQuantizer), external data.
- fallback         : q8 dynamic quantization if q4 fails argmax agreement.

Parity bar for the winner: argmax agreement 100% vs the fp16 ONNX on the same
15 prompts; report max |dlogit| and any calibration shift. The winner is copied
to final/model.onnx.
"""
import json, os, sys
import numpy as np

OUT = os.path.expanduser("~/workspace/strands-export")
EXPORT = f"{OUT}/export"
FINAL = f"{OUT}/final"
os.makedirs(FINAL, exist_ok=True)
sys.path.insert(0, OUT)


def parity_inputs():
    """Rebuild the exact parity inputs from stage 3."""
    import_module = __import__("importlib").import_module
    p3 = import_module("03_parity")
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(f"{OUT}/ckpt",
                                                     trust_remote_code=False)
    return p3.build_inputs(tok)


def run_parity(onnx_path, inputs):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    outs = []
    for name, ids, mask, opt, labels in inputs:
        o = sess.run(None, {"input_ids": ids, "attention_mask": mask,
                            "opt_idx": opt})[0][0]
        outs.append((name, o, labels))
    return outs


def main():
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer

    fp16_path = f"{EXPORT}/model_fp16.onnx"
    q4_path = f"{EXPORT}/model_q4.onnx"
    q8_path = f"{EXPORT}/model_q8.onnx"

    inputs = parity_inputs()
    print(f"parity set: {len(inputs)} prompts", flush=True)

    print("quantizing to 4-bit (weight-only, MatMul4BitsQuantizer)...", flush=True)
    quant = MatMul4BitsQuantizer(model=fp16_path, block_size=128,
                                 is_symmetric=True, nodes_to_exclude=None)
    quant.process()
    quant.model.save_model_to_file(q4_path, use_external_data_format=True)
    print("saved", q4_path, flush=True)

    ref = run_parity(fp16_path, inputs)
    q4 = run_parity(q4_path, inputs)
    worst, agree = 0.0, True
    for (n1, r, lab), (n2, q, _) in zip(ref, q4):
        d = float(np.abs(r - q).max())
        worst = max(worst, d)
        ok = int(r.argmax()) == int(q.argmax())
        agree &= ok
        print(f"[{'OK ' if ok else 'MISMATCH'}] {n1:16s} max|d|={d:.2e} "
              f"fp16={lab[int(r.argmax())]!r} q4={lab[int(q.argmax())]!r}",
              flush=True)

    report = {"q4": {"worst_abs_logit_diff": worst, "argmax_agreement": bool(agree)}}
    winner, wpath = "q4", q4_path
    if not agree:
        print("q4 failed argmax agreement -> falling back to q8 dynamic", flush=True)
        quantize_dynamic(fp16_path, q8_path, weight_type=QuantType.QUInt8)
        q8 = run_parity(q8_path, inputs)
        worst8, agree8 = 0.0, True
        for (n1, r, lab), (n2, q, _) in zip(ref, q8):
            d = float(np.abs(r - q).max())
            worst8 = max(worst8, d)
            ok = int(r.argmax()) == int(q.argmax())
            agree8 &= ok
            print(f"[{'OK ' if ok else 'MISMATCH'}] {n1:16s} max|d|={d:.2e}",
                  flush=True)
        report["q8"] = {"worst_abs_logit_diff": worst8,
                        "argmax_agreement": bool(agree8)}
        winner, wpath = "q8", q8_path
        assert agree8, "q8 also failed argmax agreement"

    print(f"\nWINNER: {winner} worst|d|={report[winner]['worst_abs_logit_diff']:.2e}",
          flush=True)
    with open(f"{OUT}/parity_quant.json", "w") as fh:
        json.dump(report, fh, indent=2)

    # Stage into final/
    import shutil
    shutil.copy(wpath, f"{FINAL}/model.onnx")
    data_src = wpath + ".data"
    if os.path.exists(data_src):
        shutil.copy(data_src, f"{FINAL}/model.onnx.data")
    shutil.copy(fp16_path, f"{FINAL}/model_fp16.onnx")
    if os.path.exists(fp16_path + ".data"):
        shutil.copy(fp16_path + ".data", f"{FINAL}/model_fp16.onnx.data")
    for f in ("tokenizer.json", "tokenizer_config.json"):
        src = f"{OUT}/ckpt/{f}"
        if os.path.exists(src):
            shutil.copy(src, f"{FINAL}/{f}")
    with open(f"{OUT}/quant_choice.txt", "w") as fh:
        fh.write(winner)
    print("QUANT STAGE DONE, winner =", winner, flush=True)


if __name__ == "__main__":
    main()
