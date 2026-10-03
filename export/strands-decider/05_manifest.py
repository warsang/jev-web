"""Stage 5: write MANIFEST.json and finalize the staged artifacts."""
import datetime, hashlib, json, os

OUT = os.path.expanduser("~/workspace/strands-export")
FINAL = f"{OUT}/final"


def sha256(path, n=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    with open(f"{OUT}/quant_choice.txt") as fh:
        quant_choice = fh.read().strip()
    with open(f"{OUT}/parity_fp16.json") as fh:
        parity_fp16 = json.load(fh)
    with open(f"{OUT}/parity_quant.json") as fh:
        parity_quant = json.load(fh)
    with open(f"{OUT}/merged/hidden_size.json") as fh:
        hidden = json.load(fh)

    files = {}
    for name in sorted(os.listdir(FINAL)):
        p = os.path.join(FINAL, name)
        if os.path.isfile(p):
            files[name] = {"bytes": os.path.getsize(p),
                           "sha256": sha256(p)}

    manifest = {
        "model": "strands-decider-2b-web",
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": {
            "base_model": "Qwen/Qwen3.5-2B-Base",
            "base_revision": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
            "checkpoint": "StrandsAgents/strands-decider-2B-hobson-v19",
            "checkpoint_revision": "bb282d786bc251fd4e3068de3ada9ddbb38127cd",
        },
        "pipeline": {
            "env": "torch CPU (download.pytorch.org), transformers>=5.0, peft, "
                   "safetensors, onnx, onnxruntime",
            "merge": "Qwen3_5ForCausalLM.from_pretrained(text_config) -> .model, "
                     "PeftModel.from_pretrained(lora) + merge_and_unload(), torso->fp16, head fp32",
            "export": "torch.onnx.dynamo_export wrapper "
                      "(inputs input_ids/attention_mask/opt_idx int64; output logits fp32 [B,K])",
            "quantization": quant_choice,
        },
        "graph_contract": {
            "inputs": {"input_ids": "int64 [B, L] dynamic",
                       "attention_mask": "int64 [B, L] dynamic",
                       "opt_idx": "int64 [B, K] dynamic, -1 padded"},
            "output": {"logits": "float32 [B, K], raw (no temperature/softmax)"},
            "hidden_size": hidden["hidden_size"],
            "pointer_dim": 256,
        },
        "parity_torch_fp16_vs_onnx_fp16": parity_fp16,
        "parity_quant_vs_fp16": parity_quant,
        "quantization_choice": quant_choice,
        "files": files,
        "notes": [
            "LoRA merged into base exactly as modeling.py::_load_torso (qwen3_5).",
            "PointerHead runs in fp32; torso in fp16.",
            "No temperature or softmax in the graph; the JS consumer applies per-kind temperatures.",
        ],
    }
    with open(f"{FINAL}/MANIFEST.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest, indent=2)[:2000])
    print("\nMANIFEST written to", f"{FINAL}/MANIFEST.json", flush=True)


if __name__ == "__main__":
    main()
