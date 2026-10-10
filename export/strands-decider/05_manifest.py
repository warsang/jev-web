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
    meta = dict(hidden)  # carries torso_family + ckpt_config from stage 1

    # Ship the checkpoint calibration verbatim: jev-web's strands.mjs reads
    # strands_decider_config.json (or hobson_config.json on v19) from the
    # model repo at load time and lets it override the pinned defaults.
    ckpt_config = meta.get("ckpt_config") or {}
    if ckpt_config:
        with open(f"{FINAL}/strands_decider_config.json", "w") as fh:
            json.dump(ckpt_config, fh, indent=2)
        print("calibration config copied into final/", flush=True)

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
            "base_model": os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-2B-Base"),
            "base_revision": meta.get("base_revision"),
            "torso_family": meta.get("torso_family", "qwen3_5"),
            "checkpoint": os.environ.get("CKPT_MODEL",
                                         "StrandsAgents/strands-decider-2B-hobson-v19"),
            "checkpoint_revision": os.environ.get("CKPT_REV"),
        },
        "pipeline": {
            "env": "torch CPU (download.pytorch.org), transformers>=5.0, peft, "
                   "safetensors, onnx, onnxruntime",
            "merge": f"{meta.get('torso_family', 'qwen3_5')} torso via "
                     "torso_support.resolve_torso_class, LoRA merged "
                     "as modeling.py::_load_torso, torso->fp16, head fp32",
            "export": "torch.onnx.dynamo_export wrapper "
                      "(inputs input_ids/attention_mask/answer_pos/option_pos "
                      "int64; output logits fp32 [B,K])",
            "quantization": quant_choice,
        },
        "graph_contract": {
            "inputs": {"input_ids": "int64 [B, L] dynamic",
                       "attention_mask": "int64 [B, L] dynamic",
                       "answer_pos": "int64 [B] dynamic (the <answer> token)",
                       "option_pos": "int64 [B, K] dynamic, -1 padded"},
            "output": {"logits": "float32 [B, K], raw (no temperature/softmax)"},
            "hidden_size": hidden["hidden_size"],
            "pointer_dim": 256,
        },
        "parity_torch_fp16_vs_onnx_fp16": parity_fp16,
        "parity_quant_vs_fp16": parity_quant,
        "quantization_choice": quant_choice,
        "files": files,
        "notes": [
            "LoRA merged into base exactly as modeling.py::_load_torso "
            "(torso_support.py selects the family).",
            "PointerHead runs in fp32; torso in fp16.",
            "No temperature or softmax in the graph; the JS consumer applies the "
            "per-kind temperatures from strands_decider_config.json, which is "
            "copied into final/ so the browser export ships it.",
        ],
    }
    with open(f"{FINAL}/MANIFEST.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest, indent=2)[:2000])
    print("\nMANIFEST written to", f"{FINAL}/MANIFEST.json", flush=True)


if __name__ == "__main__":
    main()
