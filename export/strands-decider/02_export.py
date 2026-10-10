"""Stage 2: export the fused decider graph to ONNX and tiny-validate with ORT.

Graph contract (must match jev-web/src/strands.mjs AND the published
onnx-community export — both feed answer_pos/option_pos):
  inputs:  input_ids int64 [B, L], attention_mask int64 [B, L],
           answer_pos int64 [B], option_pos int64 [B, K] (-1 = padded)
  output:  logits float32 [B, K]  (raw, no temperature/softmax)

Math mirrors modeling.py / onnx-community's fuse.py exactly:
  hidden = torso(input_ids, attention_mask).last_hidden_state      # fp16 [B,L,d]
  answer  = hidden.gather(1, answer_pos.view(B,1,1))               # fp16 [B,d]
  options = hidden.gather(1, option_pos.clamp_min(0))              # fp16 [B,K,d]
  head (fp32): LayerNorm -> q/k Linear(d,D) ->
               logits = (k_out @ q_out) * D**-0.5                 # fp32 [B,K]

answer_pos is each row's last real token (the `<answer>` pooling position);
option_pos sits at each option line's last token. The JS collate pads option
slots with -1, hence clamp_min(0) — padded slots are never read back.
"""
import json, os, sys, torch
import torch.nn as nn
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torso_support import (
    apply_qwen3_5_export_patch, resolve_torso_class, torso_family,
)

OUT = os.path.expanduser("~/workspace/strands-export")
MERGED = f"{OUT}/merged"
EXPORT = f"{OUT}/export"
os.makedirs(EXPORT, exist_ok=True)


class DeciderExport(nn.Module):
    def __init__(self, torso, head_state, hidden_size, pointer_dim=256):
        super().__init__()
        self.torso = torso
        self.norm = nn.LayerNorm(hidden_size)
        self.q = nn.Linear(hidden_size, pointer_dim)
        self.k = nn.Linear(hidden_size, pointer_dim)
        self.norm.load_state_dict({k: head_state[f"norm.{k}"]
                                   for k in ("weight", "bias")})
        self.q.load_state_dict({k: head_state[f"q.{k}"]
                                for k in ("weight", "bias")})
        self.k.load_state_dict({k: head_state[f"k.{k}"]
                                for k in ("weight", "bias")})
        for p in list(self.norm.parameters()) + list(self.q.parameters()) + list(self.k.parameters()):
            p.requires_grad_(False)
        self.scale = pointer_dim ** -0.5

    def forward(self, input_ids, attention_mask, answer_pos, option_pos):
        out = self.torso(input_ids=input_ids, attention_mask=attention_mask,
                         use_cache=False, return_dict=True)
        hidden = out.last_hidden_state                      # [B, L, d]
        d = hidden.size(-1)
        b = hidden.size(0)
        # exact-position readout: the JS runtime hands us the `<answer>` token
        # and each option line's last token (padded slots arrive as -1).
        answer = hidden.gather(                               # [B, 1, d]
            1, answer_pos.view(b, 1, 1).expand(b, 1, d)).squeeze(1)
        options = hidden.gather(                             # [B, K, d]
            1, option_pos.clamp_min(0).unsqueeze(-1).expand(-1, -1, d))
        decide = self.q(self.norm(answer.float())).unsqueeze(-1)   # [B, dim, 1]
        keys = self.k(self.norm(options.float()))                  # [B, K, dim]
        return (keys @ decide).squeeze(-1) * self.scale             # [B, K]


def load_export_model():
    import transformers
    from safetensors import safe_open
    with open(f"{MERGED}/hidden_size.json") as fh:
        meta = json.load(fh)
    hidden_size = meta["hidden_size"]
    family = meta.get("torso_family") or torso_family()
    if family == "qwen3_5":
        # Qwen3.5's GatedDeltaNet layers need the while_loop patch before the
        # model is built (bit-identical, verified — see export_patch.py).
        apply_qwen3_5_export_patch(OUT)
    print(f"building {family} torso with proper init (fp16 default dtype, no ckpt load)...",
          flush=True)
    base_cfg = transformers.AutoConfig.from_pretrained(f"{OUT}/base")
    text_cfg = base_cfg.get_text_config()
    TorsoCls, accessor = resolve_torso_class(transformers, family)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        lm = TorsoCls(base_cfg if family == "gemma4" else text_cfg)
    finally:
        torch.set_default_dtype(prev_dtype)
    torso = lm
    for part in accessor.split("."):
        torso = getattr(torso, part)
    del lm
    print("streaming fp16 weights into torso...", flush=True)
    sd = torso.state_dict()
    path = f"{MERGED}/torso_fp16.safetensors"
    with safe_open(path, framework="pt", device="cpu") as f:
        fkeys = set(f.keys())
        assert fkeys == set(sd.keys()), \
            f"key mismatch: only-in-file={fkeys - set(sd.keys())} only-in-model={set(sd.keys()) - fkeys}"
        for i, k in enumerate(f.keys()):
            sd[k].copy_(f.get_tensor(k))
            if (i + 1) % 80 == 0:
                print(f"  {i + 1}/{len(fkeys)}", flush=True)
    print("torso weights loaded", flush=True)
    head_state = load_file(f"{MERGED}/head_fp32.safetensors")
    model = DeciderExport(torso, head_state, hidden_size).eval()
    return model


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "export"
    model = load_export_model()
    torch.manual_seed(0)

    path = f"{EXPORT}/model_fp16.onnx"

    if which in ("export", "tiny"):
        # Example inputs for tracing. Seq must be within the Dim range
        # [65, 4096] (see export_patch notes on the seq=64 solver quirk).
        args = (torch.ones(1, 128, dtype=torch.long),
                torch.ones(1, 128, dtype=torch.long),
                torch.tensor([127], dtype=torch.long),
                torch.tensor([[120, 100, -1]], dtype=torch.long))
        with torch.no_grad():
            ref = model(*args)
        print("torch ref logits:", ref.flatten().tolist(), flush=True)
        print("exporting with torch.onnx.export (dynamo=True)...", flush=True)
        from torch.export import Dim
        batch = Dim("batch", min=1, max=16)
        seq = Dim("seq", min=65, max=4096)
        n_options = Dim("n_options", min=1, max=64)
        program = torch.onnx.export(
            model, args, dynamo=True,
            optimize=False,  # onnxscript optimizer hangs on this graph; skip it
            dynamic_shapes=(
                {0: batch, 1: seq},
                {0: batch, 1: seq},
                {0: batch},
                {0: batch, 1: n_options},
            ),
            input_names=["input_ids", "attention_mask",
                         "answer_pos", "option_pos"],
            output_names=["logits"],
        )
        path = f"{EXPORT}/model_fp16.onnx"
        # Free the torch model before saving: the ONNX program (3.8GB) plus
        # the torch model (3.8GB) do not fit in RAM together.
        del model
        import gc
        gc.collect()
        print("torch model released; saving ONNX via program.save()...", flush=True)
        program.save(path)
        print("saved", path, flush=True)


    if which in ("tiny", "validate"):
        tiny_validate(model, path)


def tiny_validate(model, path):
    """Memory-safe tiny validation: torch refs to disk, free model, then ORT."""
    # NOTE: torch model (3.8GB) + ORT session (3.8GB) do not fit in RAM
    # together. Compute torch refs, dump to disk, free the model, THEN
    # run ORT after release.
    print("computing torch reference logits for tiny feeds...", flush=True)
    import numpy as np
    feeds = [
        (np.ones((1, 16), np.int64), np.ones((1, 16), np.int64),
         np.array([15], np.int64), np.array([[14, 10, -1]], np.int64)),
        (np.ones((2, 32), np.int64), np.ones((2, 32), np.int64),
         np.array([31, 31], np.int64),
         np.array([[30, 20, 10, 5], [31, 21, -1, -1]], np.int64)),
    ]
    refs = {}
    for i, (ids, mask, ans, opt) in enumerate(feeds):
        with torch.no_grad():
            t = model(torch.from_numpy(ids), torch.from_numpy(mask),
                      torch.from_numpy(ans), torch.from_numpy(opt)).numpy()
        refs[f"ref{i}"] = t
        refs[f"ids{i}"] = ids
        refs[f"mask{i}"] = mask
        refs[f"ans{i}"] = ans
        refs[f"opt{i}"] = opt
        print(f"  feed {i}: torch logits {t.flatten()[:4].tolist()}",
              flush=True)
    np.savez(f"{EXPORT}/tiny_refs.npz", **refs)
    del refs, model
    import gc
    gc.collect()
    print("torch model released; validating with onnxruntime...",
          flush=True)
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print("graph inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()],
          flush=True)
    print("graph outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()],
          flush=True)
    z = np.load(f"{EXPORT}/tiny_refs.npz")
    for i in range(2):
        ids, mask, ans, opt = z[f"ids{i}"], z[f"mask{i}"], z[f"ans{i}"], z[f"opt{i}"]
        t = z[f"ref{i}"]
        o = sess.run(None, {"input_ids": ids, "attention_mask": mask,
                            "answer_pos": ans, "option_pos": opt})[0]
        d = np.abs(t - o).max()
        agree = (t.argmax(-1) == o.argmax(-1)).all()
        print(f"shape {ids.shape}/{opt.shape}: max|dtorch-donnx|={d:.2e} "
              f"argmax_agree={agree}", flush=True)
        assert agree, "ARGMAX MISMATCH on tiny validation"
        assert d < 1e-3, f"tiny parity too loose: {d}"
    print("TINY VALIDATION PASSED", flush=True)


if __name__ == "__main__":
    main()
