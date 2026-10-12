"""Stage 1: export the d3 text torso (Qwen3.5, noncausal full attention).

    input_ids      int64 [B, L]   left-padded question prompts
    attention_mask int64 [B, L]
    ->
    last           fp32 [B, D]    final-norm hidden at the last position

Loads the checkpoint exactly like a `D3` runtime instance (stock Qwen3_5Model,
sdpa) and reproduces `enable_noncausal_full_attention`'s mask contract in the
graph: full-attention layers get the 2D padding mask broadcast non-causally
(every real token sees every real token), while the GatedDeltaNet layers keep
the causal recurrence and receive the 2D padding mask so pads never leak into
the state. The Gated DeltaNet chunk scan uses the vega/strands export patch
(dynamic-seq while_loop) — 03_parity.py gates it against the unpatched
reference.

Exports fp32 (`torso_fp32.onnx`); 04_repo.py converts to the shipped fp16
graph. Needs ~8 GB RAM:

    D3_OUT=~/workspace/d3edge-export python export/d3/01_export_torso.py
"""
from __future__ import annotations

import json
import os
import sys

import torch

DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("D3_OUT", os.path.expanduser("~/workspace/d3edge-export"))
CKPT = os.environ.get("D3_CKPT", f"{OUT}/ckpt")
EXPORT = f"{OUT}/export"
os.makedirs(EXPORT, exist_ok=True)
sys.path.insert(0, DIR)
sys.path.insert(0, CKPT)

# --- Gated DeltaNet dynamic-seq patch (before the model is built) ----------
import transformers.models.qwen3_5.modeling_qwen3_5 as qm  # noqa: E402
import transformers.models.qwen3_5.modular_qwen3_5 as qmm  # noqa: E402
from export_patch import patched_torch_chunk_gated_delta_rule as _patched  # noqa: E402

qm.torch_chunk_gated_delta_rule = _patched
qmm.torch_chunk_gated_delta_rule = _patched
print("DeltaNet export patch applied", flush=True)


class D3TorsoExport(torch.nn.Module):
    """The checkpoint's text path, last-position hidden in fp32."""

    def __init__(self, backbone):
        super().__init__()
        self.language_model = backbone.language_model
        config = json.loads(open(f"{CKPT}/decision_config.json").read())
        self.noncausal = config.get("attention_mode") == "noncausal_full_attention"

    def forward(self, input_ids, attention_mask):
        padding = attention_mask.bool()
        if self.noncausal:
            masks = {
                # softmax layers see the whole prompt, pads excepted
                "full_attention": padding[:, None, None, :],
                # linear layers stay causal; the 2D mask keeps pads out of the state
                "linear_attention": padding.contiguous(),
            }
        else:
            masks = padding[:, None, None, :]
        out = self.language_model(
            input_ids=input_ids, attention_mask=masks,
            use_cache=False, return_dict=True,
        )
        return out.last_hidden_state[:, -1].float()


def main():
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    import d3_runtime
    import d3_format

    print("loading the checkpoint (fp32)...", flush=True)
    backbone = Qwen3_5Model.from_pretrained(
        CKPT, dtype=torch.float32, attn_implementation="sdpa")
    backbone.eval().requires_grad_(False)
    wrapper = D3TorsoExport(backbone).eval()

    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    config = json.loads(open(f"{CKPT}/decision_config.json").read())
    codes = config["codes"]
    state = "Our checkout is down and customers can't pay. The deploy broke production."
    prompts = []
    for question in (
        {"type": "choice", "instructions": "Which team should handle this?",
         "criteria": {"billing": "Payments", "engineering": "Bugs or outages"}},
        {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        {"type": "score", "instructions": "How urgent is this?",
         "criteria": ["Routine", "Soon", "Today"]},
    ):
        normalized = d3_runtime.normalize_question(question)
        text = d3_format.render(tokenizer, state, normalized.rendered, codes)
        prompts.append(tokenizer(text, add_special_tokens=False)["input_ids"])
    width = max(len(p) for p in prompts)
    input_ids = torch.full((len(prompts), width), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(prompts), width), dtype=torch.long)
    for b, ids in enumerate(prompts):
        input_ids[b, width - len(ids):] = torch.tensor(ids)
        attention[b, width - len(ids):] = 1
    print(f"demo batch: rows={len(prompts)} width={width} pad={tokenizer.pad_token_id}",
          flush=True)

    # The wrapper runs first (no hook), then the runtime's own noncausal path —
    # the two must agree to the last bit before exporting. Two runtime
    # generations ship: the older one installs a mask hook on the language
    # model (d3-edge/d3-lite), the newer one runs the layers through its own
    # PromptEncoder (d3-nano).
    lm = backbone.language_model
    with torch.no_grad():
        wrapped = wrapper(input_ids, attention)
        if hasattr(d3_runtime, "enable_noncausal_full_attention"):
            d3_runtime.enable_noncausal_full_attention(lm)
            reference = backbone(
                input_ids=input_ids, attention_mask=attention.bool(),
                use_cache=False, return_dict=True,
            ).last_hidden_state[:, -1].float()
            label = "hook path"
        elif hasattr(d3_runtime, "PromptEncoder"):
            embeds = lm.embed_tokens(input_ids)
            encoder = d3_runtime.PromptEncoder(lm)
            reference = encoder(embeds, attention)[:, -1].float()
            label = "PromptEncoder path"
        else:
            raise SystemExit("this d3 runtime has no noncausal entry point")
    diff = (reference - wrapped).abs().max().item()
    print(f"in-process check ({label} vs wrapper): max diff = {diff:.2e}", flush=True)
    if diff > 1e-5:
        raise SystemExit("wrapper diverges from the runtime's mask contract")

    # The wrapper builds the mask dict itself, so take the runtime hook back
    # off (it would reject the dict as a non-2D padding mask).
    lm._forward_pre_hooks.clear()

    if os.environ.get("D3_DRY"):
        print("D3_DRY set — stopping before the ONNX export", flush=True)
        return

    # --- the shipped fp16 graph: same wrapper, half weights -----------------
    print("half-precision copy...", flush=True)
    backbone.half()
    wrapper_half = D3TorsoExport(backbone).eval()
    with torch.no_grad():
        half_last = wrapper_half(input_ids, attention)
        half_diff = (half_last - wrapped).abs().max().item()
    print(f"fp16 in-process check (vs fp32 wrapper): max diff = {half_diff:.2e}", flush=True)

    from torch.export import Dim
    batch = Dim("batch", min=1, max=16)
    seq = Dim("seq", min=16, max=int(os.environ.get("D3_MAX_TOKENS", "8192")))
    print("exporting with torch.onnx.export (dynamo=True)...", flush=True)
    program = torch.onnx.export(
        wrapper_half, (input_ids, attention), dynamo=True, optimize=False,
        dynamic_shapes=({0: batch, 1: seq}, {0: batch, 1: seq}),
        input_names=["input_ids", "attention_mask"],
        output_names=["last"],
    )
    path = f"{EXPORT}/torso_fp16.onnx"
    program.save(path)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
