"""Stage 1: export the frozen Qwen3.5 torso (up to the deepest captured layer)
with in-graph span pooling.

    input_ids      int64 [B, L]      (0-padded)
    attention_mask int64 [B, L]
    pool           fp32  [B, rows, L] one-hot span means (rows = 3 + K + C)
                                      per item: state, question, options..., last,
                                      then C state-chunk rows
    ->
    pooled         fp32  [B, Lcap, rows, d]   span means per captured layer

Lcap = |feature_layers| = 2 (layers 13 and 19 of 24); the JS consumer slices
slots 0..2 (state/question/last) from every captured layer, the option spans
and chunk rows from the deepest captured layer (option_layers == [19]).

The backbone runs in fp16 (0.8B torso, ~0.9 GB); quantize afterwards with
02_quantize.py. Qwen3.5's GatedDeltaNet chunked scan is patched to a
while_loop first (export_patch.py, validated bit-identical on the strands 2B
export); this script's parity stage compares against the UNPATCHED reference.

Requires 16 GB RAM-free box only lightly: 0.8B fp16 model + ONNX program fit
in a few GB.
"""
from __future__ import annotations

import json
import os
import sys

import torch

EXPORT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("STRANDS_OUT", os.path.expanduser("~/workspace/vega-export"))
BASE = f"{OUT}/base"
CKPT = f"{OUT}/ckpt"
EXPORT = f"{OUT}/export"
os.makedirs(EXPORT, exist_ok=True)

sys.path.insert(0, EXPORT_DIR)
sys.path.insert(0, CKPT)

# --- patch Qwen3.5's GatedDeltaNet for dynamic-sequence export -------------
import transformers.models.qwen3_5.modeling_qwen3_5 as qm
import transformers.models.qwen3_5.modular_qwen3_5 as qmm
from export_patch import patched_torch_chunk_gated_delta_rule as _patched

qm.torch_chunk_gated_delta_rule = _patched
qmm.torch_chunk_gated_delta_rule = _patched
print("DeltaNet export patch applied", flush=True)

from vega_common import resolve_layers

with open(f"{CKPT}/vega_config.json") as fh:
    cfg = json.load(fh)
FEATURE_LAYERS = resolve_layers(cfg["feature_layers"], cfg["n_layers"])
OPTION_LAYERS = resolve_layers(cfg["option_layers"], cfg["n_layers"])
CAPTURE = sorted(set(FEATURE_LAYERS) | set(OPTION_LAYERS))
D_MODEL = cfg["d_model"]
MAX_LEN = cfg["max_len"]
print(f"capture layers: {CAPTURE} (feature {FEATURE_LAYERS}, option {OPTION_LAYERS}); "
      f"d_model={D_MODEL} max_len={MAX_LEN}", flush=True)


class TorsoExport(torch.nn.Module):
    """Frozen backbone + span-mean pooling, exported as one graph."""

    def __init__(self, model, capture):
        super().__init__()
        self.model = model
        self.capture = capture

    def forward(self, input_ids, attention_mask, pool):
        out = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, use_cache=False,
        )
        hs = out.hidden_states  # tuple [B, L, d]: [0] embed, [i+1] = layer i
        stacked = torch.stack([hs[l + 1] for l in self.capture], 1)  # [B, Lcap, L, d]
        # span means per captured layer: pool [B, rows, S] x [B, Lcap, S, d]
        # -> [B, Lcap, rows, d]  (rows = 3 + K + C); the JS feeds an fp32 pool
        pool = pool.to(stacked.dtype)
        return torch.einsum("brs,bLsd->bLrd", pool, stacked)


def main():
    from transformers import AutoModelForCausalLM

    print("loading frozen backbone (fp16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, attn_implementation="sdpa", dtype=torch.float16)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"backbone layers: {model.config.num_hidden_layers} "
          f"hidden: {model.config.hidden_size}", flush=True)

    wrapper = TorsoExport(model, CAPTURE).eval()

    # Demo batch: a choice, a score, a boolean question, one long state so the
    # chunk rows are exercised (C chunk rows + padded option slots).
    from vega_common import build_prompt
    tok = __import__("transformers").AutoTokenizer.from_pretrained(BASE)
    long_state = "Support log line %d.\n" % 1
    long_state = "".join(f"Support log line {i} with some filler tokens to pad the state.\n" for i in range(40))
    questions = [
        {"t": "choice", "ins": "Which team?", "crit": {"billing": "payments", "tech": "bugs"}},
        {"t": "score", "ins": "How urgent?", "crit": ["calm", "urgent"]},
        {"t": "boolean", "ins": "Does this convey urgency?", "crit": {}},
    ]
    items = []
    for q in questions:
        ids, spans = build_prompt(tok, long_state, q, 1024, opt_max_tokens=cfg["opt_max_tokens"])
        items.append((ids, spans))
    n = len(items)
    L = max(len(ids) for ids, _ in items)
    C = cfg.get("memory_chunks", 16)
    K = max(len(sp["options"]) for _, sp in items)
    rows = 3 + K + C
    pad_id = tok.pad_token_id or 0

    input_ids = torch.full((n, L), pad_id, dtype=torch.long)
    attention = torch.zeros((n, L), dtype=torch.long)
    pool = torch.zeros(n, rows, L, dtype=torch.float32)
    for b, (ids, sp) in enumerate(items):
        m = len(ids)
        input_ids[b, :m] = torch.tensor(ids, dtype=torch.long)
        attention[b, :m] = 1
        def fill(row, a, e):
            if e > a:
                pool[b, row, a:e] = 1.0 / (e - a)
        fill(0, *sp["state"]); fill(1, *sp["question"])
        for k, (a, e) in enumerate(sp["options"]):
            fill(3 + k, a, e)
        pool[b, 2, sp["last"] if "last" in sp else m - 1] = 1
    print(f"demo batch: n={n} L={L} rows={rows}", flush=True)

    with torch.no_grad():
        ref = wrapper(input_ids, attention, pool)
    print("torch pooled:", tuple(ref.shape), "sample:", ref[0, 0, 0, :3].tolist(), flush=True)

    path = f"{EXPORT}/torso_fp16.onnx"
    from torch.export import Dim
    batch = Dim("batch", min=1, max=8)
    seq = Dim("seq", min=16, max=MAX_LEN)
    nrows = Dim("rows", min=6, max=64)
    print("exporting with torch.onnx.export (dynamo=True)...", flush=True)
    program = torch.onnx.export(
        wrapper, (input_ids, attention, pool), dynamo=True,
        optimize=False,
        dynamic_shapes=(
            {0: batch, 1: seq},
            {0: batch, 1: seq},
            {0: batch, 1: nrows, 2: seq},
        ),
        input_names=["input_ids", "attention_mask", "pool"],
        output_names=["pooled"],
    )
    del wrapper, ref
    import gc; gc.collect()
    program.save(path)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
