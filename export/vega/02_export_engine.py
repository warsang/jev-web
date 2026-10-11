"""Stage 2: export the Vega physics engine.

    feats       fp32 [B, Lcap, 3, d]   per captured layer: state/question/last
    opts        fp32 [B, K, d]          option means at the option layer
    opt_mask    bool [B, K]
    qtype       int64 [B]              0 choice / 1 score / 2 boolean
    chunks      fp32 [B, C, d]          state chunk means (memory reader)
    chunk_mask  bool [B, C]
    ->
    logits      fp32 [B, K]             raw (no temperature; JS applies thermal)
    kinetic     fp32 [B]                residual kinetic energy at the end
    nearest     fp32 [B]                distance to the nearest well bottom
    steps_used  int64 [B]               steps to settle (<= n_steps)
    grad_norm   fp32 [B]                final force norm (diagnostics)

The reference VegaEngine.forward runs with halt=True / return_traj=False and
prestandardized=False (raw perception output). The 12-step integrator is
unrolled; settled items freeze via torch.where, which is exactly the
reference's early-exit behaviour with halt=True. The score-question rail is
traced in with a mixed demo batch (a choice, a score and a boolean item), so
one graph serves every qtype mix.

The engine is ~14M params -> keep fp32 (57 MB); the torso is the big graph.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

EXPORT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("STRANDS_OUT", os.path.expanduser("~/workspace/vega-export"))
CKPT = f"{OUT}/ckpt"
EXPORT = f"{OUT}/export"
os.makedirs(EXPORT, exist_ok=True)
sys.path.insert(0, CKPT)

import vega_common
from safetensors.torch import load_file

with open(f"{CKPT}/vega_config.json") as fh:
    cfg = json.load(fh)

D_MODEL = cfg["d_model"]
D = cfg["latent_dim"]
LCAP = len(vega_common.resolve_layers(cfg["feature_layers"], cfg["n_layers"]))
C = int(cfg.get("memory_chunks", 16) or 0)


class EngineExport(torch.nn.Module):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def forward(self, feats, opts, opt_mask, qtype, chunks, chunk_mask):
        # this checkpoint has a single option layer (option_layers == [19]);
        # the graph interface stays [B, K, d] and the layer axis is added here.
        out = self.engine(
            feats, opts.unsqueeze(2), opt_mask, qtype,
            halt=True, return_traj=False, prestandardized=False,
            chunks=chunks, chunk_mask=chunk_mask,
        )
        return {
            "logits": out["logits"],
            "kinetic": out["kinetic"],
            "nearest": out["nearest"],
            "steps_used": out["steps_used"].to(torch.int64),
            "grad_norm": out["grad_norm"],
        }


def main():
    engine = vega_common.VegaEngine(
        D_MODEL, LCAP, len(vega_common.resolve_layers(cfg["option_layers"], cfg["n_layers"])), cfg)
    engine.load_state_dict(load_file(f"{CKPT}/engine.safetensors"))
    engine.eval()
    engine.adapter_mode = "off"
    n_params = sum(p.numel() for p in engine.parameters())
    print(f"engine params: {n_params/1e6:.1f}M", flush=True)

    # Mixed demo batch: one choice, one score, one boolean (traces the rail).
    n = 3
    feats = torch.randn(n, LCAP, 3, D_MODEL)
    opts = torch.randn(n, 3, D_MODEL)
    opt_mask = torch.ones(n, 3, dtype=torch.bool)
    qtype = torch.tensor([0, 1, 2], dtype=torch.long)
    chunks = torch.randn(n, C, D_MODEL) if C else torch.zeros(n, 0, D_MODEL)
    chunk_mask = torch.ones(n, C, dtype=torch.bool) if C else torch.zeros(n, 0, dtype=torch.bool)

    with torch.no_grad():
        out = EngineExport(engine)(feats, opts, opt_mask, qtype, chunks, chunk_mask)
    print("torch logits:", [round(float(v), 4) for v in out["logits"].flatten().tolist()],
          "steps:", out["steps_used"].tolist(), flush=True)

    path = f"{EXPORT}/engine.onnx"
    torch.onnx.export(
        EngineExport(engine).eval(),
        (feats, opts, opt_mask, qtype, chunks, chunk_mask),
        path,
        dynamo=False,  # legacy tracer: data-dependent python control flow
                       # (bool(is_score.any())) is baked in with this demo batch
        opset_version=17,
        input_names=["feats", "opts", "opt_mask", "qtype", "chunks", "chunk_mask"],
        output_names=["logits", "kinetic", "nearest", "steps_used", "grad_norm"],
        dynamic_axes={
            "feats": {0: "batch"}, "opts": {0: "batch", 1: "options"},
            "opt_mask": {0: "batch", 1: "options"},
            "qtype": {0: "batch"}, "chunks": {0: "batch"}, "chunk_mask": {0: "batch"},
            "logits": {0: "batch", 1: "options"},
            "kinetic": {0: "batch"}, "nearest": {0: "batch"},
            "steps_used": {0: "batch"}, "grad_norm": {0: "batch"},
        },
        do_constant_folding=True,
    )
    print("saved", path, flush=True)

    # Save the demo inputs/outputs for the parity stage and the JS fixtures.
    np.savez(
        f"{EXPORT}/engine_demo.npz",
        feats=feats.numpy(), opts=opts.numpy(), opt_mask=opt_mask.numpy(),
        qtype=qtype.numpy(), chunks=chunks.numpy(), chunk_mask=chunk_mask.numpy(),
        logits=out["logits"].numpy(), kinetic=out["kinetic"].numpy(),
        nearest=out["nearest"].numpy(), steps_used=out["steps_used"].numpy(),
        grad_norm=out["grad_norm"].numpy(),
    )
    print("demo fixture saved", flush=True)


if __name__ == "__main__":
    main()
