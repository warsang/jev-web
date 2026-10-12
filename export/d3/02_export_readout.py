"""Stage 2: export the d3 code readout (255 x d, fp32).

    last    fp32 [B, D]      the torso's last-position hidden
    counts  int64 [B]        the question's option count
    ->
    logits  fp32 [B, 255]    masked code logits (beyond counts -> -inf)

1 MB fp32 — a graph of its own instead of a second parity path. The JS slices
each row to its first `counts` entries and softmaxes with the checkpoint
temperature.
"""
from __future__ import annotations

import json
import os

import torch

OUT = os.environ.get("D3_OUT", os.path.expanduser("~/workspace/d3edge-export"))
CKPT = os.environ.get("D3_CKPT", f"{OUT}/ckpt")
EXPORT = f"{OUT}/export"
os.makedirs(EXPORT, exist_ok=True)
MAX_OPTIONS = 255


class D3Readout(torch.nn.Module):
    def __init__(self, weight, temperature):
        super().__init__()
        self.register_buffer("weight", weight)  # [255, D] fp32
        self.temperature = float(temperature)

    def forward(self, last, counts):
        logits = last.float() @ self.weight.T
        invalid = torch.arange(MAX_OPTIONS, device=logits.device)[None] >= counts[:, None]
        return logits.masked_fill(invalid, float("-inf"))


def main():
    from safetensors.torch import load_file

    config = json.loads(open(f"{CKPT}/decision_config.json").read())
    weight = load_file(f"{CKPT}/readout.safetensors")["weight"]
    print("readout weight:", tuple(weight.shape), weight.dtype, flush=True)
    model = D3Readout(weight, config.get("temperature", 1.0)).eval()

    batch, dim = 3, weight.shape[1]
    last = torch.randn(batch, dim)
    counts = torch.tensor([2, 4, 3])
    with torch.no_grad():
        out = model(last, counts)
    assert out.shape == (batch, MAX_OPTIONS)
    assert torch.isinf(out[0, 2]) and not torch.isinf(out[0, 1])

    from torch.export import Dim
    b = Dim("batch", min=1, max=16)
    program = torch.onnx.export(
        model, (last, counts), dynamo=True, optimize=False,
        dynamic_shapes=({0: b}, {0: b}),
        input_names=["last", "counts"], output_names=["logits"],
    )
    path = f"{EXPORT}/readout.onnx"
    program.save(path)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
