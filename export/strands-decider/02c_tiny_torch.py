"""Compute torch reference logits for the tiny feeds (standalone torch process)."""
import os
import sys
import numpy as np
import torch

OUT = os.path.expanduser("~/workspace/strands-export")
sys.path.insert(0, OUT)
from importlib import import_module
exp = import_module("02_export")

model = exp.load_export_model()
torch.manual_seed(0)
feeds = [
    (np.ones((1, 16), np.int64), np.ones((1, 16), np.int64),
     np.array([15], np.int64), np.array([[14, 10, -1]], np.int64)),
    # batch is 1 (export is batch-1 only, like Laya's head); vary seq length
    (np.ones((1, 32), np.int64), np.ones((1, 32), np.int64),
     np.array([31], np.int64), np.array([[30, 20, 10, 5]], np.int64)),
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
    print(f"feed {i}: torch logits {t.flatten().tolist()}", flush=True)
np.savez(f"{OUT}/export/tiny_refs.npz", **refs)
print("tiny refs saved", flush=True)
