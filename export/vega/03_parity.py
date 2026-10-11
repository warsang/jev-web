"""Stage 3: parity — ONNX torso + ONNX engine vs the reference implementation.

Reference side: the author's own vega_common.Vega (Perception over the frozen
backbone + VegaEngine + thermal calibration), run with adapters="off" and
halt=True — exactly what the browser runtime reproduces.

Checks, per item from the author's own eval file:
  1. torso ONNX pooled features vs Perception.pooled (max abs diff)
  2. engine ONNX logits vs VegaEngine (argmax agreement + max abs diff)
  3. final thermal probabilities vs Vega.decide (max abs diff)

Items come from ckpt/eval/typed_decisions_items.jsonl when present, else a
small built-in set. Exit non-zero on any gate failure.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import onnxruntime as ort

OUT = os.environ.get("STRANDS_OUT", os.path.expanduser("~/workspace/vega-export"))
BASE = f"{OUT}/base"
CKPT = f"{OUT}/ckpt"
EXPORT = f"{OUT}/export"
FINAL = f"{OUT}/final"
sys.path.insert(0, CKPT)

N_ITEMS = int(os.environ.get("PARITY_ITEMS", "24"))
MAX_TOL = {"feats": 5e-3, "logits": 5e-2, "probs": 5e-3}

FALLBACK_ITEMS = [
    {"state": "I was charged twice for the same order and nobody answers my emails. I want my money back now.",
     "qs": [{"t": "choice", "ins": "Which team handles this?",
             "crit": {"billing": "payments and refunds", "tech": "bugs", "sales": "pricing"}},
            {"t": "boolean", "ins": "Does this convey urgency?", "crit": {}},
            {"t": "score", "ins": "How frustrated is the writer?",
             "crit": ["calm", "frustrated", "depressed"]}]},
    {"state": "Your support agent was wonderful, truly the best, but the product itself is garbage.",
     "qs": [{"t": "choice", "ins": "Which team handles this?",
             "crit": {"billing": "payments and refunds", "tech": "bugs", "sales": "pricing"}},
            {"t": "boolean", "ins": "Does this convey urgency?", "crit": {}}]},
]


def load_items():
    path = f"{CKPT}/eval/typed_decisions_items.jsonl"
    if os.path.exists(path):
        recs = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("qs"):
                    recs.append(rec)
                if len(recs) >= N_ITEMS:
                    break
        if recs:
            return recs
    return FALLBACK_ITEMS


def main():
    import vega_common
    from vega_api import Vega
    import torch

    cfg = json.load(open(f"{CKPT}/vega_config.json"))
    C = int(cfg.get("memory_chunks", 0) or 0)

    # --- reference ---------------------------------------------------------
    vega = Vega(CKPT, backbone=BASE, device=torch.device("cpu"), halt=True,
                adapters="off", calibration="default")
    tok = vega.perc.tok

    torso = ort.InferenceSession(f"{FINAL}/onnx/torso_fp16.onnx", providers=["CPUExecutionProvider"])
    engine = ort.InferenceSession(f"{EXPORT}/engine.onnx", providers=["CPUExecutionProvider"])

    items_recs = load_items()
    # flatten reference records into (state_text, q) pairs
    flat = []
    for rec in items_recs:
        for q in rec["qs"]:
            k = len(vega_common.option_texts(q))
            if k >= 2:
                flat.append((vega_common.serialize_state(rec["state"]), q))
    flat = flat[:N_ITEMS]
    print(f"parity on {len(flat)} questions", flush=True)

    worst = {"feats": 0.0, "logits": 0.0, "probs": 0.0}
    agree = {"feats": True, "logits": True, "probs": True}
    for idx, (state_text, q) in enumerate(flat):
        qdef = vega_common.option_texts(q)
        # --- reference: one question, one view
        ids, spans = vega_common.build_prompt(
            tok, state_text, q, cfg["max_len"],
            opt_max_tokens=cfg["opt_max_tokens"],
            query_first_max_state=cfg.get("query_first_max_state", 0))
        feats_ref, opt_ref = vega.perc.pooled([{"ids": np.asarray(ids, np.int32), "spans": spans}])
        with torch.no_grad():
            out_ref = vega.engine(
                feats_ref, opt_ref.unsqueeze(0),
                torch.tensor([[True] * len(qdef)]),
                torch.tensor([vega_common.QTYPES[q["t"]]]),
                halt=True, return_traj=False, prestandardized=False,
                chunks=None, chunk_mask=None)
        phys_ref = vega_common.thermal_features(out_ref, torch.ones(1, len(qdef), dtype=torch.bool),
                                                vega.engine.max_steps).numpy()
        probs_ref = vega_common.softmax_np(
            out_ref["logits"].float().numpy(),
            vega_common.thermal_tau(vega.theta, np.array([vega_common.QTYPES[q["t"]]]), phys_ref))[0]

        # --- onnx: same prompt, JS packing rules mirrored in numpy
        n, L = 1, len(ids)
        K = len(qdef)
        nseg = 3 + K
        rows = nseg + C
        input_ids = np.array([ids], np.int64)
        attention = np.ones((1, L), np.int64)
        pool = np.zeros((1, rows, L), np.float32)
        def fill(row, a, e):
            if e > a:
                pool[0, row, a:e] = 1.0 / (e - a)
        fill(0, *spans["state"]); fill(1, *spans["question"])
        for k, (a, e) in enumerate(spans["options"]):
            fill(3 + k, a, e)
        pool[0, 2, L - 1] = 1.0
        # chunks
        chunk_mask = np.zeros((1, C), bool)
        [s0, s1] = spans["state"]
        slen = s1 - s0
        if C and slen >= max(cfg.get("memory_min_tokens", 2048), 2):
            k_ch = max(2, min(C, slen // max(1, cfg.get("memory_chunk_tokens", 128))))
            for c in range(k_ch):
                row = nseg + c
                for t in range(s0, s1):
                    if ((t - s0) * k_ch) // slen == c:
                        pool[0, row, t] = 1.0
                pool[0, row] /= max(1, pool[0, row].sum())
                chunk_mask[0, c] = True

        pooled = torso.run(None, {"input_ids": input_ids, "attention_mask": attention,
                                  "pool": pool})[0]  # [1, Lcap, rows, d]
        d = pooled.shape[-1]
        feats_onnx = np.stack([pooled[0, :, s, :] for s in range(3)], 1)[None]  # [1, Lcap, 3, d]
        opts_onnx = pooled[0, -1, 3:3 + K, :][None]  # [1, K, d]
        chunks_onnx = pooled[0, -1, nseg:nseg + C, :][None] if C else np.zeros((1, 0, d), np.float32)

        # torso parity: reference feats are [1, Lcap, 3, d]
        feats_ref_np = feats_ref.numpy()
        d_feats = float(np.abs(feats_onnx[:, :, :3, :] - feats_ref_np).max())
        d_opt = float(np.abs(opts_onnx - opt_ref.numpy()[None, :, 0, :]).max())
        worst["feats"] = max(worst["feats"], d_feats, d_opt)

        eng = engine.run(None, {
            "feats": feats_onnx.astype(np.float32),
            "opts": opts_onnx.astype(np.float32),
            "opt_mask": np.array([[True] * K]),
            "qtype": np.array([vega_common.QTYPES[q["t"]]], np.int64),
            "chunks": chunks_onnx.astype(np.float32) if C else np.zeros((1, 0, d), np.float32),
            "chunk_mask": chunk_mask if C else np.zeros((1, 0), bool),
        })
        logits_onnx = eng[0][0]
        logits_ref = out_ref["logits"].float().numpy()[0]
        d_logits = float(np.abs(logits_onnx - logits_ref).max())
        worst["logits"] = max(worst["logits"], d_logits)
        agree["logits"] &= int(logits_onnx.argmax()) == int(logits_ref.argmax())

        # reference does not use chunks for short items; nearest/steps compare too
        phys = np.array([[
            np.log1p(float(eng[1][0])), np.log1p(float(eng[2][0])),
            float(eng[3][0]) / max(1, vega.engine.max_steps),
            np.log(max(2, K))]], np.float64)
        tau = vega_common.thermal_tau(vega.theta, np.array([vega_common.QTYPES[q["t"]]]), phys)[0]
        z = logits_onnx / tau
        z = z - z.max()
        e = np.exp(z)
        probs_onnx = e / e.sum()
        d_probs = float(np.abs(probs_onnx - probs_ref).max())
        worst["probs"] = max(worst["probs"], d_probs)

        mark = "OK " if d_probs < MAX_TOL["probs"] else "DRIFT"
        print(f"[{mark}] {idx:2d} {q['t']:7s} K={K} L={L:4d} "
              f"feats={d_feats:.2e} opt={d_opt:.2e} logits={d_logits:.2e} probs={d_probs:.2e}",
              flush=True)

    print(f"\nPARITY RESULT: worst feats={worst['feats']:.2e} logits={worst['logits']:.2e} "
          f"probs={worst['probs']:.2e} argmax_agree={agree['logits']}", flush=True)
    ok = (worst["feats"] < MAX_TOL["feats"] and worst["logits"] < MAX_TOL["logits"]
          and worst["probs"] < MAX_TOL["probs"] and agree["logits"])
    print("PARITY PASSED" if ok else "PARITY FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
