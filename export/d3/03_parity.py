"""Stage 3: parity — the ONNX torso + readout vs the reference d3 runtime.

Reference side: `d3_runtime.D3` loaded from the checkpoint (BF16 backbone,
FP32 readout, the runtime's own noncausal mask hook, the *unpatched*
GatedDeltaNet — so a patch regression fails loudly). This is the exact object
behind the published Decision Index numbers.

Checks per question: probabilities (max abs diff after the checkpoint
temperature) and argmax agreement, on a fixed set covering choice/noul/score.
Also checks the fp16 torso (once 04_repo.py built it) with a wider tolerance.
Exit non-zero on any gate failure; writes export/parity-<graph>.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

OUT = os.environ.get("D3_OUT", os.path.expanduser("~/workspace/d3edge-export"))
CKPT = os.environ.get("D3_CKPT", f"{OUT}/ckpt")
EXPORT = f"{OUT}/export"
sys.path.insert(0, CKPT)

import d3_format  # noqa: E402
import d3_runtime  # noqa: E402

TOL = {
    "torso_fp16.onnx": {"probs": 5e-2},
}

CASES = [
    {
        "state": "Our checkout is down and customers can't pay. The deploy broke production and the on-call is asleep.",
        "questions": {
            "route": {"type": "choice", "instructions": "Which team should handle this?",
                      "criteria": {"billing": "Payments, refunds and charges",
                                   "engineering": "Bugs and outages", "other": None}},
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
            "urgency": {"type": "score", "instructions": "How urgent is this?",
                        "criteria": ["Routine", "Soon", "Today"]},
        },
    },
    {
        "state": {"order": 42, "product": "blender", "paid": ["card", 19.5]},
        "questions": {
            "payment": {"type": "choice", "instructions": "How was it paid?",
                        "criteria": {"card": None, "cash": None, "gift card": "Store gift card"}},
        },
    },
    {
        "state": "The customer attached a receipt and asks for a replacement today.",
        "questions": {
            "receipt": {"type": "noul", "instructions": "Does the receipt prove the purchase?",
                        "criteria": {"false": "The receipt is missing or unreadable.",
                                     "true": "The receipt is present and readable."}},
        },
    },
]


def main(torso_graph="torso_fp16.onnx", readout_graph="readout.onnx"):
    import onnxruntime as ort

    torso_path = f"{EXPORT}/{torso_graph}"
    readout_path = f"{EXPORT}/{readout_graph}"
    for path in (torso_path, readout_path):
        if not os.path.exists(path):
            raise SystemExit(f"{path} missing — run the export stages first")
    tol = TOL.get(torso_graph, {"probs": 5e-2})
    torso = ort.InferenceSession(torso_path, providers=["CPUExecutionProvider"])
    readout = ort.InferenceSession(readout_path, providers=["CPUExecutionProvider"])

    print("loading the reference runtime (bf16)...", flush=True)
    # verify="none": the snapshot here has no assets/ (README images); the
    # text path does not touch them.
    decision = d3_runtime.D3.from_pretrained(
        CKPT, device="cpu", verify=os.environ.get("D3_VERIFY", "none"))

    failures = []
    results = []
    worst_prob = 0.0
    for case in CASES:
        prepared = decision.prepare(case["state"], case["questions"], None, None)
        if prepared.errors:
            raise SystemExit(f"reference rejected a case: {prepared.errors}")
        keys = prepared.runnable
        sequences = [prepared.sequences[k] for k in keys]
        counts_ref = [len(prepared.questions[k].keys) for k in keys]
        reference, _ = decision.run(prepared)  # {key: [probs]}
        counts = np.array(counts_ref, dtype=np.int64)

        width = max(len(s) for s in sequences)
        ids = np.full((len(sequences), width), decision.pad_id, dtype=np.int64)
        mask = np.zeros((len(sequences), width), dtype=np.int64)
        for b, seq in enumerate(sequences):
            ids[b, width - len(seq):] = seq
            mask[b, width - len(seq):] = 1

        last = torso.run(["last"], {"input_ids": ids, "attention_mask": mask})[0].astype(np.float32)
        logits = readout.run(["logits"], {"last": last, "counts": counts})[0]
        temperature = decision.temperature

        per_question = []
        for b, key in enumerate(keys):
            n = counts_ref[b]
            row = logits[b, :n] / temperature
            row = row - row.max()
            probs = np.exp(row) / np.exp(row).sum()
            ref = np.array(reference[key], dtype=np.float64)
            diff = float(np.abs(probs - ref).max())
            argmax_ok = int(probs.argmax()) == int(ref.argmax())
            per_question.append({"key": key, "probs": round(diff, 6),
                                 "argmax": bool(argmax_ok)})
            worst_prob = max(worst_prob, diff)
            if diff > tol["probs"] or not argmax_ok:
                failures.append(f"{key}: probs={diff:.4f} argmax={'ok' if argmax_ok else 'FLIPPED'}")
        results.append({"questions": per_question})
        print("  " + ", ".join(
            f"{q['key']}({q['probs']:.2e}{'' if q['argmax'] else '!argmax'})"
            for q in per_question), flush=True)

    summary = {"torso": torso_graph, "readout": readout_graph,
               "tolerance": tol, "results": results,
               "worst_probs": round(worst_prob, 6)}
    with open(f"{EXPORT}/parity-{torso_graph}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    if failures:
        raise SystemExit("parity FAILED:\n  " + "\n  ".join(failures))
    print(f"parity OK: worst probabilities {worst_prob:.2e}", flush=True)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
