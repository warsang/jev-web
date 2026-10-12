"""Stage 2: parity — the exported graph vs the reference implementation.

Reference side: Cloudflare's own `ClefModel` (joint_schema_model.py) on
records built by its `encode_record`, run on a set of fixed records that cover
choice/noul/score, descriptions, a JSON state and a sorted-key choice.

Checks per record: option logits (max abs diff), per-question probabilities
(max abs diff) and argmax agreement. Also checks the fp16 graph when it has
been built (03_repo.py), with a wider tolerance.

Exit non-zero on any gate failure; writes export/parity.json for the manifest.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

OUT = os.environ.get("CLEF_OUT", os.path.expanduser("~/workspace/clef-export"))
CKPT = os.environ.get("CLEF_CKPT", f"{OUT}/ckpt")
EXPORT = f"{OUT}/export"
sys.path.insert(0, CKPT)

import joint_schema_model as jsm  # noqa: E402

MAX_LENGTH = 16384
TOL = {
    "clef_text_fp32.onnx": {"logits": 5e-3, "probs": 5e-3},
    "clef_text_fp16.onnx": {"logits": 1e-1, "probs": 1e-2},
}

CASES = [
    {
        "id": "mixed",
        "state": "Our checkout is down and customers can't pay. The deploy broke production and the on-call is asleep.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Payments, refunds and charges",
                             "engineering": "Bugs and outages", "other": None},
            },
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
            "urgency": {"type": "score", "instructions": "How urgent is this?",
                        "criteria": ["Routine", "Soon", "Today"]},
        },
    },
    {
        "id": "json",
        "state": {"order": 42, "product": "blender", "note": "gift card was used"},
        "questions": {
            "field": {"type": "choice", "instructions": "How was it paid?",
                      "criteria": {"gift card": "Store gift card", "card": "Credit or debit card",
                                   "cash": None}},
        },
    },
    {
        "id": "custom-noul",
        "state": "The customer attached a receipt and asks for a replacement today.",
        "questions": {
            "has_receipt": {
                "type": "noul",
                "instructions": "Does the receipt prove the purchase?",
                "criteria": {"false": "The receipt is missing or unreadable.",
                             "true": "The receipt is present and readable."},
            },
        },
    },
]


def pack(encoded, length):
    """The pooling matrices + slot metadata for one reference record."""
    import torch
    Q = len(encoded.questions)
    K = max(len(q.option_spans) for q in encoded.questions)
    S = Q * K
    pool_global = torch.zeros(1, 1, length)
    pool_question = torch.zeros(1, Q, length)
    pool_option = torch.zeros(1, S, length)
    slot_question = torch.zeros(S, dtype=torch.long)
    type_ids = torch.zeros(Q, dtype=torch.long)
    opt_mask = torch.zeros(S, dtype=torch.bool)
    pool_global[0, 0, length - 1] = 1
    for q, question in enumerate(encoded.questions):
        a, e = question.question_span
        pool_question[0, q, a:e] = 1.0 / (e - a)
        type_ids[q] = question.question_type
        for k, (a, e) in enumerate(question.option_spans):
            s = q * K + k
            slot_question[s] = q
            pool_option[0, s, a:e] = 1.0 / (e - a)
            opt_mask[s] = True
    return (pool_global.numpy(), pool_question.numpy(), pool_option.numpy(),
            type_ids.numpy(), slot_question.numpy(), opt_mask.numpy())


def main(graph="clef_text_fp32.onnx"):
    import onnxruntime as ort
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = f"{EXPORT}/{graph}"
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing — run 01_export.py first")
    tol = TOL.get(graph, {"logits": 1e-1, "probs": 1e-2})
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    print("loading the reference model (fp32)...", flush=True)
    backbone = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.float32)
    head = jsm.JointSchemaHead(**json.loads(
        open(os.path.join(CKPT, "joint_head_config.json")).read()))
    head.load_state_dict(load_file(os.path.join(CKPT, "joint_head.safetensors")), strict=True)
    model = jsm.ClefModel(backbone, head).eval()
    tokenizer = AutoTokenizer.from_pretrained(CKPT)

    worst = {"logits": 0.0, "probs": 0.0}
    failures = []
    results = []
    for case in CASES:
        encoded = jsm.encode_record(tokenizer, case, max_length=MAX_LENGTH)
        length = len(encoded.input_ids)
        with torch.no_grad():
            reference = model(jsm.collate_records(
                [encoded], tokenizer.pad_token_id, torch.device("cpu")))[0]
        packed = pack(encoded, length)
        feeds = {
            "input_ids": np.array([list(encoded.input_ids)], dtype=np.int64),
            "attention_mask": np.ones((1, length), dtype=np.int64),
            "pool_global": packed[0], "pool_question": packed[1],
            "pool_option": packed[2], "type_ids": packed[3],
            "slot_question": packed[4], "opt_mask": packed[5],
        }
        exported = session.run(["logits"], feeds)[0][0]  # [S]
        opt_mask = packed[5]
        ref_logits = np.concatenate([t.numpy() for t in reference])
        exp_logits = exported[opt_mask]
        logit_diff = float(np.abs(ref_logits - exp_logits).max())

        ref_probs, exp_probs = [], []
        offset = 0
        argmax_ok = True
        for question in encoded.questions:
            k = len(question.option_spans)
            r = ref_logits[offset:offset + k]
            e = exp_logits[offset:offset + k]
            rp = np.exp(r - r.max()) / np.exp(r - r.max()).sum()
            ep = np.exp(e - e.max()) / np.exp(e - e.max()).sum()
            argmax_ok = argmax_ok and int(rp.argmax()) == int(ep.argmax())
            ref_probs.append(rp)
            exp_probs.append(ep)
            offset += k
        prob_diff = float(np.abs(np.concatenate(ref_probs) - np.concatenate(exp_probs)).max())

        ok = logit_diff <= tol["logits"] and prob_diff <= tol["probs"] and argmax_ok
        results.append({"id": case["id"], "tokens": length,
                        "logits": round(logit_diff, 5), "probs": round(prob_diff, 6),
                        "argmax": bool(argmax_ok), "ok": bool(ok)})
        print(f"  {case['id']}: tokens={length} logits={logit_diff:.2e} "
              f"probs={prob_diff:.2e} argmax={'ok' if argmax_ok else 'FLIPPED'}", flush=True)
        if not ok:
            failures.append(case["id"])
        worst["logits"] = max(worst["logits"], logit_diff)
        worst["probs"] = max(worst["probs"], prob_diff)

    summary = {"graph": graph, "tolerance": tol, "results": results,
               "worst": {k: round(v, 6) for k, v in worst.items()}}
    with open(f"{EXPORT}/parity-{graph}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    if failures:
        raise SystemExit(f"parity FAILED for {failures}")
    print(f"parity OK: worst logits {worst['logits']:.2e}, probs {worst['probs']:.2e}", flush=True)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
