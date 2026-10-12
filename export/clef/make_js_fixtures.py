"""Emit JS test fixtures from the REFERENCE clef implementation.

The JS protocol tests (test/clef.test.mjs) need ground truth the browser
runtime must reproduce exactly: the record ids and question/option spans from
Cloudflare's unchanged joint_schema_model.encode_record (run on the real
tokenizer), the piece tokenizations the JS encode() stub replays, and the
SystemOne answers from systemone_answer().

    CLEF_SRC=/path/to/clef-text-0.6b python export/clef/make_js_fixtures.py

CLEF_SRC is a checkpoint snapshot (joint_schema_model.py, tokenizer files,
joint_head_config.json); snapshot_download("FluidInference/clef-text-0.6b")
gives one.

With `--e2e` it also loads the real weights and writes
test/fixtures/clef-e2e-reference.json (per-question probabilities for one
record) — the expected side of export/clef/verify_js_local.mjs.
"""
from __future__ import annotations

import json
import os
import sys

SRC = os.environ.get("CLEF_SRC", os.path.expanduser("~/workspace/clef-export/ckpt"))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES = os.path.join(REPO, "test", "fixtures")
os.makedirs(FIXTURES, exist_ok=True)

sys.path.insert(0, SRC)
import joint_schema_model as jsm  # noqa: E402

MAX_LENGTH = 16384

CASES = [
    {
        "name": "mixed",
        "id": "checkout",
        "state": "Our checkout is down and customers can't pay. The deploy broke production and the on-call is asleep.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments, refunds and charges",
                    "engineering": "Bugs and outages",
                    "other": None,
                },
            },
            "refund": {
                "type": "noul",
                "instructions": "Is the customer asking for a refund?",
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["Routine", "Soon", "Today"],
            },
        },
    },
    {
        "name": "choice-sorted",
        "id": "payment",
        "state": {"order": 42, "product": "blender", "note": "gift card was used"},
        "questions": {
            "field": {
                "type": "choice",
                "instructions": "How was it paid?",
                "criteria": {
                    "gift card": "Store gift card",
                    "card": "Credit or debit card",
                    "cash": None,
                },
            },
        },
    },
    {
        "name": "noul-custom",
        "id": "receipt",
        "state": "The customer attached a receipt and asks for a replacement today.",
        "questions": {
            "has_receipt": {
                "type": "noul",
                "instructions": "Does the receipt prove the purchase?",
                "criteria": {
                    "false": "The receipt is missing or unreadable.",
                    "true": "The receipt is present and readable.",
                },
            },
        },
    },
    {
        "name": "long-state",
        "id": "logs",
        "state": "Support log line with filler tokens to pad the state. " * 20,
        "maxStateTokens": 40,
        "questions": {
            "severity": {
                "type": "score",
                "instructions": "How severe is this?",
                "criteria": ["Cosmetic", "Annoying", "Severe", "Outage"],
            },
        },
    },
]


def jev_question(question_id, question):
    """The same question in the shape jev-web asks for (options + criteria)."""
    if question["type"] == "choice":
        criteria = dict(question["criteria"])
        return {"id": question_id, "type": "choice",
                "instructions": question["instructions"],
                "options": list(criteria.keys()), "criteria": criteria}
    if question["type"] == "noul":
        out = {"id": question_id, "type": "noul",
               "instructions": question["instructions"]}
        if question.get("criteria"):
            out["criteria"] = dict(question["criteria"])
        return out
    return {"id": question_id, "type": "score",
            "instructions": question["instructions"],
            "options": list(question["criteria"])}



def write_e2e_reference():
    """Per-question probabilities for one record, from the real weights."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    case = CASES[0].copy()
    backbone = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32)
    head = jsm.JointSchemaHead(**json.load(open(os.path.join(SRC, "joint_head_config.json"))))
    head.load_state_dict(load_file(os.path.join(SRC, "joint_head.safetensors")), strict=True)
    model = jsm.ClefModel(backbone, head).eval()
    tokenizer = AutoTokenizer.from_pretrained(SRC)

    record = {"id": case["id"], "state": case["state"], "questions": case["questions"]}
    encoded = jsm.encode_record(tokenizer, record, max_length=MAX_LENGTH)
    with torch.no_grad():
        logits = model(jsm.collate_records(
            [encoded], tokenizer.pad_token_id, torch.device("cpu")))[0]
    out = {
        "state": case["state"],
        "questions": case["questions"],
        "inputTokens": len(encoded.input_ids),
        "perQuestion": [
            {
                "questionId": q.question_id,
                "optionIds": [str(o) for o in q.option_ids],
                "probs": [float(p) for p in torch.softmax(logit.float(), -1).tolist()],
            }
            for q, logit in zip(encoded.questions, logits)
        ],
    }
    path = os.path.join(FIXTURES, "clef-e2e-reference.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", path)


def main():
    if "--e2e" in sys.argv:
        write_e2e_reference()
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SRC)
    head_config = json.load(open(os.path.join(SRC, "joint_head_config.json")))

    # Record every piece encode_record tokenizes, so the JS encode() stub can
    # replay the reference tokenization for any piece of any case.
    pieces: list[list] = []
    seen: set[str] = set()
    original = jsm._tokens

    def recorder(tokenizer, text):
        ids = original(tokenizer, text)
        if text not in seen:
            seen.add(text)
            pieces.append([text, [int(i) for i in ids]])
        return ids

    jsm._tokens = recorder

    cases = []
    for case in CASES:
        record = {"id": case["id"], "state": case["state"], "questions": case["questions"]}
        enc = jsm.encode_record(
            tok, record, max_length=MAX_LENGTH,
            max_state_tokens=case.get("maxStateTokens"),
        )
        questions = []
        for q in enc.questions:
            n = len(q.option_ids)
            weights = [n - i for i in range(n)]
            total = sum(weights)
            probs = [w / total for w in weights]
            answer = jsm.systemone_answer(
                case["questions"][q.question_id],
                dict(zip(q.option_ids, probs)),
            )
            questions.append({
                "questionId": q.question_id,
                "questionType": q.question_type,
                "questionSpan": [int(x) for x in q.question_span],
                "optionSpans": [[int(a), int(b)] for a, b in q.option_spans],
                "optionIds": [str(o) for o in q.option_ids],
                "probsIn": [float(p) for p in probs],
                "expected": answer,
            })
        cases.append({
            "name": case["name"], "id": case["id"],
            "state": case["state"],
            "maxStateTokens": case.get("maxStateTokens"),
            "questionDefs": [jev_question(k, q) for k, q in case["questions"].items()],
            "ids": [int(i) for i in enc.input_ids],
            "questions": questions,
        })

    fixture = {
        "systemPrompt": jsm.SYSTEM_PROMPT,
        "headConfig": head_config,
        "maxLength": MAX_LENGTH,
        "pieces": pieces,
        "cases": cases,
    }
    path = os.path.join(FIXTURES, "clef-protocol.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, ensure_ascii=False, indent=1)
    print("wrote", path)
    print("pieces:", len(pieces), "cases:", [c["name"] for c in cases])
    for c in cases:
        print(f"  {c['name']}: {len(c['ids'])} tokens, "
              f"{len(c['questions'])} questions")


if __name__ == "__main__":
    main()
