"""Emit JS test fixtures from the REFERENCE d3 runtime.

The JS protocol tests (test/decision3.test.mjs) need ground truth the browser
runtime must reproduce exactly: the rendered user prompt and full chat prompt
(real tokenizer + the checkpoint's chat template), the answer-code table from
decision_config.json, and the SystemOne answers from the reference
`product_answer()` (entropy confidence, score legend).

    D3_SRC=/path/to/d3-edge python export/d3/make_js_fixtures.py

D3_SRC is a checkpoint snapshot (d3_format.py, d3_runtime.py,
decision_config.json, tokenizer files); snapshot_download("vllm-sr/d3-edge")
gives one.

With `--e2e` it also loads the reference runtime and writes
test/fixtures/d3-e2e-reference.json (per-question probabilities for the
fixture cases) — the expected side of export/d3/verify_js_local.mjs.
"""
from __future__ import annotations

import json
import os
import sys

SRC = os.environ.get("D3_SRC", os.path.expanduser("~/workspace/d3edge-export/ckpt"))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES = os.path.join(REPO, "test", "fixtures")
os.makedirs(FIXTURES, exist_ok=True)

sys.path.insert(0, SRC)
import d3_format  # noqa: E402
import d3_runtime  # noqa: E402

MAX_LENGTH = 8192

STATE = ("Our checkout is down and customers can't pay. The deploy broke "
         "production and the on-call is asleep.")
LONG_STATE = "Support log line with filler tokens to pad the state. " * 20

CASES = [
    {
        "name": "mixed",
        "state": STATE,
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
            "reason": {
                "type": "noul",
                "instructions": "Is the message about a payment problem?",
                "criteria": {"false": "No payment signal.", "true": "A payment signal is present."},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["Routine", "Soon", "Today", "Immediately"],
            },
        },
    },
    {
        "name": "json-state",
        "state": {"order": 42, "product": "blender", "paid": ["card", 19.5]},
        "questions": {
            "payment": {
                "type": "choice",
                "instructions": "How was it paid?",
                "criteria": {"card": None, "cash": None, "gift card": "Store gift card"},
            },
        },
    },
    {
        "name": "long-state",
        "state": LONG_STATE,
        "questions": {
            "severity": {
                "type": "score",
                "instructions": "How severe is this?",
                "criteria": ["Cosmetic", "Annoying", "Severe", "Outage"],
            },
        },
    },
]


def jev_question(question):
    """The same question in the shape jev-web asks for (options + criteria)."""
    if question["type"] == "choice":
        criteria = dict(question["criteria"])
        return {"type": "choice", "instructions": question["instructions"],
                "options": list(criteria.keys()), "criteria": criteria}
    if question["type"] == "noul":
        out = {"type": "noul", "instructions": question["instructions"]}
        if question.get("criteria"):
            out["criteria"] = dict(question["criteria"])
        return out
    return {"type": "score", "instructions": question["instructions"],
            "options": list(question["criteria"])}


def to_reference(question):
    """A jev-shaped fixture question -> the reference runtime's shape."""
    if question["type"] == "score":
        return {"type": "score", "instructions": question["instructions"],
                "criteria": list(question["options"])}
    out = {"type": question["type"], "instructions": question["instructions"]}
    if question.get("criteria") is not None:
        out["criteria"] = question["criteria"]
    return out


def write_e2e_reference():
    """Per-question probabilities for the fixture cases, from the real runtime."""
    fixture_path = os.path.join(FIXTURES, "d3-protocol.json")
    if not os.path.exists(fixture_path):
        raise SystemExit("run make_js_fixtures.py without --e2e first")
    fixture = json.load(open(fixture_path, encoding="utf-8"))
    decision = d3_runtime.D3.from_pretrained(SRC, device="cpu", verify="none")
    out = []
    for case in fixture["cases"]:
        questions = {q["id"]: q["questionDef"] for q in case["questions"]}
        prepared = decision.prepare(
            case["state"], {k: to_reference(v) for k, v in questions.items()}, None, None)
        if prepared.errors:
            raise SystemExit(f"reference rejected {case['name']}: {prepared.errors}")
        probs, _ = decision.run(prepared)
        per_question = {}
        for key in prepared.runnable:
            question = prepared.questions[key]
            per_question[key] = {
                "keys": [str(o) for o in question.keys],
                "probs": [float(p) for p in probs[key]],
            }
        out.append({"name": case["name"], "state": case["state"],
                    "questions": questions, "perQuestion": per_question})
    path = os.path.join(FIXTURES, "d3-e2e-reference.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", path)


def main():
    if "--e2e" in sys.argv:
        write_e2e_reference()
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SRC)
    config = json.load(open(os.path.join(SRC, "decision_config.json")))
    codes = config["codes"]

    # The checkpoint's own code table (single-token codes right after the
    # assistant prefix); the fixture records whether it matches the config.
    reference_codes, reference_ids = d3_format.answer_codes(tok)
    config_match = (
        reference_codes == list(codes)
        and reference_ids == [int(i) for i in config["token_ids"]]
    )

    cases = []
    for case in CASES:
        questions = []
        for key, question in case["questions"].items():
            normalized = d3_runtime.normalize_question(question)
            rendered = normalized.rendered
            user_prompt = d3_format.user_prompt(case["state"], rendered, codes)
            prompt = d3_format.render(tok, case["state"], rendered, codes)
            ids = tok(prompt, add_special_tokens=False)["input_ids"]
            n = len(normalized.keys)
            weights = [n - i for i in range(n)]
            total = sum(weights)
            probs = [w / total for w in weights]
            expected = d3_runtime.product_answer(normalized, probs)
            questions.append({
                "id": key,
                "kind": normalized.kind,
                "rendered": rendered,
                "questionDef": jev_question(question),
                "userPrompt": user_prompt,
                "prompt": prompt,
                "ids": [int(i) for i in ids],
                "counts": n,
                "probsIn": [float(p) for p in probs],
                "expected": expected,
            })
        cases.append({"name": case["name"], "state": case["state"], "questions": questions})

    fixture = {
        "systemPrompt": d3_format.SYSTEM_PROMPT,
        "formatId": d3_format.FORMAT_ID,
        "maxOptions": d3_format.MAX_OPTIONS,
        "codes": codes,
        "configMatch": config_match,
        "config": {
            "temperature": config["temperature"],
            "max_length": config["max_length"],
            "readout_dtype": config["readout_dtype"],
            "attention_mode": config["attention_mode"],
            "pooling": config["pooling"],
        },
        "maxLength": MAX_LENGTH,
        "cases": cases,
    }
    path = os.path.join(FIXTURES, "d3-protocol.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, ensure_ascii=False, indent=1)
    print("wrote", path)
    print("code table matches the checkpoint:", config_match)
    for c in cases:
        print(f"  {c['name']}: " + ", ".join(
            f"{q['id']}({q['kind']}, {q['counts']} opts, {len(q['ids'])} tok)"
            for q in c["questions"]))


if __name__ == "__main__":
    main()
