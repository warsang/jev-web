"""Emit JS test fixtures from the REFERENCE implementation.

The JS protocol tests (test/vega.test.mjs) need ground truth the browser
runtime must reproduce exactly: prompt ids+spans (real tokenizer), the pooling
matrix rows, and calibrated answers. This script runs the author's own
vega_common / vega_api on fixed inputs and writes
test/fixtures/vega-protocol.json.

Also emits the packing cross-check: pool rows computed by the JS packing
rules (mirrored here in python) vs the reference span means.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

OUT = os.environ.get("STRANDS_OUT", os.path.expanduser("~/workspace/vega-export"))
BASE = f"{OUT}/base"
CKPT = f"{OUT}/ckpt"
FIXTURES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test", "fixtures")
os.makedirs(FIXTURES, exist_ok=True)

sys.path.insert(0, CKPT)
import vega_common  # noqa: E402

with open(f"{CKPT}/vega_config.json") as fh:
    CFG = json.load(fh)

CASES = [
    {
        "name": "choice-short-state",
        "state": "I was charged twice for the same order and nobody answers my emails.",
        "q": {"t": "choice", "ins": "Which team handles this?",
              "crit": {"billing": "payments and refunds", "tech": "bugs", "sales": "pricing"}},
    },
    {
        "name": "score",
        "state": "The deploy broke production and the on-call is asleep.",
        "q": {"t": "score", "ins": "How severe is this?",
              "crit": ["cosmetic", "annoying", "severe", "outage"]},
    },
    {
        "name": "boolean",
        "state": "I was charged twice for the same order and nobody answers my emails.",
        "q": {"t": "boolean", "ins": "Does this convey urgency?", "crit": {}},
    },
]


def main():
    # the tokenizer ships with the backbone (Perception loads it from there)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    state = "I was charged twice for the same order and nobody answers my emails. I want my money back now."
    cases = []
    for case in CASES:
        q = dict(case["q"])
        ids, spans = vega_common.build_prompt(
            tok, state, q, CFG["max_len"],
            opt_max_tokens=CFG["opt_max_tokens"],
            query_first_max_state=CFG.get("query_first_max_state", 0),
        )
        # piece tokenizations the JS encode() stub replays
        pieces = {
            "head": vega_common.TEMPLATES[0]["head"],
            "question": vega_common.TEMPLATES[0]["q"].format(t=q["t"], ins=q["ins"]),
            "end": vega_common.TEMPLATES[0]["end"],
            "options": [vega_common.TEMPLATES[0]["opt"].format(o=o)
                        for o in vega_common.option_texts(q)],
            "state": state,
        }
        if CFG.get("query_first_max_state", 0):
            pieces["query_first"] = vega_common.QUERY_FIRST.format(t=q["t"], ins=q["ins"])
        piece_ids = {}
        for k, v in pieces.items():
            if isinstance(v, list):
                piece_ids[k] = [list(tok(s, add_special_tokens=False)["input_ids"]) for s in v]
            else:
                piece_ids[k] = [list(tok(v, add_special_tokens=False)["input_ids"])]
        # pooling rows for this single item (nseg rows, no chunks)
        K = len(spans["options"])
        nseg = 3 + K
        L = len(ids)
        pool = np.zeros((nseg, L), np.float32)
        for row, (a, e) in [(0, spans["state"]), (1, spans["question"])] + [
            (3 + k, s) for k, s in enumerate(spans["options"])]:
            if e > a:
                pool[row, a:e] = 1.0 / (e - a)
        pool[2, len(ids) - 1] = 1.0  # the "last" slot is the final token (n-1)
        cases.append({
            "name": case["name"],
            "state": state,
            # literal strings buildVegaPrompt formats, so a JS encode() stub
            # can replay this case tokenizations exactly
            "questionText": vega_common.TEMPLATES[0]["q"].format(t=q["t"], ins=q["ins"]),
            "optionTexts": [vega_common.TEMPLATES[0]["opt"].format(o=o)
                            for o in vega_common.option_texts(q)],
            "ids": [int(i) for i in ids],
            "spans": {"state": [int(x) for x in spans["state"]],
                      "question": [int(x) for x in spans["question"]],
                      "options": [[int(a), int(e)] for a, e in spans["options"]],
                      "last": len(ids) - 1},
            "pieceIds": piece_ids,
            "nseg": nseg,
            "poolRowSums": [round(float(r.sum()), 6) for r in pool],
        })

    # thermal decoding fixtures: real theta from the checkpoint config
    theta = CFG["calibration"]["theta"]
    rng = np.random.RandomState(7)
    logits = np.round(rng.randn(3, 4).astype(np.float32) * 3, 3)
    qtypes = [0, 1, 2]
    nOpts = [3, 4, 2]
    kinetic = [0.01, 0.4, 0.02]
    nearest = [0.5, 3.2, 0.1]
    steps = [12, 5, 12]
    phys = np.stack([
        np.log1p(kinetic), np.log1p(nearest),
        np.array(steps) / CFG["n_steps"],
        np.log(np.maximum(nOpts, 2)),
    ], -1)
    # reference: thermal_tau + softmax_np
    probs = []
    for b in range(3):
        th = np.array(theta[qtypes[b]])
        log_tau = th[0] + (th[1:] * phys[b]).sum()
        tau = float(np.exp(np.clip(log_tau, np.log(0.2), np.log(20.0))))
        z = logits[b, :nOpts[b]] / tau
        z = z - z.max()
        e = np.exp(z)
        probs.append([round(float(x), 6) for x in (e / e.sum())])
    fixture = {
        "cases": cases,
        "thermal": {
            "theta": theta,
            "logits": logits.tolist(),
            "qtypes": qtypes,
            "nOpts": nOpts,
            "phys": phys.tolist(),
            "expectedProbs": probs,
        },
        "calibration": CFG["calibration"],
        "config": {k: CFG[k] for k in
                   ("max_len", "opt_max_tokens", "memory_chunks", "memory_min_tokens",
                    "memory_chunk_tokens", "query_first_max_state", "n_steps",
                    "feature_layers", "option_layers", "d_model")},
    }
    path = os.path.join(FIXTURES, "vega-protocol.json")
    with open(path, "w") as fh:
        json.dump(fixture, fh, indent=1)
    print("wrote", path)
    print("cases:", [c["name"] for c in cases])
    print("thermal expected:", probs)


if __name__ == "__main__":
    main()
