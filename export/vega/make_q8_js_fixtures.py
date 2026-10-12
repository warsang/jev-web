"""Emit JS test fixtures for the SPLIT Vega export (nvkudva/vega-onnx-q8).

The split runtime does the perception in JS (pooling hidden states, M-RoPE
positions, past-state zeroing) and routes each question to an adapter's
calibration block, so the JS tests need ground truth for exactly those steps:

  1. prompt ids+spans built by the reference vega_common.build_prompt with the
     q8 export's own vega_web.json config;
  2. pooled features computed by the REFERENCE Perception._pooled_batch (run
     through a bare duck type, no backbone) on fixed hidden states -- the JS
     poolVegaHidden must reproduce it;
  3. per-row thermal calibration + adapter routing + conformal/abstain/unbound
     assembly, computed with vega_common.thermal_tau / softmax_np /
     route_adapters exactly as vega_api.Vega.decide does;
  4. the real graphs' input/output names and shapes, so the JS feed
     construction is checked against the shipped export.

Writes test/fixtures/vega-q8-protocol.json. The graph metadata block needs the
downloaded q8 graphs (VEGA_Q8_DIR, default ~/workspace/vega-q8) and is omitted
with a warning when they are absent.
"""
from __future__ import annotations

import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

OUT = os.environ.get("STRANDS_OUT", os.path.expanduser("~/workspace/vega-export"))
BASE = f"{OUT}/base"
CKPT = f"{OUT}/ckpt"
Q8 = os.environ.get("VEGA_Q8_DIR", os.path.expanduser("~/workspace/vega-q8"))
# this file lives at <repo>/export/vega/, the fixtures at <repo>/test/fixtures
FIXTURES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "test", "fixtures")
os.makedirs(FIXTURES, exist_ok=True)

sys.path.insert(0, CKPT)
import vega_common  # noqa: E402

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


class StubPerception:
    """The reference pooling methods, over given hidden states (no backbone)."""
    capture = [13, 19]
    feature_layers = [13, 19]
    option_layers = [19]
    device = torch.device("cpu")
    tok = SimpleNamespace(pad_token_id=0)
    _chunk_pool = vega_common.Perception._chunk_pool

    def __init__(self, h13, h19):
        self.h13 = torch.from_numpy(h13)
        self.h19 = torch.from_numpy(h19)

    def _run(self, ids, mask=None, cache=None):
        L = ids.shape[1]
        return {13: self.h13[:, :L], 19: self.h19[:, :L]}


def main():
    from transformers import AutoTokenizer

    web = json.load(open(f"{Q8}/vega_web.json"))
    cfg = dict(json.load(open(f"{CKPT}/vega_config.json")))
    cfg.update({k: web[k] for k in ("max_len", "opt_max_tokens", "query_first_max_state",
                                    "n_steps", "memory_chunks", "memory_min_tokens",
                                    "memory_chunk_tokens", "feature_layers", "option_layers")})
    tok = AutoTokenizer.from_pretrained(BASE)

    # --- 1. prompt fixtures -------------------------------------------------
    cases = []
    for case in CASES:
        q = dict(case["q"])
        ids, spans = vega_common.build_prompt(
            tok, case["state"], q, cfg["max_len"],
            opt_max_tokens=cfg["opt_max_tokens"],
            query_first_max_state=cfg.get("query_first_max_state", 0))
        pieces = {
            "head": vega_common.TEMPLATES[0]["head"],
            "question": vega_common.TEMPLATES[0]["q"].format(t=q["t"], ins=q["ins"]),
            "end": vega_common.TEMPLATES[0]["end"],
            "options": [vega_common.TEMPLATES[0]["opt"].format(o=o)
                        for o in vega_common.option_texts(q)],
            "state": case["state"],
        }
        if cfg.get("query_first_max_state", 0):
            pieces["query_first"] = vega_common.QUERY_FIRST.format(t=q["t"], ins=q["ins"])
        piece_ids = {}
        for k, v in pieces.items():
            if isinstance(v, list):
                piece_ids[k] = [list(tok(s, add_special_tokens=False)["input_ids"]) for s in v]
            else:
                piece_ids[k] = [list(tok(v, add_special_tokens=False)["input_ids"])]
        cases.append({
            "name": case["name"],
            "state": case["state"],
            "questionText": vega_common.TEMPLATES[0]["q"].format(t=q["t"], ins=q["ins"]),
            "optionTexts": [vega_common.TEMPLATES[0]["opt"].format(o=o)
                            for o in vega_common.option_texts(q)],
            "ids": [int(i) for i in ids],
            "spans": {"state": [int(x) for x in spans["state"]],
                      "question": [int(x) for x in spans["question"]],
                      "options": [[int(a), int(e)] for a, e in spans["options"]],
                      "last": len(ids) - 1},
            "pieceIds": piece_ids,
        })

    # --- 2. pooling fixture: reference _pooled_batch on fixed hidden states --
    # like a real prompt: the final token (the end template) sits AFTER the
    # option spans, and _pooled_batch force-assigns it to its own bucket
    D, L = 3, 446
    spans = {"state": (16, 400), "question": (400, 430),
             "options": [(430, 438), (438, 444)]}
    rng = np.random.RandomState(11)
    h13 = (rng.randn(1, L, D) * 0.4).astype(np.float32)
    h19 = (rng.randn(1, L, D) * 0.7).astype(np.float32)
    stub = StubPerception(h13, h19)
    item, opt, (chunks, cmask) = vega_common.Perception._pooled_batch(
        stub, [{"ids": np.zeros(L, np.int32), "spans": spans}],
        int(cfg["memory_chunks"]), int(cfg["memory_min_tokens"]), int(cfg["memory_chunk_tokens"]))
    pool = {
        "d": D,
        "spans": {k: ([int(v[0]), int(v[1])] if isinstance(v, tuple)
                      else [[int(a), int(b)] for a, b in v])
                  for k, v in spans.items()},
        "h13": h13[0].tolist(),
        "h19": h19[0].tolist(),
        "expected": {
            "feats": [[[round(float(x), 6) for x in row] for row in layer] for layer in item[0].tolist()],
            "opts": [[round(float(x), 6) for x in row] for row in opt[:, 0, :].tolist()],
            "chunks": [[round(float(x), 6) for x in row] for row in chunks[0].tolist()],
            "cmask": [int(x) for x in cmask[0].tolist()],
        },
    }

    # --- 3. routing + calibration fixture -----------------------------------
    # rows chosen so all four routes appear: adapter 0, adapter 0 (a type
    # without a conformal block -> null set), adapter 1, base.
    rows = [
        {"t": "choice", "qtype": 0, "logits": [1.2, -0.3, 0.4],
         "kinetic": 0.03, "nearest": 0.42, "steps": 12, "gates": [0.8, 0.1]},
        {"t": "boolean", "qtype": 2, "logits": [0.7, -0.2],
         "kinetic": 0.11, "nearest": 0.9, "steps": 7, "gates": [0.62, 0.4]},
        {"t": "score", "qtype": 1, "logits": [-0.5, 0.9, 1.4, -1.1],
         "kinetic": 0.2, "nearest": 1.5, "steps": 12, "gates": [0.2, 0.93]},
        {"t": "choice", "qtype": 0, "logits": [0.1, 0.05],
         "kinetic": 0.4, "nearest": 2.2, "steps": 12, "gates": [0.2, 0.1]},
    ]
    k_list = [len(r["logits"]) for r in rows]
    qtypes = np.array([r["qtype"] for r in rows])
    gates = np.array([r["gates"] for r in rows], np.float32)
    kinetic = np.array([r["kinetic"] for r in rows])
    nearest = np.array([r["nearest"] for r in rows])
    steps = np.array([r["steps"] for r in rows])
    phys = np.stack([np.log1p(kinetic), np.log1p(nearest), steps / cfg["n_steps"],
                     np.log(np.maximum(k_list, 2))], -1)
    thresh = web["adapters"][0]["gate_threshold"]
    route = vega_common.route_adapters(gates, thresh)
    risk = web["calibration"].get("default_risk", "0.1")
    alpha = "0.1"
    expected = []
    for r, rt in enumerate(route):
        cal = web["calibration"] if rt < 0 else web["adapters"][int(rt)]["calibration"]
        theta = np.array(cal["theta"], np.float64)
        lg = np.array(rows[r]["logits"], np.float32)[None]
        tau = vega_common.thermal_tau(theta, np.array([qtypes[r]]), phys[r:r + 1])
        probs = vega_common.softmax_np(lg, tau)[0]
        k = k_list[r]
        conf = float(probs.max())
        best = int(probs.argmax())
        th = cal.get("conformal", {}).get(rows[r]["t"], {}).get(alpha)
        pset = [i for i in range(k)
                if th is not None and (probs[i] >= max(1.0 - th, 1e-12) or i == best)]
        expected.append({
            "route": int(rt),
            "adapter": web["adapters"][int(rt)]["name"] if rt >= 0 else None,
            "gate": float(gates[r].max()),
            "probabilities": [round(float(x), 6) for x in probs],
            "confidence": round(conf, 6),
            "predictionSet": pset if th is not None else None,
            "abstain": bool(conf < cal.get("abstain_threshold", {}).get(risk, 0.0)),
            "unbound": bool(float(nearest[r]) > cal.get("unbound_threshold", math.inf)),
            "settleSteps": int(steps[r]),
            "energy": round(float(-lg[0, :k].max()), 6),
        })
    routing = {
        "alpha": alpha,
        "risk": risk,
        "gateThreshold": thresh,
        "rows": [{k: r[k] for k in ("t", "qtype", "logits", "kinetic", "nearest", "steps", "gates")}
                 for r in rows],
        "expected": expected,
    }

    # --- 4. the real graphs' contracts --------------------------------------
    graphs = {}
    try:
        import onnx
        for name, file in (("embed", "embed_q8.onnx"), ("decoder", "decoder_q8.onnx"),
                           ("engine", "engine.onnx")):
            g = onnx.load(f"{Q8}/{file}", load_external_data=False).graph
            graphs[name] = {
                "inputs": [{"name": i.name,
                            "shape": [d.dim_param or int(d.dim_value) for d in i.type.tensor_type.shape.dim]}
                           for i in g.input],
                "outputs": [o.name for o in g.output],
            }
    except Exception as exc:  # noqa: BLE001 - metadata block is optional
        print(f"WARNING: graph metadata not written ({exc}); set VEGA_Q8_DIR", file=sys.stderr)

    fixture = {
        "config": web,
        "prompt": {"config": {k: cfg[k] for k in
                              ("max_len", "opt_max_tokens", "query_first_max_state", "n_steps",
                               "memory_chunks", "memory_min_tokens", "memory_chunk_tokens",
                               "feature_layers", "option_layers")},
                   "cases": cases},
        "pool": pool,
        "routing": routing,
        "graphs": graphs or None,
    }
    path = os.path.join(FIXTURES, "vega-q8-protocol.json")
    with open(path, "w") as fh:
        json.dump(fixture, fh, indent=1)
    print("wrote", path, os.path.getsize(path), "bytes")
    print("routes:", [e["route"] for e in expected],
          "adapters:", [e["adapter"] for e in expected])


if __name__ == "__main__":
    main()
