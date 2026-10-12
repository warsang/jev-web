"""Verify the SPLIT Vega export (nvkudva/vega-onnx-q8) against the reference.

Stage A (always) — the checks the JS runtime's protocol depends on:
  1. the shipped graphs accept exactly the feeds the split runtime builds
     (names/shapes, zero-filled past_* inputs, text M-RoPE position_ids);
  2. the JS pooling rules (mirrored here in numpy) reproduce the reference
     Perception._pooled_batch on the SAME q8 hidden states;
  3. the q8 fp32 engine graph reproduces the PyTorch VegaEngine with both
     adapters attached (logits / gates / steps);
  4. adapter routing + thermal calibration assemble the reference answer.

Stage B (--deep) — the fp16 backbone reference Perception on the same prompt,
so the export's end-to-end accuracy (argmax agreement / probability drift) is
measured rather than trusted.

Graphs are downloaded to VEGA_Q8_DIR (default ~/workspace/vega-q8) on first
run. The reference side needs the export workspace (STRANDS_OUT, default
~/workspace/vega-export) with base/ and ckpt/ populated.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import urllib.request
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import torch

OUT = os.environ.get("STRANDS_OUT", os.path.expanduser("~/workspace/vega-export"))
BASE = f"{OUT}/base"
CKPT = f"{OUT}/ckpt"
sys.path.insert(0, CKPT)

import vega_common  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

Q8 = os.environ.get("VEGA_Q8_DIR", os.path.expanduser("~/workspace/vega-q8"))
HF = "https://huggingface.co/nvkudva/vega-onnx-q8/resolve/main/v2/"
WANT = ["embed_q8.onnx", "embed_q8.onnx.data", "decoder_q8.onnx", "decoder_q8.onnx.data",
        "engine.onnx", "vega_web.json"]

CASES = [
    {"name": "choice", "state": "I was charged twice for the same order and nobody answers my emails.",
     "q": {"t": "choice", "ins": "Which team handles this?",
           "crit": {"billing": "payments and refunds", "tech": "bugs", "sales": "pricing"}}},
    {"name": "boolean", "state": "I was charged twice for the same order and nobody answers my emails.",
     "q": {"t": "boolean", "ins": "Does this convey urgency?", "crit": {}}},
    {"name": "score", "state": "The deploy broke production and the on-call is asleep.",
     "q": {"t": "score", "ins": "How severe is this?",
           "crit": ["cosmetic", "annoying", "severe", "outage"]}},
]


def ensure_graphs():
    os.makedirs(Q8, exist_ok=True)
    for f in WANT:
        dest = os.path.join(Q8, f)
        if not os.path.exists(dest):
            print(f"downloading {f} ...", flush=True)
            urllib.request.urlretrieve(HF + f, dest)
    return {f: os.path.join(Q8, f) for f in WANT}


def empty_past(sess):
    """The export's emptyPast(): zero-filled past_* inputs, shapes from the graph."""
    feeds = {}
    for m in sess.get_inputs():
        if not m.name.startswith("past_"):
            continue
        shape = [d if isinstance(d, int) else (1 if d == "batch_size" else 0) for d in m.shape]
        feeds[m.name] = np.zeros(shape, np.float32)
    return feeds


def text_positions(L):
    """For text, position_ids is 0..L-1 on all three M-RoPE rows."""
    return np.tile(np.arange(L, dtype=np.int64), (3, 1))[:, None, :]  # [3, 1, L]


class StubPerception:
    """Bare duck type so the reference Perception._pooled_batch can run on given hidden states."""
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


def reference_pooled(h13, h19, ids, spans, cfg):
    stub = StubPerception(h13, h19)
    item, opt, (chunks, cmask) = vega_common.Perception._pooled_batch(
        stub, [{"ids": np.asarray(ids, np.int32), "spans": spans}],
        int(cfg["memory_chunks"]), int(cfg["memory_min_tokens"]), int(cfg["memory_chunk_tokens"]))
    return item[0].numpy(), opt[:, 0, :].numpy(), chunks[0].numpy(), cmask[0].numpy()


def js_pool(h13, h19, spans, C, min_tok, chunk_tok):
    """The pooling the split runtime performs in JS (mirror of the export's pool())."""
    B, L, D = h13.shape

    def mean(h, a, e, out, off):
        if e <= a:
            return
        out[off:off + D] = h[a:e].mean(0)

    feats = np.zeros((2, 3, D), np.float32)
    for l, h in enumerate([h13[0], h19[0]]):
        a, e = spans["state"]
        mean(h, a, e, feats[l, 0], 0)
        a, e = spans["question"]
        mean(h, a, e, feats[l, 1], 0)
        feats[l, 2] = h[L - 1]
    K = len(spans["options"])
    opts = np.zeros((K, D), np.float32)
    for k, (a, e) in enumerate(spans["options"]):
        mean(h19[0], a, e, opts[k], 0)
    chunks = np.zeros((C, D), np.float32)
    cmask = np.zeros(C, bool)
    s0, s1 = spans["state"]
    m = s1 - s0
    if m >= max(min_tok, 2):
        k = max(2, min(C, m // max(1, chunk_tok)))
        # token t belongs to chunk floor(t*k/m); chunk c covers [ceil(c*m/k), ceil((c+1)*m/k))
        for c in range(k):
            a = (c * m + k - 1) // k
            e = m if c == k - 1 else ((c + 1) * m + k - 1) // k
            mean(h19[0], s0 + a, s0 + e, chunks[c], 0)
        cmask[:k] = True
    return feats, opts, chunks, cmask


def build_reference_engine(ckpt_cfg):
    engine = vega_common.VegaEngine(ckpt_cfg["d_model"], len(ckpt_cfg["feature_layers"]),
                                    len(ckpt_cfg["option_layers"]), ckpt_cfg)
    engine.load_state_dict(load_file(f"{CKPT}/engine.safetensors"))
    engine.eval()
    for js in sorted(glob.glob(f"{CKPT}/adapters/*.json")):
        meta = json.load(open(js))
        vega_common.load_adapter(engine, load_file(js[:-len(".json")] + ".safetensors"),
                                 meta["rank"], meta["alpha"])
    engine.eval()
    engine.adapter_mode = "auto" if len(engine.gate_nets) else "off"
    return engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true", help="also run the fp16 reference perception")
    args = ap.parse_args()

    paths = ensure_graphs()
    from transformers import AutoTokenizer

    web = json.load(open(paths["vega_web.json"]))
    ckpt_cfg = json.load(open(f"{CKPT}/vega_config.json"))
    cfg = dict(ckpt_cfg)
    cfg.update({k: web[k] for k in ("max_len", "opt_max_tokens", "query_first_max_state", "n_steps",
                                    "memory_chunks", "memory_min_tokens", "memory_chunk_tokens",
                                    "feature_layers", "option_layers")})
    C = int(cfg["memory_chunks"])
    Lcap = len(cfg["feature_layers"])
    D = int(ckpt_cfg["d_model"])

    tok = AutoTokenizer.from_pretrained(BASE)
    embed = ort.InferenceSession(paths["embed_q8.onnx"], providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(paths["decoder_q8.onnx"], providers=["CPUExecutionProvider"])
    engine_onnx = ort.InferenceSession(paths["engine.onnx"], providers=["CPUExecutionProvider"])
    engine_ref = build_reference_engine(ckpt_cfg)

    # the two adapter calibrations in vega_web.json must equal the checkpoint's
    for ad in web["adapters"]:
        meta = json.load(open(f"{CKPT}/adapters/{ad['name']}.json"))
        assert meta["gate_threshold"] == ad["gate_threshold"], ad["name"]
        for key in ("theta", "conformal", "abstain_threshold", "unbound_threshold"):
            a, b = meta["calibration"].get(key), ad["calibration"].get(key)
            if isinstance(a, (int, float)):
                assert abs(a - b) < 1e-12, (ad["name"], key)
            else:
                assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), (ad["name"], key)
    print("[ok] vega_web.json adapter blocks match ckpt/adapters/*.json", flush=True)

    # graph contract: exactly the names/shapes the split runtime builds
    expects = {
        embed: ["input_ids"],
        decoder: ["inputs_embeds", "attention_mask", "position_ids"],
        engine_onnx: ["feats", "opts", "opt_mask", "qtype", "chunks", "chunk_mask"],
    }
    for sess, names in expects.items():
        have = {m.name for m in sess.get_inputs()}
        for name in names:
            assert name in have, (sess.get_modelmeta().custom_metadata_map.get("name"), name)
    past = [m for m in decoder.get_inputs() if m.name.startswith("past_")]
    print(f"[ok] graph inputs carry the expected names; decoder declares {len(past)} past_* inputs", flush=True)
    assert len(past) == 40

    worst = {"feats": 0.0, "chunks": 0.0, "logits": 0.0, "probs": 0.0}
    deep_perc = None
    if args.deep:
        deep_perc = vega_common.Perception(BASE, torch.device("cpu"),
                                           cfg["feature_layers"], cfg["option_layers"])
    for case in CASES:
        ids, spans = vega_common.build_prompt(
            tok, case["state"], case["q"], cfg["max_len"],
            opt_max_tokens=cfg["opt_max_tokens"],
            query_first_max_state=cfg["query_first_max_state"])
        L = len(ids)

        emb = embed.run(None, {"input_ids": np.array([ids], np.int64)})[0]
        assert emb.shape == (1, L, D), emb.shape
        h13, h19 = decoder.run(None, {
            **empty_past(decoder),
            "inputs_embeds": emb.astype(np.float32),
            "attention_mask": np.ones((1, L), np.int64),
            "position_ids": text_positions(L),
        })
        assert h13.shape == (1, L, D) and np.isfinite(h13).all() and np.isfinite(h19).all()

        item_ref, opt_ref, ch_ref, cm_ref = reference_pooled(h13, h19, ids, spans, cfg)
        feats_js, opts_js, chunks_js, cmask_js = js_pool(h13, h19, spans, C,
                                                         cfg["memory_min_tokens"], cfg["memory_chunk_tokens"])
        d_feats = float(np.abs(item_ref - feats_js).max())
        d_opts = float(np.abs(opt_ref - opts_js).max())
        d_chunks = float(np.abs(ch_ref - chunks_js).max())
        assert cm_ref.tolist() == cmask_js.tolist(), "chunk mask differs"
        worst["feats"] = max(worst["feats"], d_feats, d_opts)
        worst["chunks"] = max(worst["chunks"], d_chunks)

        K = len(spans["options"])
        qtype = np.array([vega_common.QTYPES[case["q"]["t"]]], np.int64)
        feeds = {
            "feats": feats_js[None].astype(np.float32),
            "opts": opts_js[None, :, None, :].astype(np.float32),
            "opt_mask": np.ones((1, K), bool),
            "qtype": qtype,
            "chunks": chunks_js[None].astype(np.float32),
            "chunk_mask": cmask_js[None],
        }
        logits_g, kinetic_g, nearest_g, steps_g, gates_g = engine_onnx.run(None, feeds)
        assert logits_g.shape == (1, K), logits_g.shape
        assert np.isfinite(logits_g).all() and np.isfinite(gates_g).all()

        with torch.no_grad():
            out_t = engine_ref(torch.from_numpy(feats_js[None]),
                               torch.from_numpy(opts_js[None, :, None, :]),
                               torch.ones(1, K, dtype=torch.bool), torch.from_numpy(qtype),
                               halt=True, return_traj=False, prestandardized=False,
                               chunks=torch.from_numpy(chunks_js[None]),
                               chunk_mask=torch.from_numpy(cmask_js[None]))
        logits_t = out_t["logits"].float().numpy()
        gates_t = out_t["gates"].float().numpy()
        d_logits = float(np.abs(logits_g - logits_t).max())
        d_gates = float(np.abs(gates_g - gates_t).max())
        worst["logits"] = max(worst["logits"], d_logits)
        assert d_gates < 1e-3, d_gates
        assert int(steps_g[0]) == int(out_t["steps_used"][0]), (steps_g, out_t["steps_used"])

        # reference calibration + routing (vega_api.Vega.decide)
        phys = np.array([[np.log1p(float(kinetic_g[0])), np.log1p(float(nearest_g[0])),
                          float(steps_g[0]) / max(1, cfg["n_steps"]), np.log(max(2, K))]])
        route = int(vega_common.route_adapters(gates_g, web["adapters"][0]["gate_threshold"])[0])
        cal = web["calibration"] if route < 0 else web["adapters"][route]["calibration"]
        probs_g = vega_common.softmax_np(logits_g, vega_common.thermal_tau(
            np.array(cal["theta"]), qtype, phys))[0]
        probs_t = vega_common.softmax_np(logits_t, vega_common.thermal_tau(
            np.array(cal["theta"]), qtype, phys))[0]
        d_probs = float(np.abs(probs_g - probs_t).max())
        worst["probs"] = max(worst["probs"], d_probs)
        assert abs(float(probs_g.sum()) - 1) < 1e-6

        print(f"[{case['name']:7s}] L={L:3d} K={K} gate={float(gates_g.max()):.3f} route={route} "
              f"feats={d_feats:.2e} opts={d_opts:.2e} chunks={d_chunks:.2e} "
              f"engine={d_logits:.2e} probs={d_probs:.2e} argmax={int(probs_g.argmax())} "
              f"p={np.round(probs_g, 3).tolist()}", flush=True)

        if deep_perc is not None:
            fr, orf = deep_perc.pooled([{"ids": np.asarray(ids, np.int32), "spans": spans}])
            with torch.no_grad():
                out_r = engine_ref(fr, orf.unsqueeze(0), torch.ones(1, K, dtype=torch.bool),
                                   torch.from_numpy(qtype), halt=True, return_traj=False,
                                   prestandardized=False, chunks=None, chunk_mask=None)
            phys_r = np.array([[np.log1p(float(out_r["kinetic"][0])), np.log1p(float(out_r["nearest"][0])),
                                float(out_r["steps_used"][0]) / max(1, cfg["n_steps"]), np.log(max(2, K))]])
            route_r = int(vega_common.route_adapters(out_r["gates"].float().numpy(),
                                                     web["adapters"][0]["gate_threshold"])[0])
            cal_r = web["calibration"] if route_r < 0 else web["adapters"][route_r]["calibration"]
            probs_r = vega_common.softmax_np(out_r["logits"].float().numpy(), vega_common.thermal_tau(
                np.array(cal_r["theta"]), qtype, phys_r))[0]
            d_feat = float(np.abs(fr.numpy()[0] - feats_js).max())
            d_opt = float(np.abs(orf.numpy()[:, 0, :] - opts_js).max())
            d_p = float(np.abs(probs_r - probs_g).max())
            kl = float(np.sum(probs_r * np.log((probs_r + 1e-12) / (probs_g + 1e-12))))
            print(f"        deep: fp16-ref feats={d_feat:.2e} opts={d_opt:.2e} probs={d_p:.2e} "
                  f"argmax {int(probs_r.argmax())}/{int(probs_g.argmax())} kl={kl:.2e}", flush=True)

    print(f"\nRESULT worst pooled-feats={worst['feats']:.2e} chunks={worst['chunks']:.2e} "
          f"engine-vs-torch={worst['logits']:.2e} probs={worst['probs']:.2e}", flush=True)
    ok = (worst["feats"] < 1e-5 and worst["chunks"] < 1e-5
          and worst["logits"] < 1e-3 and worst["probs"] < 1e-4)
    print("Q8 EXPORT CHECKS PASSED" if ok else "Q8 EXPORT CHECKS FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
