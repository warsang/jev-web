"""Stage 3: parity — torch fp16 wrapper vs exported ONNX on real prompts.

Prompts are built with the REFERENCE Python code (prompting.render_question +
schema), tokenized with the checkpoint tokenizer, option positions found with
the reference offset-mapping logic (infer._option_token_index).

Criteria: argmax agreement 100%, max |dlogit| < 0.05 across all prompts.
"""
import json, os, sys
import numpy as np
import torch

OUT = os.path.expanduser("~/workspace/strands-export")
sys.path.insert(0, os.path.expanduser("~/workspace/strands-decider/src"))
sys.path.insert(0, OUT)
from importlib import import_module
exp = import_module("02_export")

from strands_decider.prompting import render_question, render_state
from strands_decider.schema import ChoiceQuestion, ScoreQuestion, NoulQuestion

# (state, question) pairs, drawn from jev-web/demo/presets.mjs
STATES = {
    "easy": "I was charged twice for the same order and nobody answers my emails. I want my money back now.",
    "negation": "This is not about a refund. My card was stolen in Madrid and the bank refuses to help. I never received the money back I asked about.",
    "contrast": "Your support agent was wonderful, truly the best, but the product itself is garbage and I demand a full refund immediately.",
    "mixed": "I love the new design and the app is fast, but I was billed for a subscription I cancelled months ago and I am furious about it.",
    "edge": "The weather in Lisbon is lovely this time of year and the custard tarts are excellent.",
}
QUESTIONS = {
    "choice": lambda: ChoiceQuestion(
        instructions="Which product area is the message about?",
        criteria={o: o for o in ["fees & charges", "refund & dispute", "card", "other"]}),
    "noul": lambda: NoulQuestion(
        instructions="The customer is asking for a refund."),
    "score": lambda: ScoreQuestion(
        instructions="How negative is the message?",
        criteria=["very negative", "negative", "neutral", "positive", "very positive"]),
}


def build_inputs(tokenizer):
    """Return list of (name, input_ids[1,L], attention_mask[1,L], opt_idx[1,K], slot_labels)."""
    out = []
    for sname, state in STATES.items():
        for qname, qfn in QUESTIONS.items():
            rq = render_question(qfn())
            # Exact reference prompt: render_state(state) + rq.text
            text = render_state(state.strip()) + rq.text
            enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            ids = enc["input_ids"]
            offs = enc["offset_mapping"]
            opt_idx = []
            for s, e in rq.option_spans:
                last = -1
                for j, (lo, hi) in enumerate(offs):
                    if hi <= lo:
                        continue
                    if lo >= s and hi <= e:
                        last = j
                assert last >= 0, f"option span ({s},{e}) lost in {sname}/{qname}"
                opt_idx.append(last)
            out.append((f"{sname}/{qname}",
                        np.array([ids], np.int64), np.ones((1, len(ids)), np.int64),
                        np.array([opt_idx], np.int64), rq.slot_labels))
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "torch"
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(f"{OUT}/ckpt",
                                                     trust_remote_code=False)
    items = build_inputs(tok)
    print(f"{len(items)} parity prompts", flush=True)

    if which == "torch":
        # Compute torch fp16 reference logits and dump to disk. Run alone:
        # the torch model (3.8GB) must not share the process with ORT.
        model = exp.load_export_model()
        refs = {}
        for j, (name, ids, mask, opt, labels) in enumerate(items):
            with torch.no_grad():
                t = model(torch.from_numpy(ids), torch.from_numpy(mask),
                          torch.from_numpy(opt)).numpy()[0]
            refs[f"ref{j}"] = t
            refs[f"ids{j}"] = ids
            refs[f"mask{j}"] = mask
            refs[f"opt{j}"] = opt
            refs[f"name{j}"] = np.array(name)
            refs[f"labels{j}"] = np.array(labels)
            print(f"  torch {j:2d}/{len(items)} {name:16s} "
                  f"logits={t.round(3).tolist()}", flush=True)
        np.savez(f"{OUT}/parity_refs.npz", **refs)
        print("torch refs saved", flush=True)

    elif which == "onnx":
        import onnxruntime as ort
        z = np.load(f"{OUT}/parity_refs.npz", allow_pickle=True)
        n = len(items)
        sess = ort.InferenceSession(f"{OUT}/export/model_fp16.onnx",
                                    providers=["CPUExecutionProvider"])
        worst = 0.0
        agree_all = True
        for j in range(n):
            name = str(z[f"name{j}"])
            ids, mask, opt = z[f"ids{j}"], z[f"mask{j}"], z[f"opt{j}"]
            t = z[f"ref{j}"]
            labels = list(z[f"labels{j}"])
            o = sess.run(None, {"input_ids": ids, "attention_mask": mask,
                                "opt_idx": opt})[0][0]
            d = float(np.abs(t - o).max())
            worst = max(worst, d)
            at, ao = int(t.argmax()), int(o.argmax())
            ok = at == ao
            agree_all &= ok
            mark = "OK " if ok else "ARGMAX-MISMATCH"
            print(f"[{mark}] {name:16s} K={len(opt[0])} L={len(ids[0]):4d} "
                  f"max|d|={d:.2e} torch={labels[at]!r} onnx={labels[ao]!r}",
                  flush=True)
        print(f"\nPARITY RESULT: worst max|dlogit|={worst:.2e} "
              f"argmax_agree={agree_all}", flush=True)
        with open(f"{OUT}/parity_fp16.json", "w") as fh:
            json.dump({"n_prompts": n, "worst_abs_logit_diff": worst,
                       "argmax_agreement": bool(agree_all)}, fh, indent=2)
        assert agree_all, "argmax disagreement torch vs ONNX"
        assert worst < 0.05, f"logit drift too large: {worst}"
        print("PARITY PASSED", flush=True)

    else:
        raise SystemExit(f"unknown mode {which!r} (use torch|onnx)")


if __name__ == "__main__":
    main()
