"""Tiny ORT validation as a standalone process (see 02_export.py notes).

Loads export/tiny_refs.npz (torch reference logits) and compares against
export/model_fp16.onnx in onnxruntime. Must run in its own process because
the torch model + ORT session together exceed host RAM.
"""
import os
import numpy as np
import onnxruntime as ort

OUT = os.path.expanduser("~/workspace/strands-export")
EXPORT = f"{OUT}/export"

z = np.load(f"{EXPORT}/tiny_refs.npz")
path = f"{EXPORT}/model_fp16.onnx"
print("creating ORT session (basic graph optimization)...", flush=True)
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
opts.enable_mem_pattern = False
opts.intra_op_num_threads = 2
opts.inter_op_num_threads = 1
sess = ort.InferenceSession(path, sess_options=opts,
                            providers=["CPUExecutionProvider"])
print("graph inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()],
      flush=True)
print("graph outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()],
      flush=True)
ok_all = True
worst = 0.0
for i in range(2):
    ids, mask, ans, opt = z[f"ids{i}"], z[f"mask{i}"], z[f"ans{i}"], z[f"opt{i}"]
    t = z[f"ref{i}"]
    o = sess.run(None, {"input_ids": ids, "attention_mask": mask,
                        "answer_pos": ans, "option_pos": opt})[0]
    d = float(np.abs(t - o).max())
    worst = max(worst, d)
    agree = bool((t.argmax(-1) == o.argmax(-1)).all())
    ok_all &= agree
    print(f"shape {ids.shape}/{opt.shape}: max|dtorch-donnx|={d:.2e} "
          f"argmax_agree={agree}", flush=True)
assert ok_all, "ARGMAX MISMATCH on tiny validation"
assert worst < 1e-2, f"tiny parity too loose: {worst}"
print(f"TINY VALIDATION PASSED (worst |d|={worst:.2e})", flush=True)
