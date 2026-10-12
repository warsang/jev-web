"""Stage 1: export the fused clef text graph (Qwen3 torso + joint schema head).

    input_ids      int64 [1, L]
    attention_mask int64 [1, L]
    pool_global    fp32  [1, 1, L]   one-hot at the final token
    pool_question  fp32  [1, Q, L]   question-span means
    pool_option    fp32  [1, S, L]   option-span means, S = Q*K slots
                                     (question-major, masked slots zero)
    type_ids       int64 [Q]         noul 0 / choice 1 / score 2
    slot_question  int64 [S]         which question each option slot belongs to
    opt_mask       bool  [S]         valid option slots
    ->
    logits         fp32  [1, S]      one logit per option slot

The graph is the reference `ClefModel` text path (`joint_schema_model.py`,
Cloudflare, Apache-2.0) with span means arriving as pooling matrices, so the
torso never leaves the graph and no hidden states cross into JS. Everything
else — the two evidence routing layers over the record's options attending to
the whole sequence, the per-question routing softmax, the four decoder field
layers, and the prior + gated cosine/residual scoring — mirrors
JointSchemaHead exactly; 02_parity.py is the gate.

Exports fp32 (`clef_text_fp32.onnx`); 03_repo.py converts to the shipped fp16
graph. Needs ~6-8 GB RAM for the fp32 load + program:

    CLEF_OUT=~/workspace/clef-export python export/clef/01_export.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

OUT = os.environ.get("CLEF_OUT", os.path.expanduser("~/workspace/clef-export"))
CKPT = os.environ.get("CLEF_CKPT", f"{OUT}/ckpt")
EXPORT = f"{OUT}/export"
os.makedirs(EXPORT, exist_ok=True)
sys.path.insert(0, CKPT)

import joint_schema_model as jsm  # noqa: E402


class ClefTextExport(torch.nn.Module):
    """Frozen Qwen3 torso + JointSchemaHead with pooling matrices, batch 1."""

    def __init__(self, clef):
        super().__init__()
        self.language_model = clef.language_model
        self.head = clef.head

    @staticmethod
    def _evidence(layer, queries, memory, key_pad):
        """EvidenceRoutingLayer.forward with a memory padding mask."""
        routed = layer.attention(
            layer.query_norm(queries), layer.memory_norm(memory),
            layer.memory_norm(memory), key_padding_mask=key_pad, need_weights=False,
        )
        if isinstance(routed, tuple):
            routed = routed[0]
        queries = queries + layer.attention_dropout(routed)
        return queries + layer.feedforward(layer.feedforward_norm(queries))

    def forward(self, input_ids, attention_mask, pool_global, pool_question,
                pool_option, type_ids, slot_question, opt_mask):
        head = self.head
        model = self.language_model
        # fp32 IO contract for both graphs (the fp16 export casts here):
        # JS feeds fp32 pooling matrices / reads fp32 logits.
        dtype = head.hidden_norm.weight.dtype
        pool_global = pool_global.to(dtype)
        pool_question = pool_question.to(dtype)
        pool_option = pool_option.to(dtype)
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        text_model = base.model if hasattr(base, "model") else base
        if hasattr(text_model, "language_model"):
            text_model = text_model.language_model
        outputs = text_model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False, return_dict=True,
        )
        hidden = outputs.last_hidden_state

        normalized = head.hidden_norm(hidden)
        pooled_global = torch.einsum("bsl,bld->bsd", pool_global, normalized)
        question_vectors = torch.einsum("bsl,bld->bsd", pool_question, normalized)
        option_context = torch.einsum("bsl,bld->bsd", pool_option, normalized)
        embedding = base.get_output_embeddings().weight
        lexical = torch.einsum("bsl,bld->bsd", pool_option, embedding[input_ids])

        memory = head.memory_projection(normalized)
        global_vector = pooled_global[:, 0]
        key_pad = (attention_mask == 0)  # [1, L]; True marks a memory pad

        queries = (
            head.option_context_projection(option_context)
            + head.option_lexical_projection(lexical)
            + head.option_question_projection(question_vectors)[:, slot_question, :]
        )
        routed = queries
        for layer in head.evidence_layers:
            routed = self._evidence(layer, routed, memory, key_pad)

        # Per-question routing softmax over the question's valid option slots.
        base_fields = head.question_projection(question_vectors)[0]  # [Q, width]
        by_slot = base_fields[slot_question]  # [S, width]
        scores = (routed[0] * by_slot).sum(-1) / math.sqrt(routed.shape[-1])  # [S]
        q_index = torch.arange(type_ids.shape[0], device=scores.device)
        groups = slot_question[None, :] == q_index[:, None]  # [Q, S]
        valid = groups & opt_mask[None, :].bool()
        weights = torch.softmax(
            scores[None, :].expand(type_ids.shape[0], -1).masked_fill(~valid, float("-inf")),
            dim=-1,
        )  # [Q, S]
        summaries = torch.einsum("qs,sw->qw", weights, routed[0])  # [Q, width]

        fields = (
            base_fields
            + head.option_summary_norm(summaries)
            + head.global_projection(global_vector[0]).unsqueeze(0)
            + head.type_embedding(type_ids)
        ).unsqueeze(0)  # [1, Q, width]
        for layer in head.layers:
            fields = layer(fields, memory, memory_key_padding_mask=key_pad)
        fields = head.field_norm(fields[0])  # [Q, width]

        # One request's logits: prior + gated (cosine + residual), per slot.
        field_by_slot = fields[slot_question]  # [S, width]
        options = head.option_norm(routed[0])  # [S, width]
        anchor = torch.nn.functional.normalize(
            question_vectors[0] + global_vector[0], dim=-1)
        lexical_anchor = torch.nn.functional.normalize(lexical[0], dim=-1)
        prior_scale = head.prior_logit_scale.clamp(max=math.log(100.0)).exp()
        prior = prior_scale * (lexical_anchor * anchor[slot_question]).sum(-1)
        dot = (field_by_slot * options).sum(-1)
        denom = (field_by_slot.norm(dim=-1) * options.norm(dim=-1)).clamp_min(1e-8)
        cosine = dot / denom
        features = torch.cat(
            [field_by_slot, options, field_by_slot * options,
             (field_by_slot - options).abs()],
            dim=-1,
        )
        residual = head.residual_scorer(features).squeeze(-1)
        joint_scale = head.joint_logit_scale.clamp(max=math.log(100.0)).exp()
        logits = prior + torch.sigmoid(head.residual_gate) * (
            joint_scale * cosine + residual)
        return logits.unsqueeze(0).float()  # [1, S]


def demo_record():
    """One record with a choice, a noul and a score question."""
    return {
        "id": "demo",
        "state": "Our checkout is down and customers can't pay. The deploy broke production.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Payments and refunds", "engineering": "Bugs and outages",
                             "other": None},
            },
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["Routine", "Soon", "Today"],
            },
        },
    }


def pack(encoded, length):
    """The pooling matrices + slot metadata for one reference EncodedRecord."""
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
    return pool_global, pool_question, pool_option, type_ids, slot_question, opt_mask


def main():
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    print("loading the clef checkpoint (fp32)...", flush=True)
    backbone = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.float32)
    head_config = json.loads((__import__("pathlib").Path(CKPT) / "joint_head_config.json").read_text())
    head = jsm.JointSchemaHead(**head_config)
    head.load_state_dict(load_file(os.path.join(CKPT, "joint_head.safetensors")), strict=True)
    model = jsm.ClefModel(backbone, head).eval()
    wrapper = ClefTextExport(model).eval()
    print(f"head config: {head_config}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    record = demo_record()
    encoded = jsm.encode_record(tokenizer, record, max_length=16384)
    length = len(encoded.input_ids)
    tensors = pack(encoded, length)
    args = (torch.tensor([list(encoded.input_ids)], dtype=torch.long),
            torch.ones(1, length, dtype=torch.long), *tensors)

    with torch.no_grad():
        reference = model(jsm.collate_records([encoded], tokenizer.pad_token_id,
                                              torch.device("cpu")))[0]
        exported = wrapper(*args)
        ref_logits = torch.cat(reference)  # [sum K_q] in question-major order
        opt_mask = tensors[5]
        exported_valid = exported[0][opt_mask]
        diff = (ref_logits - exported_valid).abs().max().item()
    print(f"in-process check: {length} tokens, max |ref - wrapper| = {diff:.2e}", flush=True)
    if diff > 1e-4:
        raise SystemExit("wrapper diverges from the reference; fix before exporting")

    from torch.export import Dim
    seq = Dim("seq", min=32, max=16384)
    q = Dim("questions", min=1, max=64)
    s = Dim("slots", min=1, max=4096)
    print("exporting with torch.onnx.export (dynamo=True)...", flush=True)
    program = torch.onnx.export(
        wrapper, args, dynamo=True, optimize=False,
        dynamic_shapes=(
            {1: seq},          # input_ids
            {1: seq},          # attention_mask
            {2: seq},          # pool_global
            {1: q, 2: seq},    # pool_question
            {1: s, 2: seq},    # pool_option
            {0: q},            # type_ids
            {0: s},            # slot_question
            {0: s},            # opt_mask
        ),
        input_names=["input_ids", "attention_mask", "pool_global", "pool_question",
                     "pool_option", "type_ids", "slot_question", "opt_mask"],
        output_names=["logits"],
    )
    path = f"{EXPORT}/clef_text_fp32.onnx"
    program.save(path)
    print("saved", path, flush=True)

    # --- the shipped fp16 graph: the same wrapper, half weights, fp32 IO -----
    print("exporting the fp16 graph...", flush=True)
    wrapper_half = ClefTextExport(model).half().eval()
    with torch.no_grad():
        half_out = wrapper_half(*args).float()
        half_diff = (half_out[0][tensors[5]] - ref_logits).abs().max().item()
    print(f"fp16 in-process check: max |ref - fp16 wrapper| = {half_diff:.2e}", flush=True)

    program_half = torch.onnx.export(
        wrapper_half, args, dynamo=True, optimize=False,
        dynamic_shapes=(
            {1: seq},          # input_ids
            {1: seq},          # attention_mask
            {2: seq},          # pool_global
            {1: q, 2: seq},    # pool_question
            {1: s, 2: seq},    # pool_option
            {0: q},            # type_ids
            {0: s},            # slot_question
            {0: s},            # opt_mask
        ),
        input_names=["input_ids", "attention_mask", "pool_global", "pool_question",
                     "pool_option", "type_ids", "slot_question", "opt_mask"],
        output_names=["logits"],
    )
    path_half = f"{EXPORT}/clef_text_fp16.onnx"
    program_half.save(path_half)
    print("saved", path_half, flush=True)


if __name__ == "__main__":
    main()
