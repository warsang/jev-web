"""Torso families + browser-feasibility gate for the strands-decider web export.

Every strands-decider checkpoint is a LoRA adapter on a *text decoder* plus a
pointer head; only the torso family differs:

  qwen3_5  Qwen/Qwen3.5-*      transformers.Qwen3_5ForCausalLM(text_cfg).model
  gemma4   google/gemma-4-*    transformers.Gemma4ForConditionalGeneration(cfg).model.language_model

The gemma4 entry mirrors the reference loader (strands_decider/modeling.py
::_load_torso, GEMMA4_CLASSES): gemma-4 ships as multimodal checkpoints, so
the text decoder is reached through the conditional-generation class rather
than a text-only config.

BROWSER FEASIBILITY (why the 12B and the 26B-A4B are not exported):
the fused graph holds the whole torso in one ONNX file, so the download and
the resident weight set are the full parameter count at the chosen quant:

  checkpoint            params     q8 download   verdict
  2B qwen3.5 (hobson)   ~2.0B      1.8 GB today  shipped
  E2B gemma4            ~2.1B      ~2.1 GB       shipped
  E4B gemma4            ~5.7B      ~5.7 GB       opt-in (heavy but loadable)
  12B gemma4            ~13.3B     ~13 GB        refused: > MAX_DENSE_PARAMS_B
  26B-A4B gemma4        ~26B MoE   ~26 GB        refused: MoE (128 experts; a
                                                router needs *all* experts
                                                resident, and q4f16 is still
                                                ~14 GB — no browser target)

Set EXPORT_ALLOW_BIG=1 to override the size check (still refuses MoE) — useful
for a desktop/WebGPU experiment, never for a published default.
"""
from __future__ import annotations

import json
import os

# Above this dense parameter count the q8/q4f16 ONNX stops being something a
# browser should be asked to download. Tuned so E2B (~2.1B) and E4B (~5.7B)
# pass and the 12B (~13.3B) is refused.
MAX_DENSE_PARAMS_B = 6.0


class TorsoFamilyError(SystemExit):
    pass


def torso_family() -> str:
    """TORSO_FAMILY env var; defaults to the original qwen3_5 pipeline."""
    return os.environ.get("TORSO_FAMILY", "qwen3_5").strip().lower()


# family -> (transformers class, how to reach the text decoder)
TORSO_CLASSES = {
    "qwen3_5": ("Qwen3_5ForCausalLM", "model"),
    "gemma4": ("Gemma4ForConditionalGeneration", "model.language_model"),
}


def resolve_torso_class(import_transformers, family: str | None = None):
    """(class, torso_accessor_code) for the configured family."""
    fam = family or torso_family()
    if fam not in TORSO_CLASSES:
        raise TorsoFamilyError(
            f"unknown TORSO_FAMILY={fam!r}; known: {sorted(TORSO_CLASSES)}")
    cls_name, accessor = TORSO_CLASSES[fam]
    cls = getattr(import_transformers, cls_name, None)
    if cls is None:
        raise TorsoFamilyError(
            f"transformers has no {cls_name} — install a transformers with {fam} support")
    return cls, accessor


def text_config_of(base_dir: str):
    """Text-decoder config of a (possibly multimodal) checkpoint directory."""
    import transformers
    cfg = transformers.AutoConfig.from_pretrained(base_dir, trust_remote_code=False)
    return cfg.get_text_config(), cfg


def _moe_markers(text_cfg) -> dict:
    cfg = text_cfg.to_dict() if hasattr(text_cfg, "to_dict") else dict(text_cfg)
    return {
        "enable_moe_block": bool(cfg.get("enable_moe_block", False)),
        "num_local_experts": int(cfg.get("num_local_experts", 0) or 0),
        "num_experts": int(cfg.get("num_experts", 0) or 0),
    }


def estimate_dense_params_b(text_cfg) -> float:
    """Rough dense parameter estimate from the text-decoder config.

    per layer: q/k/v/o (4*h*h) + mlp (3*h*I); plus vocab embeddings twice
    (input + tied-shared lm head). Good enough for a size gate, not billing.
    """
    cfg = text_cfg.to_dict() if hasattr(text_cfg, "to_dict") else dict(text_cfg)
    h = int(cfg.get("hidden_size", 0) or 0)
    layers = int(cfg.get("num_hidden_layers", 0) or 0)
    intermediate = int(cfg.get("intermediate_size", 4 * h) or 4 * h)
    vocab = int(cfg.get("vocab_size", 0) or 0)
    if not h or not layers:
        return float("inf")
    per_layer = 4 * h * h + 3 * h * intermediate
    return (layers * per_layer + 2 * vocab * h) / 1e9


def assert_browser_feasible(text_cfg, family: str, allow_big: bool | None = None) -> float:
    """Refuse torsos that cannot become a sane browser download.

    MoE is refused unconditionally: routing needs every expert resident, so an
    'A4B active' model still ships all ~26B of weights. Dense > MAX_DENSE_PARAMS_B
    is refused unless EXPORT_ALLOW_BIG=1 (or allow_big=True).
    Returns the dense parameter estimate in billions.
    """
    allow = os.environ.get("EXPORT_ALLOW_BIG", "") == "1" if allow_big is None else allow_big
    moe = _moe_markers(text_cfg)
    if moe["enable_moe_block"] or moe["num_local_experts"] > 1 or moe["num_experts"] > 1:
        raise TorsoFamilyError(
            f"refusing to export a MoE torso ({family}: {moe}): every expert must be "
            "resident to route, so the browser download would be tens of GB. "
            "Export this server-side instead.")
    params = estimate_dense_params_b(text_cfg)
    if params > MAX_DENSE_PARAMS_B and not allow:
        raise TorsoFamilyError(
            f"refusing to export {family}: ~{params:.1f}B params "
            f"(> {MAX_DENSE_PARAMS_B}B) means a ~{params:.0f} GB q8 ONNX. "
            "Set EXPORT_ALLOW_BIG=1 to override (desktop experiment only).")
    return params


def read_ckpt_config(ckpt_dir: str) -> dict:
    """The checkpoint's strands_decider_config.json (hobson_config.json on v19).

    Carries the calibration the JS runtime reads at load time; 05_manifest.py
    copies it verbatim into the ONNX repo so the browser picks it up.
    """
    for name in ("strands_decider_config.json", "hobson_config.json"):
        path = os.path.join(ckpt_dir, name)
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
    return {}


def apply_qwen3_5_export_patch(out_dir: str) -> None:
    """Qwen3.5-only: swap the GatedDeltaNet chunk loop for a while_loop.

    Gemma4 is plain attention and needs no patch. Must run before the torso is
    built (see export/strands-decider/export_patch.py).
    """
    import sys
    import transformers.models.qwen3_5.modeling_qwen3_5 as qm
    import transformers.models.qwen3_5.modular_qwen3_5 as qmm
    if out_dir not in sys.path:
        sys.path.insert(0, out_dir)
    from export_patch import patched_torch_chunk_gated_delta_rule as patched
    qm.torch_chunk_gated_delta_rule = patched
    qmm.torch_chunk_gated_delta_rule = patched
    print("DeltaNet export patch applied", flush=True)
