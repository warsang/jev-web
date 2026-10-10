"""Download base weights + checkpoint files from the Hub (cross-platform).

The bash downloader (download.sh) is fine on a Linux build box but breaks in
two ways elsewhere: CRLF checkouts eat its line continuations, and `~` inside
WSL bash is not the Windows home the later stages use. This script is the same
file selection, run with the stage venv's python and os.path.expanduser, so it
lands in ~/workspace/strands-export on every platform.

Pins come from the environment, exactly like download.sh:

    TORSO_FAMILY=gemma4 \
    BASE_MODEL=google/gemma-4-E2B-it BASE_REV=<sha> \
    CKPT_MODEL=StrandsAgents/strands-decider-E2B-gemma4-v1-2610 CKPT_REV=<sha> \
    python download.py

Resumable: re-running only fetches what is missing.
"""
from __future__ import annotations

import os
import sys

from huggingface_hub import snapshot_download

OUT = os.environ.get("STRANDS_OUT", os.path.expanduser("~/workspace/strands-export"))
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-2B-Base")
CKPT_MODEL = os.environ.get(
    "CKPT_MODEL", "StrandsAgents/strands-decider-2B-hobson-v19")
# Defaults reproduce the original hobson-v19 run.
BASE_REV = os.environ.get("BASE_REV", "b1485b2fa6dfa1287294f269f5fb618e03d52d7c")
CKPT_REV = os.environ.get(
    "CKPT_REV", "bb282d786bc251fd4e3068de3ada9ddbb38127cd")

# The torso needs weights + the text-side configs; everything else in a
# multimodal checkpoint (docs, preprocessor extras) is skipped to save disk.
BASE_PATTERNS = [
    "*.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "special_tokens_map.json",
]

# The checkpoint ships a LoRA + the pointer head + its calibration file.
# strands_decider_config.json is the v21+ name; hobson_config.json is v19's.
CKPT_PATTERNS = [
    "lora/*",
    "head.safetensors",
    "strands_decider_config.json",
    "hobson_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def pull(repo_id: str, revision: str, patterns: list[str], local_dir: str) -> str:
    print(f"== {repo_id} @ {revision or 'main'} -> {local_dir}", flush=True)
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision or None,
        allow_patterns=patterns,
        local_dir=local_dir,
    )
    total = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(path)
        for f in files
    )
    print(f"   {total/1e9:.2f} GB on disk", flush=True)
    return path


def main() -> None:
    pull(BASE_MODEL, BASE_REV, BASE_PATTERNS, f"{OUT}/base")
    pull(CKPT_MODEL, CKPT_REV, CKPT_PATTERNS, f"{OUT}/ckpt")
    print("DOWNLOAD DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
