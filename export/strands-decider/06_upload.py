"""Upload the strands browser export to Hugging Face (needs an HF token).

    hf auth login                       # or: set HF_TOKEN=...
    python export/06_upload.py

Pins come from the environment, same as download.py / 05_manifest.py:

    TORSO_FAMILY=qwen3_5 \
    BASE_MODEL=Qwen/Qwen3.5-2B-Base BASE_REV=<sha> \
    CKPT_MODEL=StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
    CKPT_REV=<sha> \
    UPLOAD_REPO=warsang/strands-decider-2b-qwen3v1-web \
    python export/06_upload.py

Uploads final/ (the quantized graph + external data, the fp32 reference
graph, the tokenizer, and the checkpoint's strands_decider_config.json — the
JS runtime reads the calibration from it). Prints the commit sha to pin in
src/strands.mjs.
"""
from __future__ import annotations

import json
import os
import sys

from huggingface_hub import HfApi

OUT = os.environ.get("STRANDS_OUT", os.path.expanduser("~/workspace/strands-export"))
FINAL = f"{OUT}/final"
REPO = os.environ.get(
    "UPLOAD_REPO", "warsang/strands-decider-2b-qwen3v1-web")


def main():
    import hashlib
    files = {}
    for root, _, fs in os.walk(FINAL):
        for f in fs:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, FINAL).replace("\\", "/")
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            files[rel] = {"bytes": os.path.getsize(p), "sha256": h.hexdigest()}
    manifest = {}
    manifest_path = f"{FINAL}/MANIFEST.json"
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
    manifest["files"] = files
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    total_gb = sum(v["bytes"] for v in files.values()) / 1e9
    print(f"uploading {len(files)} files, {total_gb:.2f} GB, to {REPO}", flush=True)

    api = HfApi()
    api.create_repo(REPO, exist_ok=True, repo_type="model")
    commit = api.upload_folder(repo_id=REPO, folder_path=FINAL, repo_type="model")
    sha = getattr(commit, "oid", None)
    print("uploaded. pin this in src/strands.mjs:", sha)
    print(f'  export const STRANDS_DEFAULT_REVISION = "{sha}";')


if __name__ == "__main__":
    sys.exit(main())
