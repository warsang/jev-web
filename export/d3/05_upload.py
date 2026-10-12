"""Stage 5: upload the browser export to Hugging Face (needs an HF token).

    hf auth login                      # or: set HF_TOKEN=...
    python export/d3/05_upload.py      # repo: $UPLOAD_REPO (default warsang/d3-edge-web)

After the upload, pin the returned commit sha into src/decision3.mjs as
DECISION3_DEFAULT_REVISION (and the same for the d3-lite / d3-nano exports,
which reuse this pipeline with D3_OUT pointed at their checkpoint dirs).
"""
from __future__ import annotations

import json
import os

from huggingface_hub import HfApi

OUT = os.environ.get("D3_OUT", os.path.expanduser("~/workspace/d3edge-export"))
FINAL = f"{OUT}/final"
REPO = os.environ.get("UPLOAD_REPO", "warsang/d3-edge-web")


def main():
    import hashlib
    files = {}
    for root, _, fs in os.walk(FINAL):
        for f in fs:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, FINAL).replace("\\", "/")
            if rel == "MANIFEST.json":
                continue
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            files[rel] = {"bytes": os.path.getsize(p), "sha256": h.hexdigest()}
    manifest_path = f"{FINAL}/MANIFEST.json"
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    manifest["files"] = files
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    total_gb = sum(v["bytes"] for v in files.values()) / 1e9
    print(f"uploading {len(files)} files, {total_gb:.2f} GB, to {REPO}", flush=True)

    api = HfApi()
    api.create_repo(REPO, exist_ok=True, repo_type="model")
    commit = api.upload_folder(
        repo_id=REPO,
        folder_path=FINAL,
        repo_type="model",
        commit_message="d3-edge-web: browser export (fp16 noncausal torso + fp32 readout)",
    )
    sha = getattr(commit, "oid", None) or (commit.get("commitUrl") if isinstance(commit, dict) else None)
    print("uploaded. pin this in src/decision3.mjs:", sha or "<see the commit on the repo>")
    print(f"  export const DECISION3_DEFAULT_REVISION = \"{sha}\";")


if __name__ == "__main__":
    main()
