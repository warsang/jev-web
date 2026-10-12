"""Stage 4: upload the browser export to Hugging Face (needs an HF token).

    hf auth login                       # or: set HF_TOKEN=...
    python export/clef/04_upload.py     # repo: $UPLOAD_REPO (default warsang/clef-text-0.6b-web)

After the upload, pin the returned commit sha into src/clef.mjs as
CLEF_DEFAULT_REVISION.
"""
from __future__ import annotations

import json
import os

from huggingface_hub import HfApi

OUT = os.environ.get("CLEF_OUT", os.path.expanduser("~/workspace/clef-export"))
FINAL = f"{OUT}/final"
REPO = os.environ.get("UPLOAD_REPO", "warsang/clef-text-0.6b-web")


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
        commit_message="clef-text-0.6b-web: browser export (fused fp16 graph)",
    )
    sha = getattr(commit, "oid", None) or (commit.get("commitUrl") if isinstance(commit, dict) else None)
    print("uploaded. pin this in src/clef.mjs:", sha or "<see the commit on the repo>")
    print(f"  export const CLEF_DEFAULT_REVISION = \"{sha}\";")


if __name__ == "__main__":
    main()
