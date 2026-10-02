#!/bin/bash
# Download base model + checkpoint files from Hugging Face (public, no auth).
set -e
OUT=~/workspace/strands-export
mkdir -p "$OUT/base" "$OUT/ckpt"

BASE_REV=b1485b2fa6dfa1287294f269f5fb618e03d52d7c
CKPT_REV=bb282d786bc251fd4e3068de3ada9ddbb38127cd

echo "== base model file list =="
curl -s "https://huggingface.co/api/models/Qwen/Qwen3.5-2B-Base/revision/$BASE_REV" \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
for s in d.get('siblings',[]):
    print(s['rfilename'])
" | tee "$OUT/base_filelist.txt"

echo "== downloading base safetensors + configs =="
grep -E '\.safetensors$|config\.json$|generation_config\.json$|tokenizer' "$OUT/base_filelist.txt" \
  | while read -r f; do
  dest="$OUT/base/$f"
  if [ -f "$dest" ]; then echo "skip $f"; continue; fi
  mkdir -p "$(dirname "$dest")"
  echo "get $f"
  curl -sL --retry 3 -o "$dest" "https://huggingface.co/Qwen/Qwen3.5-2B-Base/resolve/$BASE_REV/$f"
done

echo "== checkpoint files =="
for f in lora/adapter_model.safetensors lora/adapter_config.json head.safetensors \
         hobson_config.json tokenizer.json tokenizer_config.json; do
  dest="$OUT/ckpt/$f"
  if [ -f "$dest" ]; then echo "skip $f"; continue; fi
  mkdir -p "$(dirname "$dest")"
  echo "get $f"
  curl -sL --retry 3 -o "$dest" \
    "https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19/resolve/$CKPT_REV/$f"
done

echo "== sizes =="
du -sh "$OUT/base" "$OUT/ckpt"
ls -la "$OUT/ckpt" "$OUT/ckpt/lora"
