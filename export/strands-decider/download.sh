#!/bin/bash
# Download base model + checkpoint files from Hugging Face (public, no auth).
#
# Torso family is selected with TORSO_FAMILY (qwen3_5|gemma4) and pinned via
# env vars; defaults reproduce the original hobson-v19 run:
#   BASE_MODEL / BASE_REV      base weights of the torso
#   CKPT_MODEL / CKPT_REV     the strands-decider LoRA checkpoint
set -e
OUT=~/workspace/strands-export
mkdir -p "$OUT/base" "$OUT/ckpt"

PYTHON_BIN=${PYTHON_BIN:-python3}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-2B-Base}
CKPT_MODEL=${CKPT_MODEL:-StrandsAgents/strands-decider-2B-hobson-v19}
BASE_REV=${BASE_REV:-b1485b2fa6dfa1287294f269f5fb618e03d52d7c}
CKPT_REV=${CKPT_REV:-bb282d786bc251fd4e3068de3ada9ddbb38127cd}

echo "== base model file list =="
curl -s "https://huggingface.co/api/models/$BASE_MODEL/revision/$BASE_REV" |
  "$PYTHON_BIN" -c "
import json,sys
d=json.load(sys.stdin)
for s in d.get('siblings',[]):
    print(s['rfilename'])
" | tee "$OUT/base_filelist.txt"

echo "== downloading base safetensors + configs =="
grep -E '\.safetensors$|config\.json$|generation_config\.json$|tokenizer' "$OUT/base_filelist.txt" |
  while read -r f; do
  dest="$OUT/base/$f"
  if [ -f "$dest" ]; then echo "skip $f"; continue; fi
  mkdir -p "$(dirname "$dest")"
  echo "get $f"
  curl -sL --retry 3 -o "$dest" "https://huggingface.co/$BASE_MODEL/resolve/$BASE_REV/$f"
done

echo "== checkpoint files =="
# strands_decider_config.json is the v21+ calibration file; hobson_config.json
# is the v19 name. Either is fine — download.sh keeps whichever exists.
CONFIG_FILE=strands_decider_config.json
if ! curl -sfL -o /dev/null "https://huggingface.co/$CKPT_MODEL/resolve/$CKPT_REV/$CONFIG_FILE"; then
  CONFIG_FILE=hobson_config.json
fi
echo "calibration file: $CONFIG_FILE"
for f in lora/adapter_model.safetensors lora/adapter_config.json head.safetensors "$CONFIG_FILE" tokenizer.json tokenizer_config.json; do
  dest="$OUT/ckpt/$f"
  if [ -f "$dest" ]; then echo "skip $f"; continue; fi
  mkdir -p "$(dirname "$dest")"
  echo "get $f"
  curl -sL --retry 3 -o "$dest" \
    "https://huggingface.co/$CKPT_MODEL/resolve/$CKPT_REV/$f"
done

echo "== sizes =="
du -sh "$OUT/base" "$OUT/ckpt"
ls -la "$OUT/ckpt" "$OUT/ckpt/lora"
