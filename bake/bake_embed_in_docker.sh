#!/usr/bin/env bash
# Low-RAM bake of an embed-intN derivative inside the vLLM XPU image.
# Example:
#   SRC=/path/to/Qwen3.8-27B-GPTQ-Int4-baked-v2 \
#   DST=/path/to/Qwen3.8-27B-GPTQ-Int4-baked-v2-embed-int8 \
#   bash bake/bake_embed_in_docker.sh 8
set -euo pipefail
BITS="${1:-8}"
IMAGE="${IMAGE:-vllm/vllm-openai-xpu}"
SRC="${SRC:?set SRC to baked-v2 (or equivalent) checkpoint dir}"
DST="${DST:-${SRC}-embed-int${BITS}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAKE_PY="${BAKE_PY:-$SCRIPT_DIR/bake_embed_quant.py}"
CHUNK="${CHUNK:-2048}"

if [[ ! -f "$BAKE_PY" ]]; then
  echo "missing $BAKE_PY" >&2
  exit 1
fi
if [[ ! -d "$SRC" ]]; then
  echo "missing SRC $SRC" >&2
  exit 1
fi

echo "bake embed-int${BITS}: $SRC -> $DST (chunk=$CHUNK)"
rm -rf "$DST"

docker run --rm --entrypoint bash \
  -v "$(dirname "$SRC"):$(dirname "$SRC")" \
  -v "$(dirname "$DST"):$(dirname "$DST")" \
  -v "$BAKE_PY:/bake.py:ro" \
  "$IMAGE" -lc \
  "python /bake.py --src '$SRC' --bits $BITS --dst '$DST' --chunk $CHUNK"

ls -la "$DST/model-embed-int${BITS}.safetensors"
python3 - <<PY
import json
from pathlib import Path
dst = Path("$DST")
c = json.loads((dst / "config.json").read_text())
print("embed_tokens_quant", c.get("embed_tokens_quant"))
idx = json.loads((dst / "model.safetensors.index.json").read_text())
print("weight map", idx["weight_map"].get("model.language_model.embed_tokens.weight"))
print("scale in map", "model.language_model.embed_tokens.weight_scale" in idx["weight_map"])
PY
echo BAKE_OK
