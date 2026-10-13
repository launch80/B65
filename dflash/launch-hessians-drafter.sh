#!/usr/bin/env bash
# DRAFT=~/models/dflash2-qwen3.8-27b TARGET=~/models/swift-bake/out OUT=~/bake-dflash ./launch-hessians-drafter.sh --group A
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
IMAGE="${IMAGE:-vllm/vllm-openai-xpu@sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d}"
TARGET="${TARGET:-$HOME/models/swift-bake/out}"; DRAFT="${DRAFT:-$HOME/models/dflash2-qwen3.8-27b}"
OUT="${OUT:-$HOME/bake-dflash}"; NAME="${NAME:-df-hessians}"; GPU_BDF="${GPU_BDF:-0000:86:00.0}"
HERE="$(cd "$(dirname "$0")" && pwd)"; W="${W:-$HERE}"; PATCH_DIR="${PATCH_DIR:-$HERE/../patches}"
BYPATH="/tmp/df-bake-bypath"; RENDER="$(readlink -f "/dev/dri/by-path/pci-${GPU_BDF}-render")"
rm -rf "$BYPATH"; mkdir -p "$BYPATH"; ln -s "../$(basename "$RENDER")" "$BYPATH/pci-${GPU_BDF}-render"
docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run -d --name "$NAME" --device "$RENDER" --ipc=host \
  -v "$BYPATH:/dev/dri/by-path:ro" -v "$TARGET:/model:ro" -v "$DRAFT:/draft:ro" -v "$W:/w:ro" -v "$OUT:/out" -v "$PATCH_DIR:/patches:ro" \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=0 -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e MODEL_PATH=/model -e DRAFT_PATH=/draft -e HF_HUB_OFFLINE=1 \
  --entrypoint bash "$IMAGE" -lc \
  "python /patches/patch_mtp_nightly.py; python /patches/patch_mtp_boundary.py; python /patches/patch_gdn_mixed_split_v5.py;
   stdbuf -oL python /w/bake_hessians_drafter.py --out /out --prompts /out/prompts.json $* 2>&1 | stdbuf -oL grep -vE 'it/s|toks/s|Adding requests|CCL_WARN|Loading safetensors|Token indices|min_frames|max_frames|use_fast' | tee -a /out/hessians-run.log"
