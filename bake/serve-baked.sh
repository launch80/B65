#!/usr/bin/env bash
# Serve the BAKED checkpoint with NO weight patches. The point of the bake is that
# R3 and M1 live on disk now, so this deliberately runs only the three vLLM
# correctness fixes (MTP nightly, MTP boundary, GDN mixed-split) plus the R6 vocab
# prefix, which is a runtime slice and cannot be a weight. If this serves at the same
# quality and speed as the patched stock checkpoint, the bake is correct.
#
#   BAKED=$HOME/bake-model ./serve-baked.sh
set -euo pipefail
BAKED="${BAKED:?path to the assembled baked checkpoint}"
IMAGE="${IMAGE:-vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f}"
SERVED="${SERVED:-launch80/Qwen3.8-27B-GPTQ-Int4-baked}"
PORT="${PORT:-8002}"; NAME="${NAME:-vllm-baked}"; GPU_BDF="${GPU_BDF:-0000:86:00.0}"
MAX_LEN="${MAX_LEN:-32768}"; UTIL="${UTIL:-0.92}"; MAX_SEQS="${MAX_SEQS:-32}"
SPEC="${SPEC:-6}"; DRAFT_VOCAB="${DRAFT_VOCAB:-32768}"
PATCH_DIR="${PATCH_DIR:-$HOME/Documents/project608/parity/patches}"
R3_DIR="${R3_DIR:-$HOME/Documents/project608/research/r3}"
BYPATH_DIR="$HOME/.cache/vllm-dri-by-path-baked"
RENDER="$(readlink -f "/dev/dri/by-path/pci-${GPU_BDF}-render")"
rm -rf "$BYPATH_DIR"; mkdir -p "$BYPATH_DIR"; ln -s "../$(basename "$RENDER")" "$BYPATH_DIR/pci-${GPU_BDF}-render"
# No patch_draft_mtp_int4 (M1) and P608_R3_LMHEAD_INT4=0: both are on disk now.
# B70_MTP_BF16_DRAFT is deliberately NOT set: the cookbook MTP patch forces the draft
# to a dense build when it is 1, which would discard the baked INT4 draft on disk.
# patch_r3 is still loaded ONLY for its R6 draft-vocab prefix, gated by P608_DRAFT_VOCAB.
PATCH_CMDS='python /patches/patch_mtp_nightly.py; python /patches/patch_mtp_boundary.py'
[[ "$SPEC" != "0" ]] && PATCH_CMDS="$PATCH_CMDS; python /patches/patch_gdn_mixed_split_v5.py"
[[ "$DRAFT_VOCAB" != "0" ]] && PATCH_CMDS="$PATCH_CMDS; python /r3/patch_r3_lmhead_int4.py"
ARGS=(vllm serve /model --host 0.0.0.0 --port 8000 --served-model-name "$SERVED"
  --quantization gptq --dtype float16 --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$UTIL" --kv-cache-dtype fp8 --max-num-seqs "$MAX_SEQS"
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3)
[[ "$SPEC" != "0" ]] && ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC}}")
docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run --rm --name "$NAME" --device "$RENDER" --ipc=host \
  -v "$BYPATH_DIR:/dev/dri/by-path:ro" -v "$BAKED:/model:ro" \
  -v "$PATCH_DIR:/patches:ro" -v "$R3_DIR:/r3:ro" -p "${PORT}:8000" \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e P608_R3_LMHEAD_INT4=0 -e "P608_DRAFT_VOCAB=$DRAFT_VOCAB" -e B70_DRAFT_MTP_INT4=0 \
  --entrypoint bash "$IMAGE" -lc "set -e; ${PATCH_CMDS}; exec $(printf '%q ' "${ARGS[@]}")"
