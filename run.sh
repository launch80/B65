#!/usr/bin/env bash
# One-command vLLM XPU server with the INT4 lm_head patch applied.
#
#   ./run.sh                      # your model from $MODEL, port 8000
#   MODEL=/path/to/model ./run.sh
#   SPEC=0 ./run.sh               # without speculative decoding
#
# Everything is overridable by env; nothing is specific to our box except the
# defaults. See README.md for what this does and what it costs.
set -euo pipefail

IMAGE="${IMAGE:-vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f}"
MODEL="${MODEL:?set MODEL to a local model directory}"
PORT="${PORT:-8000}"
NAME="${NAME:-vllm-p608}"
MAX_LEN="${MAX_LEN:-32768}"
UTIL="${UTIL:-0.88}"
# Hybrid (GDN / Mamba-style) models allocate one recurrent-state block per decode
# sequence, so vLLM's default max_num_seqs of 256 fails engine start with
# "exceeds available Mamba cache blocks". 32 is a safe serving value; use 1 for
# single-stream benchmarking.
MAX_SEQS="${MAX_SEQS:-32}"
SPEC="${SPEC:-4}"
LMHEAD_INT4="${LMHEAD_INT4:-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# HuggingFace cache dirs store snapshots as symlinks into ../../blobs, which dangle
# if only the snapshot is bind-mounted. Detect that and mount the repo root instead.
MODEL_MOUNT=(-v "$MODEL:/model:ro")
MODEL_PATH=/model
case "$MODEL" in
  */snapshots/*)
    REPO="${MODEL%%/snapshots/*}"
    if [[ -d "$REPO/blobs" ]]; then
      MODEL_MOUNT=(-v "$REPO:/repo:ro")
      MODEL_PATH="/repo/snapshots/${MODEL##*/snapshots/}"
      echo "note: HF cache layout detected, mounting repo root so blob symlinks resolve"
    fi ;;
esac

# GPU selection: pass GPU_BDF to pin one card by PCI address on a multi-GPU host,
# otherwise every render node is exposed and ZE_AFFINITY_MASK picks one.
DEVFLAGS=(--device /dev/dri)
if [[ -n "${GPU_BDF:-}" ]]; then
  RENDER="$(readlink -f "/dev/dri/by-path/pci-${GPU_BDF}-render")"
  BP="${TMPDIR:-/tmp}/p608-by-path-$$"; rm -rf "$BP"; mkdir -p "$BP"
  ln -s "../$(basename "$RENDER")" "$BP/pci-${GPU_BDF}-render"
  # oneCCL's fd manager scans /dev/dri/by-path and aborts if it is missing
  DEVFLAGS=(--device "$RENDER" -v "$BP:/dev/dri/by-path:ro")
fi

PATCHES='true'
[[ "$LMHEAD_INT4" == "1" ]] && PATCHES='python /p608/patches/p608_lmhead_int4.py'

ARGS=(vllm serve "$MODEL_PATH" --host 0.0.0.0 --port 8000
  --quantization gptq --dtype float16
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$UTIL"
  --kv-cache-dtype fp8 --max-num-seqs "$MAX_SEQS"
  --no-enable-prefix-caching --language-model-only)
[[ "$SPEC" != "0" ]] && ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC}}")

docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run --rm --name "$NAME" "${DEVFLAGS[@]}" --ipc=host \
  "${MODEL_MOUNT[@]}" -v "$HERE:/p608:ro" -p "${PORT}:8000" \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}" \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e "P608_R3_LMHEAD_INT4=$LMHEAD_INT4" \
  --entrypoint bash "$IMAGE" -lc "set -e; ${PATCHES}; exec $(printf '%q ' "${ARGS[@]}")"
