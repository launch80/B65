#!/usr/bin/env bash
# Serve with the Launch80 W3A16 body. Same stack as serve-b65.sh with two changes:
#
#   * the p608-sycl image, because libw3_l80.so needs the oneAPI SYCL runtime that
#     the stock vLLM image does not carry
#   * W3=1 swaps the 256 body linears from GPTQ-int4 to W3A16
#
# W3=0 runs the identical script without that swap, so a perplexity comparison
# between the two arms differs in exactly one thing.
#
#   W3=0 ./serve-w3.sh     # baseline arm
#   W3=1 ./serve-w3.sh     # candidate arm
set -euo pipefail

IMAGE="${IMAGE:-p608-sycl:latest}"
MODEL_REPO="${MODEL_REPO:-$HOME/.cache/huggingface/hub/models--SergiioB--Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16}"
MODEL_REV="${MODEL_REV:-e28c5f952bdd5d814297a07d85a064a87af26a3f}"
SERVED="${SERVED:-SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16}"
PORT="${PORT:-8002}"
NAME="${NAME:-vllm-w3}"
GPU_BDF="${GPU_BDF:-0000:86:00.0}"
MAX_LEN="${MAX_LEN:-8192}"
# Lower than the daily driver's 0.92: the W3 install loads 3-bit weights while
# freeing the int4 ones module by module, and this leaves headroom for the churn.
UTIL="${UTIL:-0.85}"
MAX_SEQS="${MAX_SEQS:-8}"
# Perplexity needs speculation OFF - prompt_logprobs returns NaN under MTP on this
# stack. Default accordingly; override for throughput runs.
SPEC="${SPEC:-0}"
R3="${R3:-1}"
MTP_INT4="${MTP_INT4:-1}"
W3="${W3:-1}"
W3_DIR="${W3_DIR:-$HOME/w3full}"
W3_SKIP="${W3_SKIP:-}"
KERNEL_DIR="${KERNEL_DIR:-$HOME/Documents/project608/research/r8/l80kernel}"
PATCH_DIR="${PATCH_DIR:-$HOME/Documents/project608/parity/patches}"
R3_DIR="${R3_DIR:-$HOME/Documents/project608/research/r3}"
R9_DIR="${R9_DIR:-$HOME/Documents/project608/research/r9}"
BYPATH_DIR="${BYPATH_DIR:-$HOME/.cache/vllm-dri-by-path-w3}"

RENDER="$(readlink -f "/dev/dri/by-path/pci-${GPU_BDF}-render" 2>/dev/null || true)"
[[ -n "$RENDER" && -e "$RENDER" ]] || { echo "ERROR: no render node for ${GPU_BDF}" >&2; exit 1; }
rm -rf "$BYPATH_DIR"; mkdir -p "$BYPATH_DIR"
ln -s "../$(basename "$RENDER")" "$BYPATH_DIR/pci-${GPU_BDF}-render"

PATCH_CMDS='python /patches/patch_mtp_nightly.py; python /patches/patch_mtp_boundary.py'
[[ "$MTP_INT4" == "1" ]] && PATCH_CMDS="$PATCH_CMDS; python /patches/patch_draft_mtp_int4.py"
[[ "$R3" == "1" ]] && PATCH_CMDS="$PATCH_CMDS; python /r3/patch_r3_lmhead_int4.py"
[[ "$W3" == "1" ]] && PATCH_CMDS="$PATCH_CMDS; python /r9/patch_w3.py"

SERVE_ARGS=(vllm serve "/model-repo/snapshots/$MODEL_REV"
  --host 0.0.0.0 --port 8000
  --served-model-name "$SERVED"
  --quantization gptq --dtype float16
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$UTIL"
  --max-num-seqs "$MAX_SEQS")
[[ "$SPEC" != "0" ]] && SERVE_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC}}")

docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run --rm --name "$NAME" \
  --device "$RENDER" --ipc=host \
  -v "$BYPATH_DIR:/dev/dri/by-path:ro" \
  -v "$MODEL_REPO:/model-repo:ro" \
  -v "$PATCH_DIR:/patches:ro" -v "$R3_DIR:/r3:ro" -v "$R9_DIR:/r9:ro" \
  -v "$W3_DIR:/w3:ro" -v "$KERNEL_DIR:/kernel:ro" \
  -p "${PORT}:8000" \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e "P608_R3_LMHEAD_INT4=$R3" -e "B70_DRAFT_MTP_INT4=$MTP_INT4" \
  -e "P608_W3_DIR=$([[ "$W3" == "1" ]] && echo /w3 || echo '')" \
  -e "P608_W3_LIB=/kernel/libw3_l80.so" \
  -e "P608_W3_SKIP=$W3_SKIP" \
  --entrypoint bash "$IMAGE" -lc \
  "set -e; ${PATCH_CMDS}; exec $(printf '%q ' "${SERVE_ARGS[@]}")"
