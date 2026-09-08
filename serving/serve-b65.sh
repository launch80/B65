#!/usr/bin/env bash
# Daily driver for the Arc Pro B65 — direct vLLM, no router.
#
# Replaces llama-swap (dropped 2026-09-06). Runs the current stack proven in
# P0.5: vLLM 0.27.2rc1 + kernels 0.1.12.3 + XPU graph capture, which is faster
# than the old 0.21.0-xpu + --enforce-eager path it supersedes.
#
#   ./serve-b65.sh              # foreground
#   SPEC=4 ./serve-b65.sh       # with MTP speculative decoding
set -euo pipefail

IMAGE="${IMAGE:-vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f}"
MODEL_REPO="${MODEL_REPO:-$HOME/.cache/huggingface/hub/models--SergiioB--Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16}"
MODEL_REV="${MODEL_REV:-e28c5f952bdd5d814297a07d85a064a87af26a3f}"
SERVED="${SERVED:-SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16}"
PATCHES="${PATCHES:-$HOME/Documents/project608/parity/patches}"
PORT="${PORT:-8000}"
NAME="${NAME:-vllm-b65}"
GPU_BDF="${GPU_BDF:-0000:86:00.0}"
MAX_LEN="${MAX_LEN:-32768}"
UTIL="${UTIL:-0.92}"
MAX_SEQS="${MAX_SEQS:-32}"     # a serving value, not the benchmark's 1
SPEC="${SPEC:-0}"
# R3: shared lm_head -> INT4 g128. Gated 2026-09-07: perplexity +0.840% (<1%),
# GSM8K 90.0 vs 88.0 (McNemar chi2 0.250, not significant). Worth +36.8% at MTP4.
R3="${R3:-1}"
# Phase M1 (SergiioB cookbook): the MTP draft layer's own linears, 0.849 GB BF16,
# read once per draft pass. INT4 -> 0.219 GB. Draft-only, so the target still
# verifies every token and no quality gate is needed. Worth +10.4% at MTP4.
# Re-enabled after isolation: the garbage output on very short prompts is an MTP
# speculative-decoding bug, present with the STOCK fp16 head and no M1 at all.
# Verified: MTP4+R3+M1 reproduces the no-speculation greedy output on 7/8 normal
# prompts (the 8th diverges late on a near-tie). Worth +10.4%.
MTP_INT4="${MTP_INT4:-1}"
# R6: the DRAFT reads only the first N vocabulary rows of lm_head. qweight is
# [K/8, V] with stride (1, K/8), so the first N columns are a contiguous memory
# prefix - a slice, not a gather. Draft-only, so the target still verifies every
# token: this costs ACCEPTANCE, never correctness. 0.656 -> 0.087 GB per draft.
# Measured: 95.9% of true argmaxes fall inside the first 32768 rows.
DRAFT_VOCAB="${DRAFT_VOCAB:-32768}"
# GDN mixed spec + non-spec split. The fused torch.ops._xpu_C.gdn_attention host
# binding REFUSES a batch that mixes spec-decode tokens with prefill/decode tokens:
#   "causal_conv1d does not support spec-decode and non-spec (prefill + decode)
#    tokens in the same invocation"
# That is a hard TORCH_CHECK and it kills EngineCore, after which every request gets
# a connection error. It never fires on single-stream benchmarks, which is why this
# was missed - it needs a concurrent client where one sequence prefills while others
# spec-decode, i.e. any real agent harness. The cookbook flags it as Cn correctness,
# C1 speed-flat, so there is no reason not to run it whenever SPEC is on.
GDN_V5="${GDN_V5:-1}"
# XPU graph capture is OFF by default as of 2026-09-08. Measured on this image
# (vLLM 0.27.2, inductor compile always on): capture changes the target pass by
# 0.0% (32.26 vs 32.26 ms) and slows the draft slope ~1%. Off is +2% end to end,
# repeated twice. The +13% the ladder credits to XPU_GRAPH=1 was measured on 0.21
# with --enforce-eager, where compile was off; it does not apply here.
PATCH_DIR="${PATCH_DIR:-$HOME/Documents/project608/parity/patches}"
R3_DIR="${R3_DIR:-$HOME/Documents/project608/research/r3}"
BYPATH_DIR="${BYPATH_DIR:-$HOME/.cache/vllm-dri-by-path}"

RENDER="$(readlink -f "/dev/dri/by-path/pci-${GPU_BDF}-render" 2>/dev/null || true)"
[[ -n "$RENDER" && -e "$RENDER" ]] || { echo "ERROR: no render node for ${GPU_BDF}" >&2; exit 1; }
rm -rf "$BYPATH_DIR"; mkdir -p "$BYPATH_DIR"
ln -s "../$(basename "$RENDER")" "$BYPATH_DIR/pci-${GPU_BDF}-render"

# Tool calling needs BOTH flags or clients sending tool_choice:"auto" get a hard 400.
# The parser must be qwen3_xml (Qwen3.8 emits XML-nested calls), never hermes.
PATCH_CMDS='python /patches/patch_mtp_nightly.py; python /patches/patch_mtp_boundary.py'
[[ "$MTP_INT4" == "1" ]] && PATCH_CMDS="$PATCH_CMDS; python /patches/patch_draft_mtp_int4.py"
[[ "$GDN_V5" == "1" && "$SPEC" != "0" ]] && PATCH_CMDS="$PATCH_CMDS; python /patches/patch_gdn_mixed_split_v5.py"
[[ "$R3" == "1" ]] && PATCH_CMDS="$PATCH_CMDS; python /r3/patch_r3_lmhead_int4.py"

SERVE_ARGS=(vllm serve "/model-repo/snapshots/$MODEL_REV"
  --host 0.0.0.0 --port 8000
  --served-model-name "$SERVED"
  --quantization gptq --dtype float16
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$UTIL"
  --kv-cache-dtype fp8 --max-num-seqs "$MAX_SEQS"
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3)
[[ "$SPEC" != "0" ]] && SERVE_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC}}")

docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run --rm --name "$NAME" \
  --device "$RENDER" --ipc=host \
  -v "$BYPATH_DIR:/dev/dri/by-path:ro" \
  -v "$MODEL_REPO:/model-repo:ro" \
  -v "$PATCHES:/patches:ro" \
  -v "$R3_DIR:/r3:ro" \
  -p "${PORT}:8000" \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 -e "VLLM_XPU_ENABLE_XPU_GRAPH=${XPU_GRAPH:-0}" \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e "P608_R3_LMHEAD_INT4=$R3" \
  -e "P608_DRAFT_VOCAB=$DRAFT_VOCAB" \
  -e "B70_DRAFT_MTP_INT4=$MTP_INT4" \
  --entrypoint bash "$IMAGE" -lc \
  "set -e; ${PATCH_CMDS}; exec $(printf '%q ' "${SERVE_ARGS[@]}")"
