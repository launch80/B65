#!/usr/bin/env python3
"""Project 608 · R3 step 1 — INT4 the TARGET lm_head, not just the draft's.

Why this is not the cookbook's Phase S
--------------------------------------
SergiioB's patch_draft_lmhead_int4.py quantizes the *draft's* lm_head and leaves
the target fp16, so verification stays bit-exact. It relies on the draft owning a
private copy (`DraftModelProposer._maybe_share_lm_head` being a no-op). On our
build that is false — vLLM logs "Sharing target model lm_head weights with the
draft model" — which is why their patch has nothing to quantize and hangs.

That same sharing is what makes this patch cheap: ONE INT4 copy, keyed by the
shared weight's storage pointer, serves the target and the draft alike. Under
MTP4 the fp16 head is read 5x per step (4 drafts + 1 verify) = 12.7 GB/step at
~98% of DRAM peak. Routing all five through INT4 g128 takes that to ~3.3 GB/step.

The cost, stated plainly: the verify pass now reads INT4 too, so output is NOT
bit-identical to fp16. That is deliberate — it is the open ground the cookbook
explicitly does not touch — and it is why R3's gate is KL divergence against the
fp16 head rather than top-1 agreement. Gate before you ship.

Quantizer and packing format are adapted from SergiioB's Phase S helper (GPTQ
INT4 g128 sym, the layout torch.ops._xpu_C.int4_gemm_w4a16 already consumes).

Env gate: P608_R3_LMHEAD_INT4=1   (default off = stock behaviour)
"""
from __future__ import annotations

import os
import sys

MARKER = "P608_R3_LMHEAD_INT4"
HELPER_MODULE = "p608_r3_lmhead_int4.py"

HELPER_SOURCE = '''\
"""Project 608 R3 runtime helper: shared lm_head -> INT4 g128 sym.

One quantized copy is cached per underlying fp16 storage, so when vLLM shares
lm_head between the target and the MTP draft both paths hit the same tensors.
Quantizer adapted from SergiioB's B70 cookbook Phase S helper.
"""
from __future__ import annotations

import os

import torch

# storage data_ptr -> (qweight, scales, qzeros, group_size)
_CACHE: dict[int, tuple] = {}
_PREFIX: dict = {}
_FAILED: set[int] = set()


def quantize_lmhead_to_int4(weight: torch.Tensor, group_size: int = 128):
    """fp16 [N, K] -> GPTQ INT4 g128 sym in int4_gemm_w4a16 layout.

    qweight int32 [K//8, N] NT, nibbles LSB-first, stored = q + 8
    scales  fp16  [K//group_size, N]
    qzeros  int8  [8]  -> symmetric branch
    """
    device = weight.device
    N, K = weight.shape
    num_groups = K // group_size
    chunk = 4096
    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28],
                          dtype=torch.int32, device=device)
    parts, scale_parts = [], []
    for i in range(0, N, chunk):
        wc = weight[i:i + chunk].float()
        wg = wc.view(wc.shape[0], num_groups, group_size)
        scale = wg.abs().amax(dim=-1) / 7.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = (wg / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int32)
        qv = (q + 8).view(wc.shape[0], num_groups, group_size // 8, 8)
        parts.append((qv << shifts).sum(dim=-1).to(torch.int32)
                     .reshape(wc.shape[0], K // 8))
        scale_parts.append(scale.half())
    qweight = torch.cat(parts, dim=0).t()                 # [K//8, N]
    scales = torch.cat(scale_parts, dim=0).t().contiguous()  # [g, N]
    qzeros = torch.tensor([8], dtype=torch.int8, device=device)
    return qweight, scales, qzeros, group_size


def int4_lmhead_logits(x, qweight, scales, qzeros, group_size):
    flat = x.reshape(-1, x.shape[-1])
    out = torch.ops._xpu_C.int4_gemm_w4a16(
        flat, qweight, None, scales, qzeros, group_size, None)
    return out.reshape(*x.shape[:-1], qweight.shape[1])


@torch.no_grad()
def _build(weight: torch.Tensor):
    key = weight.untyped_storage().data_ptr()
    if key in _CACHE:
        return _CACHE[key]
    if key in _FAILED:
        return None
    N, K = weight.shape
    if K % 128 or weight.dtype not in (torch.float16, torch.bfloat16):
        print(f"[P608-R3] lm_head {tuple(weight.shape)} {weight.dtype} "
              "unsupported; staying fp16", flush=True)
        _FAILED.add(key)
        return None
    print(f"[P608-R3] quantizing SHARED lm_head {tuple(weight.shape)} "
          "-> INT4 g128 sym (one-time)", flush=True)
    try:
        packed = quantize_lmhead_to_int4(weight.detach())
    except Exception as exc:                      # noqa: BLE001
        print(f"[P608-R3] quantization failed ({exc}); staying fp16", flush=True)
        _FAILED.add(key)
        return None
    qw, sc = packed[0], packed[1]
    fp16_b = weight.numel() * weight.element_size()
    int4_b = qw.numel() * qw.element_size() + sc.numel() * sc.element_size()
    print(f"[P608-R3] lm_head ready: {fp16_b/1e9:.3f} GB fp16 -> "
          f"{int4_b/1e9:.3f} GB INT4 ({fp16_b/int4_b:.2f}x, "
          f"{(fp16_b-int4_b)/1e9:.3f} GB saved per read)", flush=True)
    _CACHE[key] = packed
    return packed


def _build_prefix(weight, n):
    """A draft-only head over the first n vocabulary rows.

    qweight is [K//8, V] with strides (1, K//8), so columns 0..n-1 are a CONTIGUOUS
    prefix of memory - the draft reads a smaller slice of the same tensor rather than
    gathering scattered rows. Scales are re-packed contiguously once.
    """
    key = (weight.untyped_storage().data_ptr(), n)
    if key in _PREFIX:
        return _PREFIX[key]
    full = _build(weight)
    if full is None:
        _PREFIX[key] = None
        return None
    qw, sc, qz, g = full
    pq = qw[:, :n].contiguous().t().contiguous().t()   # keep NT layout, strides[-2]==1
    ps = sc[:, :n].contiguous()
    _PREFIX[key] = (pq, ps, qz, g)
    b = pq.numel()*pq.element_size() + ps.numel()*ps.element_size()
    fb = qw.numel()*qw.element_size() + sc.numel()*sc.element_size()
    print(f"[P608-R6] draft head truncated to first {n} of {qw.shape[1]} rows: "
          f"{fb/1e9:.3f} -> {b/1e9:.3f} GB per draft pass "
          f"({fb/b:.1f}x, {(fb-b)/1e9:.3f} GB saved x4/step)", flush=True)
    return _PREFIX[key]


def r3_draft_logits(module, hidden_states):
    """Draft-only path. The target verifies every proposed token, so a truncated
    vocabulary here costs ACCEPTANCE, never correctness - which is why this is
    applied to the MTP draft and never to the target."""
    n = int(os.environ.get("P608_DRAFT_VOCAB", "0"))
    if n <= 0:
        return None
    head = getattr(module, "lm_head", None)
    pc = getattr(getattr(module, "vllm_config", None), "parallel_config", None)
    if (getattr(pc, "tensor_parallel_size", 1) or 1) > 1:
        return None
    weight = getattr(head, "weight", None)
    qw_loaded = getattr(head, "qweight", None)
    if os.environ.get("P608_R6_DEBUG") == "1" and not getattr(module, "_r6_dbg", False):
        module._r6_dbg = True
        print(f"[P608-R6-DEBUG] head={type(head).__name__} weight={None if weight is None else tuple(weight.shape)} "
              f"qweight={None if qw_loaded is None else tuple(qw_loaded.shape)} n={n} "
              f"attrs={[a for a in ('weight','qweight','scales','qzeros','g_idx') if hasattr(head,a)]}", flush=True)
    if weight is not None:
        full_v = weight.shape[0]
        if n >= full_v:
            return None
        packed = _build_prefix(weight, n)
    elif qw_loaded is not None:
        # P608_R6_QUANTIZED_HEAD: the head is already GPTQ on disk (the baked
        # checkpoint). vLLM's INC wna16 method leaves it as qweight [K/8, V] with
        # strides (1, K/8), scales [K/G, V], qzeros = tensor([8]) - the layout
        # _build() would have produced - so slice it instead of quantizing fp16.
        sc_loaded = head.scales                         # [K/G, V]
        full_v = sc_loaded.shape[1]
        K8 = qw_loaded.numel() // full_v                 # K/8
        if n >= full_v:
            return None
        key = (qw_loaded.untyped_storage().data_ptr(), n)
        if key not in _PREFIX:
            # The head's qweight is a [V, K/8] row-major tensor as loaded by the INC
            # method (measured: shape (248320, 640)), so the first n vocabulary rows
            # are a contiguous prefix; transposing that slice gives the [K/8, n] view
            # with strides (1, K/8) that int4_gemm_w4a16 expects. Handle the
            # already-transposed [K/8, V] view too, in case a future vLLM changes it.
            if tuple(qw_loaded.shape) == (full_v, K8):
                pq = qw_loaded[:n].contiguous().t()
            else:
                pq = qw_loaded[:, :n].contiguous().t().contiguous().t()
            ps = sc_loaded[:, :n].contiguous()
            qz = torch.tensor([8], dtype=torch.int8, device=qw_loaded.device)
            gs = (K8 * 8) // sc_loaded.shape[0]
            _PREFIX[key] = (pq, ps, qz, gs)
            b = pq.numel() * 4 + ps.numel() * 2
            fb = qw_loaded.numel() * 4 + sc_loaded.numel() * 2
            print(f"[P608-R6] draft head (INT4 on disk, qweight {tuple(qw_loaded.shape)} stride {tuple(qw_loaded.stride())}) "
                  f"truncated to first {n} of {full_v} rows: {fb/1e9:.3f} -> {b/1e9:.3f} GB per draft pass "
                  f"({fb/b:.1f}x); prefix view {tuple(pq.shape)} stride {tuple(pq.stride())}", flush=True)
        packed = _PREFIX[key]
    else:
        return None
    if packed is None:
        return None
    logits = int4_lmhead_logits(hidden_states, *packed)
    # Pad back to full width with -inf so downstream shape checks and argmax work.
    lp = getattr(module, "logits_processor", None)
    org = getattr(lp, "org_vocab_size", None) or full_v
    out = torch.full((*logits.shape[:-1], org), float("-inf"),
                     dtype=logits.dtype, device=logits.device)
    out[..., :min(n, org)] = logits[..., :min(n, org)]
    scale = getattr(lp, "scale", 1.0)
    if scale != 1.0:
        out = out * scale
    return out


def r3_logits(module, hidden_states):
    """INT4 logits, or None to fall through to the stock fp16 path."""
    if os.environ.get("P608_R3_LMHEAD_INT4") != "1":
        return None
    head = getattr(module, "lm_head", None)
    weight = getattr(head, "weight", None)
    if weight is None:
        return None
    pc = getattr(getattr(module, "vllm_config", None), "parallel_config", None)
    tp = getattr(pc, "tensor_parallel_size", 1)
    if tp and tp > 1:                     # fail closed: TP splits the vocab
        return None
    packed = _build(weight)
    if packed is None:
        return None
    logits = int4_lmhead_logits(hidden_states, *packed)
    lp = getattr(module, "logits_processor", None)
    org = getattr(lp, "org_vocab_size", None)
    if org is not None and logits.shape[-1] > org:
        logits = logits[..., :org]
    scale = getattr(lp, "scale", 1.0)
    if scale != 1.0:
        logits = logits * scale
    return logits
'''

TARGET_OLD = (
    "    def compute_logits(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor | None:\n"
    "        return self.logits_processor(self.lm_head, hidden_states)\n"
)
TARGET_NEW = (
    "    def compute_logits(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor | None:\n"
    "        from vllm.model_executor.models.p608_r3_lmhead_int4 import r3_logits\n"
    "        _r3 = r3_logits(self, hidden_states)\n"
    "        if _r3 is not None:\n"
    "            return _r3\n"
    "        # P608_R3_LMHEAD_INT4\n"
    "        return self.logits_processor(self.lm_head, hidden_states)\n"
)
DRAFT_OLD = (
    "    def compute_logits(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "        spec_step_idx: int = 0,\n"
    "    ) -> torch.Tensor | None:\n"
    "        return self.logits_processor(self.lm_head, hidden_states)\n"
)
DRAFT_NEW = (
    "    def compute_logits(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "        spec_step_idx: int = 0,\n"
    "    ) -> torch.Tensor | None:\n"
    "        from vllm.model_executor.models.p608_r3_lmhead_int4 import (\n"
    "            r3_logits, r3_draft_logits)\n"
    "        _rd = r3_draft_logits(self, hidden_states)\n"
    "        if _rd is not None:\n"
    "            return _rd\n"
    "        _r3 = r3_logits(self, hidden_states)\n"
    "        if _r3 is not None:\n"
    "            return _r3\n"
    "        # P608_R3_LMHEAD_INT4\n"
    "        return self.logits_processor(self.lm_head, hidden_states)\n"
)


def _write_helper(vllm_dir: str) -> None:
    path = os.path.join(vllm_dir, "model_executor", "models", HELPER_MODULE)
    if os.path.exists(path) and open(path).read() == HELPER_SOURCE:
        print(f"helper already present {path}")
        return
    open(path, "w").write(HELPER_SOURCE)
    print(f"helper written {path}")


def _patch(vllm_dir: str, fname: str, old: str, new: str, label: str) -> None:
    path = os.path.join(vllm_dir, "model_executor", "models", fname)
    if not os.path.exists(path):
        sys.exit(f"{fname} not found at {path}")
    text = open(path).read()
    if MARKER in text:
        print(f"already patched {label} ({fname})")
        return
    if old not in text:
        sys.exit(f"anchor not found in {fname}: compute_logits ({label})")
    text = text.replace(old, new, 1)
    compile(text, path, "exec")
    open(path, "w").write(text)
    print(f"patched {label} ({fname})")


def main() -> None:
    import vllm
    vllm_dir = os.path.dirname(vllm.__file__)
    _write_helper(vllm_dir)
    _patch(vllm_dir, "qwen3_5.py", TARGET_OLD, TARGET_NEW, "TARGET verify head")
    _patch(vllm_dir, "qwen3_5_mtp.py", DRAFT_OLD, DRAFT_NEW, "MTP draft head")


if __name__ == "__main__":
    main()
