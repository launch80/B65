#!/usr/bin/env python3
"""Stage 7 - wire the Launch80 W3A16 kernel into vLLM.

Same shape as research/r3/patch_r3_lmhead_int4.py: write a helper into vLLM's
model_executor/models/ and add one idempotent call site, so the patch is re-runnable
and leaves the image otherwise untouched.

The 3-bit weights REPLACE the int4 ones rather than sitting beside them. Both
resident would be ~28 GB of 30, leaving nothing for KV cache. So each module's int4
qweight/scales are dropped as its 3-bit pair is loaded, module by module, to keep the
peak down.

Env:
  P608_W3_DIR   directory of w3-layers-*.safetensors from quantize_w3.py
  P608_W3_LIB   path to libw3_l80.so
  P608_W3_SKIP  comma-separated substrings of module names to leave at int4
"""
import os, sys

MARK = "P608_W3_HOOK"
HELPER = "p608_w3_linear.py"

SRC = '''\
"""Launch80 W3A16 linear method (Project 608, Stage 7)."""
import glob, os
import torch

_S = {"done": False}


class W3Method:
    """Wraps the module's original quant method. Decode (M=1) goes to the W3 GEMV;
    prefill dequantizes once and uses a normal GEMM, because at large M the problem
    is compute-bound and a batch-1 kernel is the wrong tool."""

    def __init__(self, orig):
        self._orig = orig

    def __getattr__(self, k):
        return getattr(self._orig, k)

    def apply(self, layer, x, bias=None):
        qw, sc, K = layer.w3_qweight, layer.w3_scales, layer.w3_K
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        if x2.shape[0] == 1:
            out = torch.ops.p608w3.gemv_w3(x2, qw, sc, False, 2)
        else:
            W = torch.ops.p608w3.dequant_w3(qw, sc, K)      # [N, K] fp16
            out = torch.nn.functional.linear(x2, W)
            del W
        if bias is not None:
            out = out + bias
        return out.reshape(*shp[:-1], out.shape[-1])


def install(model):
    if _S["done"]:
        return
    _S["done"] = True
    d = os.environ.get("P608_W3_DIR")
    lib = os.environ.get("P608_W3_LIB")
    if not d or not lib:
        print("[P608-W3] P608_W3_DIR / P608_W3_LIB unset - staying on int4", flush=True)
        return
    torch.ops.load_library(lib)
    from safetensors.torch import load_file

    skip = [s for s in os.environ.get("P608_W3_SKIP", "").split(",") if s]
    shards = sorted(glob.glob(os.path.join(d, "w3-layers-*.safetensors")))
    if not shards:
        print(f"[P608-W3] no shards in {d} - staying on int4", flush=True)
        return

    mods = {n: m for n, m in model.named_modules() if hasattr(m, "qweight")}
    dev = next(iter(mods.values())).qweight.device
    n_done, n_skip, mb = 0, 0, 0.0
    for shard in shards:
        t = load_file(shard)
        names = sorted({k.rsplit(".", 1)[0] for k in t if k.endswith(".qweight3")})
        for name in names:
            mod = mods.get(name)
            if mod is None:
                continue
            if any(s in name for s in skip):
                n_skip += 1
                continue
            qw = t[name + ".qweight3"].to(dev)
            sc = t[name + ".scales3"].to(dev)
            mod.w3_K = int(mod.qweight.shape[1]) * 8
            mod.w3_qweight, mod.w3_scales = qw, sc
            # drop the int4 pair now, not at the end, so the peak stays low
            for attr in ("qweight", "scales", "qzeros", "g_idx"):
                if hasattr(mod, attr):
                    try:
                        delattr(mod, attr)
                    except Exception:
                        setattr(mod, attr, None)
            mod.quant_method = W3Method(mod.quant_method)
            mb += (qw.numel() * 4 + sc.numel() * 2) / 1e6
            n_done += 1
        del t
    torch.xpu.empty_cache()
    print(f"[P608-W3] {n_done} linears on W3A16 ({mb/1000:.2f} GB), "
          f"{n_skip} left at int4", flush=True)
'''

OLD = ("    def compute_logits(\\n"
       "        self,\\n"
       "        hidden_states: torch.Tensor,\\n"
       "    ) -> torch.Tensor | None:\\n")
NEW = ("    def compute_logits(\\n"
       "        self,\\n"
       "        hidden_states: torch.Tensor,\\n"
       "    ) -> torch.Tensor | None:\\n"
       "        from vllm.model_executor.models.p608_w3_linear import install as _w3\\n"
       "        _w3(self)  # P608_W3_HOOK\\n")


def main():
    import vllm
    md = os.path.join(os.path.dirname(vllm.__file__), "model_executor", "models")
    open(os.path.join(md, HELPER), "w").write(SRC)
    p = os.path.join(md, "qwen3_5.py")
    t = open(p).read()
    if MARK in t:
        print("W3 hook already installed")
        return
    if OLD not in t:
        sys.exit("anchor not found in qwen3_5.py")
    t = t.replace(OLD, NEW, 1)
    compile(t, p, "exec")
    open(p, "w").write(t)
    print(f"W3 hook installed in {p}")


main()
