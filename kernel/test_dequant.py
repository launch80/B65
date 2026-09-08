#!/usr/bin/env python3
"""The prefill dequant path is new. It must agree with the GEMV bit-for-bit, or
decode and prefill silently compute different functions of the same weights - the
kind of bug that shows up as a mysterious quality regression and nothing else.

Three checks:
  1. dequant_w3 vs an independent torch unpacking of the same bits
  2. gemv_w3(x) vs x @ dequant_w3(...)          - the two kernels agree
  3. both vs a reference built from the ORIGINAL codes, before packing
"""
import os, sys, torch, vllm  # noqa
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "research", "r9"))
sys.path.insert(0, "/r9")
from w3_pack_torch import pack_columns

H = os.path.dirname(os.path.abspath(__file__))
torch.ops.load_library(os.path.join(H, "libw3_l80.so"))
dev, G = "xpu", 128
ok = True
for (K, N) in [(512, 64), (5120, 256), (1024, 128)]:
    rng = np.random.default_rng(K + N)
    codes = torch.from_numpy(rng.integers(0, 8, size=(K, N)).astype(np.int64)).to(dev)
    s3 = (torch.rand(N, K // G, device=dev) * 0.02 + 0.005).half()
    qw = pack_columns(codes)

    # reference straight from the codes, never through the packed form
    ref_W = ((codes.float() - 4.0)
             * s3.t().float().repeat_interleave(G, dim=0))          # [K, N]

    W = torch.ops.p608w3.dequant_w3(qw, s3, K)                       # [N, K]
    torch.xpu.synchronize()
    e1 = (W.float().t() - ref_W).abs().max().item()

    x = (torch.randn(1, K, dtype=torch.float16, device=dev) * 0.05)
    g = torch.ops.p608w3.gemv_w3(x, qw, s3, False, 2)
    p = torch.nn.functional.linear(x, W)
    torch.xpu.synchronize()
    e2 = ((g.float() - p.float()).norm() / p.float().norm()).item()
    e3 = ((g.float() - (x.float() @ ref_W)).norm()
          / (x.float() @ ref_W).norm()).item()

    bad = (e1 > 1e-6) or (e2 > 3e-3) or (e3 > 3e-3)
    ok &= not bad
    print(f"  K={K:<5d} N={N:<5d}  dequant-vs-codes max abs {e1:.2e}   "
          f"gemv-vs-prefill {e2:.2e}   gemv-vs-ref {e3:.2e}  {'FAIL' if bad else 'ok'}")
print("\n  " + ("ALL PASS - decode and prefill compute the same function"
                if ok else "FAILED"))
raise SystemExit(0 if ok else 1)
