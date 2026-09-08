#!/usr/bin/env python3
"""P0 — measure the B65's achievable memory bandwidth.

A STREAM-style benchmark on real kernels, so the answer is what the memory system
actually delivers rather than the 608 GB/s datasheet figure.

HONEST LIMITATION: this is torch-XPU, not hand-written SYCL. The pinned vLLM image
ships no SYCL compiler (no icpx/dpcpp), and pulling the oneAPI base toolkit to hand-
tune load widths is a bigger job than this measurement needs. What that costs: no
control over vectorization, work-group shape or cache hints, so these numbers are a
LOWER BOUND on the hardware - a floor on the wall, which is exactly what P0 needs to
decide whether kernel work has anything left to give.

Bytes counted the same way the project's byte budgets are: decimal GB (1e9), matching
the safetensors-header arithmetic behind the 434 GB/s anchor.
"""
import argparse, time, torch

ap = argparse.ArgumentParser()
ap.add_argument("--gib", type=float, default=2.0, help="buffer size per array")
ap.add_argument("--reps", type=int, default=12)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--dtype", default="float16")
a = ap.parse_args()

dev = "xpu"
assert torch.xpu.is_available(), "no XPU"
p = torch.xpu.get_device_properties(0)
dt = getattr(torch, a.dtype)
esz = torch.tensor([], dtype=dt).element_size()
n = int(a.gib * (1 << 30) // esz)
nbytes = n * esz

print(f"device      : {torch.xpu.get_device_name(0)}")
print(f"total mem   : {p.total_memory/2**30:.2f} GiB")
print(f"buffer      : {nbytes/1e9:.3f} GB  ({n:,} x {a.dtype})")
print(f"reps        : {a.reps} (best-of, {a.warmup} warmup)\n")

A = torch.randn(n, device=dev, dtype=dt)
B = torch.randn(n, device=dev, dtype=dt)
C = torch.empty(n, device=dev, dtype=dt)
torch.xpu.synchronize()

def bench(fn, traffic, label):
    for _ in range(a.warmup):
        fn()
    torch.xpu.synchronize()
    best = float("inf"); times = []
    for _ in range(a.reps):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        dt_ = time.perf_counter() - t0
        times.append(dt_); best = min(best, dt_)
    gbs = traffic / best / 1e9
    med = traffic / sorted(times)[len(times)//2] / 1e9
    print(f"  {label:26s} {gbs:7.1f} GB/s   (median {med:7.1f})")
    return gbs

print("kernel                        best        ")
print("-" * 52)
res = {}
res["read (sum)"]      = bench(lambda: A.sum(),            nbytes,     "read   : sum(A)")
res["copy"]            = bench(lambda: C.copy_(A),         2*nbytes,   "copy   : C = A")
res["scale"]           = bench(lambda: torch.mul(A, 2.0, out=C), 2*nbytes, "scale  : C = 2*A")
res["add"]             = bench(lambda: torch.add(A, B, out=C),   3*nbytes, "add    : C = A+B")
res["triad"]           = bench(lambda: torch.add(A, B, alpha=2.0, out=C), 3*nbytes, "triad  : C = A+2*B")
# (A*B).sum() materialises an N-element temporary: reads A, reads B, writes tmp,
# reads tmp = 4N, not 2N. Counting it as 2N understates it by half.
res["dot"]             = bench(lambda: (A*B).sum(),        4*nbytes,   "dot    : sum(A*B) [4N]")

# Decode reads PACKED int4 weights: contiguous, read-only, one pass. The int32
# view models that element width without changing the byte traffic.
Ai = A.view(torch.int32) if esz == 2 else A
res["read int32 view"] = bench(lambda: Ai.sum(dtype=torch.int64), nbytes, "read   : sum(A as int32)")

# Strided read - the pessimistic end: what happens when access is not contiguous.
S = 2
res["strided read x2"] = bench(lambda: A[::S].sum(), nbytes // S, f"read   : A[::{S}] strided")

print("-" * 52)
peak = max(res.values())
print(f"  {'PEAK MEASURED':26s} {peak:7.1f} GB/s")
print(f"  {'datasheet':26s} {608.0:7.1f} GB/s   ({peak/608*100:.0f}% achieved)")
print(f"  {'project anchor (decode)':26s} {434.0:7.1f} GB/s   ({434/peak*100:.0f}% of measured peak)")
