#!/usr/bin/env python3
"""Optimised Triton int4 GEMV: load x once per group, one reduction per group."""
import itertools, time, torch, triton, triton.language as tl
import vllm  # noqa
DEV = "xpu"


@triton.jit
def gemv(X, QW, SC, OUT, K, N, stride_qn, stride_sk,
         BLOCK_N: tl.constexpr, W8: tl.constexpr, NGROUP: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    r8 = tl.arange(0, W8)
    r_j = tl.arange(0, 8)
    for g in range(0, NGROUP):
        k8 = g * W8 + r8
        w = tl.load(QW + k8[:, None] + offs_n[None, :] * stride_qn,
                    mask=mask_n[None, :], other=0)                    # [W8, BLOCK_N]
        # x for this group, as [W8, 8] - ONE global load, reshaped in registers
        xk = (k8[:, None] * 8 + r_j[None, :])
        xv = tl.load(X + xk, mask=xk < K, other=0.0).to(tl.float32)   # [W8, 8]
        part = tl.zeros((W8, BLOCK_N), dtype=tl.float32)
        for j in tl.static_range(8):
            nib = ((w >> (4 * j)) & 0xF).to(tl.float32) - 8.0
            part += nib * tl.sum(tl.where(r_j[None, :] == j, xv, 0.0), axis=1)[:, None]
        s = tl.load(SC + g * stride_sk + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += tl.sum(part, axis=0) * s                               # ONE reduction
    tl.store(OUT + offs_n, acc.to(tl.float16), mask=mask_n)


def run(x, qw, sc, BN, nw, stages):
    K, N = qw.shape[0]*8, qw.shape[1]
    out = torch.empty((1, N), dtype=torch.float16, device=DEV)
    gemv[(triton.cdiv(N, BN),)](x, qw, sc, out, K, N, qw.stride(1), sc.stride(0),
                                BLOCK_N=BN, W8=16, NGROUP=K//128,
                                num_warps=nw, num_stages=stages)
    return out


def make(K, N, G=128):
    qw = torch.randint(-2**31, 2**31-1, (N, K//8), dtype=torch.int32, device=DEV).t()
    sc = (torch.randn(N, K//G, dtype=torch.float16, device=DEV).abs()*0.02+0.005).t().contiguous()
    return (torch.randn(1, K, dtype=torch.float16, device=DEV)*0.05, qw, sc,
            torch.tensor([8], dtype=torch.int8, device=DEV))


def bench(fn, reps=30, warm=8):
    for _ in range(warm): fn()
    torch.xpu.synchronize(); best = 1e9
    for _ in range(reps):
        torch.xpu.synchronize(); t0=time.perf_counter(); fn(); torch.xpu.synchronize()
        best = min(best, time.perf_counter()-t0)
    return best


K, N = 5120, 17408
x, qw, sc, qz = make(K, N)
nb = qw.numel()*4 + sc.numel()*2
ref = torch.ops._xpu_C.int4_gemm_w4a16(x, qw, None, sc, qz, 128, None)
tv = bench(lambda: torch.ops._xpu_C.int4_gemm_w4a16(x, qw, None, sc, qz, 128, None))
print(f"  vendor int4_gemm_w4a16: {tv*1e6:7.1f} us  {nb/tv/1e9:6.1f} GB/s   (target)\n")
print(f"  {'BLOCK_N':>8s} {'warps':>6s} {'stages':>7s} {'us':>9s} {'GB/s':>8s} {'vs vendor':>10s} {'err':>9s}")
print("  " + "-"*62)
best = (None, 1e9)
for BN, nw, st in itertools.product((64,128,256), (4,8,16), (1,2,3)):
    try:
        got = run(x, qw, sc, BN, nw, st)
        err = ((got.float()-ref.float()).norm()/ref.float().norm()).item()
        t = bench(lambda: run(x, qw, sc, BN, nw, st))
    except Exception as e:
        continue
    if t < best[1]: best = ((BN,nw,st), t)
    if t < tv*3:
        print(f"  {BN:>8d} {nw:>6d} {st:>7d} {t*1e6:>8.1f} {nb/t/1e9:>8.1f} {tv/t:>9.2f}x {err:>9.1e}")
print("  " + "-"*62)
print(f"  best: BLOCK_N={best[0][0]} warps={best[0][1]} stages={best[0][2]} -> "
      f"{nb/best[1]/1e9:.1f} GB/s ({tv/best[1]:.2f}x vendor)")
