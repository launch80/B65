#!/usr/bin/env python3
"""Stage 4: Launch80 W3A16 vs Intel's W4A16, same session, same protocol.

The win condition is NOT a bandwidth ratio - it is TIME. 3-bit moves 34.82 MB where
the vendor's 4-bit moves 46.0 MB on the same shape, so:

    break-even rate  B = (34.82/46.0) * V = 0.757 * V
    at equal GB/s    3-bit is 1.32x faster than the vendor

Both are measured here in one process, cache-cold with rotated buffers, because the
absolute vendor number moves with card temperature (Stage 0 measured 473 on a cold
card, 527 warm) while the ratio does not.
"""
import argparse, time, os, torch, vllm  # noqa: F401
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w3_format import pack, PAIR, SPARE

H = os.path.dirname(os.path.abspath(__file__))
torch.ops.load_library(os.path.join(H, "libw4_l80.so"))
torch.ops.load_library(os.path.join(H, "libw3_l80.so"))

dev, G = "xpu", 128
ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--footprint-mb", type=float, default=512.0)
a = ap.parse_args()

props = torch.xpu.get_device_properties(0)
if props.total_memory < 20e9:
    raise SystemExit("REFUSING: not the B65. Use --device /dev/dri/renderD130.")
print(f"device: {torch.xpu.get_device_name(0)}   {props.total_memory/1e9:.1f} GB\n")


def pack_planes(codes):
    """codes uint8 [K, N] in 0..7 -> int32 [N, 3*K/32] with the three-plane layout."""
    K, N = codes.shape
    nblk = K // 32
    # [N, nblk, 32] : block b of column n holds weights k = 32b..32b+31
    c = codes.T.reshape(N, nblk, 32)
    w = pack(c.reshape(-1, 32)).reshape(N, nblk, 3)      # -> 3 words per block
    planes = np.concatenate([w[:, :, 0], w[:, :, 1], w[:, :, 2]], axis=1)  # [N, 3*nblk]
    return torch.from_numpy(planes.astype(np.int32)).to(dev).contiguous()


SHAPES = [(5120, 17408, 128), (17408, 5120, 64), (5120, 10240, 48),
          (5120, 12288, 16), (6144, 5120, 64), (5120, 6144, 48)]
print(f"  {'K x N':>16s} | {'4b MB':>6s} {'3b MB':>6s} | {'rel err':>9s} | "
      f"{'vendor':>8s} {'W3 GB/s':>8s} | {'us v':>7s} {'us W3':>7s} {'SPEEDUP':>8s} {'no-strag':>8s}")
print("  " + "-" * 92)

rows = []
for K, N, cnt in SHAPES:
    b4 = (K // 8) * N * 4 + (K // G) * N * 2
    b3 = (K // 32) * 3 * N * 4 + (K // G) * N * 2
    nbuf = max(2, int(a.footprint_mb * 1e6 / b4 + 0.5))
    rng = np.random.default_rng(608 + K + N)

    vend, w3 = [], []
    ref_deq = None
    for j in range(nbuf):
        qw = torch.randint(-2**31, 2**31-1, (N, K//8), dtype=torch.int32, device=dev).t()
        ws = (torch.randn(N, K//G, dtype=torch.float16, device=dev).abs()+0.01).t().contiguous()
        vend.append((qw, ws))
        codes = rng.integers(0, 8, size=(K, N)).astype(np.uint8)
        s3 = (torch.randn(N, K//G, dtype=torch.float16, device=dev).abs()*0.02+0.005).contiguous()
        w3.append((pack_planes(codes), s3))
        if j == 0:
            cf = torch.from_numpy(codes.astype(np.int16)).to(dev).float() - 4.0
            sf = s3.t().float().repeat_interleave(G, dim=0)      # [K, N]
            ref_deq = (cf * sf).half()

    qz = torch.tensor([8], dtype=torch.int8, device=dev)
    x = torch.randn(1, K, dtype=torch.float16, device=dev) * 0.05

    got = torch.ops.p608w3.gemv_w3(x, w3[0][0], w3[0][1], False, 2)
    ref = (x.float() @ ref_deq.float())
    torch.xpu.synchronize()
    err = ((got.float()-ref).norm()/ref.norm()).item()
    del ref_deq, cf, sf

    def bench(fn):
        for i in range(5): fn(i)
        torch.xpu.synchronize()
        best = float("inf")
        for _ in range(8):
            torch.xpu.synchronize(); t0 = time.perf_counter()
            for i in range(a.reps): fn(i)
            torch.xpu.synchronize()
            best = min(best, (time.perf_counter()-t0)/a.reps)
        return best

    tv = bench(lambda i: torch.ops._xpu_C.int4_gemm_w4a16(
        x, vend[i%nbuf][0], None, vend[i%nbuf][1], qz, G, None))
    t3 = bench(lambda i: torch.ops.p608w3.gemv_w3(x, w3[i%nbuf][0], w3[i%nbuf][1], False, 2))
    # ablation: same loads, straggler math removed. Numerically wrong on 2 of every
    # 32 weights - purely to bound how much of the gap that path still owns.
    tns = bench(lambda i: torch.ops.p608w3.gemv_w3(x, w3[i%nbuf][0], w3[i%nbuf][1], True, 2))
    gv, g3 = b4/tv/1e9, b3/t3/1e9
    rows.append((K, N, cnt, b4, b3, gv, g3, tv, t3, err, tns))
    flag = "" if err < 2e-3 else "  <-- ERR"
    print(f"  {K:>6d}x{N:<7d} | {b4/1e6:>6.1f} {b3/1e6:>6.1f} | {err:>9.2e} | "
          f"{gv:>8.1f} {g3:>8.1f} | {tv*1e6:>7.1f} {t3*1e6:>7.1f} {tv/t3:>7.2f}x"
          f" {b3/tns/1e9:>8.1f}{flag}")
    del vend, w3

print("  " + "-" * 92)
tv_ = sum(t*c for _,_,c,_,_,_,_,t,_,_,_ in rows)
t3_ = sum(t*c for _,_,c,_,_,_,_,_,t,_,_ in rows)
b4_ = sum(b*c for _,_,c,b,_,_,_,_,_,_,_ in rows)
b3_ = sum(b*c for _,_,c,_,b,_,_,_,_,_,_ in rows)
print(f"  over the 400-linear mix: vendor {tv_*1e3:.2f} ms ({b4_/tv_/1e9:.0f} GB/s)   "
      f"W3 {t3_*1e3:.2f} ms ({b3_/t3_/1e9:.0f} GB/s)   SPEEDUP {tv_/t3_:.2f}x")

r = [t for t in rows if (t[0],t[1])==(5120,17408)][0]
V, W, B = r[5], r[6], 0.757*r[5]
print(f"\n{'='*80}")
print(f"  GATE (5120x17408)   vendor V = {V:.1f} GB/s   break-even B = {B:.1f} GB/s")
print(f"    W3 rate      {W:.1f} GB/s   {'PASS' if W>=B else 'FAIL'}  (need >= {B:.0f})")
print(f"    W3 time      {r[8]*1e6:.1f} us vs vendor {r[7]*1e6:.1f} us -> {r[7]/r[8]:.2f}x")
print(f"    correctness  rel err {r[9]:.2e}  {'PASS' if r[9]<2e-3 else 'FAIL'}")
e2e = 1.0/(0.82*(t3_/tv_)+0.18)
print(f"\n  body {t3_/tv_:.3f}x of vendor -> end-to-end {e2e:.3f}x -> "
      f"{77.2*e2e:.1f} tok/s from 77.2")
print(f"{'='*80}")
