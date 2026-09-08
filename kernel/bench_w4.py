#!/usr/bin/env python3
"""Stage 2 kill gate: our W4A16 GEMV vs Intel's, same data, same protocol.

Gate: rel err < 1e-3 against torch.ops._xpu_C.int4_gemm_w4a16, AND rate >= 0.90*V
where V is the vendor's cache-cold burst rate measured here in the same process.

Correctness is checked against the vendor op rather than a CPU reference because
the vendor is what we must replace; if we match it to fp16 rounding on real data,
the repack and the kernel are both right.
"""
import argparse, time, torch, vllm  # noqa: F401
import os

torch.ops.load_library(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "libw4_l80.so"))
dev, G = "xpu", 128
ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--footprint-mb", type=float, default=512.0)
ap.add_argument("--ncol", type=int, default=2)
a = ap.parse_args()
NCOL = a.ncol

props = torch.xpu.get_device_properties(0)
if props.total_memory < 20e9:
    raise SystemExit(f"REFUSING: {props.total_memory/1e9:.1f} GB device, not the B65. "
                     "Pass --device /dev/dri/renderD130.")
print(f"device: {torch.xpu.get_device_name(0)}   {props.total_memory/1e9:.1f} GB   NCOL={NCOL}\n")

# Our nibble permutation: weight 2t -> bit 4t, weight 2t+1 -> bit 16+4t.
# So a (w >> 4t) & 0x000F000F lands the pair (w_2t, w_2t+1) in the two fp16 halves.
PERM = [0, 16, 4, 20, 8, 24, 12, 28]


def repack(qw_nt):
    """vendor [K/8, N] NT (contiguous buffer [N, K/8]) -> our [N, K/8] contiguous."""
    src = qw_nt.t().contiguous()                      # [N, K/8], nibble j at bit 4j
    out = torch.zeros_like(src)
    for j, p in enumerate(PERM):
        nib = (src >> (4 * j)) & 0xF
        out |= nib << p
    return out


SHAPES = [(5120, 17408, 128), (17408, 5120, 64), (5120, 10240, 48),
          (5120, 12288, 16), (6144, 5120, 64), (5120, 6144, 48)]

print(f"  {'K x N':>16s} {'MB':>7s} | {'rel err':>9s} | {'vendor':>9s} {'ours':>9s} "
      f"{'ratio':>7s} {'%wall':>7s}")
print("  " + "-" * 78)

tot = []
for K, N, cnt in SHAPES:
    nbytes = (K // 8) * N * 4 + (K // G) * N * 2
    nbuf = max(2, int(a.footprint_mb * 1e6 / nbytes + 0.5))

    vend, ours = [], []
    for _ in range(nbuf):
        qw = torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int32, device=dev).t()
        ws = (torch.randn(N, K // G, dtype=torch.float16, device=dev).abs() + 0.01).t().contiguous()
        vend.append((qw, ws))
        ours.append((repack(qw), ws.t().contiguous()))
    qz = torch.tensor([8], dtype=torch.int8, device=dev)
    x = torch.randn(1, K, dtype=torch.float16, device=dev)

    ref = torch.ops._xpu_C.int4_gemm_w4a16(x, vend[0][0], None, vend[0][1], qz, G, None)
    got = torch.ops.p608.gemv_w4(x, ours[0][0], ours[0][1], NCOL)
    torch.xpu.synchronize()
    err = ((got.float() - ref.float()).norm() / ref.float().norm()).item()

    def bench(fn):
        for i in range(5):
            fn(i)
        torch.xpu.synchronize()
        best = float("inf")
        for _ in range(8):
            torch.xpu.synchronize(); t0 = time.perf_counter()
            for i in range(a.reps):
                fn(i)
            torch.xpu.synchronize()
            best = min(best, (time.perf_counter() - t0) / a.reps)
        return best

    tv = bench(lambda i: torch.ops._xpu_C.int4_gemm_w4a16(
        x, vend[i % nbuf][0], None, vend[i % nbuf][1], qz, G, None))
    to = bench(lambda i: torch.ops.p608.gemv_w4(x, ours[i % nbuf][0], ours[i % nbuf][1], NCOL))
    gv, go = nbytes / tv / 1e9, nbytes / to / 1e9
    tot.append((K, N, cnt, nbytes, gv, go, err))
    flag = "" if err < 1e-3 else "   <-- ERR"
    print(f"  {K:>6d}x{N:<7d} {nbytes/1e6:>7.1f} | {err:>9.2e} | {gv:>9.1f} {go:>9.1f} "
          f"{go/gv:>6.2f}x {go/587*100:>6.1f}%{flag}")

print("  " + "-" * 78)
bw = sum(b * c for _, _, c, b, _, _, _ in tot)
tv_ = sum(b * c / (g * 1e9) for _, _, c, b, g, _, _ in tot)
to_ = sum(b * c / (g * 1e9) for _, _, c, b, _, g, _ in tot)
print(f"  weighted over the 400-linear mix: vendor {bw/tv_/1e9:.1f} GB/s   "
      f"ours {bw/to_/1e9:.1f} GB/s   ratio {tv_/to_:.2f}x")

K, N = 5120, 17408
r = [t for t in tot if (t[0], t[1]) == (K, N)][0]
V, O, E = r[4], r[5], r[6]
print(f"\n{'='*72}")
print(f"  GATE (5120x17408):  vendor V = {V:.1f}   ours = {O:.1f}   ratio {O/V:.3f}")
print(f"    correctness  rel err {E:.2e}  {'PASS' if E < 1e-3 else 'FAIL'} (need < 1e-3)")
print(f"    rate         {O/V:.3f} x V      {'PASS' if O >= 0.90*V else 'FAIL'} (need >= 0.90)")
if O >= V:
    print(f"\n  WE ARE FASTER THAN THE VENDOR AT 4 BITS: {O/V:.2f}x")
print(f"{'='*72}")
