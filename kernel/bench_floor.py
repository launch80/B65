#!/usr/bin/env python3
"""Stage 0 - rebuild the measurement floor for the W3A16 project.

Every gate in the kernel plan is expressed relative to the vendor's rate, so that
rate has to be right. Two things were wrong with how we measured it before.

1. We quoted 394 GB/s, which is a PER-CALL number. It syncs around every
   invocation and so carries ~30 us of dispatch cost - visible in the 2.7 MB
   shape, which should take 5 us at the wall and took 37. The served model runs
   under VLLM_XPU_ENABLE_XPU_GRAPH=1, where most of that is gone. The honest
   reference is the amortised rate, which release/bench_gemv.py measured at 533.

2. But that amortised loop re-reads ONE 46 MB buffer 50 times against 18 MB of
   L2, so up to ~39% of it can be served from cache. docs/10 flags this and
   argues the 46 MB shapes are safe because they exceed any plausible L2. That
   is an argument, not a measurement. This settles it by rotating over enough
   distinct buffers that nothing can be resident.

The difference decides whether the project is comfortable or brittle:

    break-even for 3-bit = (34.8 MB / 46.0 MB) * V = 0.757 * V

    V = 440  ->  B = 333   comfortable
    V = 533  ->  B = 403   brittle

Reports all three protocols side by side so the gap is attributable.
"""
import argparse, time, torch, vllm  # noqa: F401  (importing vllm registers _xpu_C)

WALL = 587.0
L2_MB = 18.0
SHAPES = [
    (5120, 17408, 128, "mlp.gate/up_proj"),
    (6144,  5120,  64, "linear_attn.out_proj"),
    (17408, 5120,  64, "mlp.down_proj"),
    (5120, 10240,  48, "linear_attn.in_proj_qkv"),
    (5120,  6144,  48, "linear_attn.in_proj_z"),
    (5120,  1024,  32, "self_attn.k/v_proj"),
    (5120, 12288,  16, "self_attn.q_proj"),
]
G, dev = 128, "xpu"

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=50, help="calls per burst")
ap.add_argument("--best-of", type=int, default=8)
ap.add_argument("--footprint-mb", type=float, default=512.0,
                help="total distinct weight bytes to rotate over, per shape")
ap.add_argument("--only", default=None, help="substring filter on shape name")
ap.add_argument("--allow-any-device", action="store_true",
                help="skip the B65 identity check")
a = ap.parse_args()


def make_weights(K, N):
    """NT layout: the op requires strides[-2] == 1, i.e. a transposed view of a
    contiguous [N, K//8] buffer - the same shape the R3 packer produces."""
    qw = torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int32, device=dev).t()
    ws = (torch.randn(N, K // G, dtype=torch.float16, device=dev).abs() + 0.01).t().contiguous()
    return qw, ws


def time_burst(fn, reps, best_of):
    """One sync around a burst of `reps` launches. Returns best per-call seconds."""
    for _ in range(3):
        fn(0)
    torch.xpu.synchronize()
    best = float("inf")
    for _ in range(best_of):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for i in range(reps):
            fn(i)
        torch.xpu.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best / reps


# --- device guard -----------------------------------------------------------
# This machine has TWO Battlemage cards: a G21 (Arc B580, 0000:04:00.0) and the
# G31 target (0000:86:00.0). Exposing all of /dev/dri and setting
# ZE_AFFINITY_MASK=0 silently selects the B580, and a full run then reports
# plausible-looking numbers for the wrong hardware. Refuse to run unless the
# device matches, unless explicitly overridden.
DEV_NAME = torch.xpu.get_device_name(0)
props = torch.xpu.get_device_properties(0)
total_gb = getattr(props, "total_memory", 0) / 1e9
n_eu = getattr(props, "gpu_eu_count", None)
if not a.allow_any_device:
    if total_gb < 20:
        raise SystemExit(
            f"\nREFUSING TO RUN: device is '{DEV_NAME}' with {total_gb:.1f} GB.\n"
            f"The B65 (BMG-G31, 0000:86:00.0) has 32 GB. You have most likely\n"
            f"selected the B580 (BMG-G21, 0000:04:00.0) or the iGPU.\n"
            f"Pass --device /dev/dri/renderD130 to docker, not all of /dev/dri.\n"
            f"Override with --allow-any-device only if you mean it.\n")

print(f"device: {DEV_NAME}   {total_gb:.1f} GB"
      + (f"   {n_eu} EU" if n_eu else ""))
print(f"wall (P0 measured read): {WALL:.0f} GB/s   L2: {L2_MB:.0f} MB   "
      f"reps/burst: {a.reps}  best-of: {a.best_of}\n")
print("  protocols:  per-call = sync around each launch (carries dispatch cost)")
print("              hot      = one sync around a burst, ONE weight buffer reused")
print("              cold     = one sync around a burst, rotating distinct buffers\n")
print(f"  {'K x N':>16s} {'MB':>7s} {'bufs':>5s} | {'per-call':>9s} "
      f"{'hot GB/s':>9s} {'cold GB/s':>10s} {'%wall':>6s} | {'hot/cold':>8s} {'infl':>6s}")
print("  " + "-" * 96)

tot_b = tot_t_cold = tot_t_hot = 0.0
rows = []
for K, N, cnt, name in SHAPES:
    if a.only and a.only not in name:
        continue
    nbytes = (K // 8) * N * 4 + (K // G) * N * 2
    nbuf = max(2, int(a.footprint_mb * 1e6 / nbytes + 0.5))

    bufs = [make_weights(K, N) for _ in range(nbuf)]
    qz = torch.tensor([8], dtype=torch.int8, device=dev)
    x16 = torch.randn(1, K, dtype=torch.float16, device=dev)

    def call(i, _b=bufs, _x=x16, _z=qz):
        qw, ws = _b[i % len(_b)]
        torch.ops._xpu_C.int4_gemm_w4a16(_x, qw, None, ws, _z, G, None)

    def call_hot(i, _b=bufs, _x=x16, _z=qz):
        qw, ws = _b[0]
        torch.ops._xpu_C.int4_gemm_w4a16(_x, qw, None, ws, _z, G, None)

    # per-call: sync around every launch
    for _ in range(8):
        call_hot(0)
    torch.xpu.synchronize()
    per = float("inf")
    for _ in range(20):
        torch.xpu.synchronize(); t0 = time.perf_counter()
        call_hot(0)
        torch.xpu.synchronize(); per = min(per, time.perf_counter() - t0)

    t_hot = time_burst(call_hot, a.reps, a.best_of)
    t_cold = time_burst(call, a.reps, a.best_of)

    g_per, g_hot, g_cold = (nbytes / t / 1e9 for t in (per, t_hot, t_cold))
    tot_b += nbytes * cnt
    tot_t_cold += t_cold * cnt
    tot_t_hot += t_hot * cnt
    rows.append((K, N, cnt, name, nbytes, g_per, g_hot, g_cold))

    print(f"  {K:>6d}x{N:<7d} {nbytes/1e6:>7.1f} {nbuf:>5d} | {g_per:>9.1f} "
          f"{g_hot:>9.1f} {g_cold:>10.1f} {g_cold/WALL*100:>5.1f}% | "
          f"{g_hot/g_cold:>7.2f}x {(g_hot/g_cold-1)*100:>5.1f}%")
    del bufs

print("  " + "-" * 96)
agg_cold = tot_b / tot_t_cold / 1e9
agg_hot = tot_b / tot_t_hot / 1e9
print(f"  over all 400 body linears:  hot {agg_hot:.1f} GB/s   "
      f"cold {agg_cold:.1f} GB/s ({agg_cold/WALL*100:.0f}% of wall)")

ref = [r for r in rows if (r[0], r[1]) == (5120, 17408)]
if ref:
    V = ref[0][7]
    B = 0.757 * V
    print("\n" + "=" * 78)
    print(f"  GATE: vendor reference V on 5120x17408, cache-cold = {V:.1f} GB/s")
    print(f"        3-bit break-even  B = 0.757 * V           = {B:.1f} GB/s")
    print(f"        3-bit at V parity                          = {V:.1f} GB/s -> 1.32x on the body")
    print()
    for tgt in (B, 450, 500, V):
        ratio = (34.8 / 46.0) * (V / tgt)          # body time vs today
        e2e = 1.0 / (0.82 * ratio + 0.18)
        print(f"        3-bit at {tgt:>5.0f} GB/s -> body x{ratio:.3f} -> "
              f"end-to-end x{e2e:.3f} -> {77.2*e2e:.1f} tok/s")
    print("=" * 78)
    if agg_hot / agg_cold > 1.10:
        print(f"  !! hot overstates by {(agg_hot/agg_cold-1)*100:.0f}% - the old 533 was "
              f"cache-inflated and every gate derived from it was too high.")
    else:
        print(f"  hot and cold agree to {(agg_hot/agg_cold-1)*100:.1f}% - "
              f"the 46 MB shapes really do exceed L2, as docs/10 argued.")
