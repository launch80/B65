#!/usr/bin/env python3
"""P2/R1 — int4 GEMV at M=1 on the model's real shapes, W4A16 vs W4A8, against P0's wall.

Two questions at once:
  1. Is the incumbent kernel the reason a decode step gets 434 GB/s when the memory
     system delivers 587?
  2. R1's headline was "delete the dequantize step" via a W4A8 path. But
     `_xpu_C.int4_gemm_w4a8` ALREADY EXISTS in vllm-xpu-kernels 0.1.12.3. So the
     question is no longer whether to build it - it is whether it is faster here.

Shapes come from the checkpoint headers: qweight [K//8, N] int32, scales [K/G, N].
"""
import time, torch, vllm  # noqa: F401  (importing vllm registers the _xpu_C ops)

WALL = 587.0
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


def timeit(fn, reps=30, warm=8):
    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    best = float("inf")
    for _ in range(reps):
        torch.xpu.synchronize(); t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize(); best = min(best, time.perf_counter() - t0)
    return best


print(f"device: {torch.xpu.get_device_name(0)}   wall (P0 read): {WALL:.0f} GB/s\n")
print(f"  {'K x N':>16s} {'MB':>7s} | {'W4A16 us':>9s} {'GB/s':>7s} {'%wall':>6s} |"
      f" {'W4A8 us':>8s} {'GB/s':>7s} {'%wall':>6s} | {'A8/A16':>7s}")
print("  " + "-" * 92)

tb16 = tt16 = tb8 = tt8 = 0.0
for K, N, cnt, name in SHAPES:
    # NT layout: the op requires strides[-2] == 1, i.e. a transposed view of a
    # contiguous [N, K//8] buffer - the same shape the R3 packer produces.
    qw = torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int32, device=dev).t()
    ws = (torch.randn(N, K // G, dtype=torch.float16, device=dev).abs() + 0.01).t().contiguous()
    qz = torch.tensor([8], dtype=torch.int8, device=dev)
    nbytes = qw.numel() * 4 + ws.numel() * 2

    x16 = torch.randn(1, K, dtype=torch.float16, device=dev)
    t16 = timeit(lambda: torch.ops._xpu_C.int4_gemm_w4a16(x16, qw, None, ws, qz, G, None))
    g16 = nbytes / t16 / 1e9

    # W4A8: int8 activations with their own scale
    x8 = torch.randint(-127, 127, (1, K), dtype=torch.int8, device=dev)
    xs = torch.ones(1, dtype=torch.float16, device=dev)
    xz = torch.zeros(1, dtype=torch.int8, device=dev)
    wz = torch.tensor([8], dtype=torch.int8, device=dev)
    try:
        t8 = timeit(lambda: torch.ops._xpu_C.int4_gemm_w4a8(x8, xs, xz, qw, ws, wz, G, None, None))
        g8 = nbytes / t8 / 1e9
        a8 = f"{t16/t8:.2f}x"
    except Exception as e:
        t8, g8, a8 = float("nan"), float("nan"), f"ERR {str(e)[:14]}"

    tb16 += nbytes * cnt; tt16 += t16 * cnt
    if g8 == g8:
        tb8 += nbytes * cnt; tt8 += t8 * cnt
    print(f"  {K:>6d}x{N:<7d} {nbytes/1e6:>7.1f} | {t16*1e6:>8.1f} {g16:>7.1f} "
          f"{g16/WALL*100:>5.1f}% | {t8*1e6:>7.1f} {g8:>7.1f} {g8/WALL*100:>5.1f}% | {a8:>7s}")

print("  " + "-" * 92)
a16 = tb16 / tt16 / 1e9
print(f"  W4A16 over all 400 body linears: {tb16/1e9:.3f} GB in {tt16*1e3:.2f} ms "
      f"= {a16:.1f} GB/s ({a16/WALL*100:.0f}% of wall)")
if tt8:
    a8g = tb8 / tt8 / 1e9
    print(f"  W4A8  over all 400 body linears: {tb8/1e9:.3f} GB in {tt8*1e3:.2f} ms "
          f"= {a8g:.1f} GB/s ({a8g/WALL*100:.0f}% of wall)")
print(f"\n  whole decode step sustains 434 GB/s (74% of wall); step time 1/31.7 = 31.5 ms")
print(f"  body GEMVs alone account for {tt16*1e3:.2f} ms of that")

# ---------------------------------------------------------------------------
# The numbers above are per-call, so each carries dispatch latency. The small
# 5120x1024 shape exposes the floor: 2.7 MB should take ~5 us at the wall and
# takes 37, so roughly 30 us is fixed cost. In the served model graph capture
# removes most of that. Re-measure with the sync OUTSIDE a run of back-to-back
# calls, which is what the captured graph actually does, to separate KERNEL
# efficiency from LAUNCH cost.
print("\n" + "=" * 78)
print("  amortised: 50 back-to-back calls, one sync (what graph capture approximates)")
print(f"  {'K x N':>16s} {'MB':>7s} | {'per-call us':>12s} {'GB/s':>8s} {'%wall':>7s} | "
      f"{'vs isolated':>12s}")
print("  " + "-" * 76)
tb = tt = 0.0
for K, N, cnt, name in SHAPES:
    qw = torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int32, device=dev).t()
    ws = (torch.randn(N, K // G, dtype=torch.float16, device=dev).abs() + 0.01).t().contiguous()
    qz = torch.tensor([8], dtype=torch.int8, device=dev)
    x16 = torch.randn(1, K, dtype=torch.float16, device=dev)
    nbytes = qw.numel() * 4 + ws.numel() * 2
    R = 50

    def burst():
        for _ in range(R):
            torch.ops._xpu_C.int4_gemm_w4a16(x16, qw, None, ws, qz, G, None)

    for _ in range(3):
        burst()
    torch.xpu.synchronize()
    best = float("inf")
    for _ in range(8):
        torch.xpu.synchronize(); t0 = time.perf_counter()
        burst()
        torch.xpu.synchronize(); best = min(best, time.perf_counter() - t0)
    per = best / R
    gbs = nbytes / per / 1e9
    tb += nbytes * cnt; tt += per * cnt
    iso = timeit(lambda: torch.ops._xpu_C.int4_gemm_w4a16(x16, qw, None, ws, qz, G, None), reps=10)
    print(f"  {K:>6d}x{N:<7d} {nbytes/1e6:>7.1f} | {per*1e6:>11.1f} {gbs:>8.1f} "
          f"{gbs/WALL*100:>6.1f}% | {iso/per:>11.2f}x")
print("  " + "-" * 76)
agg = tb / tt / 1e9
print(f"  amortised over all 400 body linears: {tb/1e9:.3f} GB in {tt*1e3:.2f} ms "
      f"= {agg:.1f} GB/s ({agg/WALL*100:.0f}% of wall)")
print(f"\n  => launch cost in the model is bounded by the isolated-vs-amortised gap.")
print(f"     Body pass amortised: {tt*1e3:.2f} ms of the 31.5 ms step, "
      f"leaving {31.5-tt*1e3:.2f} ms for lm_head + GDN + KV + attention.")
