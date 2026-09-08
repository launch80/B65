#!/usr/bin/env python3
"""W4A8 (dp4a) vs W4A16 (ours) vs Intel's int4_gemm_w4a16 - one process, one protocol.

Reports two errors, because they are different questions:
  kernel err  - vs a reference using the SAME quantized activations. Tests the kernel.
  a8 err      - vs the fp16 reference. This is the cost of quantizing activations, and
                it is a quality change that would need a gate before shipping.
"""
import argparse, time, os, torch, vllm  # noqa
H = os.path.dirname(os.path.abspath(__file__))
torch.ops.load_library(os.path.join(H, "libw4_l80.so"))
torch.ops.load_library(os.path.join(H, "libw4a8_l80.so"))
dev, G = "xpu", 128
ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--ncol", type=int, default=2); a = ap.parse_args()

PERM = [0, 16, 4, 20, 8, 24, 12, 28]
def repack(qw_nt):
    src = qw_nt.t().contiguous(); out = torch.zeros_like(src)
    for j, p in enumerate(PERM):
        out |= ((src >> (4 * j)) & 0xF) << p
    return out

def prep_x(xf, K):
    """int8 activations, pre-interleaved into the even/odd nibble order dp4a needs."""
    xs = (xf.abs().max() / 127.0).item()
    xi = torch.clamp(torch.round(xf / xs), -127, 127).to(torch.int8).reshape(K)
    w = xi.reshape(K // 8, 8).to(torch.int32)
    def pack4(t):
        t = t & 0xFF
        return (t[:, 0] | (t[:, 1] << 8) | (t[:, 2] << 16) | (t[:, 3] << 24))
    even = pack4(w[:, [0, 2, 4, 6]]); odd = pack4(w[:, [1, 3, 5, 7]])
    xq = torch.stack([even, odd], 1).reshape(-1).to(torch.int32).contiguous()
    sumx = xi.reshape(K // 32, 32).to(torch.int32).sum(1).to(torch.int32).contiguous()
    return xq, sumx, xs, xi

print(f"device: {torch.xpu.get_device_name(0)}   NCOL={a.ncol}\n")
print(f"  {'K x N':>15s} | {'kernel err':>10s} {'a8 err':>8s} | {'vendor':>7s} "
      f"{'W4A16':>7s} {'W4A8':>7s} | {'us v':>6s} {'us a8':>6s} {'SPEEDUP':>8s}")
print("  " + "-" * 92)
rows = []
for K, N, cnt in [(5120,17408,128),(17408,5120,64),(5120,10240,48),
                  (5120,12288,16),(6144,5120,64),(5120,6144,48)]:
    nbytes = (K//8)*N*4 + (K//G)*N*2
    nbuf = max(2, int(512e6/nbytes+0.5))
    vend, ours, plain = [], [], []
    for _ in range(nbuf):
        qw = torch.randint(-2**31,2**31-1,(N,K//8),dtype=torch.int32,device=dev).t()
        ws = (torch.randn(N,K//G,dtype=torch.float16,device=dev).abs()+.01).t().contiguous()
        vend.append((qw, ws))
        ours.append((repack(qw), ws.t().contiguous()))
        # dp4a needs NO repacking: & 0x0F0F0F0F already selects nibbles
        # 0,2,4,6 in the stock GPTQ order. Only the fp16 half2 path needs
        # the PERM shuffle.
        plain.append((qw.t().contiguous(), ws.t().contiguous()))
    qz = torch.tensor([8],dtype=torch.int8,device=dev)
    xf = torch.randn(1,K,dtype=torch.float16,device=dev)*0.05
    xq, sumx, xs, xi = prep_x(xf, K)

    ref16 = torch.ops._xpu_C.int4_gemm_w4a16(xf, vend[0][0], None, vend[0][1], qz, G, None)
    got8 = torch.ops.p608a8.gemv_w4a8(xq, sumx, plain[0][0], plain[0][1], xs, a.ncol)
    # reference using the SAME quantized activations, to isolate kernel correctness
    refq = torch.ops._xpu_C.int4_gemm_w4a16(
        (xi.float()*xs).half().reshape(1,K), vend[0][0], None, vend[0][1], qz, G, None)
    torch.xpu.synchronize()
    ek = ((got8.float()-refq.float()).norm()/refq.float().norm()).item()
    ea = ((got8.float()-ref16.float()).norm()/ref16.float().norm()).item()

    def bench(fn):
        for i in range(5): fn(i)
        torch.xpu.synchronize(); best=1e30
        for _ in range(8):
            torch.xpu.synchronize(); t0=time.perf_counter()
            for i in range(a.reps): fn(i)
            torch.xpu.synchronize(); best=min(best,(time.perf_counter()-t0)/a.reps)
        return best
    tv = bench(lambda i: torch.ops._xpu_C.int4_gemm_w4a16(xf,vend[i%nbuf][0],None,vend[i%nbuf][1],qz,G,None))
    t16 = bench(lambda i: torch.ops.p608.gemv_w4(xf,ours[i%nbuf][0],ours[i%nbuf][1],a.ncol))
    t8 = bench(lambda i: torch.ops.p608a8.gemv_w4a8(xq,sumx,plain[i%nbuf][0],plain[i%nbuf][1],xs,a.ncol))
    # W4A8 moves fewer activation bytes but the same weight bytes; weights dominate
    rows.append((K,N,cnt,nbytes,tv,t16,t8))
    print(f"  {K:>6d}x{N:<7d} | {ek:>10.2e} {ea:>8.2e} | {nbytes/tv/1e9:>7.1f} "
          f"{nbytes/t16/1e9:>7.1f} {nbytes/t8/1e9:>7.1f} | {tv*1e6:>6.1f} {t8*1e6:>6.1f} "
          f"{tv/t8:>7.2f}x")
    del vend, ours, plain
print("  " + "-" * 92)
tv_=sum(t*c for _,_,c,_,t,_,_ in rows); t16_=sum(t*c for _,_,c,_,_,t,_ in rows)
t8_=sum(t*c for _,_,c,_,_,_,t in rows); b_=sum(b*c for _,_,c,b,_,_,_ in rows)
print(f"  over the 400-linear mix:  vendor {tv_*1e3:.2f} ms ({b_/tv_/1e9:.0f} GB/s)")
print(f"                            W4A16  {t16_*1e3:.2f} ms ({b_/t16_/1e9:.0f} GB/s)  {tv_/t16_:.2f}x")
print(f"                            W4A8   {t8_*1e3:.2f} ms ({b_/t8_/1e9:.0f} GB/s)  {tv_/t8_:.2f}x")
print(f"\n  read ceiling for this pattern: 579.4 GB/s -> {b_/579.4e9*1e3:.2f} ms floor")
