#!/usr/bin/env python3
"""R2 feasibility: can product quantization get the body below 4 bits/weight?

The roadmap's headline: the body is 12.703 GB/token, 92.8% of the byte budget, and
P2 showed the GEMV that reads it is already at 93% of the memory wall. So the only
large lever left is making those bytes fewer. R2 proposes 8-bit indices into a
256-entry codebook of 4-dim subvectors = 2 bits/weight, decoded with surplus compute.

This measures whether the accuracy survives, BEFORE any kernel work. The reference is
the DEQUANTIZED GPTQ-Int4 weight, because that is what the server actually multiplies
by today - not some fp16 original we do not have.
"""
import argparse, json, struct, time, torch

ap = argparse.ArgumentParser()
ap.add_argument("--shard", required=True)
ap.add_argument("--layer", default="model.language_model.layers.9.mlp.gate_proj")
ap.add_argument("--iters", type=int, default=15)
ap.add_argument("--probe", type=int, default=64, help="activation rows for output error")
a = ap.parse_args()
dev = "xpu"

# ---- read the GPTQ tensors --------------------------------------------------
with open(a.shard, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(n))
    base = fh.tell()

    def get(name):
        m = hdr[name]
        s, e = m["data_offsets"]
        fh.seek(base + s)
        raw = fh.read(e - s)
        dt = {"I32": torch.int32, "F16": torch.float16}[m["dtype"]]
        return torch.frombuffer(bytearray(raw), dtype=dt).reshape(m["shape"])

    qw = get(a.layer + ".qweight").to(dev)      # [K/8, N] int32
    sc = get(a.layer + ".scales").to(dev)       # [K/G, N] fp16
    gi = get(a.layer + ".g_idx").to(dev)        # [K]

K, N, G = qw.shape[0] * 8, qw.shape[1], 128
print(f"  {a.layer}")
print(f"  K={K} N={N} group={G}  ({K*N/1e6:.1f}M weights)")

# ---- dequantize: sym, zero = 8 ---------------------------------------------
shifts = torch.arange(8, device=dev, dtype=torch.int32) * 4
nib = ((qw.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF).reshape(K, N)   # [K, N]
W = ((nib.to(torch.float16) - 8.0) * sc[gi.long()])                     # [K, N] fp16
print(f"  dequantized: {tuple(W.shape)} {W.dtype}, "
      f"|W| mean {W.abs().mean().item():.5f}, max {W.abs().max().item():.4f}")
bits_int4 = 4 + (16 / G)          # 4 bits + fp16 scale per 128
print(f"  GPTQ-Int4 costs {bits_int4:.3f} bits/weight "
      f"({K*N*bits_int4/8/1e6:.1f} MB for this matrix)")

# ---- product quantization ---------------------------------------------------
@torch.no_grad()
def pq_fit(W, sub, ncent, iters):
    """Split each row into sub-dim chunks; one shared codebook per matrix."""
    Kd, Nd = W.shape
    X = W.t().reshape(-1, sub).float()                 # [N*K/sub, sub]
    g = torch.Generator(device="cpu").manual_seed(608)
    C = X[torch.randperm(X.shape[0], generator=g)[:ncent].to(dev)].clone()
    # cdist allocates chunk x ncent floats; scale the chunk so that stays bounded
    CH = max(1 << 15, (1 << 28) // ncent)
    for it in range(iters):
        idx = torch.empty(X.shape[0], dtype=torch.long, device=dev)
        for s in range(0, X.shape[0], CH):
            idx[s:s+CH] = torch.cdist(X[s:s+CH], C).argmin(1)
        newC = torch.zeros_like(C); cnt = torch.zeros(ncent, device=dev)
        newC.index_add_(0, idx, X); cnt.index_add_(0, idx, torch.ones(X.shape[0], device=dev))
        alive = cnt > 0
        newC[alive] /= cnt[alive].unsqueeze(1)
        newC[~alive] = C[~alive]
        C = newC
    Wq = C[idx].reshape(Nd, Kd).t().to(torch.float16)
    bits = (torch.log2(torch.tensor(float(ncent))) / sub).item()
    cb_bytes = ncent * sub * 2
    return Wq, bits, cb_bytes

@torch.no_grad()
def pq_fit_scaled(W, sub, ncent, iters, group=128):
    """PQ on per-group-normalised weights.

    Plain PQ has to spend codebook entries representing magnitude, which varies a
    lot across a weight matrix. Dividing each 128-element group by its max-abs
    first - exactly what GPTQ-Int4 does - leaves PQ to encode only SHAPE, and the
    scale is stored separately at 16 bits per 128 weights, the same overhead Int4
    already pays.
    """
    Kd, Nd = W.shape
    Wg = W.float().reshape(Kd // group, group, Nd)
    s2 = Wg.abs().amax(1, keepdim=True).clamp_min(1e-8)
    Wn = (Wg / s2).reshape(Kd, Nd)
    Wq, bits, cb = pq_fit(Wn, sub, ncent, iters)
    Wr = (Wq.float().reshape(Kd // group, group, Nd) * s2).reshape(Kd, Nd)
    return Wr.to(torch.float16), bits + 16.0 / group, cb


x = torch.randn(a.probe, K, device=dev, dtype=torch.float16)
ref = (x.float() @ W.float())
print(f"\n  {'config':>22s} {'bits/w':>7s} {'MB':>8s} {'vs Int4':>8s} "
      f"{'W rel err':>10s} {'out rel err':>12s}")
print("  " + "-" * 76)
for sub, ncent, label in [(2, 256, "PQ 2-dim x 256"), (4, 4096, "PQ 4-dim x 4096"),
                          (4, 256, "PQ 4-dim x 256"), (8, 256, "PQ 8-dim x 256")]:
    t0 = time.time()
    Wq, bits, cb = pq_fit(W, sub, ncent, a.iters)
    werr = ((Wq.float() - W.float()).norm() / W.float().norm()).item()
    out = (x.float() @ Wq.float())
    oerr = ((out - ref).norm() / ref.norm()).item()
    mb = K * N * bits / 8 / 1e6 + cb / 1e6
    print(f"  {label:>22s} {bits:>7.2f} {mb:>8.1f} {mb/(K*N*bits_int4/8/1e6):>7.2f}x "
          f"{werr:>10.4f} {oerr:>12.4f}   ({time.time()-t0:.0f}s)")

# Int4's own round-trip error is zero by construction here (it IS the reference),
# so give the reader a scale: what a plain 3-bit and 2-bit RTN would cost.
print("  " + "-" * 76)
for sub, ncent, label in [(2, 256, "scaled PQ 2d x 256"), (4, 4096, "scaled PQ 4d x 4096"),
                          (4, 256, "scaled PQ 4d x 256"), (8, 256, "scaled PQ 8d x 256")]:
    Wq, bits, cb = pq_fit_scaled(W, sub, ncent, a.iters)
    werr = ((Wq.float() - W.float()).norm() / W.float().norm()).item()
    oerr = (((x.float() @ Wq.float()) - ref).norm() / ref.norm()).item()
    mb = K * N * bits / 8 / 1e6 + cb / 1e6
    print(f"  {label:>22s} {bits:>7.2f} {mb:>8.1f} {mb/(K*N*bits_int4/8/1e6):>7.2f}x "
          f"{werr:>10.4f} {oerr:>12.4f}")

print("  " + "-" * 76)
for b in (3, 2):
    lv = 2 ** (b - 1) - 1
    Wg = W.float().reshape(K // G, G, N)
    s2 = Wg.abs().amax(1, keepdim=True) / lv
    Wr = ((Wg / s2).round().clamp(-lv - 1, lv) * s2).reshape(K, N)
    werr = ((Wr - W.float()).norm() / W.float().norm()).item()
    oerr = (((x.float() @ Wr) - ref).norm() / ref.norm()).item()
    print(f"  {'RTN ' + str(b) + '-bit g128':>22s} {b + 16/G:>7.2f} "
          f"{K*N*(b+16/G)/8/1e6:>8.1f} {(b+16/G)/bits_int4:>7.2f}x "
          f"{werr:>10.4f} {oerr:>12.4f}")
