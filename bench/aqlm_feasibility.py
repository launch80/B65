#!/usr/bin/env python3
"""R2, second attempt: can AQLM-class methods reach 2 bits where plain PQ could not?

Plain product quantization gave 31% output error at 2 bits/weight. AQLM beats naive PQ
with two ingredients, and this isolates each so we learn WHICH one matters here:

  1. ACTIVATION-AWARE objective. Minimise ||X(W-W')||, not ||W-W'||. Real activations are
     anisotropic - some input channels carry far more energy - so uniform weight error
     spends bits in directions the layer never uses. Implemented as per-input-channel
     scaling by the measured second moment, which is exactly the reparameterisation that
     turns a weighted least-squares into an unweighted one.

  2. ADDITIVE codebooks. w = C1[i] + C2[j] instead of a single lookup. Two 256-entry
     codebooks over 8-dim groups is the same 2 bits/weight as one codebook over 4-dim
     groups, but the representable set is a sum of two, which is vastly larger.

Everything is scored on REAL captured activations, against the dequantized GPTQ-Int4
weight the server actually multiplies by today.
"""
import argparse, json, os, struct, time, torch

ap = argparse.ArgumentParser()
ap.add_argument("--shard", required=True)
ap.add_argument("--acts", required=True)
ap.add_argument("--layer", default="model.language_model.layers.9.mlp.gate_proj")
ap.add_argument("--iters", type=int, default=12)
a = ap.parse_args()
dev = "xpu"

with open(a.shard, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(n)); base = fh.tell()
    def get(name):
        m = hdr[name]; s, e = m["data_offsets"]; fh.seek(base + s)
        dt = {"I32": torch.int32, "F16": torch.float16}[m["dtype"]]
        return torch.frombuffer(bytearray(fh.read(e - s)), dtype=dt).reshape(m["shape"])
    qw = get(a.layer + ".qweight").to(dev)
    sc = get(a.layer + ".scales").to(dev)
    gi = get(a.layer + ".g_idx").to(dev)

K, N, G = qw.shape[0] * 8, qw.shape[1], 128
shifts = torch.arange(8, device=dev, dtype=torch.int32) * 4
nib = ((qw.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF).reshape(K, N)
W = ((nib.to(torch.float16) - 8.0) * sc[gi.long()]).float()          # [K, N]
X = torch.load(a.acts, map_location="cpu").to(dev).float()[:, :K]     # [rows, K]
bad = torch.isnan(X).any(1) | torch.isinf(X).any(1)
if int(bad.sum()):
    print(f"  dropped {int(bad.sum())} non-finite activation row(s) of {X.shape[0]}")
    X = X[~bad]
ref = X @ W
print(f"  weight {tuple(W.shape)}   activations {tuple(X.shape)} (REAL, captured)")

# how anisotropic are the activations? this is what ingredient 1 exploits
e = (X * X).mean(0)
print(f"  per-channel energy: max/median = {(e.max()/e.median()).item():,.0f}x, "
      f"top 1% of channels hold {e.sort(descending=True).values[:K//100].sum()/e.sum()*100:.1f}% of it")


def kmeans(Xs, ncent, iters):
    g = torch.Generator(device="cpu").manual_seed(608)
    C = Xs[torch.randperm(Xs.shape[0], generator=g)[:ncent].to(dev)].clone()
    CH = max(1 << 15, (1 << 28) // ncent)
    for _ in range(iters):
        idx = torch.empty(Xs.shape[0], dtype=torch.long, device=dev)
        for s in range(0, Xs.shape[0], CH):
            idx[s:s+CH] = torch.cdist(Xs[s:s+CH], C).argmin(1)
        newC = torch.zeros_like(C); cnt = torch.zeros(ncent, device=dev)
        newC.index_add_(0, idx, Xs); cnt.index_add_(0, idx, torch.ones(Xs.shape[0], device=dev))
        alive = cnt > 0
        newC[alive] /= cnt[alive].unsqueeze(1); newC[~alive] = C[~alive]
        C = newC
    return C, idx


@torch.no_grad()
def quantize(Wm, sub, ncent, books, iters):
    """books=1 plain PQ, books=2 additive. Returns reconstruction."""
    Kd, Nd = Wm.shape
    V = Wm.t().reshape(-1, sub)               # subvectors along the input dim
    recon = torch.zeros_like(V)
    resid = V.clone()
    for _ in range(books):
        C, idx = kmeans(resid, ncent, iters)
        recon += C[idx]
        resid = V - recon
    return recon.reshape(Nd, Kd).t()


print(f"\n  {'method':>34s} {'bits/w':>7s} {'output rel err':>15s}")
print("  " + "-" * 60)
sqrt_e = e.clamp_min(1e-12).sqrt()            # per-input-channel importance
Ws = W * sqrt_e.unsqueeze(1)                  # reparameterise: uniform error here == weighted error there

CONFIGS = [
    ("plain PQ, 1 codebook",        4, 256, 1, False),
    ("+ activation-aware",          4, 256, 1, True),
    ("additive, 2 codebooks",       8, 256, 2, False),
    ("+ activation-aware  (AQLM-like)", 8, 256, 2, True),
    ("additive 2cb @ 3 bits",       8, 4096, 2, True),
]
for label, sub, nc, books, aware in CONFIGS:
    t0 = time.time()
    if aware:
        Wq = quantize(Ws, sub, nc, books, a.iters) / sqrt_e.unsqueeze(1)
    else:
        Wq = quantize(W, sub, nc, books, a.iters)
    err = ((X @ Wq - ref).norm() / ref.norm()).item()
    bits = books * torch.log2(torch.tensor(float(nc))).item() / sub
    print(f"  {label:>34s} {bits:>7.2f} {err*100:>14.2f}%   ({time.time()-t0:.0f}s)")
print("  " + "-" * 60)
print(f"  {'GPTQ-Int4 (what we serve today)':>34s} {4.125:>7.2f} {0.0:>14.2f}%   reference")
