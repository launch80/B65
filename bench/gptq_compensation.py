#!/usr/bin/env python3
"""Does ERROR COMPENSATION rescue low-bit quantization where better codebooks did not?

Naive product quantization at 2 bits gives ~27% output error, and neither of AQLM's
representational ingredients - activation weighting, additive codebooks - moved it.
That points at the optimiser rather than the representation.

The optimiser trick every strong method shares (GPTQ, QuIP#, AQLM) is error
compensation: quantize one input channel at a time and push the resulting error onto
the channels not yet quantized, weighted by the inverse Hessian of the real
activations. Errors then cancel instead of accumulating.

This implements it properly (OBQ/GPTQ) and compares against round-to-nearest at the
same bit widths, on the real weight and the real captured activations.
"""
import argparse, json, struct, time, torch

ap = argparse.ArgumentParser()
ap.add_argument("--shard", required=True); ap.add_argument("--acts", required=True)
ap.add_argument("--layer", default="model.language_model.layers.9.mlp.gate_proj")
ap.add_argument("--group", type=int, default=128)
a = ap.parse_args()
dev = "xpu"

with open(a.shard, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n)); base = fh.tell()
    def get(nm):
        m = hdr[nm]; s, e = m["data_offsets"]; fh.seek(base + s)
        dt = {"I32": torch.int32, "F16": torch.float16}[m["dtype"]]
        return torch.frombuffer(bytearray(fh.read(e - s)), dtype=dt).reshape(m["shape"])
    qw = get(a.layer + ".qweight").to(dev); sc = get(a.layer + ".scales").to(dev)
    gi = get(a.layer + ".g_idx").to(dev)

K, N = qw.shape[0] * 8, qw.shape[1]
sh = torch.arange(8, device=dev, dtype=torch.int32) * 4
W = ((((qw.unsqueeze(1) >> sh.view(1, 8, 1)) & 0xF).reshape(K, N).to(torch.float16) - 8.0)
     * sc[gi.long()]).float()
X = torch.load(a.acts, map_location="cpu").to(dev).float()[:, :K]
bad = torch.isnan(X).any(1) | torch.isinf(X).any(1); X = X[~bad]
ref = X @ W
print(f"  weight {tuple(W.shape)}  activations {tuple(X.shape)}  (dropped {int(bad.sum())} bad rows)")

# Hessian of the layer objective ||X(W-W')||^2  ->  H = 2 X^T X
H = (X.t() @ X).double()
dead = torch.diag(H) == 0
H[dead, dead] = 1.0
damp = 0.01 * torch.diag(H).mean()
H += torch.eye(K, device=dev, dtype=torch.float64) * damp
Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()
print(f"  Hessian {tuple(H.shape)}, damping {damp.item():.4g}")


def rtn(Wm, bits, group):
    lv = 2 ** (bits - 1) - 1
    g = Wm.reshape(K // group, group, N)
    s = g.abs().amax(1, keepdim=True).clamp_min(1e-9) / lv
    return ((g / s).round().clamp(-lv - 1, lv) * s).reshape(K, N)


@torch.no_grad()
def gptq(Wm, bits, group, blocksize=128):
    """OBQ/GPTQ: quantize column-by-column, push error onto columns not yet done."""
    lv = 2 ** (bits - 1) - 1
    Wc = Wm.clone(); Q = torch.zeros_like(Wc)
    for i0 in range(0, K, blocksize):
        i1 = min(i0 + blocksize, K)
        Wb = Wc[i0:i1].clone(); Eb = torch.zeros_like(Wb)
        Hb = Hinv[i0:i1, i0:i1]
        for j in range(i1 - i0):
            col = Wb[j]
            if (i0 + j) % group == 0:                      # refresh the group scale
                gsl = Wc[i0 + j: i0 + j + group]
                s = gsl.abs().amax(0).clamp_min(1e-9) / lv
            q = (col / s).round().clamp(-lv - 1, lv) * s
            Q[i0 + j] = q
            d = Hb[j, j]
            err = (col - q) / d
            Wb[j:] -= torch.outer(Hb[j, j:], err)          # compensate within block
            Eb[j] = err
        Wc[i1:] -= Hinv[i0:i1, i1:].t() @ Eb               # compensate the rest
    return Q


print(f"\n  {'method':>30s} {'bits/w':>7s} {'output rel err':>15s}")
print("  " + "-" * 56)
for bits in (4, 3, 2):
    t0 = time.time(); Wq = rtn(W, bits, a.group)
    e1 = ((X @ Wq - ref).norm() / ref.norm()).item()
    t1 = time.time(); Wg = gptq(W, bits, a.group)
    e2 = ((X @ Wg - ref).norm() / ref.norm()).item()
    b = bits + 16.0 / a.group
    print(f"  {'round-to-nearest ' + str(bits) + '-bit':>30s} {b:>7.2f} {e1*100:>14.2f}%")
    print(f"  {'GPTQ ' + str(bits) + '-bit (compensated)':>30s} {b:>7.2f} {e2*100:>14.2f}%"
          f"   {e1/e2:>5.1f}x better  ({time.time()-t1:.0f}s)")
print("  " + "-" * 56)
print(f"  {'plain PQ @ 2 bits (earlier)':>30s} {2.00:>7.2f} {26.80:>14.2f}%")
