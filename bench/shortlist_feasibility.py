#!/usr/bin/env python3
"""R3 step 2 feasibility: can a cluster shortlist find the true top-k tokens?

The roadmap proposes a two-stage vocabulary projection: a small head over ~2048
clusters picks candidate clusters, and only those rows of the full lm_head are read.
That is only worth building if the shortlist actually RETAINS the distribution -
sampling and MTP verification both need a faithful tail, so the gate is recall and
KL, not top-1 agreement.

This answers the question offline, on real captured hidden states, before any kernel
work. Cheap to run, and a negative result here saves days.
"""
import argparse, math, time, torch

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/capture")
ap.add_argument("--clusters", type=int, default=2048)
ap.add_argument("--iters", type=int, default=12)
ap.add_argument("--topk", type=int, default=20, help="tail depth that must be preserved")
ap.add_argument("--probe", type=int, default=256, help="hidden states to evaluate")
ap.add_argument("--always", type=int, default=0,
                help="always read the top-N rows by weight norm, on top of the shortlist")
a = ap.parse_args()

dev = "xpu"
W = torch.load(f"{a.dir}/lm_head.pt", map_location="cpu")     # [V, D] fp16
H = torch.load(f"{a.dir}/hidden.pt", map_location="cpu")[:a.probe]
V, D = W.shape
print(f"lm_head {tuple(W.shape)} {W.dtype} | hidden {tuple(H.shape)}")
W = W.to(dev); H = H.to(dev, torch.float16)

# ---- ground truth -----------------------------------------------------------
t0 = time.time()
logits = (H.float() @ W.float().T)                              # [P, V]
true_top = logits.topk(a.topk, dim=-1)
print(f"exact logits computed in {time.time()-t0:.1f}s\n")

# ---- spherical k-means over the vocabulary ---------------------------------
# Direction is what a dot product ranks on once norms are folded in, so cluster on
# normalised rows and keep the norm as a separate per-row scale.
g = torch.Generator(device="cpu").manual_seed(608)
Wn = torch.nn.functional.normalize(W.float(), dim=1)
idx = torch.randperm(V, generator=g)[:a.clusters].to(dev)
Cen = Wn[idx].clone()
CH = 16384
for it in range(a.iters):
    assign = torch.empty(V, dtype=torch.long, device=dev)
    for s in range(0, V, CH):
        sim = Wn[s:s+CH] @ Cen.T
        assign[s:s+CH] = sim.argmax(-1)
    newC = torch.zeros_like(Cen)
    cnt = torch.zeros(a.clusters, device=dev)
    newC.index_add_(0, assign, Wn)
    cnt.index_add_(0, assign, torch.ones(V, device=dev))
    empty = cnt == 0
    newC[~empty] = newC[~empty] / cnt[~empty].unsqueeze(1)
    newC[empty] = Cen[empty]
    Cen = torch.nn.functional.normalize(newC, dim=1)
    if it % 4 == 0 or it == a.iters - 1:
        print(f"  kmeans iter {it:2d}  empty={int(empty.sum())}  "
              f"largest={int(cnt.max())}  median={int(cnt.median())}")

# per-cluster max row-norm: lets the shortlist score account for magnitude.
# Done on CPU - index_reduce_/scatter_reduce_ amax is beta on XPU and asserts here.
norms = W.float().norm(dim=1)
assign_c = assign.cpu()
cmaxnorm = torch.zeros(a.clusters).scatter_reduce_(
    0, assign_c, norms.cpu(), reduce="amax", include_self=False).to(dev)
sizes = torch.bincount(assign_c, minlength=a.clusters)
print(f"\n  cluster sizes: min={int(sizes.min())} median={int(sizes.median())} "
      f"max={int(sizes.max())} empty={int((sizes==0).sum())}")

# ---- evaluate the shortlist (fully vectorised) ------------------------------
Hn = H.float()
cluster_score = (Hn @ Cen.T) * cmaxnorm.unsqueeze(0)      # norm-aware stage-1 score
P = Hn.shape[0]
gt = true_top.indices                                      # [P, topk]
p_full = torch.softmax(logits, -1).gather(1, gt)
p_full = p_full / p_full.sum(1, keepdim=True)

print(f"\nrecall of the true top-{a.topk}, and share of lm_head rows read:")
print(f"  {'clusters':>9s} {'rows read':>11s} {'% of head':>10s} {'bytes/read':>11s} "
      f"{'top-1':>7s} {'recall':>8s} {'KL':>10s}")
always_mask = torch.zeros(V, dtype=torch.bool, device=dev)
if a.always:
    # A cheap, legitimate prior: rows with large weight norm produce large logits
    # more often. Needs no corpus statistics and no peeking at the answer.
    always_mask[norms.topk(a.always).indices] = True
    print(f"  always-read set: top {a.always} rows by weight norm "
          f"({a.always/V*100:.2f}% of head)")

for M in (16, 32, 64, 128, 256, 512):
    sel = torch.zeros(P, a.clusters, dtype=torch.bool, device=dev)
    sel.scatter_(1, cluster_score.topk(M, dim=-1).indices, True)
    row_sel = sel[:, assign]                               # [P, V] bool
    if a.always:
        row_sel = row_sel | always_mask.unsqueeze(0)
    rows_read = row_sel.sum(1).float().mean().item()
    masked = logits.masked_fill(~row_sel, float("-inf"))
    sl = masked.topk(a.topk, dim=-1).indices
    top1 = (sl[:, 0] == gt[:, 0]).float().mean().item() * 100
    # recall = |shortlist top-k INTERSECT true top-k| / k
    hit = (sl.unsqueeze(2) == gt.unsqueeze(1)).any(2).float().sum(1) / a.topk
    recall = hit.mean().item() * 100
    q = torch.softmax(masked, -1).gather(1, gt).clamp_min(1e-12)
    q = q / q.sum(1, keepdim=True)
    kl = (p_full * (p_full.log() - q.log())).sum(1).mean().item()
    frac = rows_read / V
    print(f"  {M:>9d} {rows_read:>11,.0f} {frac*100:>9.2f}% "
          f"{frac*2.543:>10.3f}GB {top1:>6.1f}% {recall:>7.1f}% {kl:>10.5f}")

print(f"\n  full fp16 head reads 2.543 GB; R3 step 1 (blanket INT4) reads 0.656 GB.")
print( "  A shortlist only wins if it beats 0.656 GB at acceptable recall/KL -")
print( "  and the two compose: shortlist rows can themselves be INT4.")
