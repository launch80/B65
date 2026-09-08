#!/usr/bin/env python3
"""Are the tokens the model actually predicts concentrated at LOW token IDs?

BPE vocabularies are built by merge order, which correlates with corpus frequency,
so low IDs tend to be the common tokens. If the argmax on real hidden states lands
in the first N rows of lm_head, the draft can read a contiguous PREFIX - no
reordering, no gather, just a smaller slice of the same tensor.

Measured on real captured hidden states against the real head.
"""
import os, torch
CAP = os.environ.get("P608_CAP", "/p608/research/r3/capture")
W = torch.load(f"{CAP}/lm_head.pt", map_location="cpu")
H = torch.load(f"{CAP}/hidden.pt", map_location="cpu")
V, D = W.shape
dev = "xpu"
logits = H.to(dev).float() @ W.to(dev).float().T
t1 = logits.argmax(-1)
t5 = logits.topk(5, -1).indices
print(f"  {H.shape[0]} real hidden states, vocab {V:,}\n")
print(f"  {'prefix N':>10s} {'% of head':>10s} {'GB/draft':>9s} {'top-1 in':>9s} {'top-5 in':>9s}")
print("  " + "-" * 54)
for N in (4096, 8192, 16384, 32768, 65536, 131072, 151936):
    c1 = (t1 < N).float().mean().item()*100
    c5 = (t5 < N).float().mean().item()*100
    print(f"  {N:>10,} {N/V*100:>9.1f}% {0.656*N/V:>8.3f}G {c1:>8.1f}% {c5:>8.1f}%")
print("  " + "-" * 54)
q = torch.tensor([.5,.9,.95,.99,1.0], device=dev)
print("  argmax id percentiles:", [int(x) for x in torch.quantile(t1.float(), q).tolist()])
print("  max argmax id observed:", int(t1.max()))
