# P0 — the measured wall, and R3 step 2 — a negative result

2026-09-07. Two results that between them redirect the roadmap.

## P0 · The wall is 587 GB/s, and P2 should not close

Published as *B65 P0-01*. Harness: [`../research/p0/streaming.py`](../research/p0/streaming.py).

| kernel | GB/s |
|---|---|
| **contiguous read** (`sum(A)`) | **587** — 97% of the 608 datasheet |
| copy / scale / add / triad | 534–539 |
| `sum(A*B)`, 4N traffic | 550 |
| strided read, stride 2 | 299 (useful bytes) |
| **decode achieves** | **434 = 74% of the wall** |

Stable across 1, 2 and 4 GiB buffers (577 / 587 / 587). **Headroom to the wall: 1.35×.**

**This refutes two roadmap claims.** The wall was estimated at 450–500; it is 587. And the
roadmap concluded headroom was *"perhaps 1.25×, not 3×"* with P2 possibly closing unstarted.
It should not close — roughly a third more is available from access-pattern and kernel work,
before any byte reduction. The two levers compose: byte reduction multiplies forward passes,
closing the 74% gap raises the rate the bytes move at.

**Method limitation, stated plainly.** This is torch-XPU, not hand-written SYCL — the pinned
image ships no `icpx`/`dpcpp`. No control over vectorisation or cache hints, so these are a
**lower bound**, which is the conservative direction for the conclusion.

**One number not to trust:** the int32-view read came out at 119.7 GB/s. That is arithmetic-
bound (int64 accumulation), not a memory result. Recorded only so nobody later quotes it as an
integer-read penalty.

## R3 step 2 · The shortlist head does not beat blanket INT4

Study: [`../research/r3/shortlist_study.py`](../research/r3/shortlist_study.py), run against
the real fp16 head and **512 real captured hidden states**, before writing any kernel.

Spherical k-means, 2048 clusters, 12 iterations (median cluster 102, none empty), norm-aware
stage-1 scoring against centroids.

| clusters kept | rows read | bytes | top-1 | **top-20 recall** | **KL** |
|---|---|---|---|---|---|
| 16 | 0.69% | 0.018 GB | 95.7% | 72.3% | 1.209 |
| 128 | 4.70% | 0.119 GB | 97.7% | 91.8% | 0.578 |
| 512 | 19.1% | 0.485 GB | 98.8% | **97.0%** | **0.262** |

Against R3 step 1: **0.656 GB at KL 0.012.** So the best shortlist buys **26% fewer bytes for
~20× worse fidelity**.

### The finding worth keeping is about the gate

**Top-1 agreement stays 95.7–98.8% in every configuration**, including one reading under 1% of
the head. Anyone gating on top-1 would have shipped this. Meanwhile the top-20 recall runs
72–97% — the shortlist keeps the argmax and **loses the tail**, which is exactly what sampling
and MTP verification depend on. The roadmap specified a distributional gate for R3 from the
outset; this measurement is why that was right.

### A prior that failed

Always reading the top-8192 rows by **weight norm** on top of the shortlist recovers almost
nothing — recall 72.3 → 72.8% at 16 clusters, 97.0 → 97.1% at 512 — while bytes rise. **Weight
norm does not predict which vocabulary rows carry probability mass.** Recorded so it is not
retried.

### Why we stopped rather than tuned

After step 1 the per-token budget is **13.689 GB**:

| | GB | share |
|---|---|---|
| language body | 12.703 | **92.8%** |
| `lm_head` | 0.656 | 4.8% |
| GDN state | 0.30 | 2.2% |
| KV | 0.03 | 0.2% |

A *perfect* shortlist eliminating the head entirely is worth **under 5%**. Step 1 already took
the large share. **The remaining bytes are in the body — R2.** That reprioritisation is this
study's real output.

What would make a shortlist work, for anyone who wants it: exact bound-based selection so recall
is guaranteed rather than measured, or clustering learned against the hidden-state distribution
rather than weight geometry. Both cost more than the 4.8% they compete for.

## NaN bug — investigated, not fixed, deliberately

Instrumented `RejectionSampler._get_logprobs_tensors`: **the sampler path is clean**, zero NaN
at every stage on every call, all rows of `final_logits` covered. So the NaN enters *downstream*,
in output processing — and an earlier guess in our draft (uninitialised rejected-draft positions)
is **disproved**. See [`../upstream/vllm-xpu-mtp-logprobs-nan.md`](../upstream/vllm-xpu-mtp-logprobs-nan.md).

Not fixed, because it does not block us: head quality is measured with speculation off (which
*isolates* the variable), and acceptance comes from `vllm:spec_decode_*` counters, which work
fine under MTP.

---

## P2 / R1 · The int4 GEMV is not the bottleneck, and R1's premise is refuted

Published as *B65 P2-01*. Harness: [`../research/p2/gemv_bench.py`](../research/p2/gemv_bench.py).

P0 implied 1.35× of kernel headroom. Tested directly on all seven real weight shapes.

| K × N | MB | per-call GB/s | **amortised GB/s** | % wall |
|---|---|---|---|---|
| 5120×17408 ×128 | 46.0 | 425 | **533** | 91% |
| 17408×5120 ×64 | 46.0 | 454 | **557** | 95% |
| 5120×12288 ×16 | 32.4 | 405 | 533 | 91% |
| 5120×10240 ×48 | 27.0 | 404 | 572 | 97% |
| 6144×5120 ×64 | 16.2 | 385 | 627 | *106% — cache* |
| 5120×6144 ×48 | 16.2 | 324 | 535 | 91% |
| 5120×1024 ×32 | 2.7 | 73 | 363 | 62% |
| **all 400 linears** | | **404 (69%)** | **547 (93%)** | |

**Two measurements per shape, because it matters.** Per-call syncs around each invocation and
carries ~30 µs of dispatch cost — visible in the smallest shape, where 2.7 MB should take 5 µs
and takes 37. Amortised syncs once around 50 back-to-back calls, closer to what the captured
graph does. **The 1.23–5.09× spread between them *is* the launch cost.**

**Caveat that bounds the claim.** Repeating a weight 50× lets cache serve part of it — the
16.2 MB shape reporting 627 GB/s, *above* the streaming wall, proves it. The two 46 MB shapes
exceed any plausible L2 and read 533/557 (91/95%). The conclusion rests on those; they are 192
of 400 linears and most of the bytes.

### R1 is refuted twice over

1. **`int4_gemm_w4a8` already ships** in vllm-xpu-kernels 0.1.12.3. Nothing to invent.
2. **It is 1.03× faster than W4A16** at M=1 — because at 2 op/byte the dequantize step was never
   the constraint. The bus was.
3. And the kernel already runs at **93% of wall**, so there was little room regardless.

**Third roadmap premise overturned by measurement**, after the graph-capture claim and
`--language-model-only`. The pattern: assumptions about what upstream *hasn't* done age badly.

### Where the time actually is

| | |
|---|---|
| decode step (31.7 t/s) | 31.50 ms |
| body GEMVs @ 547 GB/s | 23.21 ms (74%) |
| everything else | 8.29 ms |
| …its bytes (0.986 GB) at the wall | 1.68 ms |
| **unexplained** | **6.61 ms = 21% of the step** |

**That residue is the new P2 target** — and it is not a GEMV rewrite. Candidates: GDN state
update across 48 layers, attention on the 16 full-attention layers, sampling over a 248,320
vocabulary, and whatever dispatch survives graph capture.

The one GEMV-shaped exception: **5120×1024 k/v projections run at 62% of wall even amortised**
because at 2.7 MB they are latency-bound. There are 32; fusing them is worth ~1 ms.
