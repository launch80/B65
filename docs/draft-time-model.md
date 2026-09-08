# The draft is latency-bound: a time model for MTP speculation on the B65

## Why this exists

`docs/11` sized the MTP draft passes at 20% of the per-token byte budget and called
them "quality-free ground". Both halves of that were stale or wrong, and the whole
"optimise the draft" direction was built on them. This replaces the byte model with a
measured time model.

## Method

`research/r11_accept.py` runs a fixed 8-prompt workload against the live server and
diffs `vllm:spec_decode_*` counters, giving acceptance, tokens/step and observed
tok/s. `research/r11_ksweep.sh` runs it at k = 0, 2, 4, 6, 8 on the daily driver
(M1 on, R6 on). Step time = tokens/step / tok/s.

The absolute numbers are workload-specific (this workload runs ~68 tok/s at k=6
where BetterBench runs ~77, because acceptance is 35% here vs 44% there). The SHAPE
- the slope - is what is load-bearing.

## The time model

```
  k  accept  tok/step   tok/s  step ms
  0    0.0%      1.00    31.0    32.27
  2   63.5%      2.27    59.0    38.45
  4   47.6%      2.91    68.3    42.55
  6   35.4%      3.12    67.8    46.06
  8   27.8%      3.23    60.0    53.80

  fit:  step_ms = 32.49 + 2.534 * k
        target pass   32.49 ms
        draft pass     2.53 ms
```

A draft pass moves 0.306 GB (INT4 MTP layer 0.219 + 32768-row lm_head prefix 0.087),
which at the ~405 GB/s this card sustains is **0.76 ms**. It costs **2.53 ms**.
**A draft pass costs 3.3x its bytes.** It is latency-bound, not bandwidth-bound.

The fit also explains a result that was already in the repo and that the byte model
got backwards: k=8 is ~10% slower than k=6 (`results/p05-r6-k8-decode.json`). The
byte model predicted k=8 would win by 3.5%. Marginal tokens fall off
(2.91 -> 3.12 -> 3.23) while each pass costs a flat 2.5 ms.

Where the 2.5 ms comes from: the target runs 64 layers in 32 ms, ~0.5 ms per layer.
One draft layer takes 2.5 ms - five times a target layer. That is per-forward fixed
cost (attention-metadata rebuild, launch chain, `compute_logits` + argmax, and four
`.item()` host syncs in `llm_base_proposer.py` at lines 1147/1158/1269/1270), paid
once by the target and k times by the draft. At k=6 that is ~12 ms of a 46 ms step,
**~26%**, and it is not bytes.

## Consequences

1. **The byte budget table in `docs/11` is stale.** After R6 the draft is 11.8% of
   bytes, not 20%. And bytes are not what it costs: in time it is 32% of the step.
2. **Shrinking draft weights cannot buy much.** Only 0.76 ms of each 2.53 ms pass is
   bytes. Quantizing the draft further - including with the W3 kernel, which
   `docs/15` floated for exactly this - can recover at most ~0.3 ms per pass.
3. **M1 and R6 need re-evaluating at k=6.** Both traded acceptance for bytes. If bytes
   were never the cost, they may be trading the thing that multiplies throughput for
   a thing that was free. Neither was re-measured after k moved from 4 to 6.
   `research/r11_sweep.sh` runs that 2x2. (Result appended below when it lands.)
4. **k=6 is right on BetterBench.** With ~77% conditional acceptance and a 2.5 ms
   pass, k=4 gives 74.5 and k=6 gives 76.8 tok/s. k=4 only wins on low-acceptance
   workloads.

## XPU graph capture is worth 0.0%

`research/r11_graphtest.sh` runs k=0 and k=6 with `VLLM_XPU_ENABLE_XPU_GRAPH` on and
off. The toggle is verified in the logs (`cudagraph_mode: FULL_AND_PIECEWISE` vs
`NONE`, "Skipping CUDA graph capture").

```
                k=0 (target)      k=6     draft ms/pass
  graph ON          32.26 ms   45.96 ms       2.283
  graph OFF         32.26 ms   45.50 ms       2.206
```

Target: **32.26 vs 32.26 ms.** Identical to two decimals. The draft slope is ~1%
better with capture off. Repeated twice at k=6: off 69.4 vs on 68.0 tok/s, **+2%**.

The ladder credits `XPU_GRAPH=1` with +13%. That was measured on vLLM 0.21 with
`--enforce-eager`, where torch.compile was off and capture was the only thing
removing launch overhead. On this image `enforce_eager=False` and
`mode: VLLM_COMPILE` regardless; inductor already does that work. **The daily driver
now defaults capture OFF** (`serve-b65.sh`, `XPU_GRAPH=0`).

This also closes the hypothesis that the draft's 2.5 ms was uncaptured launch
overhead: the proposer (`llm_base_proposer.py`, not the 22-line `eagle.py` shim) does
dispatch PIECEWISE capture, and turning it off changes nothing.

## The daily driver was crashing under concurrent load

Separate finding, same day. The fused `gdn_attention` host binding has a hard
`TORCH_CHECK`: *"causal_conv1d does not support spec-decode and non-spec (prefill +
decode) tokens in the same invocation"*. It kills EngineCore, after which every
request gets a connection error. It never fires single-stream, so BetterBench never
saw it; any concurrent client (an agent harness) hits it within minutes.

The fix - `patches/patch_gdn_mixed_split_v5.py` - already existed, was documented in
the B70 cookbook as required "after the two MTP patches", and had a `GDN_V5` knob in
`serve-parity.sh`. It was never added to `serve-b65.sh`. It is now on by default
whenever `SPEC != 0`. Verified: 8/8 concurrent mixed prefill/decode requests, engine
alive, zero `causal_conv1d` errors.

**Every throughput number this project has quoted was measured on a config that
crashes under the load its users actually generate.** None of the numbers are
invalidated - single-stream never triggers it - but the benchmarked config and the
used config were not the same config.

## Status of the 85 tok/s target

From 77.2, 85 needs +10.1%. Levers, measured:

| lever | size | status |
|---|---|---|
| capture off | +2% | **shipped** |
| M1/R6 re-evaluation at k=6 | unknown, real mechanism | see below |
| W4A16 kernel (`kernel/w4_l80.cpp`) | +1.5% | built, unwired |
| k | 0 - k=6 is right | closed |
| draft per-forward fixed cost | ~26% of step | vLLM internals; days |

The last row is where 85 lives if the config levers fall short.
