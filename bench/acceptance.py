#!/usr/bin/env python3
"""Measure MTP acceptance per workload, by diffing vllm:spec_decode_* counters.

The cookbook benchmarks p512/g128 - a 512-token prompt continued for 128 tokens.
Our BetterBench corpus is a reasoning-heavy mix. Acceptance is workload-dependent,
so a like-for-like comparison against their catalog requires their workload.
"""
import json, re, time, urllib.request, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16")
a = ap.parse_args()

def metrics():
    t = urllib.request.urlopen(a.endpoint + "/metrics", timeout=20).read().decode()
    out = {}
    for k in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        m = re.search(rf'vllm:spec_decode_{k}_total\{{[^}}]*\}} ([\d.e+]+)', t)
        out[k] = float(m.group(1)) if m else 0.0
    for i in range(4):
        m = re.search(rf'vllm:spec_decode_num_accepted_tokens_per_pos_total\{{[^}}]*position="{i}"[^}}]*\}} ([\d.e+]+)', t)
        out[f"pos{i}"] = float(m.group(1)) if m else 0.0
    return out

FILLER = ("The history of computer architecture is a history of moving data. Every "
          "generation of processor has been faster at arithmetic than at fetching the "
          "operands that arithmetic consumes, and every generation has answered that "
          "imbalance with caches, prefetchers, wider buses and deeper pipelines. ")

# Realistic ~512-token English prose, no self-repetition. A prompt built by
# repeating one sentence is trivially predictable and yields 100% acceptance,
# which flatters the result and is not what a real benchmark measures.
REAL = """Memory bandwidth has quietly become the defining constraint of modern
accelerator design. For most of the last three decades, the story of processor
performance was told in floating point operations per second, and the industry
optimised relentlessly for that number. Arithmetic units multiplied, clock rates
climbed, and vector widths doubled. But the memory system did not keep pace. Each
generation widened the gap between how fast a chip could compute and how fast it
could be fed. Designers responded with deeper cache hierarchies, more aggressive
prefetchers, wider buses and eventually stacked memory sitting on the same package
as the die. These mitigations bought time without changing the underlying trend.
Language model inference makes the imbalance impossible to ignore. When a model
generates text one token at a time, it must read every weight it owns to produce a
single output. There is no batch across which to amortise that read, and no reuse
to exploit. The arithmetic per byte fetched is minimal, so the processor spends
most of its time waiting. Larger models make this worse in direct proportion to
their size. The practical consequences shape how systems are built. Quantization
matters less for the arithmetic it saves than for the traffic it avoids. A format
that halves the bytes per weight nearly halves the time per token, whether or not
the arithmetic units were ever the limit. Speculative decoding attacks the problem
from a different angle, amortising one expensive read across several candidate
tokens. Both techniques are ultimately about the same scarce resource, approached
from opposite directions, and they compose because they act on different terms of
the same equation."""

WORKLOADS = [
    ("p512/g128 REAL prose", REAL, 128),
    ("p512/g128 repetitive filler", (FILLER * 30)[:2048], 128),
    ("short prompt, long gen",      "Write a long essay about memory bandwidth.", 256),
    ("code continuation",           "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n", 128),
    ("reasoning (our corpus style)", "Solve step by step: a train leaves at 3pm going 60mph, "
                                     "another at 4pm going 80mph. When does the second catch the first?", 256),
]

print(f"  {'workload':32s} {'tok/s':>7s} {'tok/step':>9s} {'accept':>8s}  per-position")
print("  " + "-" * 84)
for name, prompt, mx in WORKLOADS:
    before = metrics()
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        a.endpoint + "/v1/completions",
        json.dumps({"model": a.model, "prompt": prompt, "max_tokens": mx,
                    "temperature": 0.0, "ignore_eos": True}).encode(),
        {"Content-Type": "application/json"}), timeout=600))
    dt = time.perf_counter() - t0
    after = metrics()
    n = r["usage"]["completion_tokens"]
    d = {k: after[k] - before[k] for k in after}
    drafts = d["num_drafts"] or 1
    tps = n / dt
    tok_step = (d["num_accepted_tokens"] + drafts) / drafts
    acc = d["num_accepted_tokens"] / (d["num_draft_tokens"] or 1) * 100
    pos = " / ".join(f"{d[f'pos{i}']/drafts*100:4.1f}" for i in range(4))
    print(f"  {name:32s} {tps:7.1f} {tok_step:9.2f} {acc:7.1f}%  {pos}")
print("  " + "-" * 84)
print("  their implied figure at p512/g128: 4.76 tok/step, ~94% acceptance")
