#!/usr/bin/env python3
"""Decode throughput vs CONTEXT DEPTH, counting TOKENS not SSE events.

Under speculative decoding one streaming chunk carries several tokens, so counting
chunks undercounts throughput by the acceptance factor. Ask the server for a usage
block and trust that instead.
"""
import json, re, sys, time, urllib.request
EP = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8002"
M = "SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"

def met():
    t = urllib.request.urlopen(EP+"/metrics", timeout=20).read().decode()
    g = lambda k: float((re.search(rf'vllm:spec_decode_{k}_total\{{[^}}]*\}} ([\d.e+]+)', t) or [0,0])[1])
    return g("num_drafts"), g("num_accepted_tokens")

FILL = ("The system reads every weight from memory to produce a single token, which makes "
        "decoding a bandwidth problem rather than an arithmetic one. Caches, prefetchers and "
        "wider buses have all been aimed at the same imbalance for thirty years. ")
def ntok(s):
    return json.load(urllib.request.urlopen(urllib.request.Request(
        EP+"/tokenize", json.dumps({"model":M,"prompt":s}).encode(),
        {"Content-Type":"application/json"}), timeout=180))["count"]

print(f"  {'context':>9s} {'decode t/s':>11s} {'tok/step':>9s} {'chunks/s':>9s} {'ttft':>8s}")
print("  " + "-"*52)
for target in (500, 2000, 6000, 12000, 21000, 28000):
    p = FILL * max(1, target//45)
    while ntok(p) > target: p = p[:int(len(p)*0.93)]
    d0,a0 = met(); t0=time.perf_counter(); first=None; chunks=0; ntokens=None
    body = {"model":M,"prompt":p,"max_tokens":160,"temperature":0,"ignore_eos":True,
            "stream":True,"stream_options":{"include_usage":True}}
    req = urllib.request.Request(EP+"/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            ln = raw.decode().strip()
            if not ln.startswith("data: "): continue
            if ln[6:] == "[DONE]": break
            o = json.loads(ln[6:])
            if o.get("usage"): ntokens = o["usage"]["completion_tokens"]
            ch = (o.get("choices") or [{}])[0]
            if ch.get("text"):
                if first is None: first = time.perf_counter()
                chunks += 1
    end=time.perf_counter(); d1,a1=met(); dd=max(1.0,d1-d0)
    n = ntokens if ntokens else chunks
    print(f"  {target:>9,} {n/(end-first):>11.1f} {(a1-a0)/dd+1:>9.2f} "
          f"{chunks/(end-first):>9.1f} {(first-t0)*1000:>7.0f}ms")
