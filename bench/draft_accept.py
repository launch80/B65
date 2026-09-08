#!/usr/bin/env python3
"""Acceptance vs draft cost: is the current draft configuration still optimal at k=6?

M1 (INT4 draft linears) and R6 (32768-row draft lm_head prefix) both make the draft
CHEAPER and WORSE. Each was measured as a net win - but at MTP4. The optimum depends
on k, because the byte saving scales with k while the acceptance loss compounds
through the accepted-prefix length. At k=6 the balance may have moved.

This measures acceptance on a FIXED workload and combines it with the byte model:

    GB/step  = target + k * draft
    tokens   = 1 + accepted/drafts
    GB/token = GB/step / tokens          <- the thing that sets throughput

Byte model (GB), measured elsewhere in this project:
    target body 12.703 + target lm_head INT4 0.656 + GDN/KV 0.330 = 13.689
    draft pass  = MTP layer (BF16 0.849 | INT4 0.219)
                + draft lm_head (full INT4 0.656 | 32768-prefix 0.087)
"""
import argparse, json, re, statistics, time, urllib.request

TARGET = 13.689
MTP_BF16, MTP_INT4 = 0.849, 0.219
LMH_FULL, LMH_PREFIX = 0.656, 0.087

ap = argparse.ArgumentParser()
ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16")
ap.add_argument("--label", required=True)
ap.add_argument("--m1", type=int, required=True, help="1 = INT4 draft linears")
ap.add_argument("--r6", type=int, required=True, help="1 = 32768 draft vocab prefix")
ap.add_argument("--out", default=None)
a = ap.parse_args()

# A fixed, realistic mix: prose continuation, code, and reasoning. Acceptance is
# strongly workload-dependent, so every config must see exactly these prompts.
PROMPTS = [
 "Write a Python function that merges two sorted lists into one sorted list, with a docstring.",
 "Explain, in three sentences, why memory bandwidth rather than compute limits LLM decoding.",
 "def binary_search(arr, target):\n    # complete this function\n",
 "Summarize the causes of the 1929 stock market crash in one paragraph.",
 "Write a SQL query that returns the top 5 customers by total order value, with a CTE.",
 "What is 17 * 23? Show your working step by step.",
 "Rewrite this sentence to be clearer: 'The thing that was done by him was not good.'",
 "List the steps to set up a Python virtual environment and install a package.",
]

def metrics():
    t = urllib.request.urlopen(a.endpoint + "/metrics", timeout=30).read().decode()
    def g(p):
        m = re.search(rf'vllm:spec_decode_{p}_total\{{[^}}]*\}} ([\d.e+]+)', t)
        return float(m.group(1)) if m else 0.0
    return g("num_drafts"), g("num_draft_tokens"), g("num_accepted_tokens")

def run(p):
    body = {"model": a.model, "messages": [{"role": "user", "content": p}],
            "max_tokens": 200, "temperature": 0, "stream": False}
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        a.endpoint + "/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"}), timeout=300))
    dt = time.perf_counter() - t0
    return r["usage"]["completion_tokens"], dt

for p in PROMPTS[:2]:            # warm up, not measured
    run(p)
d0, t0_, a0 = metrics()
toks, secs = 0, 0.0
for p in PROMPTS:
    n, dt = run(p); toks += n; secs += dt
d1, t1_, a1 = metrics()

nd, ndt, nat = d1 - d0, t1_ - t0_, a1 - a0
k = int(round(ndt / nd)) if nd else 0
acc = nat / ndt if ndt else 0
tps = 1 + nat / nd if nd else 1
draft = (MTP_INT4 if a.m1 else MTP_BF16) + (LMH_PREFIX if a.r6 else LMH_FULL)
gb_step = TARGET + k * draft
res = dict(label=a.label, m1=a.m1, r6=a.r6, k=k, drafts=nd, drafted=ndt, accepted=nat,
           acceptance=acc, tokens_per_step=tps, draft_gb=draft, gb_step=gb_step,
           gb_per_token=gb_step / tps, observed_tok_s=toks / secs)
print(f"  {a.label:<22s} k={k}  accept={acc*100:5.1f}%  tok/step={tps:.2f}  "
      f"draft={draft:.3f} GB  GB/tok={res['gb_per_token']:.3f}  "
      f"observed={res['observed_tok_s']:.1f} tok/s")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
