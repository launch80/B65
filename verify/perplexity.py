#!/usr/bin/env python3
"""Perplexity over the BetterBench corpus, via vLLM prompt_logprobs.

This is R3's shipping gate. P3/R2 specify "perplexity delta < 1%", and until now
the project had no way to measure it.

Method: tokenize each passage with the server's own /tokenize, then request
prompt_logprobs and read the logprob assigned to the ACTUAL next token at every
position. PPL = exp(-mean logprob). Teacher forcing is inherent here - every
position is scored against the true prefix - so the fp16 and INT4 heads are
compared on identical inputs.

Speculation must be OFF: prompt_logprobs returns NaN under MTP on this stack.
"""
import argparse, json, math, urllib.request, glob, os

ap = argparse.ArgumentParser()
ap.add_argument("--endpoint", default="http://127.0.0.1:8002/v1")
ap.add_argument("--base", default="http://127.0.0.1:8002")
ap.add_argument("--model", default="SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16")
ap.add_argument("--corpus", default=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ppl-corpus.json"))
ap.add_argument("--out", required=True)
a = ap.parse_args()


def post(url, body, timeout=900):
    return json.load(urllib.request.urlopen(urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}),
        timeout=timeout))


passages = [(p["source"], p["id"], p["text"])
            for p in json.load(open(a.corpus))]

print(f"  {len(passages)} passages, {len({c for c,_,_ in passages})} sources")

total_lp, total_tok = 0.0, 0
per_cat = {}
detail = []
for cat, pid, text in passages:
    tok = post(a.base + "/tokenize", {"model": a.model, "prompt": text})
    ids = tok["tokens"]
    r = post(a.endpoint + "/completions",
             {"model": a.model, "prompt": text, "max_tokens": 1,
              "temperature": 0.0, "prompt_logprobs": 0})
    pl = r["choices"][0].get("prompt_logprobs") or []
    lps = []
    for i, pos in enumerate(pl):
        if not pos or i == 0 or i >= len(ids):
            continue
        entry = pos.get(str(ids[i]))
        if entry is None:
            continue
        lp = entry["logprob"] if isinstance(entry, dict) else float(entry)
        if lp is None or math.isnan(lp) or math.isinf(lp):
            continue
        lps.append(lp)
    if not lps:
        print(f"  WARN no usable logprobs for {pid}")
        continue
    s = sum(lps)
    total_lp += s; total_tok += len(lps)
    c = per_cat.setdefault(cat, [0.0, 0])
    c[0] += s; c[1] += len(lps)
    detail.append({"category": cat, "id": pid, "tokens": len(lps),
                   "mean_logprob": s / len(lps),
                   "ppl": math.exp(-s / len(lps))})
    print(f"  {cat:20s} {pid[:26]:28s} n={len(lps):5d}  ppl={math.exp(-s/len(lps)):8.3f}")

ppl = math.exp(-total_lp / total_tok)
print(f"\n  TOTAL: {total_tok} tokens, mean logprob {total_lp/total_tok:.5f}")
print(f"  PERPLEXITY = {ppl:.5f}")
out = {"perplexity": ppl, "tokens": total_tok,
       "mean_logprob": total_lp / total_tok,
       "per_category": {k: {"ppl": math.exp(-v[0]/v[1]), "tokens": v[1]}
                        for k, v in per_cat.items()},
       "passages": detail}
json.dump(out, open(a.out, "w"), indent=1)
print(f"  wrote {a.out}")
