#!/usr/bin/env python3
"""Greedy-equivalence check: speculative decoding must reproduce the target's
greedy output EXACTLY. If it does not, the verification path is broken and any
throughput number measured on it is worthless.

Capture with --out on a reference config, then compare on a candidate config.
"""
import argparse, json, urllib.request, sys, os

ap = argparse.ArgumentParser()
ap.add_argument("--endpoint", default="http://127.0.0.1:8002/v1")
ap.add_argument("--model", default="SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16")
ap.add_argument("--out"); ap.add_argument("--ref")
ap.add_argument("--max-tokens", type=int, default=48)
a = ap.parse_args()

PROMPTS = [
  "France is a country in Europe. Its capital city is",
  "Question: what is 2 + 2? Answer:",
  "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =",
  "The three primary colours of light are red, green and",
  "In machine learning, gradient descent is an algorithm that",
  "SELECT name, age FROM users WHERE age > 30 ORDER BY",
  "Water freezes at zero degrees Celsius and boils at",
  "The purpose of a cache in a processor is to",
]
out = []
for p in PROMPTS:
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        a.endpoint + "/completions",
        json.dumps({"model": a.model, "prompt": p, "max_tokens": a.max_tokens,
                    "temperature": 0.0}).encode(),
        {"Content-Type": "application/json"}), timeout=300))
    out.append(r["choices"][0]["text"])

if a.out:
    json.dump(out, open(a.out, "w")); print(f"  reference captured -> {a.out}")
if a.ref:
    ref = json.load(open(a.ref))
    same = sum(x == y for x, y in zip(ref, out))
    print(f"  identical greedy completions: {same}/{len(ref)}")
    for i, (x, y) in enumerate(zip(ref, out)):
        if x != y:
            print(f"    [{i}] MISMATCH\n        ref : {x[:70]!r}\n        cand: {y[:70]!r}")
    sys.exit(0 if same == len(ref) else 1)
