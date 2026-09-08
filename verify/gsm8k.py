#!/usr/bin/env python3
"""GSM8K subset accuracy — R3's reasoning gate.

A fixed subset (seeded, so both arms see identical problems in identical order),
greedy, answer taken as the last integer in the completion. Reports a Wilson 95%
interval, because at n=100 a few points of difference is noise and quoting a bare
percentage would overstate what the sample can support.

Speculative decoding may be left on: MTP verification preserves the target
distribution, so it accelerates without changing which head is being measured.
"""
import argparse, json, math, random, re, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--endpoint", default="http://127.0.0.1:8002/v1")
ap.add_argument("--model", default="SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16")
ap.add_argument("--parquet", default=None)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed", type=int, default=608)
ap.add_argument("--max-tokens", type=int, default=768)
ap.add_argument("--out", required=True)
a = ap.parse_args()

import os
pq_path = a.parquet or os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsm8k.parquet")
import pyarrow.parquet as pq
rows = pq.read_table(pq_path).to_pylist()
random.Random(a.seed).shuffle(rows)
rows = rows[:a.n]

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
def final_num(text):
    m = NUM.findall(text.replace("$", ""))
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None

def gold(ans):
    return float(ans.split("####")[-1].strip().replace(",", ""))

ok = n = 0
trunc = 0
recs = []
t_start = time.time()
for i, r in enumerate(rows):
    prompt = (r["question"].strip() +
              "\n\nSolve this step by step, then give the final numeric answer "
              "on its own line after 'Answer:'.")
    body = {"model": a.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": a.max_tokens, "temperature": 0.0}
    try:
        resp = json.load(urllib.request.urlopen(urllib.request.Request(
            a.endpoint + "/chat/completions", json.dumps(body).encode(),
            {"Content-Type": "application/json"}), timeout=900))
    except Exception as exc:                                # noqa: BLE001
        print(f"  [{i+1}] request failed: {str(exc)[:80]}")
        continue
    ch = resp["choices"][0]
    txt = (ch["message"].get("content") or "")
    if ch.get("finish_reason") == "length":
        trunc += 1
    got, want = final_num(txt), gold(r["answer"])
    hit = got is not None and abs(got - want) < 1e-4
    ok += hit; n += 1
    recs.append({"i": i, "gold": want, "got": got, "hit": bool(hit),
                 "finish": ch.get("finish_reason")})
    if (i + 1) % 10 == 0:
        print(f"  [{i+1:3d}/{len(rows)}] running acc {ok/n*100:5.1f}%  "
              f"trunc {trunc}  ({time.time()-t_start:.0f}s)")

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h)*100, min(1, c+h)*100)

lo, hi = wilson(ok, n)
print(f"\n  GSM8K n={n}  correct={ok}  accuracy={ok/n*100:.1f}%  "
      f"95% CI [{lo:.1f}, {hi:.1f}]  truncated={trunc}")
json.dump({"n": n, "correct": ok, "accuracy": ok/n*100, "ci95": [lo, hi],
           "truncated": trunc, "seed": a.seed, "max_tokens": a.max_tokens,
           "records": recs}, open(a.out, "w"), indent=1)
print(f"  wrote {a.out}")
