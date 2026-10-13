#!/usr/bin/env python3
"""Small, fast decode probe: 8 fixed prompts (one per BetterBench category), greedy, streamed.

Per prompt: completion tokens, decode tok/s (tokens after the first / wall after the first
chunk), and acceptance from the engine's vllm:spec_decode_* counters diffed around the request
(tokens per verify step = 1 + accepted/drafts). Not a benchmark: a go/no-go before BetterBench.

  python decode_probe.py --endpoint http://127.0.0.1:8002 --model ID --max-tokens 384 --out probe.json
"""
import argparse, json, re, time, urllib.request, statistics as st
ap = argparse.ArgumentParser()
ap.add_argument("--endpoint", default="http://127.0.0.1:8002"); ap.add_argument("--model", required=True)
ap.add_argument("--max-tokens", type=int, default=384); ap.add_argument("--out", default=None)
ap.add_argument("--label", default="")
a = ap.parse_args()

PROMPTS = [
 ("chat", "I'm switching from Python to Rust for a CLI tool. What are the three biggest mindset changes I should expect, briefly?"),
 ("code", "Write a Python function `merge_intervals(intervals)` that merges overlapping [start, end] intervals, with a docstring and three unit tests using pytest."),
 ("file_edit", "Here is a config file:\n\n[server]\nport = 8080\nworkers = 4\nlog_level = \"info\"\n\n[db]\nurl = \"postgres://localhost/app\"\npool = 10\n\nChange the port to 9090, set log_level to debug, and add a `timeout = 30` key under [db]. Return the full updated file only."),
 ("json", "Return a JSON object describing three fictional planets, each with name, radius_km, has_rings (bool), moons (array of names), and a one-sentence description. JSON only."),
 ("math", "A train leaves at 9:15 traveling 72 km/h. A second train leaves the same station at 9:45 at 96 km/h on the same track. At what time does the second train catch up? Show your work."),
 ("prose", "Write a 250-word reflective paragraph about a lighthouse keeper's last night on duty before the light is automated."),
 ("reasoning", "Three boxes are labeled Apples, Oranges, and Mixed. Every label is wrong. You may draw one fruit from one box. Explain how to relabel all boxes correctly and why it works."),
 ("summarization", "Summarize in four bullet points: Speculative decoding runs a small draft model to propose several tokens, then the large target model verifies them in one forward pass. Accepted tokens are kept; the first rejected position is resampled from the target's distribution, so the output distribution is unchanged. The speedup depends on acceptance rate and on the draft's cost relative to the target. Multi-token-prediction heads attached to the target itself are a cheap drafter, while separate draft models can be stronger but cost more per step."),
]
def metrics():
    t = urllib.request.urlopen(a.endpoint + "/metrics", timeout=20).read().decode()
    out = {}
    for k in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        m = re.search(rf'vllm:spec_decode_{k}_total\{{[^}}]*\}} ([\d.e+]+)', t)
        out[k] = float(m.group(1)) if m else 0.0
    return out
def run(prompt):
    body = json.dumps({"model": a.model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": a.max_tokens, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(a.endpoint + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; n_chunks = 0; usage = None; text = ""
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices") or []
            delta = (ch[0].get("delta") or {}) if ch else {}
            if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                if t_first is None: t_first = time.time()
                n_chunks += 1
                text += delta.get("content") or ""
    t_end = time.time()
    ct = (usage or {}).get("completion_tokens", 0)
    return {"completion_tokens": ct, "ttft_s": (t_first or t_end) - t0, "decode_s": t_end - (t_first or t_end),
            "decode_tps": (ct - 1) / (t_end - t_first) if t_first and ct > 1 else 0.0, "chunks": n_chunks, "text_tail": text[-80:]}

# warm (not counted)
run("Say hi.")
rows = []
for cat, p in PROMPTS:
    b = metrics(); r = run(p); m = metrics()
    drafts = m["num_drafts"] - b["num_drafts"]; acc = m["num_accepted_tokens"] - b["num_accepted_tokens"]
    dt = m["num_draft_tokens"] - b["num_draft_tokens"]
    r.update({"category": cat, "drafts": drafts, "accepted": acc, "draft_tokens": dt,
              "tokens_per_step": (1 + acc / drafts) if drafts else 1.0, "accept_rate": (acc / dt) if dt else 0.0})
    rows.append(r)
    print(f"  {cat:<14} {r['completion_tokens']:>4} tok  {r['decode_tps']:6.1f} tok/s  tok/step {r['tokens_per_step']:.2f}  accept {r['accept_rate']*100:5.1f}%  ttft {r['ttft_s']:.2f}s", flush=True)
W = {"code": .3, "reasoning": .2, "prose": .15, "json": .15, "file_edit": .1, "summarization": .1}
by = {r["category"]: r for r in rows}
weighted = sum(W[c] * by[c]["decode_tps"] for c in W)
tps_mean = st.mean(r["decode_tps"] for r in rows); tps = st.mean(r["tokens_per_step"] for r in rows)
print(f"\n[{a.label or a.model}] weighted decode {weighted:.1f} tok/s (harness weights) | mean {tps_mean:.1f} tok/s | mean tokens/step {tps:.2f}")
if a.out:
    json.dump({"label": a.label, "model": a.model, "max_tokens": a.max_tokens, "weighted_decode_tps": weighted,
               "mean_decode_tps": tps_mean, "mean_tokens_per_step": tps, "rows": rows}, open(a.out, "w"), indent=1)
