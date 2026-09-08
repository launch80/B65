#!/usr/bin/env python3
"""Mixed precision fallback: choose which modules stay at int4.

The plan budgeted for this. Modules are ranked by measured output error and the
worst are excluded until the remaining 3-bit share still pays for itself. Every
module left at int4 costs bytes, so this prints what the retained speedup is.
"""
import json, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--threshold", type=float, default=0.15, help="skip modules above this out_rel_err")
a = ap.parse_args()
m = json.load(open(a.manifest))
b4 = {k: (v["K"] // 8) * v["N"] * 4 + (v["K"] // 128) * v["N"] * 2 for k, v in m.items()}
b3 = {k: (v["K"] // 32) * 3 * v["N"] * 4 + (v["K"] // 128) * v["N"] * 2 for k, v in m.items()}
skip = [k for k, v in m.items() if v["out_rel_err"] > a.threshold]
tot4 = sum(b4.values())
kept = sum(b3[k] for k in m if k not in skip) + sum(b4[k] for k in skip)
errs = sorted(v["out_rel_err"] for v in m.values())
print(f"modules {len(m)}   out_rel_err median {errs[len(errs)//2]*100:.1f}%  "
      f"p95 {errs[int(len(errs)*0.95)]*100:.1f}%  max {errs[-1]*100:.1f}%")
print(f"threshold {a.threshold*100:.0f}%  ->  skip {len(skip)} modules "
      f"({len(skip)/len(m)*100:.0f}%)")
print(f"body bytes: int4 {tot4/1e9:.2f} GB -> mixed {kept/1e9:.2f} GB "
      f"({kept/tot4:.3f}x)  all-3bit would be {sum(b3.values())/1e9:.2f} GB")
# de-duplicate to the shortest distinguishing substrings vLLM's name matching needs
print("\nP608_W3_SKIP=" + ",".join(sorted(skip)))
