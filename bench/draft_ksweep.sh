#!/usr/bin/env bash
# What does a draft pass actually COST?
#
# The byte model says 0.306 GB per pass = ~0.76 ms at the rate this card sustains,
# and it predicts k=8 should beat k=6 by ~3.5%. Measured, k=8 is ~10% SLOWER. So the
# model is missing a per-pass cost that is not bytes, and the "draft = 20% of the byte
# budget" framing that motivated this whole direction needs replacing with a time model.
#
# Decompose it directly:  step_time(k) = target_time + k * draft_time
# by measuring step time and tokens/step across k on one fixed workload.
set -uo pipefail
OUT="${OUT:-$HOME/w3gate/ksweep}"; mkdir -p "$OUT"
ACC="$(dirname "$0")/draft_accept.py"
SERVE="$(dirname "$0")/../serving/serve-b65.sh"
cleanup(){ docker rm -f vllm-b65 >/dev/null 2>&1||true
           systemctl --user start vllm-b65.service >/dev/null 2>&1||true
           echo "  (daily driver restarted)"; }
trap cleanup EXIT
systemctl --user stop vllm-b65.service >/dev/null 2>&1||true
docker rm -f vllm-b65 >/dev/null 2>&1||true; sleep 3

for K in 0 2 4 6 8; do
  docker rm -f vllm-b65 >/dev/null 2>&1||true
  SPEC="$K" nohup "$SERVE" > "$OUT/serve-k$K.log" 2>&1 &
  for i in $(seq 1 90); do curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break; sleep 5; done
  curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "  k=$K: no server"; tail -8 "$OUT/serve-k$K.log"; continue; }
  python3 "$ACC" --label "k=$K" --m1 1 --r6 1 --out "$OUT/k$K.json"
  docker rm -f vllm-b65 >/dev/null 2>&1||true; sleep 4
done

python3 - "$OUT" <<'PY'
import glob, json, os, sys
rows=[]
for f in glob.glob(os.path.join(sys.argv[1],"k*.json")):
    r=json.load(open(f)); rows.append(r)
rows.sort(key=lambda r: r["k"])
print(f"\n  {'k':>3s} {'accept':>7s} {'tok/step':>9s} {'tok/s':>7s} {'step ms':>8s} "
      f"{'GB/step':>8s} {'pred tok/s':>11s}")
base=None
for r in rows:
    step_ms = r["tokens_per_step"]/r["observed_tok_s"]*1000 if r["observed_tok_s"] else 0
    r["step_ms"]=step_ms
    print(f"  {r['k']:>3d} {r['acceptance']*100:>6.1f}% {r['tokens_per_step']:>9.2f} "
          f"{r['observed_tok_s']:>7.1f} {step_ms:>8.2f} {r['gb_step']:>8.2f} "
          f"{r['gb_step']/r['gb_per_token']:>11.1f}")
# fit step_ms = a + b*k
ks=[r["k"] for r in rows]; ts=[r["step_ms"] for r in rows]
if len(ks)>=2:
    n=len(ks); sx=sum(ks); sy=sum(ts); sxx=sum(k*k for k in ks); sxy=sum(k*t for k,t in zip(ks,ts))
    b=(n*sxy-sx*sy)/(n*sxx-sx*sx); a=(sy-b*sx)/n
    print(f"\n  fit: step_ms = {a:.2f} + {b:.3f} * k")
    print(f"    target pass  {a:.2f} ms")
    print(f"    draft pass   {b:.3f} ms   <- byte model predicts 0.306 GB / ~405 GB/s = 0.76 ms")
    if b > 1.2:
        print(f"    => a draft pass costs {b/0.76:.1f}x its bytes. The draft is LATENCY-bound,")
        print(f"       not bandwidth-bound, so making its weights smaller cannot help much.")
PY
