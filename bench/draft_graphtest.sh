#!/usr/bin/env bash
# Is the draft pass eager? Test the mechanism, don't infer it from a grep.
#
# With XPU graph capture ON:  step_ms = 32.49 + 2.534*k
# eagle.py (the MTP proposer) has no capture code, so the prediction is:
#   turning capture OFF should make the TARGET intercept worse (it is captured today)
#   and leave the DRAFT slope unchanged (it was never captured).
# If instead the slope also moves, the drafts were captured and the 2.53 ms is
# something else.
set -uo pipefail
OUT="${OUT:-$HOME/w3gate/graphtest}"; mkdir -p "$OUT"
ACC="$(dirname "$0")/draft_accept.py"
SERVE="$(dirname "$0")/../serving/serve-b65.sh"
cleanup(){ docker rm -f vllm-b65 >/dev/null 2>&1||true
           systemctl --user start vllm-b65.service >/dev/null 2>&1||true; }
trap cleanup EXIT
systemctl --user stop vllm-b65.service >/dev/null 2>&1||true
docker rm -f vllm-b65 >/dev/null 2>&1||true; sleep 3

run_one(){ local k=$1 graph=$2 lbl="k${1}-graph${2}"
  docker rm -f vllm-b65 >/dev/null 2>&1||true
  SPEC="$k" XPU_GRAPH="$graph" nohup "$SERVE" > "$OUT/$lbl.log" 2>&1 &
  for i in $(seq 1 90); do curl -sf http://127.0.0.1:8000/health>/dev/null 2>&1 && break; sleep 5; done
  curl -sf http://127.0.0.1:8000/health>/dev/null 2>&1 || { echo "  $lbl: no server"; tail -6 "$OUT/$lbl.log"; return; }
  python3 "$ACC" --label "$lbl" --m1 1 --r6 1 --out "$OUT/$lbl.json"
  docker rm -f vllm-b65 >/dev/null 2>&1||true; sleep 4; }

for g in 1 0; do for k in 0 6; do run_one $k $g; done; done

python3 - "$OUT" <<'PY'
import glob,json,os,sys
d={}
for f in glob.glob(os.path.join(sys.argv[1],"*.json")):
    r=json.load(open(f)); d[r["label"]]=r["tokens_per_step"]/r["observed_tok_s"]*1000
print(f"\n  {'':<12s} {'k=0 (target)':>14s} {'k=6':>10s} {'draft ms/pass':>15s}")
for g in (1,0):
    a,b=d.get(f"k0-graph{g}"),d.get(f"k6-graph{g}")
    if a and b:
        print(f"  graph={g}     {a:>14.2f} {b:>10.2f} {(b-a)/6:>15.3f}")
if all(k in d for k in ("k0-graph1","k0-graph0","k6-graph1","k6-graph0")):
    s1=(d["k6-graph1"]-d["k0-graph1"])/6; s0=(d["k6-graph0"]-d["k0-graph0"])/6
    print(f"\n  target intercept: {d['k0-graph1']:.2f} -> {d['k0-graph0']:.2f} ms "
          f"({(d['k0-graph0']/d['k0-graph1']-1)*100:+.1f}% without capture)")
    print(f"  draft slope:      {s1:.3f} -> {s0:.3f} ms ({(s0/s1-1)*100:+.1f}%)")
    print("\n  If the target moved and the draft slope did not, the drafts were never")
    print("  captured - and capturing them is worth roughly (slope - 0.76 ms) * k per step.")
PY
