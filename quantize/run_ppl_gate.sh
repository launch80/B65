#!/usr/bin/env bash
# Perplexity gate for W3A16: same server script, same corpus, one variable.
#
# Speculation is OFF in both arms - prompt_logprobs returns NaN under MTP on this
# stack, and teacher forcing is what makes the comparison meaningful anyway: every
# position is scored against the true prefix, so both arms see identical inputs.
set -uo pipefail
OUT="${OUT:-$HOME/w3gate}"; mkdir -p "$OUT"
SERVE="$HOME/Documents/project608/serving/serve-w3.sh"
VERIFY="$HOME/Documents/L80/B65-repo/verify"
PORT=8002

run_arm () {
  local arm="$1" w3="$2" skip="${3:-}"
  echo "=== arm: $arm (W3=$w3 skip='${skip}') ==="
  docker rm -f vllm-w3 >/dev/null 2>&1 || true
  W3="$w3" W3_SKIP="$skip" PORT=$PORT nohup "$SERVE" > "$OUT/serve-$arm.log" 2>&1 &
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 5
  done
  if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "  server never came up; tail of log:"; tail -25 "$OUT/serve-$arm.log"; return 1
  fi
  grep -E "P608-W3|P608_W3" "$OUT/serve-$arm.log" | tail -2
  python3 "$VERIFY/perplexity.py" --endpoint "http://127.0.0.1:$PORT/v1" \
      --base "http://127.0.0.1:$PORT" --out "$OUT/ppl-$arm.json" 2>&1 | tail -4
  docker rm -f vllm-w3 >/dev/null 2>&1 || true
  sleep 5
}

run_arm baseline 0
run_arm w3 1

python3 - "$OUT" <<'PY'
import json, sys, os
o = sys.argv[1]
try:
    b = json.load(open(f"{o}/ppl-baseline.json"))
    w = json.load(open(f"{o}/ppl-w3.json"))
except Exception as e:
    sys.exit(f"could not read both results: {e}")
def ppl(d):
    for k in ("perplexity", "ppl", "value"):
        if k in d: return d[k]
    for v in d.values():
        if isinstance(v, dict):
            for k in ("perplexity", "ppl"):
                if k in v: return v[k]
    return None
pb, pw = ppl(b), ppl(w)
print(f"\n{'='*64}")
print(f"  PERPLEXITY GATE   (gate: W3 within +1.5%; R3 shipped at +0.840%)")
print(f"    int4 baseline : {pb}")
print(f"    W3A16         : {pw}")
if pb and pw:
    d = (pw/pb - 1) * 100
    print(f"    delta         : {d:+.3f}%   {'PASS' if d <= 1.5 else 'FAIL'}")
print(f"{'='*64}")
PY
