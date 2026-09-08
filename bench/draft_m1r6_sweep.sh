#!/usr/bin/env bash
# Re-evaluate the draft configuration at k=6.
#
# M1 (INT4 draft linears) and R6 (32768-row draft lm_head prefix) each make the draft
# cheaper and worse. Both were measured as net wins at MTP4. The balance depends on k,
# so this re-runs the 2x2 at the k we actually serve, on one fixed workload.
#
# Leaves the daily driver running when it finishes, whatever happens.
set -uo pipefail
OUT="${OUT:-$HOME/w3gate/draft}"; mkdir -p "$OUT"
SPEC_K="${SPEC_K:-6}"
ACC="$(dirname "$0")/draft_accept.py"
SERVE="$(dirname "$0")/../serving/serve-b65.sh"

cleanup() {
  docker rm -f vllm-b65 >/dev/null 2>&1 || true
  systemctl --user start vllm-b65.service >/dev/null 2>&1 || true
  echo "  (daily driver restarted)"
}
trap cleanup EXIT

systemctl --user stop vllm-b65.service >/dev/null 2>&1 || true
docker rm -f vllm-b65 >/dev/null 2>&1 || true
sleep 3

arm () {
  local label="$1" m1="$2" vocab="$3"
  local r6=1; [[ "$vocab" == "0" ]] && r6=0
  docker rm -f vllm-b65 >/dev/null 2>&1 || true
  SPEC="$SPEC_K" MTP_INT4="$m1" DRAFT_VOCAB="$vocab" \
    nohup "$SERVE" > "$OUT/serve-$label.log" 2>&1 &
  for i in $(seq 1 90); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break
    sleep 5
  done
  if ! curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "  $label: server did not come up"; tail -12 "$OUT/serve-$label.log"; return 1
  fi
  python3 "$ACC" --label "$label" --m1 "$m1" --r6 "$r6" --out "$OUT/$label.json"
  docker rm -f vllm-b65 >/dev/null 2>&1 || true
  sleep 4
}

echo "  k=$SPEC_K, fixed 8-prompt workload, warm-up excluded"
echo "  ------------------------------------------------------------------------"
arm "m1+r6 (current)" 1 32768
arm "m1 only"         1 0
arm "r6 only"         0 32768
arm "neither"         0 0

echo "  ------------------------------------------------------------------------"
python3 - "$OUT" <<'PY'
import glob, json, os, sys
rows=[json.load(open(f)) for f in sorted(glob.glob(os.path.join(sys.argv[1],"*.json")))]
if not rows: sys.exit("  no results")
rows.sort(key=lambda r: r["gb_per_token"])
print(f"  {'config':<20s} {'accept':>7s} {'tok/step':>9s} {'draft GB':>9s} "
      f"{'GB/step':>8s} {'GB/token':>9s} {'obs tok/s':>10s}")
for r in rows:
    print(f"  {r['label']:<20s} {r['acceptance']*100:>6.1f}% {r['tokens_per_step']:>9.2f} "
          f"{r['draft_gb']:>9.3f} {r['gb_step']:>8.2f} {r['gb_per_token']:>9.3f} "
          f"{r['observed_tok_s']:>10.1f}")
b=rows[0]
print(f"\n  best by GB/token: {b['label']}  ({b['gb_per_token']:.3f})")
print("  GB/token is the byte model's prediction; obs tok/s is what actually happened.")
print("  They should rank the same way - if they do not, the byte model is missing something.")
PY
