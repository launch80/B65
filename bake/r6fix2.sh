#!/usr/bin/env bash
set -uo pipefail
G=$HOME/bake/gates; SB=$HOME/Documents/project608/research/bake/serve-baked.sh
M=launch80/Qwen3.8-27B-GPTQ-Int4-baked; EP=http://127.0.0.1:8002
log(){ echo "[r6fix2 $(date +%H:%M:%S)] $*"; }
systemctl --user stop vllm-b65.service >/dev/null 2>&1; docker rm -f vllm-b65 vllm-baked >/dev/null 2>&1 || true; sleep 4
BAKED=$HOME/bake-model PORT=8002 SPEC=6 nohup $SB > $G/r6fix2-serve.log 2>&1 &
for i in $(seq 1 100); do curl -sf $EP/health >/dev/null 2>&1 && break; sleep 5; done
curl -sf $EP/health >/dev/null 2>&1 || { log "server failed"; tail -12 $G/r6fix2-serve.log; systemctl --user start vllm-b65.service; exit 1; }
curl -s $EP/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to five.\"}],\"max_tokens\":24}" >/dev/null
sleep 3; R6=$(grep -m1 'P608-R6\] draft head' $G/r6fix2-serve.log | cut -c1-230); log "R6: ${R6:-NOT ENGAGED}"
if [ -z "$R6" ]; then grep -E "P608|Traceback|Error" $G/r6fix2-serve.log | grep -v "ERROR\]   File" | tail -6 | cut -c1-180; docker rm -f vllm-baked >/dev/null 2>&1; systemctl --user start vllm-b65.service; exit 1; fi
log "acceptance + tok/s (fixed workload)"
python3 $HOME/Documents/project608/research/r11_accept.py --endpoint $EP --model $M --label "baked k=6 R6 on" --m1 1 --r6 1 --out $G/accept-baked-k6-r6on.json 2>&1 | tail -1
log "BetterBench (k=6)"
$HOME/bench/BetterBench/.venv/bin/betterbench run --endpoint $EP/v1 --model $M --out $G/betterbench-baked-r6on.json > $G/betterbench-baked-r6on.log 2>&1
python3 - "$G/betterbench-baked-r6on.json" <<'PY'
import json,sys
C=json.load(open(sys.argv[1]))["single_stream"]; W={"code":.3,"reasoning":.2,"prose":.15,"json":.15,"file_edit":.1,"summarization":.1}; w=0
for c,runs in C.items():
    if c=="chat": continue
    v=[r["decode_tps"] for r in runs if isinstance(r,dict) and isinstance(r.get("decode_tps"),(int,float))]; m=sum(v)/len(v) if v else float("nan"); w+=W.get(c,0)*m
    print(f"    {c:<14s} {m:6.1f}")
print(f"    WEIGHTED       {w:6.1f}   (baseline 77.8, baked R6-off 67.9/69.2)")
PY
docker rm -f vllm-baked >/dev/null 2>&1 || true; systemctl --user start vllm-b65.service; log "DONE (daily driver restarted)"
