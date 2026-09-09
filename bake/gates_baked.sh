#!/usr/bin/env bash
# Quality + speed gates for the baked checkpoint, against :8002 (serve-baked.sh).
# Order: greedy-equivalence (k=6, vs the current-stack reference), GSM8K (k=6),
# BetterBench (k=6), then a SPEC=0 relaunch for perplexity (NaN under MTP).
set -uo pipefail
G=$HOME/bake/gates; V=$HOME/Documents/L80/B65-repo/verify
EP=http://127.0.0.1:8002; M=launch80/Qwen3.8-27B-GPTQ-Int4-baked
log(){ echo "[gates $(date +%H:%M:%S)] $*"; }

log "1/4 greedy equivalence vs current stack (k=6)"
python3 $V/greedy_equiv.py --endpoint $EP/v1 --model $M --max-tokens 48 \
  --out $G/baked-k6.json --ref $G/ref-current-k6.json 2>&1 | tail -6 | sed 's/^/    /'

log "2/4 GSM8K n=100 (k=6)"
# gsm8k.py needs pyarrow: use the BetterBench venv python if it has it, else the container
PY=$HOME/bench/BetterBench/.venv/bin/python; $PY -c "import pyarrow" 2>/dev/null || PY=python3
$PY $V/gsm8k.py --endpoint $EP/v1 --model $M --n 100 --parquet $HOME/Documents/project608/research/r3/gsm8k.parquet --out $G/gsm8k-baked.json 2>&1 | tail -4 | sed 's/^/    /'

log "3/4 BetterBench decode (k=6)"
$HOME/bench/BetterBench/.venv/bin/betterbench run --endpoint $EP/v1 --model $M \
  --out $G/betterbench-baked.json > $G/betterbench-baked.log 2>&1
python3 - "$G/betterbench-baked.json" <<'PY' 2>/dev/null | sed 's/^/    /'
import json,sys
d=json.load(open(sys.argv[1]))
def walk(o,p=''):
    if isinstance(o,dict):
        for k,v in o.items():
            if k=='decode_tps' and isinstance(v,(int,float)): print(f"{p}: {v:.1f}")
            walk(v,p+'/'+k)
    elif isinstance(o,list):
        for i,v in enumerate(o): walk(v,f"{p}[{i}]")
walk(d)
PY

log "4/4 perplexity: relaunch baked with SPEC=0"
docker rm -f vllm-baked >/dev/null 2>&1 || true; sleep 4
BAKED=$HOME/bake-model PORT=8002 SPEC=0 nohup $HOME/Documents/project608/research/bake/serve-baked.sh > $G/baked-serve-nospec.log 2>&1 &
for i in $(seq 1 100); do curl -sf $EP/health >/dev/null 2>&1 && break; sleep 5; done
curl -sf $EP/health >/dev/null 2>&1 || { log "nospec baked server failed"; tail -10 $G/baked-serve-nospec.log; }
python3 $V/perplexity.py --endpoint $EP/v1 --base $EP --model $M --out $G/ppl-baked.json 2>&1 | tail -3 | sed 's/^/    /'
docker rm -f vllm-baked >/dev/null 2>&1 || true
log "DONE - baked arms complete; current-stack PPL/GSM8K arms still needed (driver, SPEC=0 / SPEC=6)"
systemctl --user start vllm-b65.service
