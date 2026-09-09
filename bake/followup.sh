#!/usr/bin/env bash
# After gates_baked.sh: (1) GSM8K at a 1536-token budget on BOTH arms, to separate a
# verbosity shift from a reasoning loss; (2) the spec-on-vs-off invariant on the baked
# weights. Leaves the daily driver running.
set -uo pipefail
G=$HOME/bake/gates; V=$HOME/Documents/L80/B65-repo/verify; PQ=$HOME/Documents/project608/research/r3/gsm8k.parquet
PY=$HOME/bench/BetterBench/.venv/bin/python
SB=$HOME/Documents/project608/research/bake/serve-baked.sh
MB=launch80/Qwen3.8-27B-GPTQ-Int4-baked; MC=SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
log(){ echo "[followup $(date +%H:%M:%S)] $*"; }
up(){ for i in $(seq 1 100); do curl -sf $1/health >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
gsm(){ $PY $V/gsm8k.py --endpoint $1/v1 --model $2 --n 100 --max-tokens 1536 --parquet $PQ --out $3 2>&1 | tail -1 | sed 's/^/    /'; }

systemctl --user stop vllm-b65.service; docker rm -f vllm-b65 vllm-baked >/dev/null 2>&1 || true; sleep 4
log "A. baked k=6: GSM8K @1536"
BAKED=$HOME/bake-model PORT=8002 SPEC=6 nohup $SB > $G/fu-baked-k6.log 2>&1 &
up http://127.0.0.1:8002 && gsm http://127.0.0.1:8002 $MB $G/gsm8k-baked-1536.json || log "baked k6 failed to start"
docker rm -f vllm-baked >/dev/null 2>&1 || true; sleep 4

log "B. baked SPEC=0: greedy invariant vs baked k=6 (same weights)"
BAKED=$HOME/bake-model PORT=8002 SPEC=0 nohup $SB > $G/fu-baked-k0.log 2>&1 &
up http://127.0.0.1:8002 && python3 $V/greedy_equiv.py --endpoint http://127.0.0.1:8002/v1 --model $MB --max-tokens 48 \
   --out $G/baked-k0.json --ref $G/baked-k6.json 2>&1 | tail -3 | sed 's/^/    /' || log "baked k0 failed"
docker rm -f vllm-baked >/dev/null 2>&1 || true; sleep 4

log "C. current stack k=6: GSM8K @1536"
systemctl --user start vllm-b65.service
up http://127.0.0.1:8000 && gsm http://127.0.0.1:8000 $MC $G/gsm8k-current-1536.json || log "driver failed to start"
log "DONE (daily driver left running)"
