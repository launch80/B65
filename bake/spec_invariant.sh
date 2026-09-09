#!/usr/bin/env bash
# The real greedy-equivalence invariant for the bake: SAME weights, speculation ON vs
# OFF must produce byte-identical greedy output. Run after gates_baked.sh.
set -uo pipefail
G=$HOME/bake/gates; V=$HOME/Documents/L80/B65-repo/verify
EP=http://127.0.0.1:8002; M=launch80/Qwen3.8-27B-GPTQ-Int4-baked
systemctl --user stop vllm-b65.service; docker rm -f vllm-b65 vllm-baked >/dev/null 2>&1 || true; sleep 4
BAKED=$HOME/bake-model PORT=8002 SPEC=0 nohup $HOME/Documents/project608/research/bake/serve-baked.sh > $G/baked-nospec-inv.log 2>&1 &
for i in $(seq 1 100); do curl -sf $EP/health >/dev/null 2>&1 && break; sleep 5; done
curl -sf $EP/health >/dev/null 2>&1 || { echo "[inv] baked SPEC=0 failed"; tail -8 $G/baked-nospec-inv.log; }
echo "[inv] baked SPEC=0 vs baked SPEC=6 (same weights):"
python3 $V/greedy_equiv.py --endpoint $EP/v1 --model $M --max-tokens 48 --out $G/baked-k0.json --ref $G/baked-k6.json 2>&1 | tail -4 | sed 's/^/    /'
docker rm -f vllm-baked >/dev/null 2>&1 || true
systemctl --user start vllm-b65.service; echo "[inv] DONE, daily driver restarted"
