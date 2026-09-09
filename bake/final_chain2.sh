#!/usr/bin/env bash
# Wait for today's current-stack BetterBench -> L80 publish (summary <= 600 chars) ->
# fill BOTH card columns + L80 URL -> HF upload. Detached; logs everything.
set -uo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin:$PATH
export L80_BETTERBENCH=$HOME/bench/BetterBench/.venv/bin/betterbench
G=$HOME/bake/gates; CARD=$HOME/Documents/project608/research/bake/README-modelcard.md
BB=$G/betterbench-baked-r6on.json; BC=$G/betterbench-current-today.json; REPO=launch80/Qwen3.8-27B-GPTQ-Int4-baked
log(){ echo "[final2 $(date +%H:%M:%S)] $*"; }
weighted(){ python3 - "$1" <<'PY'
import json,sys
C=json.load(open(sys.argv[1]))["single_stream"]; Wt={"code":.3,"reasoning":.2,"prose":.15,"json":.15,"file_edit":.1,"summarization":.1}; w=0
for c,runs in C.items():
    if c=="chat": continue
    v=[r["decode_tps"] for r in runs if isinstance(r,dict) and isinstance(r.get("decode_tps"),(int,float))]
    if v: w+=Wt.get(c,0)*sum(v)/len(v)
print(f"{w:.1f}")
PY
}
until [ -s "$BC" ] && ! pgrep -f "betterbench run" >/dev/null; do sleep 30; done
WB=$(weighted "$BB"); WC=$(weighted "$BC")
log "weighted decode: baked $WB   current stack (today, GDN v5 + capture off) $WC   (old 09-07 baseline 77.8)"

TITLE="Project 608 · baked checkpoint (GPTQ lm_head + MTP draft on disk) · Qwen3.8-27B · Arc Pro B65 · MTP k=6"
SUMMARY="SergiioB's Qwen3.8-27B GPTQ-Int4 with the lm_head and MTP draft quantized to INT4 on disk (Hessian-compensated) instead of at boot by patches: stock vLLM loads it with zero weight patches. Weighted decode ${WB} tok/s vs ${WC} for the patched stock checkpoint, same B65, same day, same config (vLLM 0.27.2-xpu, capture off). Perplexity 6.133 vs 6.144; GSM8K paired 91 vs 89; spec-on/off greedy 8/8 identical. Checkpoint: huggingface.co/${REPO}; scripts and raw JSON: github.com/launch80/B65."
log "summary length: ${#SUMMARY} (limit 600)"
[ ${#SUMMARY} -le 600 ] || { log "summary too long"; exit 1; }
log "L80 dry run"; L80 betterbench --results "$BB" --title "$TITLE" --summary "$SUMMARY" --note engine=vllm-0.27.2-xpu --note model=$REPO --note hardware="Arc Pro B65 32GB" --dry-run 2>&1 | tail -3 | sed 's/^/    /'
log "L80 publish"
OUT=$(L80 betterbench --results "$BB" --title "$TITLE" --summary "$SUMMARY" --note engine=vllm-0.27.2-xpu --note model=$REPO --note hardware="Arc Pro B65 32GB" 2>&1)
echo "$OUT" | tail -3 | sed 's/^/    /'
URL=$(echo "$OUT" | grep -oE 'https://(www\.)?launch80\.com/a/[A-Za-z0-9-]+' | head -1)
[ -n "$URL" ] || { log "no L80 URL captured - stopping before HF upload"; exit 1; }
log "L80 page: $URL"
python3 - "$CARD" "$WB" "$WC" "$URL" <<'PY'
import sys,re; p,wb,wc,url=sys.argv[1:]; s=open(p).read()
s=s.replace("| BetterBench decode, MTP k=6, weighted | _pending R6 fix_ | 77.8 | — |",
            f"| BetterBench decode, MTP k=6, weighted, same day/config | **{wb}** | {wc} | — |")
s=s.replace("Layer-wise output error on held-out activations",
            f"BetterBench run page (Launch80): {url} — a self-reported rendering; free-tier pages expire,\nso the results JSON in the GitHub repo is the citation of record.\n\nLayer-wise output error on held-out activations",1)
assert "_pending" not in s; open(p,'w').write(s); print("    card: both BetterBench columns + L80 URL filled")
PY
log "HF upload -> $REPO"; bash $G/publish_run.sh 2>&1 | sed 's/^/    /'
log "DONE"
