#!/usr/bin/env bash
# Publish ~/bake-model to the Hub. The token is read from .env into HF_TOKEN for the
# child process only; it is never echoed, logged, or written anywhere else.
set -uo pipefail
REPO="${REPO:-launch80/Qwen3.8-27B-GPTQ-Int4-baked}"
DIR=$HOME/bake-model; CARD=$HOME/Documents/project608/research/bake/README-modelcard.md
grep -q "_pending" "$CARD" && { echo "[publish] card still has _pending fields - refusing"; exit 1; }
cp "$CARD" "$DIR/README.md"; echo "[publish] card installed as README.md ($(wc -c < "$DIR/README.md") bytes)"
tok=$(sed -n 's/^HUGGINGFACE_TOKEN=//p' "$HOME/Documents/L80/B65-repo/.env" | tr -d '"' | tr -d "'")
[ -n "$tok" ] || { echo "[publish] no HUGGINGFACE_TOKEN in .env"; exit 1; }
# huggingface_hub lives only in the container; no GPU needed, just network + the folder
run(){ docker run --rm -v "$DIR:/model:ro" -v "$HOME/Documents/project608/research/bake:/w:ro" \
        -e HF_TOKEN="$tok" -e HF_HUB_ENABLE_HF_TRANSFER=0 --entrypoint bash p608-sycl:latest \
        -lc "python /w/publish.py --dir /model --repo $REPO $1; exit \${PIPESTATUS[0]:-\$?}" 2>&1 | grep -viE "hf_[A-Za-z0-9]{10}|Level Zero|device count|UserWarning|return _enum|count = torch"; }
echo "[publish] dry run:"; run --dry-run
echo "[publish] uploading (16 GB)..."
out=$(run ""); rc=$?; echo "$out"
if [ $rc -ne 0 ] || echo "$out" | grep -qE "Error|Forbidden|403|401"; then echo "[publish] FAILED (see above)"; exit 1; fi
echo "[publish] DONE -> https://huggingface.co/$REPO"
