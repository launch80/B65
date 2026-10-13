#!/usr/bin/env python3
"""P608 mixed-dtype DFlash: target fp16, drafter in its native bf16 (set P608_DRAFT_DTYPE=bfloat16).

Why: vLLM copies the target dtype onto the draft ModelConfig; on the fp16 XPU stack the bf16-trained
DFlash2 drafter then runs in fp16 and proposes garbage (0 acceptance, 2026-10-13). Running the whole
target in bf16 fixes acceptance but costs ~5% decode. This keeps the target fp16 and casts at the
drafter's boundaries:
  1. config/speculative.py       draft ModelConfig dtype <- $P608_DRAFT_DTYPE
  2. v1/spec_decode/llm_base_proposer.py   proposer buffers use the draft dtype
  3. models/qwen3_dflash.py      draft params from draft dtype; casts: shared embedding -> draft dtype,
                                 target aux hidden -> fc dtype, context states -> fused KV dtype,
                                 draft hidden -> shared lm_head dtype (GPTQ scales are fp16)
  4. models/qwen3_dflash2.py     same lm_head cast for the candidate top-k
Idempotent; no-op when the env var is unset except for the dtype-neutral casts.
"""
import os, sys, glob
roots = sorted(set(p for p in ["/opt/venv/lib/python3.12/site-packages/vllm", "/workspace/vllm/vllm", "/opt/vllm/vllm"] if os.path.isdir(p)))
try:
    import vllm; roots.append(os.path.dirname(vllm.__file__))
except Exception: pass
roots = sorted(set(os.path.realpath(r) for r in roots))
HELPER = '''
def _p608_cast_for_head(lm_head, h):  # P608 mixed dtype: shared target head may be fp16 / GPTQ fp16 scales
    for a in ("scales", "weight"):
        t = getattr(lm_head, a, None)
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            return h.to(t.dtype) if h.dtype != t.dtype else h
    return h
'''
EDITS = {
 "config/speculative.py": [
  ("dtype=self.target_model_config.dtype,",
   'dtype=(__import__("os").environ.get("P608_DRAFT_DTYPE") or self.target_model_config.dtype),  # P608')],
 "v1/spec_decode/llm_base_proposer.py": [
  ("        self.dtype = vllm_config.model_config.dtype\n",
   '        self.dtype = (self.draft_model_config.dtype if __import__("os").environ.get("P608_DRAFT_DTYPE") else vllm_config.model_config.dtype)  # P608\n')],
 "model_executor/models/qwen3_dflash.py": [
  ("import torch.nn.functional as F\n", "import torch.nn.functional as F\n" + HELPER),
  ("params_dtype=vllm_config.model_config.dtype", "params_dtype=vllm_config.speculative_config.draft_model_config.dtype"),
  ("torch.zeros(self.config.hidden_size, dtype=vllm_config.model_config.dtype)",
   "torch.zeros(self.config.hidden_size, dtype=vllm_config.speculative_config.draft_model_config.dtype)"),
  ("        embeds = self.embed_tokens(input_ids)\n",
   "        embeds = self.embed_tokens(input_ids)\n        embeds = embeds.to(self.mask_embedding.dtype)  # P608: shared target embedding may be fp16\n"),
  ("        result = self.model.fc(hidden_states)\n",
   "        result = self.model.fc(_p608_cast_for_head(self.model.fc, hidden_states))  # P608 (fc may be GPTQ)\n"),
  ("        num_ctx = context_states.shape[0]\n        L = self._num_attn_layers\n",
   "        context_states = context_states.to(self._fused_kv_weight.dtype)  # P608\n        num_ctx = context_states.shape[0]\n        L = self._num_attn_layers\n"),
  ("        logits = self.logits_processor(self.lm_head, hidden_states)\n        if self.draft_id_to_target_id is None:",
   "        hidden_states = _p608_cast_for_head(self.lm_head, hidden_states)\n        logits = self.logits_processor(self.lm_head, hidden_states)\n        if self.draft_id_to_target_id is None:")],
 "model_executor/models/qwen3_dflash2.py": [
  ("params_dtype=vllm_config.model_config.dtype", "params_dtype=vllm_config.speculative_config.draft_model_config.dtype"),
  ("            self.lm_head, hidden_states, self.model.candidate_selector.top_k\n",
   "            self.lm_head, _p608_cast_for_head(self.lm_head, hidden_states), self.model.candidate_selector.top_k\n"),
  ("from vllm.model_executor.layers.logits_processor import LogitsProcessor\n",
   "from vllm.model_executor.layers.logits_processor import LogitsProcessor\nfrom vllm.model_executor.models.qwen3_dflash import _p608_cast_for_head  # P608\n")],
}
ok = True
for root in roots:
    for rel, edits in EDITS.items():
        p = os.path.join(root, rel)
        if not os.path.exists(p): print(f"  skip (missing) {p}"); continue
        s = open(p).read(); n = 0
        for old, new in edits:
            if new in s: continue  # already applied
            if old not in s: print(f"  ANCHOR MISSING in {p}: {old[:60]!r}"); ok = False; continue
            s = s.replace(old, new); n += 1
        open(p, "w").write(s); print(f"  patched {p} ({n} edits)")
print("P608_DRAFT_DTYPE =", os.environ.get("P608_DRAFT_DTYPE"))
sys.exit(0 if ok else 1)
