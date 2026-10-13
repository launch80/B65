#!/usr/bin/env python3
"""P608 R6 draft-vocab prefix for DFlash/DFlash2 (set P608_DRAFT_VOCAB=32768).

The drafter shares the target's GPTQ INT4 lm_head (248k rows). Draft logits only need to be good
enough to propose; the target verifies. As with the MTP R6 prefix (research/r3), the draft scores
only the first n vocabulary rows (a contiguous prefix of the on-disk qweight, ~13% of the read).
Reuses r3_draft_logits from the helper that patch_r3_lmhead_int4.py installs (run it first).
Patched: DFlashQwen3ForCausalLM.compute_logits, DFlash2Qwen3ForCausalLM.compute_candidates.
"""
import os, sys
roots = sorted(set(os.path.realpath(p) for p in ["/opt/venv/lib/python3.12/site-packages/vllm", "/workspace/vllm/vllm", "/opt/vllm/vllm"] if os.path.isdir(p)))
try:
    import vllm; roots = sorted(set(roots + [os.path.realpath(os.path.dirname(vllm.__file__))]))
except Exception: pass
HELPER = '''
def _p608_r6_logits(module, h):  # P608 R6: draft-only vocab prefix over the shared INT4 head
    import os as _os
    if int(_os.environ.get("P608_DRAFT_VOCAB", "0") or 0) <= 0:
        return None
    try:
        from vllm.model_executor.models.p608_r3_lmhead_int4 import r3_draft_logits
    except Exception:
        return None
    head = getattr(module, "lm_head", None)
    for a in ("scales", "weight"):
        t = getattr(head, a, None)
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            if h.dtype != t.dtype:
                h = h.to(t.dtype)
            break
    return r3_draft_logits(module, h)
'''
EDITS = {
 "model_executor/models/qwen3_dflash.py": [
  ("import torch.nn.functional as F\n", "import torch.nn.functional as F\n" + HELPER),
  ("        logits = self.logits_processor(self.lm_head, hidden_states)\n        if self.draft_id_to_target_id is None:",
   "        _r6 = _p608_r6_logits(self, hidden_states)\n        logits = _r6 if _r6 is not None else self.logits_processor(self.lm_head, hidden_states)\n        if self.draft_id_to_target_id is None:")],
 "model_executor/models/qwen3_dflash2.py": [
  ("from vllm.model_executor.layers.logits_processor import LogitsProcessor\n",
   "from vllm.model_executor.layers.logits_processor import LogitsProcessor\nfrom vllm.model_executor.models.qwen3_dflash import _p608_r6_logits  # P608 R6\n"),
  ("        return self.candidate_logits_processor.get_top_k_tokens(\n",
   "        _r6 = _p608_r6_logits(self, hidden_states)\n"
   "        if _r6 is not None:\n"
   "            _v, _i = torch.topk(_r6, self.model.candidate_selector.top_k, dim=-1)\n"
   "            _v = _v.float(); _lp = self.candidate_logits_processor\n"
   "            if _lp.scale != 1.0: _v = _v * _lp.scale\n"
   "            if _lp.soft_cap is not None: _v = torch.tanh(_v / _lp.soft_cap) * _lp.soft_cap\n"
   "            return _i, _v\n"
   "        return self.candidate_logits_processor.get_top_k_tokens(\n")],
}
ok = True
for root in roots:
    for rel, edits in EDITS.items():
        p = os.path.join(root, rel)
        if not os.path.exists(p): print(f"  skip (missing) {p}"); continue
        s = open(p).read(); n = 0
        for old, new in edits:
            if new in s: continue
            if old not in s: print(f"  ANCHOR MISSING in {p}: {old[:70]!r}"); ok = False; continue
            s = s.replace(old, new, 1); n += 1
        compile(s, p, "exec"); open(p, "w").write(s); print(f"  patched {p} ({n} edits)")
print("P608_DRAFT_VOCAB =", os.environ.get("P608_DRAFT_VOCAB")); sys.exit(0 if ok else 1)
