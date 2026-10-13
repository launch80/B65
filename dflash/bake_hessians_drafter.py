#!/usr/bin/env python3
"""Hessians for the DFlash2 drafter's linears, captured in-process while vLLM serves the baked
target with the drafter attached (target bf16 so the drafter runs in its native dtype).

  MODEL_PATH=/model DRAFT_PATH=/draft python bake_hessians_drafter.py --out /out --prompts /out/prompts.json --group A

group A: layers.*.self_attn.o_proj, mlp.gate_up_proj and fc (~3.5 GB of fp32 H); B1/B2: mlp.down_proj layers 0-2 / 3-4 (1.2 GB each)
(fp32 Hessians live outside vLLM's budget: util 0.74 + max_num_batched_tokens 512 leaves ~8 GB on the 32 GB card)
Writes hessians-drafter-<group>.pt: {"H","Xe","nrows","gen_tokens"} keyed by vLLM module name.
"""
import argparse, json, os, re, time, torch
from vllm import LLM, SamplingParams
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--prompts", required=True)
ap.add_argument("--group", required=True, choices=["A", "B1", "B2"])
ap.add_argument("--gen", type=int, default=128); ap.add_argument("--spec", type=int, default=7)
ap.add_argument("--util", type=float, default=0.80); ap.add_argument("--max-model-len", type=int, default=2048)
ap.add_argument("--heldout-every", type=int, default=64); ap.add_argument("--heldout-cap", type=int, default=2048)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
EVERY, EVAL_CAP = a.heldout_every, a.heldout_cap
PAT = {"A": r"^model\.(layers\.\d+\.(self_attn\.o_proj|mlp\.gate_up_proj)|fc)$",   # qkv stays dense (context-KV precompute)
       "B1": r"^model\.layers\.[012]\.mlp\.down_proj$", "B2": r"^model\.layers\.[34]\.mlp\.down_proj$"}[a.group]
llm = LLM(model=os.environ["MODEL_PATH"], quantization="gptq", dtype="bfloat16",
          max_model_len=a.max_model_len, gpu_memory_utilization=a.util, max_num_seqs=4, max_num_batched_tokens=512, enforce_eager=True, kv_cache_dtype="fp8",
          speculative_config={"method": "dflash", "model": os.environ["DRAFT_PATH"], "num_speculative_tokens": a.spec})
def find_runner(eng):
    for path in (("model_executor", "driver_worker", "model_runner"),
                 ("engine_core", "engine_core", "model_executor", "driver_worker", "model_runner"),
                 ("engine_core", "model_executor", "driver_worker", "model_runner")):
        o = eng
        try:
            for p in path: o = getattr(o, p)
            if hasattr(o, "model"): print(f"  model_runner via {'.'.join(path)}", flush=True); return o
        except AttributeError: continue
    raise RuntimeError("model runner not found; is VLLM_ENABLE_V1_MULTIPROCESSING=0 set?")
runner = find_runner(llm.llm_engine); target = runner.model; draft = None
for k, v in list(vars(runner).items()):
    cand = getattr(v, "model", None) if not isinstance(v, torch.nn.Module) else v
    if isinstance(cand, torch.nn.Module) and cand is not target and any(n.endswith(".fc") or n == "fc" for n, _ in cand.named_modules()):
        print(f"  draft model via runner.{k}: {type(cand).__name__}", flush=True); draft = cand; break
assert draft is not None, "DFlash draft model not found on the runner"
MODS = {n: m for n, m in draft.named_modules() if re.match(PAT, n)}
assert MODS, f"no drafter modules matched {PAT}; have: {[n for n,_ in draft.named_modules()][:40]}"
for n, m in MODS.items():
    w = getattr(m, "weight", None); assert w is not None and w.dtype in (torch.float16, torch.bfloat16), f"{n}: not dense"
    print(f"  hook {n:<42s} weight {tuple(w.shape)} {w.dtype}", flush=True)
tok = llm.get_tokenizer()
items = json.load(open(a.prompts))["mix"]
prompts = [tok.apply_chat_template(it, tokenize=False, add_generation_prompt=True) if isinstance(it, list) else it for it in items]
prompts = [p for p in prompts if p.strip()]
H, nrows, Xe = {}, {}, {}
def mk(nm):
    def hook(mod, inp):
        x = inp[0]
        if x is None: return
        x = x.detach().reshape(-1, x.shape[-1]).float()
        if nm not in H:
            H[nm] = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.float32, device=x.device); nrows[nm] = 0; Xe[nm] = []
        off = nrows[nm] % EVERY
        idx = torch.arange(off, x.shape[0], EVERY, device=x.device) if off < x.shape[0] else torch.empty(0, dtype=torch.long, device=x.device)
        if idx.numel() and sum(t.shape[0] for t in Xe[nm]) < EVAL_CAP: Xe[nm].append(x[idx].clone())
        keep = torch.ones(x.shape[0], dtype=torch.bool, device=x.device); keep[idx] = False
        xh = x[keep]; H[nm] += xh.t() @ xh; nrows[nm] += x.shape[0]
    return hook
hooks = [m.register_forward_pre_hook(mk(n)) for n, m in MODS.items()]
t0 = time.time()
outs = llm.generate(prompts, SamplingParams(max_tokens=a.gen, temperature=0), use_tqdm=False)
for h in hooks: h.remove()
gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
print(f"\n[mix/{a.group}] {len(prompts)} prompts, {gen_tokens} generated tokens in {time.time()-t0:.0f}s", flush=True)
missing = [n for n in MODS if n not in H]; assert not missing, f"no activations for {missing}"
for n in MODS: print(f"  {n:<42s} rows {nrows[n]:>8d} heldout {sum(t.shape[0] for t in Xe[n]):>5d} K={H[n].shape[0]}", flush=True)
dst = os.path.join(a.out, f"hessians-drafter-{a.group}.pt")
torch.save({"H": {k: v.cpu() for k, v in H.items()}, "Xe": {k: [t.cpu() for t in v] for k, v in Xe.items()},
            "nrows": {k: int(v) for k, v in nrows.items()}, "gen_tokens": gen_tokens, "prompts": len(prompts)}, dst)
print(f"  -> {dst}", flush=True)
