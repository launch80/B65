#!/usr/bin/env python3
"""Bake R3 + M1 into the checkpoint: GPTQ-quantize lm_head and the MTP draft layer.

Today both are quantized to INT4 *in VRAM at boot* by patches, with round-to-nearest.
This produces the same INT4 g128 symmetric GPTQ tensors on disk, with Hessian error
compensation, in exactly the layout the body already uses - so stock vLLM loads them
with no weight patches at all.

Targets (HF names in the checkpoint; the draft's vLLM module names drop "mtp." -> "model."):
    lm_head.weight                       [248320, 5120]   shared target/draft head
    mtp.fc.weight                        [5120, 10240]
    mtp.layers.0.self_attn.{q,k,v}_proj  share one input  (vLLM: qkv_proj)
    mtp.layers.0.self_attn.o_proj
    mtp.layers.0.mlp.{gate,up}_proj      share one input  (vLLM: gate_up_proj)
    mtp.layers.0.mlp.down_proj

Run with NO patches (dense weights), MTP on so the draft executes. Every hooked module
must expose a dense `.weight`; the script refuses otherwise.

Output layout per tensor, matching the body byte-for-byte:
    qweight int32 [K/8, N]     nibble j at bit 4j = code of row 8w+j
    scales  fp16  [K/128, N]
    qzeros  int32 [K/128, N/8] every nibble 7   (sym zero 8, stored as zero-1)
    g_idx   int32 [K]          k // 128
"""
import argparse, json, os, sys, time, urllib.request
import torch
from vllm import LLM, SamplingParams
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--prompts", type=int, default=256)
ap.add_argument("--seq-len", type=int, default=384)
ap.add_argument("--gen", type=int, default=128)
ap.add_argument("--damp", type=float, default=0.01)
ap.add_argument("--util", type=float, default=0.80)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
G, DEV = 128, "xpu"

CALIB = ("https://raw.githubusercontent.com/pytorch/examples/main/"
         "word_language_model/data/wikitext-2/valid.txt")
cache = os.path.join(a.out, "calib.txt")
if not os.path.exists(cache):
    open(cache, "wb").write(urllib.request.urlopen(CALIB, timeout=60).read())
text = open(cache).read()

llm = LLM(model=os.environ["MODEL_PATH"], quantization="gptq", dtype="float16",
          max_model_len=1024, gpu_memory_utilization=a.util, max_num_seqs=4,
          enforce_eager=True,
          speculative_config={"method": "mtp", "num_speculative_tokens": 6})
runner = llm.llm_engine.model_executor.driver_worker.model_runner
target = runner.model
drafter = getattr(runner, "drafter", None)
draft = getattr(drafter, "model", None)
assert draft is not None, "could not find the MTP draft model on the runner"
tok = llm.get_tokenizer()

# ---------------------------------------------------------------- targets
# (HF tensor name, module owning the INPUT we need, slice of that module's output)
tmods = {n: m for n, m in target.named_modules()}
dmods = {n: m for n, m in draft.named_modules()}
def find(mods, suffix):
    hits = [n for n in mods if n.endswith(suffix)]
    assert len(hits) == 1, f"{suffix}: {hits}"
    return hits[0], mods[hits[0]]

lm_name, lm_mod = find(tmods, "lm_head")
# ParallelLMHead is never called through forward(): compute_logits runs
# logits_processor(lm_head, hidden_states). Hook the processor and take inp[1].
# Target and draft share the head, so both processors feed the same Hessian.
lp_t_name, lp_t = find(tmods, "logits_processor")
lp_d_name, lp_d = find(dmods, "logits_processor")
fc_name, fc_mod = find(dmods, ".fc")
qkv_name, qkv_mod = find(dmods, "layers.0.self_attn.qkv_proj")
o_name, o_mod = find(dmods, "layers.0.self_attn.o_proj")
gu_name, gu_mod = find(dmods, "layers.0.mlp.gate_up_proj")
dn_name, dn_mod = find(dmods, "layers.0.mlp.down_proj")

for n, m in [(lm_name, lm_mod), (fc_name, fc_mod), (qkv_name, qkv_mod),
             (o_name, o_mod), (gu_name, gu_mod), (dn_name, dn_mod)]:
    w = getattr(m, "weight", None)
    assert w is not None and w.dtype in (torch.float16, torch.bfloat16), \
        f"{n} is not a dense fp16/bf16 module - run with NO weight patches"
    print(f"  hook {n:<45s} weight {tuple(w.shape)} {w.dtype}")

# how the merged vLLM weights split back into HF tensors (rows = output features)
cfg = llm.llm_engine.model_config.hf_config
tc = getattr(cfg, "text_config", cfg)
nh, nkv, hd = tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim
q_n, kv_n = nh * hd, nkv * hd
inter = tc.intermediate_size
SPLITS = {
    lm_name:  [("lm_head.weight", 0, None)],
    fc_name:  [("mtp.fc.weight", 0, None)],
    qkv_name: [("mtp.layers.0.self_attn.q_proj.weight", 0, q_n),
               ("mtp.layers.0.self_attn.k_proj.weight", q_n, q_n + kv_n),
               ("mtp.layers.0.self_attn.v_proj.weight", q_n + kv_n, q_n + 2 * kv_n)],
    o_name:   [("mtp.layers.0.self_attn.o_proj.weight", 0, None)],
    gu_name:  [("mtp.layers.0.mlp.gate_proj.weight", 0, inter),
               ("mtp.layers.0.mlp.up_proj.weight", inter, 2 * inter)],
    dn_name:  [("mtp.layers.0.mlp.down_proj.weight", 0, None)],
}
MODS = {lm_name: lm_mod, fc_name: fc_mod, qkv_name: qkv_mod,
        o_name: o_mod, gu_name: gu_mod, dn_name: dn_mod}

# ---------------------------------------------------------------- calibration
ids = tok(text, add_special_tokens=False)["input_ids"]
prompts = [tok.decode(ids[i * a.seq_len:(i + 1) * a.seq_len]) for i in range(a.prompts)]
prompts = [p for p in prompts if p.strip()]
H, nrows, Xe = {}, {}, {}
EVERY, EVAL_CAP = 64, 1024

def mk(nm, arg=0):
    def hook(mod, inp):
        x = inp[arg] if len(inp) > arg else None
        if x is None:
            return
        x = x.detach().reshape(-1, x.shape[-1]).float()
        if nm not in H:
            H[nm] = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.float32, device=x.device)
            nrows[nm] = 0; Xe[nm] = []
        off = nrows[nm] % EVERY
        # the draft's hooks see 1-7 rows per call; arange(off, n) throws when off > n
        idx = (torch.arange(off, x.shape[0], EVERY, device=x.device) if off < x.shape[0]
               else torch.empty(0, dtype=torch.long, device=x.device))
        if idx.numel() and sum(t.shape[0] for t in Xe[nm]) < EVAL_CAP:
            Xe[nm].append(x[idx].clone())
        keep = torch.ones(x.shape[0], dtype=torch.bool, device=x.device); keep[idx] = False
        xh = x[keep]; H[nm] += xh.t() @ xh; nrows[nm] += x.shape[0]
    return hook

hooks = [m.register_forward_pre_hook(mk(n)) for n, m in MODS.items() if n != lm_name]
hooks += [lp_t.register_forward_pre_hook(mk(lm_name, 1)),
          lp_d.register_forward_pre_hook(mk(lm_name, 1))]
t0 = time.time()
llm.generate(prompts, SamplingParams(max_tokens=a.gen, temperature=0), use_tqdm=False)
for h in hooks: h.remove()
print(f"\ncalibration: {len(prompts)} prompts x {a.gen} gen tokens in {time.time()-t0:.0f}s")
missing = [n for n in MODS if n not in H]
assert not missing, f"no activations captured for {missing} - hook did not fire"
for n in MODS: print(f"  {n:<45s} rows {nrows.get(n,0):>7d}   K={H[n].shape[0]}")
# 18 minutes of calibration is worth saving before anything else can fail
torch.save({"H": {k: v.cpu() for k, v in H.items()},
            "Xe": {k: [t.cpu() for t in v] for k, v in Xe.items()}, "nrows": nrows},
           os.path.join(a.out, "hessians.pt"))
print("  hessians checkpointed")

# ---------------------------------------------------------------- GPTQ
@torch.no_grad()
def inv_hessian(Hm, K):
    d = torch.diag(Hm); Hm[d == 0, d == 0] = 1.0
    Hm += torch.eye(K, device=Hm.device) * (a.damp * d.mean())
    try:
        return torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hm)), upper=True)
    except Exception:
        Hc = Hm.double().cpu()
        return torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hc)), upper=True).float().to(Hm.device)

@torch.no_grad()
def gptq_int4(W, Hinv):
    """W [K, N] fp32 -> codes uint8 [K,N] in 0..15 (value = (code-8)*s), scales [K/G, N]."""
    K, N = W.shape
    Wc = W.clone(); codes = torch.zeros((K, N), dtype=torch.uint8, device=W.device)
    scales = torch.zeros((K // G, N), dtype=torch.float32, device=W.device)
    for i0 in range(0, K, G):
        i1 = i0 + G; Wb = Wc[i0:i1].clone(); Eb = torch.zeros_like(Wb); Hb = Hinv[i0:i1, i0:i1]
        s = Wc[i0:i1].abs().amax(0).clamp_min(1e-9) / 7.0        # sym: -8..7, +amax -> 15
        scales[i0 // G] = s
        for j in range(G):
            col = Wb[j]
            qi = torch.clamp(torch.round(col / s), -8, 7)
            codes[i0 + j] = (qi + 8).to(torch.uint8)
            err = (col - qi * s) / Hb[j, j]
            Wb[j:] -= torch.outer(Hb[j, j:], err); Eb[j] = err
        if i1 < K: Wc[i1:] -= Hinv[i0:i1, i1:].t() @ Eb
    return codes, scales

def pack_gptq(codes, scales):
    """codes [K,N] 0..15, scales [K/G,N] -> the four GPTQ tensors, body layout."""
    K, N = codes.shape
    c = codes.to(torch.int32).reshape(K // 8, 8, N)
    sh = (torch.arange(8, device=c.device, dtype=torch.int32) * 4).view(1, 8, 1)
    qweight = (c << sh).sum(1, dtype=torch.int64).to(torch.int32)          # wraps bit 31 correctly
    qzeros = torch.full((K // G, N // 8), 0x77777777, dtype=torch.int32, device=c.device)
    g_idx = (torch.arange(K, device=c.device, dtype=torch.int32) // G)
    return {"qweight": qweight.contiguous(), "scales": scales.half().contiguous(),
            "qzeros": qzeros, "g_idx": g_idx}

def unpack_check(t, codes):
    """independent unpack of what we wrote, must equal codes bit-exactly"""
    qw = t["qweight"]; K8, N = qw.shape
    sh = (torch.arange(8, device=qw.device, dtype=torch.int32) * 4).view(1, 8, 1)
    back = ((qw.unsqueeze(1) >> sh) & 0xF).reshape(K8 * 8, N).to(torch.uint8)
    return torch.equal(back, codes)

out, manifest = {}, {}
with torch.inference_mode(False), torch.no_grad():
    for k in list(H): H[k] = H[k].detach().clone()
    for k in list(Xe): Xe[k] = torch.cat([t.detach().clone() for t in Xe[k]], 0)[:EVAL_CAP] if Xe[k] else None
    for mname, mod in MODS.items():
        Wfull = mod.weight.detach().float()                    # [N_merged, K]
        K = Wfull.shape[1]
        Hi = inv_hessian(H.pop(mname), K)
        for hf_name, r0, r1 in SPLITS[mname]:
            Wt = Wfull[r0:r1].t().contiguous()                 # [K, N_i]
            # GPTQ is exact per output column, so chunk N: lm_head is 5.1 GB fp32 and
            # the loop clones it, which does not fit beside the model.
            CH = 32768
            parts = [gptq_int4(Wt[:, c0:c0 + CH].contiguous(), Hi)
                     for c0 in range(0, Wt.shape[1], CH)]
            codes = torch.cat([p_[0] for p_ in parts], 1); s = torch.cat([p_[1] for p_ in parts], 1)
            del parts
            deq = (codes.float() - 8.0) * s.repeat_interleave(G, dim=0)
            werr = ((deq - Wt).norm() / Wt.norm()).item()
            xe = Xe.get(mname)
            oerr = float("nan")
            if xe is not None:
                ref = xe @ Wt; oerr = (((xe @ deq) - ref).norm() / ref.norm()).item()
            # RTN comparison = what the boot-time patches do today
            s_rtn = Wt.reshape(K // G, G, -1).abs().amax(1).clamp_min(1e-9) / 7.0
            rtn = (torch.clamp(torch.round(Wt.reshape(K // G, G, -1) / s_rtn.unsqueeze(1)), -8, 7)
                   * s_rtn.unsqueeze(1)).reshape(K, -1)
            rerr = (((xe @ rtn) - ref).norm() / ref.norm()).item() if xe is not None else float("nan")
            t = pack_gptq(codes, s)
            assert unpack_check(t, codes), f"{hf_name}: pack/unpack mismatch"
            base = hf_name[:-len(".weight")]
            for kk, v in t.items(): out[f"{base}.{kk}"] = v.cpu()
            manifest[hf_name] = dict(K=K, N=int(Wt.shape[1]), rows=nrows.get(mname, 0),
                                     out_err_gptq=oerr, out_err_rtn_today=rerr, weight_err=werr)
            print(f"  {hf_name:<44s} K={K:<6d} N={Wt.shape[1]:<7d} "
                  f"OUT gptq {oerr*100:5.2f}%  (rtn today {rerr*100:5.2f}%)")
            del Wt, codes, s, deq, rtn
        del Hi, Wfull
        torch.xpu.empty_cache()

save_file(out, os.path.join(a.out, "bake-int4.safetensors"), metadata={"format": "pt"})
json.dump(manifest, open(os.path.join(a.out, "bake-manifest.json"), "w"), indent=1)
print(f"\nwrote {len(out)} tensors -> {a.out}/bake-int4.safetensors")
