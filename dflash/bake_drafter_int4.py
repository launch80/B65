#!/usr/bin/env python3
"""GPTQ INT4 (sym, g128) bake of the DFlash2 drafter's o_proj/gate/up/down/fc (q/k/v stay dense: the context-KV precompute reads qkv_proj.weight) from in-engine Hessians, plus assembly
of a loadable GPTQ drafter checkpoint.

  python bake_drafter_int4.py --hess /out --src /draft --dst /models/dflash2-int4 --damps 0.005,0.01,0.02

Per linear: GPTQ for each damp, scored by relative output L2 on held-out rows (vs RTN); best damp
packed in the body's GPTQ layout. Unquantized tensors (norms, conv kernels + projections, candidate
selector, hidden_norm) are copied as-is. config.json gets a gptq quantization_config with those
excluded via `dynamic`.
"""
import argparse, json, os, shutil, time, torch
from safetensors import safe_open
from safetensors.torch import save_file
ap = argparse.ArgumentParser()
ap.add_argument("--hess", required=True); ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
ap.add_argument("--damps", default="0.005,0.01,0.02"); ap.add_argument("--eval-cap", type=int, default=4096)
ap.add_argument("--device", default="xpu")
a = ap.parse_args(); G, DEV = 128, a.device; DAMPS = [float(x) for x in a.damps.split(",")]
os.makedirs(a.dst, exist_ok=True)
NL = 5
MAP = {"model.fc": ["fc.weight"]}
for i in range(NL):
    # qkv_proj stays dense: DFlashQwen3Model._build_context_kv_buffers slices qkv_proj.weight for the fused context-KV GEMM
    MAP[f"model.layers.{i}.self_attn.o_proj"] = [f"layers.{i}.self_attn.o_proj.weight"]
    MAP[f"model.layers.{i}.mlp.gate_up_proj"] = [f"layers.{i}.mlp.gate_proj.weight", f"layers.{i}.mlp.up_proj.weight"]
    MAP[f"model.layers.{i}.mlp.down_proj"] = [f"layers.{i}.mlp.down_proj.weight"]
cks = {}
for g in ("A", "B1", "B2"):
    p = os.path.join(a.hess, f"hessians-drafter-{g}.pt")
    if os.path.exists(p):
        ck = torch.load(p, map_location="cpu"); cks.update({k: (ck["H"][k], ck["Xe"][k], ck["nrows"][k]) for k in ck["H"]})
        print(f"  {os.path.basename(p)}: {len(ck['H'])} modules, gen tokens {ck.get('gen_tokens')}")
missing = [m for m in MAP if m not in cks]; assert not missing, f"no Hessians for {missing}"
src = os.path.join(a.src, "model.safetensors")
with safe_open(src, framework="pt") as f:
    all_names = list(f.keys())
def load(name):
    with safe_open(src, framework="pt") as f: return f.get_tensor(name)
@torch.no_grad()
def inv_hessian(Hm, K, damp):
    Hm = Hm.to(DEV).float().clone(); d = torch.diag(Hm); Hm[d == 0, d == 0] = 1.0
    Hm += torch.eye(K, device=DEV) * (damp * d.mean())
    try: return torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hm)), upper=True)
    except Exception:
        Hc = Hm.double().cpu(); return torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hc)), upper=True).float().to(DEV)
@torch.no_grad()
def gptq_int4(W, Hinv):
    K, N = W.shape; Wc = W.clone(); codes = torch.zeros((K, N), dtype=torch.uint8, device=W.device)
    scales = torch.zeros((K // G, N), dtype=torch.float32, device=W.device)
    for i0 in range(0, K, G):
        i1 = i0 + G; Wb = Wc[i0:i1].clone(); Eb = torch.zeros_like(Wb); Hb = Hinv[i0:i1, i0:i1]
        s = Wc[i0:i1].abs().amax(0).clamp_min(1e-9) / 7.0; scales[i0 // G] = s
        for j in range(G):
            col = Wb[j]; qi = torch.clamp(torch.round(col / s), -8, 7); codes[i0 + j] = (qi + 8).to(torch.uint8)
            err = (col - qi * s) / Hb[j, j]; Wb[j:] -= torch.outer(Hb[j, j:], err); Eb[j] = err
        if i1 < K: Wc[i1:] -= Hinv[i0:i1, i1:].t() @ Eb
    return codes, scales
def gptq_chunked(Wt, Hinv, CH=32768):
    parts = [gptq_int4(Wt[:, c0:c0 + CH].contiguous(), Hinv) for c0 in range(0, Wt.shape[1], CH)]
    return torch.cat([p[0] for p in parts], 1), torch.cat([p[1] for p in parts], 1)
def pack_gptq(codes, scales):
    K, N = codes.shape; c = codes.to(torch.int32).reshape(K // 8, 8, N)
    sh = (torch.arange(8, device=c.device, dtype=torch.int32) * 4).view(1, 8, 1)
    qweight = (c << sh).sum(1, dtype=torch.int64).to(torch.int32)
    qzeros = torch.full((K // G, N // 8), 0x77777777, dtype=torch.int32, device=c.device)
    g_idx = torch.arange(K, device=c.device, dtype=torch.int32) // G
    return {"qweight": qweight.contiguous(), "scales": scales.half().contiguous(), "qzeros": qzeros, "g_idx": g_idx}
def unpack_codes(qweight):
    K8, N = qweight.shape; sh = (torch.arange(8, device=qweight.device, dtype=torch.int32) * 4).view(1, 8, 1)
    return ((qweight.unsqueeze(1) >> sh) & 0xF).reshape(K8 * 8, N).to(torch.uint8)
def deq(codes, scales): return (codes.float() - 8.0) * scales.float().repeat_interleave(G, dim=0)
def rel_err(xe, W, Wq): ref = xe @ W; return (((xe @ Wq) - ref).norm() / ref.norm()).item()
out, manifest, t0, quantized = {}, {}, time.time(), set()
with torch.no_grad():
    for mname, hf_names in MAP.items():
        Hm, XeL, nrows = cks[mname]; xe = torch.cat(XeL, 0)[:a.eval_cap].to(DEV).float(); K = Hm.shape[0]
        Hinvs = {d: inv_hessian(Hm, K, d) for d in DAMPS}
        for hf in hf_names:
            Wt = load(hf).to(DEV).float().t().contiguous(); assert Wt.shape[0] == K, (hf, Wt.shape, K)
            base = hf[:-len(".weight")]
            s_rtn = Wt.reshape(K // G, G, -1).abs().amax(1).clamp_min(1e-9) / 7.0
            rtn = (torch.clamp(torch.round(Wt.reshape(K // G, G, -1) / s_rtn.unsqueeze(1)), -8, 7) * s_rtn.unsqueeze(1)).reshape(K, -1)
            rec = {"K": K, "N": int(Wt.shape[1]), "rows": int(nrows), "eval_rows": int(xe.shape[0]), "rtn_err": rel_err(xe, Wt, rtn), "settings": {}}
            del rtn, s_rtn; best = None
            for d, Hi in Hinvs.items():
                codes, s = gptq_chunked(Wt, Hi); e = rel_err(xe, Wt, deq(codes, s)); rec["settings"][str(d)] = e
                print(f"  {hf:<40s} damp {d:<6} OUT {e*100:6.3f}%  (rtn {rec['rtn_err']*100:6.3f}%)  {time.time()-t0:5.0f}s", flush=True)
                if best is None or e < best[0]: best = (e, d, codes.cpu(), s.cpu())
                del codes, s
            e, d, codes, s = best; rec["chosen"] = {"damp": d, "out_err": e}
            t = pack_gptq(codes.to(DEV), s.to(DEV)); assert torch.equal(unpack_codes(t["qweight"]), codes.to(DEV))
            for kk, v in t.items(): out[f"{base}.{kk}"] = v.cpu()
            manifest[hf] = rec; quantized.add(hf); del Wt, t, codes, s
        del Hinvs, xe
        if DEV == "xpu": torch.xpu.empty_cache()
# unquantized tensors copied as-is
for n in all_names:
    if n not in quantized: out[n] = load(n)
save_file(out, os.path.join(a.dst, "model.safetensors"), metadata={"format": "pt"})
for fn in os.listdir(a.src):
    if fn.endswith(".json") and fn != "config.json": shutil.copy(os.path.join(a.src, fn), os.path.join(a.dst, fn))
cfg = json.load(open(os.path.join(a.src, "config.json")))
QC = {"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": True, "desc_act": False, "lm_head": False,
      "dynamic": {"-:.*qkv_proj.*": {}, "-:.*conv.*": {}, "-:.*candidate_selector.*": {}, "-:.*hidden_norm.*": {}},
      # vLLM decides per-layer quantization by substring-matching these against the RUNTIME prefix, which for a
      # draft is offset by the target's layer count (layers.64...), so the names must carry no layer index.
      "modules_in_block_to_quantize": ["self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.gate_up_proj", "mlp.down_proj", "fc"]}  # fused name too: the draft quant config gets no name mapper
cfg["quantization_config"] = QC
json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=2)
json.dump(QC, open(os.path.join(a.dst, "quantize_config.json"), "w"), indent=2)
json.dump({"source": a.src, "method": "GPTQ INT4 sym g128 desc_act=false, Hessians from the drafter's own inputs in-engine",
           "linears": manifest}, open(os.path.join(a.dst, "BAKE.json"), "w"), indent=1)
qb = sum(v.numel() * v.element_size() for k, v in out.items() if k.split(".")[-1] in ("qweight", "scales", "qzeros", "g_idx"))
ub = sum(v.numel() * v.element_size() for k, v in out.items() if k.split(".")[-1] not in ("qweight", "scales", "qzeros", "g_idx"))
print(f"\nwrote {len(out)} tensors -> {a.dst}: quantized {qb/1e9:.2f} GB + bf16 {ub/1e9:.2f} GB; {len(quantized)} linears baked")
