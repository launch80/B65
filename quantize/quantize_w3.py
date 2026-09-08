#!/usr/bin/env python3
"""Stage 6 - requantize the model body from GPTQ-int4 to the Launch80 W3 format.

Method is GPTQ/OBQ error compensation against the real inverse Hessian of the real
activations. That choice is not cosmetic: on this model, at 3 bits, round-to-nearest
costs 22.34% output error and compensation costs 5.13%.

Two things drive the structure:

  * 99 GB of Hessians if you hold them all (down_proj is K=17408, so 1.21 GB each),
    against ~11 GB free once the model is resident. So layers are processed in
    chunks: hook the chunk, run calibration, quantize, free, next chunk. The model is
    loaded once and the calibration prefill is repeated per chunk, which is the cheap
    half of the trade.

  * Calibration text must NOT be the evaluation text. verify/ppl-corpus.json and
    gsm8k are the quality gates; calibrating on them would make the gates flatter
    than they are. This uses wikitext-2 validation, which is neither.

The dequantisation of the source int4 weights is VERIFIED against the module's own
forward output before anything is quantized, because a transposed or mis-strided
layout yields weights with a perfectly plausible distribution and no error anywhere.
"""
import argparse, json, os, sys, time, urllib.request
import torch
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w3_pack_torch import pack_columns

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--layers-per-chunk", type=int, default=2)
ap.add_argument("--calib-tokens", type=int, default=65536)
ap.add_argument("--seq-len", type=int, default=512)
ap.add_argument("--group", type=int, default=128)
ap.add_argument("--damp", type=float, default=0.01)
ap.add_argument("--limit-layers", type=int, default=0, help="0 = all; else first N layers")
ap.add_argument("--only-layers", default="", help="comma list, e.g. 9,10,32,33")
# Diagnostic. research/r2/gptq_study.py reported 5.13% output error at 3 bits using
# 1024 activation rows for a K=5120 Hessian, and scored the SAME rows it fitted. With
# rank 1024 << 5120 the compensation can cancel error almost perfectly inside the
# sampled subspace, so that number may be in-sample overfitting rather than a
# generalising result. This flag reproduces those conditions exactly.
ap.add_argument("--eval-in-sample", action="store_true",
                help="score the rows that went into the Hessian (reproduces the study)")
ap.add_argument("--util", type=float, default=0.70)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
DEV = "xpu"
# The SOURCE checkpoint is GPTQ int4 at group 128 (quantize_config.json). That is
# fixed and independent of --group, which sets the group of the 3-bit weights we
# produce. Conflating the two silently reshapes the dequantised source.
SRC_GROUP = 128

# ---------------------------------------------------------------- calibration text
CALIB_URL = ("https://raw.githubusercontent.com/pytorch/examples/main/"
             "word_language_model/data/wikitext-2/valid.txt")
cache = os.path.join(a.out, "calib.txt")
if not os.path.exists(cache):
    open(cache, "wb").write(urllib.request.urlopen(CALIB_URL, timeout=60).read())
text = open(cache).read()

llm = LLM(model=os.environ["MODEL_PATH"], quantization="gptq", dtype="float16",
          max_model_len=max(2048, a.seq_len), gpu_memory_utilization=a.util,
          max_num_seqs=4, enforce_eager=True)
runner = llm.llm_engine.model_executor.driver_worker.model_runner
model = runner.model
tok = llm.get_tokenizer()

ids = tok(text, add_special_tokens=False)["input_ids"]
nseq = max(1, a.calib_tokens // a.seq_len)
prompts = [tok.decode(ids[i * a.seq_len:(i + 1) * a.seq_len]) for i in range(nseq)]
prompts = [p for p in prompts if p.strip()]
print(f"\ncalibration: {len(prompts)} sequences x {a.seq_len} tokens "
      f"= ~{len(prompts)*a.seq_len} rows (wikitext-2 valid, disjoint from the gates)")

QMODS = [(n, m) for n, m in model.named_modules() if hasattr(m, "qweight")]
def layer_of(name):
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return -1
LAYERS = sorted({layer_of(n) for n, _ in QMODS if layer_of(n) >= 0})
if a.only_layers:
    want = {int(v) for v in a.only_layers.split(",")}
    LAYERS = [l for l in LAYERS if l in want]
elif a.limit_layers:
    LAYERS = LAYERS[:a.limit_layers]
print(f"{len(QMODS)} quantized modules across {len(LAYERS)} layers\n")


def dequant(mod):
    """vLLM GPTQ int4 (always group 128) -> dense fp32 [K, N].
    Layout is verified, not assumed."""
    qw, sc = mod.qweight, mod.scales
    N, K8 = qw.shape
    K = K8 * 8
    sh = (torch.arange(8, device=qw.device, dtype=torch.int32) * 4).view(1, 8, 1)
    nib = ((qw.t().unsqueeze(1) >> sh) & 0xF).reshape(K, N)     # [K, N]
    s = sc.to(torch.float32)                                     # [K/G, N]
    return (nib.to(torch.float32) - 8.0) * s.repeat_interleave(SRC_GROUP, dim=0)


@torch.no_grad()
def verify_dequant():
    """A wrong layout produces weights with the right variance and no error. Catch it
    by checking against the module's own forward."""
    name, mod = QMODS[0]
    W = dequant(mod)
    K, N = W.shape
    x = torch.randn(1, K, dtype=torch.float16, device=DEV) * 0.05
    with torch.no_grad():
        out = mod(x)
    out = out[0] if isinstance(out, tuple) else out
    ref = (x.to(torch.float32) @ W)
    err = ((out.float() - ref).norm() / ref.norm()).item()
    print(f"dequant check on {name}: W{tuple(W.shape)} rel err vs module forward "
          f"= {err:.3e}  {'OK' if err < 5e-3 else 'FAILED'}")
    del W
    return err < 5e-3


with torch.inference_mode(False):
    _ok = verify_dequant()
if not _ok:
    raise SystemExit("dequantisation layout is wrong - refusing to quantize")


@torch.no_grad()
def gptq_codes(W, Hinv, bits=3, group=128):
    """OBQ/GPTQ: quantize one input channel at a time, push the error onto the
    channels not yet done, weighted by the inverse Hessian. Returns integer codes,
    not dequantized weights."""
    K, N = W.shape
    lv = 2 ** (bits - 1) - 1                       # 3  -> grid -4..3
    Wc = W.clone()
    codes = torch.zeros((K, N), dtype=torch.uint8, device=W.device)
    scales = torch.zeros((K // group, N), dtype=torch.float32, device=W.device)
    s = None
    for i0 in range(0, K, group):
        i1 = i0 + group
        Wb = Wc[i0:i1].clone()
        Eb = torch.zeros_like(Wb)
        Hb = Hinv[i0:i1, i0:i1]
        s = Wc[i0:i1].abs().amax(0).clamp_min(1e-9) / lv        # one scale per group
        scales[i0 // group] = s
        for j in range(group):
            col = Wb[j]
            qi = torch.clamp(torch.round(col / s), -lv - 1, lv)
            codes[i0 + j] = (qi + 4).to(torch.uint8)             # 0..7
            err = (col - qi * s) / Hb[j, j]
            Wb[j:] -= torch.outer(Hb[j, j:], err)
            Eb[j] = err
        if i1 < K:
            Wc[i1:] -= Hinv[i0:i1, i1:].t() @ Eb
    return codes, scales


def inv_hessian(H, K):
    d = torch.diag(H)
    H[d == 0, d == 0] = 1.0
    H += torch.eye(K, device=H.device, dtype=H.dtype) * (a.damp * d.mean())
    try:
        L = torch.linalg.cholesky(H)
        Hi = torch.cholesky_inverse(L)
        return torch.linalg.cholesky(Hi, upper=True)
    except Exception as e:                      # fp32 XPU can fail; CPU fp64 is exact
        print(f"    cholesky fell back to CPU fp64 ({type(e).__name__})")
        Hc = H.double().cpu()
        L = torch.linalg.cholesky(Hc)
        Hi = torch.cholesky_inverse(L)
        return torch.linalg.cholesky(Hi, upper=True).float().to(H.device)


SP = SamplingParams(max_tokens=1, temperature=0)
manifest, t_start = {}, time.time()

for c0 in range(0, len(LAYERS), a.layers_per_chunk):
    chunk = LAYERS[c0:c0 + a.layers_per_chunk]
    mods = [(n, m) for n, m in QMODS if layer_of(n) in chunk]
    H, hooks, nrows, Xe = {}, [], {}, {}
    EVERY, EVAL_CAP = 64, 1024      # hold out every 64th row, cap the eval set

    def mk(nm):
        def hook(mod, inp):
            x = inp[0]
            if x is None:
                return
            x = x.detach().reshape(-1, x.shape[-1]).float()
            if nm not in H:
                H[nm] = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.float32,
                                    device=x.device)
                nrows[nm] = 0
                Xe[nm] = []
            # eval rows are EXCLUDED from the Hessian: measuring error on the rows the
            # compensation was fitted to would flatter it
            off = nrows[nm] % EVERY
            idx = torch.arange(off, x.shape[0], EVERY, device=x.device)
            if idx.numel() and sum(t.shape[0] for t in Xe[nm]) < EVAL_CAP:
                Xe[nm].append(x[idx].clone())
            if a.eval_in_sample:
                xh = x                       # eval rows stay in the Hessian
            else:
                keep = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
                keep[idx] = False
                xh = x[keep]
            H[nm] += xh.t() @ xh
            nrows[nm] += x.shape[0]
        return hook

    for n, m in mods:
        hooks.append(m.register_forward_pre_hook(mk(n)))
    t0 = time.time()
    llm.generate(prompts, SP, use_tqdm=False)
    for h in hooks:
        h.remove()
    t_cal = time.time() - t0

    # vLLM generates under torch.inference_mode(), so the Hessians the hooks built
    # are inference tensors and cannot be updated in place. Everything from here is
    # ordinary tensor work, so step out of it and re-materialise them.
    tensors = {}
    with torch.inference_mode(False), torch.no_grad():
      for k in list(H.keys()):
          H[k] = H[k].detach().clone()
      for k in list(Xe.keys()):
          Xe[k] = torch.cat([t.detach().clone() for t in Xe[k]], 0)[:EVAL_CAP]
      for n, m in mods:
          K = m.qweight.shape[1] * 8
          W = dequant(m)
          Hi = inv_hessian(H.pop(n), K)
          codes, s3 = gptq_codes(W, Hi, 3, a.group)
          del Hi
          # OUTPUT error on held-out activations - the metric that matters.
          # Weight error is not it: GPTQ deliberately trades weight error for output
          # error, and at 3 bits (8 levels) even plain RTN sits near 30-40% weight
          # error while the output error is a small fraction of that.
          deq3 = (codes.float() - 4.0) * s3.repeat_interleave(a.group, dim=0)
          werr = ((deq3 - W).norm() / W.norm()).item()
          xe = Xe.get(n)
          if xe is not None and xe.numel():
              ref = xe @ W
              oerr = (((xe @ deq3) - ref).norm() / ref.norm()).item()
              del ref
          else:
              oerr = float("nan")
          del deq3, W
          tensors[n + ".qweight3"] = pack_columns(codes).cpu()
          tensors[n + ".scales3"] = s3.t().contiguous().half().cpu()
          manifest[n] = {"K": K, "N": int(m.qweight.shape[0]),
                         "out_rel_err": round(oerr, 5),
                         "weight_rel_err": round(werr, 5), "rows": nrows.get(n, 0)}
          del codes, s3
          print(f"  {n:<56s} K={K:<6d} rows={nrows.get(n,0):<7d} "
                f"OUT={oerr*100:5.2f}%  (w={werr*100:4.1f}%)")
    from safetensors.torch import save_file
    save_file(tensors, os.path.join(a.out, f"w3-layers-{chunk[0]:03d}-{chunk[-1]:03d}.safetensors"))
    del tensors, H, Xe
    torch.xpu.empty_cache()
    print(f"  [chunk {chunk[0]}-{chunk[-1]}] calib {t_cal:.0f}s  "
          f"elapsed {(time.time()-t_start)/60:.1f} min\n", flush=True)

json.dump(manifest, open(os.path.join(a.out, "w3-manifest.json"), "w"), indent=1)
errs = sorted(v["out_rel_err"] for v in manifest.values())
n_ = len(errs)
print(f"\n{len(manifest)} modules. OUTPUT rel err: median {errs[n_//2]*100:.2f}%  "
      f"p95 {errs[int(n_*0.95)]*100:.2f}%  max {errs[-1]*100:.2f}%")
bad = [k for k, v in manifest.items() if v["out_rel_err"] > 0.12]
print(f"  gate: <=6% for 95% of modules -> "
      f"{'PASS' if errs[int(n_*0.95)] <= 0.06 else 'MISS'};  "
      f"{len(bad)} modules above 12%" + (f": {bad[:5]}" if bad else ""))
