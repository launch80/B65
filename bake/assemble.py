#!/usr/bin/env python3
"""Assemble the baked checkpoint.

Physically rebuilds every shard without the 9 dense tensors being replaced, adds the
baked GPTQ tensors, rewrites the index, and edits BOTH quantize_config.json and
config.json:quantization_config (vLLM reads the latter) so that
    lm_head: true            -> gptq_utils honours lm_head_quantized
    dynamic: {}              -> qwen3_5_mtp no longer forces the draft unquantized
Rebuilding shards rather than just re-indexing is deliberate: vLLM's weight iterator
walks *.safetensors files, and a stale lm_head.weight left in an old shard would be
loaded over the top of the baked one.
"""
import argparse, json, os, shutil, struct
from safetensors import safe_open
from safetensors.torch import save_file
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--bake", required=True)
ap.add_argument("--dst", required=True)
a = ap.parse_args()
os.makedirs(a.dst, exist_ok=True)

REPLACED = {"lm_head.weight", "mtp.fc.weight"} | {
    f"mtp.layers.0.{n}.weight" for n in
    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}

idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
wm = idx["weight_map"]; new_wm = {}
shards = sorted(set(wm.values()))
for sh in shards:
    keep = [k for k, v in wm.items() if v == sh and k not in REPLACED]
    dropped = [k for k, v in wm.items() if v == sh and k in REPLACED]
    with safe_open(os.path.join(a.src, sh), framework="pt") as f:
        tensors = {k: f.get_tensor(k) for k in keep}
    save_file(tensors, os.path.join(a.dst, sh), metadata={"format": "pt"})
    for k in keep: new_wm[k] = sh
    print(f"  {sh}: kept {len(keep)}, dropped {len(dropped)}" + (f"  {dropped}" if dropped else ""))
    del tensors

bake = os.path.join(a.bake, "bake-int4.safetensors")
with safe_open(bake, framework="pt") as f:
    names = list(f.keys())
shutil.copy(bake, os.path.join(a.dst, "model-bake-int4.safetensors"))
for k in names: new_wm[k] = "model-bake-int4.safetensors"
print(f"  bake shard: {len(names)} tensors")

missing = REPLACED - set(wm)
assert not missing, f"expected tensors not in source: {missing}"
assert not (REPLACED & set(new_wm)), "a replaced tensor survived"
for base in {n.rsplit(".", 1)[0] for n in names}:
    for suf in ("qweight", "scales", "qzeros", "g_idx"):
        assert f"{base}.{suf}" in new_wm, f"{base}.{suf} missing"

total = sum(os.path.getsize(os.path.join(a.dst, s)) for s in set(new_wm.values()))
json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_wm.items()))},
          open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=2)

for fn in os.listdir(a.src):
    p = os.path.join(a.src, fn)
    if fn.endswith(".safetensors") or fn == "model.safetensors.index.json" or not os.path.isfile(p):
        continue
    shutil.copy(p, os.path.join(a.dst, fn))

def edit(qc):
    qc["lm_head"] = True
    qc["dynamic"] = {}
    return qc
qc = edit(json.load(open(os.path.join(a.dst, "quantize_config.json"))))
json.dump(qc, open(os.path.join(a.dst, "quantize_config.json"), "w"), indent=2)
cfg = json.load(open(os.path.join(a.dst, "config.json")))
cfg["quantization_config"] = edit(cfg["quantization_config"])
json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=2)
print(f"\n  configs: lm_head=true, dynamic={{}}   total {total/1e9:.2f} GB   -> {a.dst}")
