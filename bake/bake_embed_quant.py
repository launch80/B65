#!/usr/bin/env python3
"""Embed-only INT8/INT4 derivative of a baked GPTQ checkpoint.

Quantizes *only* `model.language_model.embed_tokens.weight`. Body / lm_head /
MTP GPTQ shards are hardlinked (or copied) unchanged.

INT8 is the accepted path (~1.15 GiB weight reclaim on Qwen3.8-27B baked-v2,
graphs+MTP6 decode within ~1% of FP16-embed baseline). INT4 embed failed
acceptance (~-21% decode via MTP accept drop) — keep bits=4 for repro only.

Runtime note
------------
AutoWeightsLoader expects an FP16 `embed_tokens.weight`. By default this bake
writes the quantized tensors to `model-embed-int{N}.safetensors` and remaps the
index so `.weight` still points at the original FP16 shard (scale removed from
the map). A boot patch (`patches/patch_embed_int8.py`) then swaps the resident
tensor to int8/int4 in `process_weights_after_loading`.

Pass `--store-quant-in-index` only if you have a loader that understands the
quant side file directly (not stock vLLM today).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EMBED_KEY = "model.language_model.embed_tokens.weight"
EMBED_SCALE_KEY = "model.language_model.embed_tokens.weight_scale"


def hl_copy(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in files:
            sp = Path(root) / name
            dp = dst / rel / name
            if dp.exists():
                continue
            try:
                os.link(sp, dp)
            except OSError:
                shutil.copy2(sp, dp)


def break_hardlink(path: Path) -> None:
    data = path.read_bytes()
    path.unlink()
    path.write_bytes(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="baked GPTQ checkpoint dir")
    ap.add_argument("--dst", required=True, help="output derivative dir")
    ap.add_argument("--bits", type=int, choices=[8, 4], default=8)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument(
        "--store-quant-in-index",
        action="store_true",
        help="map embed keys to the quant side file (breaks stock vLLM load)",
    )
    args = ap.parse_args()
    src, dst, bits, chunk = Path(args.src), Path(args.dst), args.bits, args.chunk

    if bits == 4:
        print(
            "WARNING: INT4 embed failed speed acceptance on baked-v2 "
            "(~-21% decode). Prefer --bits 8.",
            file=sys.stderr,
        )

    if dst.exists():
        print("removing existing dst")
        shutil.rmtree(dst)
    print("hardlink copy...")
    hl_copy(src, dst)

    idx = json.loads((dst / "model.safetensors.index.json").read_text())
    fp16_shard = idx["weight_map"][EMBED_KEY]
    shard = src / fp16_shard
    print("quantizing from", shard)

    with safe_open(str(shard), framework="pt", device="cpu") as f:
        try:
            sl = f.get_slice(EMBED_KEY)
            shape = tuple(sl.get_shape())
            print("slice shape", shape)
            V, H = shape
            if bits == 8:
                q = torch.empty(V, H, dtype=torch.int8)
            else:
                q = torch.empty(V, (H + 1) // 2, dtype=torch.int8)
            scale = torch.empty(V, 1, dtype=torch.float16)
            mae_acc = 0.0
            mae_n = 0
            max_err = 0.0
            for i0 in range(0, V, chunk):
                i1 = min(V, i0 + chunk)
                rows = sl[i0:i1].float()
                if bits == 8:
                    sc = rows.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
                    qq = torch.clamp(torch.round(rows / sc), -128, 127).to(torch.int8)
                    recon = qq.float() * sc
                    q[i0:i1] = qq
                else:
                    sc = rows.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 7.0
                    q4 = torch.clamp(torch.round(rows / sc), -8, 7).to(torch.int8)
                    if H % 2 == 1:
                        q4 = torch.nn.functional.pad(q4, (0, 1))
                    lo = q4[:, 0::2] & 0x0F
                    hi = q4[:, 1::2] & 0x0F
                    qq = (lo | (hi << 4)).to(torch.uint8).view(torch.int8)
                    q[i0:i1] = qq
                    recon = q4[:, :H].float() * sc
                scale[i0:i1] = sc.to(torch.float16)
                err = (recon - rows).abs()
                mae_acc += err.sum().item()
                mae_n += err.numel()
                max_err = max(max_err, err.max().item())
                if (i0 // chunk) % 10 == 0:
                    print(f"  rows {i0}:{i1}/{V}", flush=True)
            print(f"mae={mae_acc / mae_n:.6f} max={max_err:.6f}")
        except Exception as e:
            print("slice failed, fallback full load:", e)
            w = f.get_tensor(EMBED_KEY)
            V, H = w.shape
            wf = w.float()
            if bits == 8:
                scale = (
                    wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
                ).to(torch.float16)
                q = torch.clamp(
                    torch.round(wf / scale.float()), -128, 127
                ).to(torch.int8)
            else:
                scale = (
                    wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 7.0
                ).to(torch.float16)
                q4 = torch.clamp(
                    torch.round(wf / scale.float()), -8, 7
                ).to(torch.int8)
                if H % 2 == 1:
                    q4 = torch.nn.functional.pad(q4, (0, 1))
                q = (
                    (q4[:, 0::2] & 0x0F) | ((q4[:, 1::2] & 0x0F) << 4)
                ).to(torch.uint8).view(torch.int8)

    embed_file = f"model-embed-int{bits}.safetensors"
    outp = dst / embed_file
    if outp.exists():
        outp.unlink()
    save_file(
        {EMBED_KEY: q.contiguous(), EMBED_SCALE_KEY: scale.contiguous()},
        str(outp),
    )
    print("wrote", outp, "gib", outp.stat().st_size / 1024**3)

    for name in ("model.safetensors.index.json", "config.json"):
        break_hardlink(dst / name)

    idx = json.loads((dst / "model.safetensors.index.json").read_text())
    if args.store_quant_in_index:
        idx["weight_map"][EMBED_KEY] = embed_file
        idx["weight_map"][EMBED_SCALE_KEY] = embed_file
        loader_note = "quant_keys_in_index"
    else:
        # Stock loader: FP16 shard for .weight; strip scale; side file for patch.
        idx["weight_map"][EMBED_KEY] = fp16_shard
        idx["weight_map"].pop(EMBED_SCALE_KEY, None)
        loader_note = "load_fp16_then_swap_int_via_patch"
    idx.setdefault("metadata", {})["embed_tokens_quant"] = {
        "bits": bits,
        "scheme": "per_row_absmax_symmetric",
        "packed": bits == 4,
        "side_file": embed_file,
        "loader": loader_note,
    }
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(idx, indent=2) + "\n"
    )

    cfg = json.loads((dst / "config.json").read_text())
    cfg["embed_tokens_quant"] = {
        "bits": bits,
        "scheme": "per_row_absmax_symmetric",
        "packed": bits == 4,
        "key_weight": EMBED_KEY,
        "key_scale": EMBED_SCALE_KEY,
        "side_file": embed_file,
    }
    cfg["_derivative_of"] = str(src.resolve())
    cfg["_derivative_note"] = (
        f"only language_model.embed_tokens quantized to int{bits}"
    )
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print("DONE", dst, "loader", loader_note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
