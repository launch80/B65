#!/usr/bin/env python3
"""Remove the host<->GPU syncs from GDNAttentionMetadataBuilder.build on the pure spec-decode path.

On a batch where every real sequence is a spec decode (no prefills, no plain decodes: single-stream DFlash/MTP
decode), build() indexes two GPU tensors with a CPU boolean mask:
    block_table_tensor[spec_sequence_masks_cpu, : self.num_spec + 1]
    num_accepted_tokens[spec_sequence_masks_cpu]
Boolean-mask indexing has a data-dependent output size, so each one waits for the device queue to drain. That stalls
the host every step and cancels async scheduling's CPU/GPU overlap (py-spy on the B65, 2026-11-03: build() ~60% of the k=7 engine
thread, no time left waiting on the step's output event).
On that path padded sequences are always at the back (vLLM's own comment, and spec_query_start_loc already relies on it),
so the mask equals "first num_spec_decodes rows" and a slice gives identical tensors without a sync.
The mixed spec/non-spec path is left untouched. Works on stock gdn_attn.py and after patches/patch_mtp_boundary.py
(which re-indents the same block; run this one after it). Idempotent.
"""
import re
from pathlib import Path
import vllm

MARK = "P608_GDN_SPEC_NOSYNC"
paths = {Path(vllm.__file__).resolve().parent / "v1/attention/backends/gdn_attn.py",
         Path("/workspace/vllm/vllm/v1/attention/backends/gdn_attn.py"),
         Path("/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/gdn_attn.py")}
# pure-spec branch: the masked block_table index that is immediately followed by `non_spec_state_indices_tensor = None`
PURE = re.compile(r"(?P<i>[ ]+)# Filter by spec_sequence_masks to exclude padded sequences\n"
                  r"(?P=i)spec_state_indices_tensor = block_table_tensor\[\n"
                  r"(?P=i)    spec_sequence_masks_cpu, : self\.num_spec \+ 1\n"
                  r"(?P=i)\]\n"
                  r"(?P=i)non_spec_state_indices_tensor = None\n")
FIN = re.compile(r"(?P<i>[ ]+)num_accepted_tokens = num_accepted_tokens\[spec_sequence_masks_cpu\]\n")


def pure(m):
    i = m["i"]
    return (f"{i}# {MARK}: padded sequences are at the back, so the mask is a prefix; slice (no sync)\n"
            f"{i}spec_state_indices_tensor = block_table_tensor[\n{i}    :num_spec_decodes, : self.num_spec + 1\n{i}]\n"
            f"{i}non_spec_state_indices_tensor = None\n")


def fin(m):
    i = m["i"]
    return (f"{i}if num_prefills == 0 and num_decodes == 0:  # {MARK}: pure-spec batch, mask is a prefix\n"
            f"{i}    num_accepted_tokens = num_accepted_tokens[:num_spec_decodes]\n"
            f"{i}else:\n{i}    num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]\n")


for p in sorted(x for x in paths if x.is_file()):
    s = p.read_text()
    if MARK in s:
        print(f"[gdn-spec-nosync] already patched {p}"); continue
    s, n1 = PURE.subn(pure, s)
    s, n2 = FIN.subn(fin, s)
    if (n1, n2) != (1, 1):
        raise SystemExit(f"[gdn-spec-nosync] anchors {n1},{n2} != 1,1 in {p}")
    compile(s, str(p), "exec")
    p.write_text(s)
    print(f"[gdn-spec-nosync] patched {p}")
