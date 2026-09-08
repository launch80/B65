#!/usr/bin/env python3
"""Torch/XPU packer for the Launch80 W3 format, validated against the numpy reference.

The numpy packer in kernel/w3_format.py is the spec and the test oracle, but it is far
too slow for a real checkpoint (down_proj alone is 89M weights). This is the same
format on the GPU. It is checked against the numpy version rather than trusted, and
the adversarial suite is re-run through it, because a packer that is subtly wrong
produces weights with the right variance and no error anywhere.
"""
import torch

PAIR = [(p // 5, 3 * (p % 5)) for p in range(15)]
SPARE = [[(2, 15), (1, 15), (0, 15)], [(2, 31), (1, 31), (0, 31)]]


def pack_blocks(q):
    """q: uint8/int64 [B, 32] values 0..7  ->  int32 [B, 3].

    Accumulates in int64 because a shift into bit 31 overflows int32, then truncates
    to int32, which keeps the bit pattern (the sign bit is payload here, not sign)."""
    q = q.to(torch.int64)
    w = torch.zeros((q.shape[0], 3), dtype=torch.int64, device=q.device)
    for p, (m, s) in enumerate(PAIR):
        w[:, m] |= (q[:, 2 * p] & 7) << s
        w[:, m] |= (q[:, 2 * p + 1] & 7) << (s + 16)
    for i, bits in enumerate(SPARE):
        v = q[:, 30 + i]
        for b, (m, bit) in enumerate(bits):
            w[:, m] |= ((v >> b) & 1) << bit
    return (w & 0xFFFFFFFF).to(torch.int32)


def pack_columns(codes):
    """codes: [K, N] values 0..7  ->  int32 [N, 3*K/32] in the three-plane layout.

    Plane m holds word m of every 32-weight block, contiguous down the column, so one
    load per plane gives the kernel a whole block."""
    K, N = codes.shape
    nblk = K // 32
    c = codes.t().reshape(N * nblk, 32)          # column-major, block-major
    w = pack_blocks(c).reshape(N, nblk, 3)
    return torch.cat([w[:, :, 0], w[:, :, 1], w[:, :, 2]], dim=1).contiguous()


def _selftest():
    import numpy as np, sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "..", "L80", "B65-repo", "kernel"))
    sys.path.insert(0, "/kernel")
    from w3_format import pack as np_pack, unpack as np_unpack, run_tests

    dev = "xpu" if torch.xpu.is_available() else "cpu"
    rng = np.random.default_rng(608)
    ok = True

    # 1. agree with the numpy spec on random data
    q = rng.integers(0, 8, size=(4096, 32)).astype(np.uint8)
    a = np_pack(q).astype(np.int64)
    b = pack_blocks(torch.from_numpy(q.astype(np.int64)).to(dev)).cpu().numpy().astype(np.int64) & 0xFFFFFFFF
    if not np.array_equal(a, b):
        d = np.flatnonzero((a != b).any(1))[:3]
        print(f"  FAIL torch vs numpy packer differ at blocks {d.tolist()}"); ok = False

    # 2. and on the adversarial patterns, not just random
    k = np.arange(97 * 32).reshape(97, 32)
    q2 = ((7 * (k // 32) + 3 * (k % 32)) % 8).astype(np.uint8)
    a2 = np_pack(q2).astype(np.int64)
    b2 = pack_blocks(torch.from_numpy(q2.astype(np.int64)).to(dev)).cpu().numpy().astype(np.int64) & 0xFFFFFFFF
    if not np.array_equal(a2, b2):
        print("  FAIL torch packer disagrees on ordinal pattern"); ok = False

    # 3. round-trip through the INDEPENDENT numpy unpacker
    if not np.array_equal(np_unpack(b.astype(np.uint32)), q):
        print("  FAIL torch pack -> numpy unpack round-trip"); ok = False

    # 4. three-plane assembly puts the right word in the right plane
    K, N = 128, 7
    codes = rng.integers(0, 8, size=(K, N)).astype(np.uint8)
    pl = pack_columns(torch.from_numpy(codes.astype(np.int64)).to(dev)).cpu().numpy()
    pl = pl.astype(np.int64) & 0xFFFFFFFF          # int32 bit patterns -> unsigned
    nblk = K // 32
    for n in range(N):
        for bi in range(nblk):
            want = np_pack(codes[bi*32:(bi+1)*32, n][None, :].astype(np.uint8))[0]
            got = [pl[n, bi], pl[n, nblk + bi], pl[n, 2 * nblk + bi]]
            if [int(v) for v in want] != [int(v) for v in got]:
                print(f"  FAIL plane layout at col {n} block {bi}"); ok = False; break
    print("  numpy spec suite:", "PASS" if run_tests() else "FAIL")
    print("  torch packer:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
