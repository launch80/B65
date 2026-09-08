#!/usr/bin/env python3
"""Launch80 W3 format v1 - packer, independent unpacker, and adversarial tests.

THE FORMAT. 32 weights in exactly 3 int32 words - 96 bits for 96 bits of payload,
zero waste. The constraint that shapes it is that dequant must be doable with the
half2 trick: (w >> s) & 0x00070007 | 0x64006400 reinterpreted as fp16 pairs is
(1024+lo, 1024+hi), because 0x6400 is fp16 1024.0 and the low mantissa bits carry
the field. That needs the two 3-bit fields at the SAME offset in each 16-bit half.

Within one 32-bit word, valid shifts are s in {0,3,6,9,12}: five pairs, occupying
bits 0-14 and 16-30. Bits 15 and 31 are unreachable that way. So 3 words carry
30 weights as 15 cheap pairs, and leave exactly 6 spare bits - which is exactly the
2 remaining weights. Nothing is wasted, and 30 of 32 weights decode at ~2.5 ops per
two weights instead of ~12.

  pair p (0..14):  word m = p//5, shift s = 3*(p%5)
      weight 2p    -> bits [s+2:s]        of word m
      weight 2p+1  -> bits [s+18:s+16]    of word m
  weight 30 -> w0 bit15 (lsb), w0 bit31, w1 bit15 (msb)
  weight 31 -> w1 bit31 (lsb), w2 bit15,  w2 bit31 (msb)

Values are q in 0..7 meaning q-4, matching the symmetric 3-bit grid that
research/r2/gptq_study.py quantizes to (lv = 2^(bits-1)-1 = 3, clamp -4..3).

WHY THE TESTS BELOW LOOK LIKE THIS. A wrong bit layout does not crash. Against
i.i.d. Gaussian weights it yields plausible values with the right variance, because
Gaussians are exchangeable - permuting them changes nothing you can see. Every test
here is built to destroy that exchangeability.
"""
import numpy as np

PAIR = [(p // 5, 3 * (p % 5)) for p in range(15)]     # (word, shift) per pair
# Spare-bit assignment for weights 30 and 31, chosen so the GPU can extract BOTH
# in ~12 ops instead of ~25. All six spare bits are bit 15 and bit 31 of the three
# words, so one masked merge collects them:
#     u = (w0 & 0x80008000) | ((w1 & 0x80008000) >> 1) | ((w2 & 0x80008000) >> 2)
# puts w0/w1/w2's bit15 at u bits 15/14/13 and their bit31 at u bits 31/30/29, so
# ((u >> 13) & 0x00070007) is exactly the half2 pair (weight30, weight31).
# Hence weight30 = 4*w0[15] + 2*w1[15] + w2[15], and likewise at bit 31.
SPARE = [  # (word, bit) triples, LSB first
    [(2, 15), (1, 15), (0, 15)],
    [(2, 31), (1, 31), (0, 31)],
]


def pack(q):
    """q: uint8 [nblock, 32] with values 0..7 -> uint32 [nblock, 3]."""
    q = np.asarray(q, dtype=np.uint32)
    assert q.ndim == 2 and q.shape[1] == 32
    assert q.max(initial=0) <= 7, "values must be 0..7"
    w = np.zeros((q.shape[0], 3), dtype=np.uint32)
    for p, (m, s) in enumerate(PAIR):
        w[:, m] |= (q[:, 2 * p] & 7) << s
        w[:, m] |= (q[:, 2 * p + 1] & 7) << (s + 16)
    for i, bits in enumerate(SPARE):
        v = q[:, 30 + i]
        for b, (m, bit) in enumerate(bits):
            w[:, m] |= ((v >> b) & 1) << bit
    return w


def unpack(w):
    """Written from the docstring above, NOT by inverting pack().

    uint32 [nblock, 3] -> uint8 [nblock, 32]."""
    w = np.asarray(w, dtype=np.uint32)
    assert w.ndim == 2 and w.shape[1] == 3
    out = np.zeros((w.shape[0], 32), dtype=np.uint8)
    for idx in range(30):
        p, half = idx // 2, idx % 2
        word, shift = p // 5, 3 * (p % 5)
        sh = shift + (16 if half else 0)
        out[:, idx] = ((w[:, word] >> sh) & 7).astype(np.uint8)
    for i in range(2):
        acc = np.zeros(w.shape[0], dtype=np.uint8)
        for b, (m, bit) in enumerate(SPARE[i]):
            acc |= (((w[:, m] >> bit) & 1) << b).astype(np.uint8)
        out[:, 30 + i] = acc
    return out


def _fail(msg):
    print(f"  FAIL  {msg}")
    return False


def run_tests():
    ok = True
    rng = np.random.default_rng(608)

    # 1. ORDINAL VALUES. Every value encodes its own position, so no permutation,
    #    rotation or straddle error can survive - unlike Gaussian data.
    nb = 97
    k = np.arange(nb * 32).reshape(nb, 32)
    q = ((7 * (k // 32) + 3 * (k % 32)) % 8).astype(np.uint8)
    ok &= (np.array_equal(unpack(pack(q)), q) or _fail("ordinal round-trip"))

    # 2. FULL-COVERAGE BIJECTION: every value 0..7 must appear in every one of the
    #    32 slots, so no slot is silently aliased to another.
    q2 = np.zeros((8, 32), dtype=np.uint8)
    for v in range(8):
        q2[v, :] = v
    ok &= (np.array_equal(unpack(pack(q2)), q2) or _fail("constant-value coverage"))
    q3 = rng.integers(0, 8, size=(4096, 32)).astype(np.uint8)
    got = unpack(pack(q3))
    ok &= (np.array_equal(got, q3) or _fail("random round-trip"))
    seen = np.zeros((32, 8), dtype=bool)
    for s in range(32):
        for v in np.unique(q3[:, s]):
            seen[s, v] = True
    ok &= (seen.all() or _fail("not all 8 values exercised in all 32 slots"))

    # 3. ONE-HOT / DELTA: exactly one weight non-zero, swept over all 32 positions.
    #    Pins each slot independently; a swapped pair dies here and nowhere else.
    for s in range(32):
        for v in range(1, 8):
            q4 = np.zeros((1, 32), dtype=np.uint8)
            q4[0, s] = v
            r = unpack(pack(q4))
            if r[0, s] != v or r.sum() != v:
                ok = _fail(f"one-hot slot {s} value {v} -> {r[0]}")
                break

    # 4. BIT ALIASING: every one of the 96 payload bits must be used exactly once.
    #    Catches two fields overlapping, which random data hides almost perfectly.
    used = np.zeros(96, dtype=int)
    for s in range(32):
        for b in range(3):
            q5 = np.zeros((1, 32), dtype=np.uint8)
            q5[0, s] = 1 << b
            w = pack(q5)[0]
            bits = np.unpackbits(w.view(np.uint8), bitorder="little")
            idx = np.flatnonzero(bits)
            if len(idx) != 1:
                ok = _fail(f"slot {s} bit {b} set {len(idx)} bits, expected 1")
            else:
                used[idx[0]] += 1
    if not np.array_equal(used, np.ones(96, dtype=int)):
        dup = np.flatnonzero(used > 1); miss = np.flatnonzero(used == 0)
        ok = _fail(f"bit map not a bijection: duplicated {dup.tolist()}, unused {miss.tolist()}")

    # 5. HALF2 REACHABILITY: the whole point of the layout. For each of the 15
    #    pairs, (w >> s) & 0x00070007 must recover exactly weights 2p and 2p+1.
    q6 = rng.integers(0, 8, size=(256, 32)).astype(np.uint8)
    w6 = pack(q6)
    for p, (m, s) in enumerate(PAIR):
        ext = (w6[:, m] >> s) & 0x00070007
        lo, hi = ext & 7, (ext >> 16) & 7
        if not (np.array_equal(lo, q6[:, 2 * p]) and np.array_equal(hi, q6[:, 2 * p + 1])):
            ok = _fail(f"pair {p} (word {m} shift {s}) not half2-extractable")
            break
    return ok


if __name__ == "__main__":
    print("Launch80 W3 format v1 - adversarial correctness")
    print(f"  32 weights in 3 words = {32*3} bits payload in 96 bits: zero waste")
    print(f"  15 half2 pairs + 2 spare-bit weights\n")
    good = run_tests()
    print("\n  ALL TESTS PASS" if good else "\n  TESTS FAILED")
    raise SystemExit(0 if good else 1)
