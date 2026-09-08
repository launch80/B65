#!/usr/bin/env python3
"""Stage 0b gate: does a hand-written SYCL custom op survive XPU graph capture?

If it does not, the W3A16 kernel cannot be used in the served model as planned and
the whole project's target changes (an eager-mode kernel competes against an
eager-mode vendor baseline, ~395 GB/s rather than 473).

The dangerous outcome is not an exception - it is SILENCE. A SYCL op that submits
on its own queue is never recorded by stream capture, so replay reuses whatever was
in the output buffer from the warm-up. The numbers look plausible and the model is
wrong. So the probe checks the OWN-QUEUE op too, and expects it to FAIL, to prove
the test can actually see the failure it is looking for.
"""
import torch, os, sys

so = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libp608probe.so")
torch.ops.load_library(so)

# fake/meta impls: needed the moment anything traces or compiles through the op
for name in ("scale_stream", "scale_ownqueue"):
    torch.library.register_fake(f"p608probe::{name}",
                                lambda x, s: torch.empty_like(x))

dev = "xpu"
N = 1 << 20
print(f"device: {torch.xpu.get_device_name(0)}   torch {torch.__version__}")
print(f"has XPUGraph: {hasattr(torch.xpu, 'XPUGraph')}   "
      f"has xpu.graph: {hasattr(torch.xpu, 'graph')}\n")


def probe(opname, expect_capture):
    op = getattr(torch.ops.p608probe, opname)
    static_in = torch.ones(N, device=dev, dtype=torch.float32)

    # eager correctness first - if this is wrong, nothing else means anything
    got = op(static_in, 3.0)
    torch.xpu.synchronize()
    if not torch.allclose(got, torch.full_like(got, 3.0)):
        return f"EAGER BROKEN (got {got[0].item()}, want 3.0)"

    # warm-up on a side stream, as graph capture requires
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(3):
            op(static_in, 3.0)
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()

    g = torch.xpu.XPUGraph()
    try:
        with torch.xpu.graph(g):
            static_out = op(static_in, 3.0)
    except Exception as e:
        return f"CAPTURE RAISED: {type(e).__name__}: {str(e)[:90]}"

    # The real test: change the input, replay, and see whether the output follows.
    # If the kernel was never recorded, static_out keeps its warm-up value and the
    # mismatch below is what catches it.
    static_in.fill_(2.0)
    g.replay()
    torch.xpu.synchronize()
    want = 6.0
    lo, hi = static_out.min().item(), static_out.max().item()
    if abs(lo - want) < 1e-4 and abs(hi - want) < 1e-4:
        return "REPLAY OK"
    return f"REPLAY STALE/WRONG (out in [{lo:.3f},{hi:.3f}], want {want})"


results = {}
for opname, expect in (("scale_stream", True), ("scale_ownqueue", False)):
    try:
        r = probe(opname, expect)
    except Exception as e:
        r = f"EXCEPTION {type(e).__name__}: {str(e)[:90]}"
    results[opname] = r
    print(f"  {opname:16s} -> {r}")

print()
ok_stream = results["scale_stream"] == "REPLAY OK"
ctrl_failed = results["scale_ownqueue"] != "REPLAY OK"
if ok_stream and ctrl_failed:
    print("  GATE PASS: a SYCL kernel submitted on torch's current XPU stream is")
    print("             captured and replays correctly. The control (private queue)")
    print("             failed as designed, so the probe can see the silent failure.")
elif ok_stream and not ctrl_failed:
    print("  GATE PASS (weak): the stream op works, but the private-queue control ALSO")
    print("             'passed', so this probe cannot distinguish captured from")
    print("             not-captured. Treat the result as unproven.")
else:
    print("  GATE FAIL: custom SYCL ops do not survive XPU graph capture here.")
    print("             The kernel would have to run eager; re-baseline against the")
    print("             vendor's per-call 395 GB/s, not its captured 473.")
sys.exit(0 if ok_stream else 1)
