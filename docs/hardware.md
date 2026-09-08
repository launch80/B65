# The machine

## Workstation

ASUS ROG STRIX Z890-H · Core Ultra 7 270K (24 c, **AVX2 only**) · 30 GB RAM ·
Ubuntu 26.04, kernel 7.x.

| GPU | PCI | Slot | Role |
|---|---|---|---|
| **Arc Pro B65 32 GB** (ASRock Creator) | `0000:86:00.0` | chipset, PCIe 4.0 x4 | local AI, headless, `xe` driver, `renderD129` |
| **Arc B580** | `0000:04:00.0` | CPU x16 | gaming; monitor attached |
| Arrow Lake iGPU | — | — | fallback display |

The B65 deliberately gets the **chipset x4 slot**, measured at **7.16 GB/s**. Gaming pays PCIe
cost every frame; decode reads weights from VRAM, not the bus. (sysfs reports the card's
internal bridge as 2.5 GT/s x1 — cosmetic, ignore it.)

That x4 link is also why **PCIe weight streaming and CPU offload are both dead ends**: 7.16 GB/s
is ~80× slower than VRAM, and there is only 30 GB of system RAM behind it.

## Why the card is capped at 200 W

The original question that started the project. The answer: **nothing is throttling it.**

- The 200 W ceiling is **firmware PL1**, exposed at `power1_cap` in hwmon.
- Every throttle reason reads zero — `reason_pl1`, `reason_pl2`, `reason_pl4`, `reason_thermal`,
  `reason_vr_tdc`, `reason_prochot`, `reason_ratl`.
- Idle draw is 44.2 W.

A 600 W cable from a 1000 W PSU is a **supply** ceiling. 200 W is a **demand** ceiling set in
firmware. The cable was never the constraint.

Relevant hwmon nodes: `power1_cap` (PL1), `power1_crit`, `power1_cap_interval`,
`energy1_input`; frequency at `rp0_freq` / `rpa_freq` / `act_freq`.

Raising the cap needs root and this session has no TTY for `sudo`, so it must be run by the
user directly:

```bash
! echo 220000000 | sudo tee /sys/class/hwmon/hwmonN/power1_cap
```

**This matters for parity.** The B70 cookbook runs a **230 W** cap and its tables report
195–229 W actually drawn — so the cap is load-bearing for them. Part of the 1.33× stack gap
may simply not be available to us at 200 W.

## Xe2 / Battlemage capabilities

| Quantity | Value | Basis |
|---|---|---|
| XMX bf16 peak | **88.3 TFLOP/s** | measured at prefill, 2026-09-01 |
| XMX int8 peak | ~176 TOP/s | architectural 2× — **unverified**, R0 checks |
| XMX int4 peak | ~353 TOP/s | architectural 4× — **unverified**, R0 checks |
| FP8 on XMX | **no native path** | bf16/fp16/int8/int4 only; FP8 emulated at half rate |
| Memory bandwidth | 608 GB/s | datasheet (19 Gbps GDDR6 × 256-bit); real wall unmeasured |
| Machine balance (int4) | ~581 op/byte | 353 TOP/s ÷ 0.608 TB/s |
| Decode GEMV intensity | 2 op/byte | inherent to M=1 |
| **XMX utilization at decode** | **~0.34%** | 2 ÷ 581 |

**No native FP8 XMX is the single most consequential hardware fact here.** It is why block-FP8
sustains only 194 GB/s at M=1, why the FP8 GEMM ceiling is ~44.5 TFLOP/s (half of bf16), and
ultimately why **GPTQ-INT4 beats FP8 on these cards** — on compute as well as bandwidth.

## Shell quirk

Every `Bash` tool command in a Claude session on this box must begin with:

```bash
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
```

The harness shell inherits only `~/.local/bin`. The user's own `.bashrc` PATH bug was fixed
2026-08-30; this harness quirk persists independently.
