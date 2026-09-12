# Feature batch of 2026-09-12 — read this first

54 commits on `main` from `048716a` ("baseline: carsim before feature batch"),
which is the rollback point. 31 files, +9807 / −226.

**The suite is green.** Measured on a quiet machine at the end:

```
python3 -m drive.validate            ->  82/82  pass  0 HARD  0 soft  [113.0 s]
python3 -m drive.validate --modules  -> 100/100 pass  0 HARD  0 soft  [235.3 s]
```

identical to the baseline recorded in `00-baseline.md` before anything was
touched. Module self-checks: `tyre` pass, `powertrain` **90/90** (was 82),
`vehicle` **33/33** (was 32), `input` pass, `render` **26/26** (was 19),
`garage` pass, `aero.wing` **27/27** (was 21), `cars` pass, `ml` **11/11**.

> **Caveat on one number.** Two of the agents and I all independently saw
> `render.py`'s `V22 frame budget` fail while other work was running on the
> machine. It is a **wall-clock** test (`p99 <= 16 ms` over 600 frames) and it
> is load-flaky: 7 identical runs at load average 20.7 gave FAIL/FAIL/ok/FAIL/
> ok/ok/FAIL with p99 11.19–22.45 ms. On a quiet machine it passes. Nothing in
> this batch can reach it. If you see 99/100, check `uptime` before believing it.

## One file per task

| note | task | one line |
|---|---|---|
| `00-baseline.md` | — | the true baseline: 82/82 and 100/100, and why "100/100" is the `--modules` figure |
| `08-steering.md` | **8** | the owner's IMPORTANT, diagnosed and fixed: TC and the steer aid's β term |
| `04-wings-audit.md` | **4** | the three-wing audit. 5 areas PASS with numbers, **1 major bug**, 10 smaller findings |
| `03-wing-mount.md` | **3** | pylon / endplate / none on the rear wing, derived from the VLM |
| `02-optimiser-order.md` | **2** | the orders already agreed; the real bug was the span band |
| `01-chase3d.md` | **1** | the 3-D chase camera |
| `05-when-automatic.md` | **5** | "When automatic" — my interpretation is at the top. 6 defects, 6 now fixed |
| `06-car-library.md` | **6** | three cars, and the one-tyre-of-data finding |
| `07-weight.md` | **7** | adjustable mass as point masses, with ballast station |
| `09-ml.md` | **9** | the ES driving agent: +9.22 % a lap over a hand-written baseline |

`salvage/` holds the patch recovered from a worker that was killed mid-flight
early on; it has been reviewed and applied, and is kept only as provenance.

## The three things most worth your attention

1. **`04-wings-audit.md` section 3(b)** — the flank panel could never change
   flanks after the first corner, and never stowed. It sat on the *inner*
   (wrong) flank for **36.6 % of cornering time** on a scripted lap, worth
   −1.94 % instead of +3.82 %, and left 302 N of uncommanded side force on
   every straight. Fixed (`ceb5e04`). Everything you have previously measured
   about the device on a *lap* was measured with this bug present.
2. **`08-steering.md`** — the traction control was giving a wheel carrying a
   sixth of the axle load full authority over the engine, so the car
   *decelerated at full throttle* mid-corner (`ax = −0.206 m/s²`). And the
   steer aid's `1.2·|β|` term was positive feedback that opened the limit to
   full mechanical lock in a slide.
3. **`06-car-library.md`, the tyre section** — there is exactly **one tyre's
   worth of coefficient data** in `tyre_data/`. Only 5 of 12 `.tir` files load,
   and all 5 are the same coefficient set with different geometry. Every
   grip difference between cars is therefore a labelled `mu_scale`
   calibration, not data.

## Findings reported upward, deliberately NOT fixed

Beyond CONTRACT section 10's own two (`qss.residuals`'s `Y_r`, and
`alpha_peak_deg = 7.0`):

* **The tyre is not odd under combined slip.** `drive/tyre.py` already knew and
  says so; `RHX1/RHY1/RHY2` are kept on purpose and that is incompatible with
  CONTRACT section 2's "exactly odd to 1e-9" claim. Newly quantified: 0.70 % in
  pure lateral, 8.5 % combined, and at the car level 0.8 % of `a_y` in steady
  cornering but a **1.5 m/s² transient when braking and steering together**.
  Zeroing `RHY1` would make it exact and move every calibrated number.
* **`cfg.h_aero` is dead code** (`dV` exactly 0.000e+00 when changed 0.55→1.50).
* **`Fy_body` never enters the lateral load transfer** — 4.85 % of `dFz_tot` at
  the plate limit, and the code comment claiming algebraic identity with `qss`
  is false.
* **`drive/plots.py` holds its own `CorsaC()`**, so a 540i telemetry trace gets
  the Corsa's g-g envelope. The telemetry *sidecar* is correct.
* Residual auto-gearbox hunting on a 3–6 % grade at mid speed (6 of 33 cells)
  is genuinely between-gears and curing it needs new state.

## Contract changes

`drive/CONTRACT.md` was extended or corrected in the same commit as its code
every time, as the brief requires: section 1 (module map), 2 (`tyre_for`,
`mu_curve_matches`), 4 (the `Mz_dev` algebra **was wrong**, `wing_side` is the
TURN sign and **was documented backwards**, the per-car scaling block, the
added-mass block), 7 (`set_car`, `HudData`), 8 (`--car`, `--ballast`,
`--ml-drive`, `Settings`), and reconciliation 9 (`power_scale` now carries two
factors and only one is an aid).
