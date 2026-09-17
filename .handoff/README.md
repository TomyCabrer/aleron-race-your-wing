# Feature batch of 2026-09-12 — read this first

54 commits on `main` from `048716a` ("baseline: carsim before feature batch"),
which is the rollback point. 31 files, +9807 / −226.

**The suite is green.** Measured on a quiet machine at the end:

```
python3 -m drive.validate            ->  82/82  pass  0 HARD  0 soft  [113.2 s]
python3 -m drive.validate --modules  -> 100/100 pass  0 HARD  0 soft  [238.9 s]
```

`--only` no longer lies: a filter that matches nothing prints what it matched
against and exits 2, instead of `0/0 pass` after 324 s of work.

identical to the baseline recorded in `00-baseline.md` before anything was
touched. Module self-checks: `tyre` pass, `powertrain` **90/90** (was 82),
`vehicle` **33/33** (was 32), `input` pass, `render` **26/26** (was 19),
`garage` pass, `aero.wing` **27/27** (was 21), `cars` pass, `ml` **11/11**.

> **All eight wall-clock checks are now load-normalised** (`render.py`'s V22,
> the two open-map frames, the menu overlay and the chase budget; the garage's
> three page draws) against a reference workload timed in the same run.
> Verified at load average 40 and with twelve saturated cores: 100/100 and
> 27/27. The caveat below is kept for the record.
>
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

### Wave 2

| note | item | one line |
|---|---|---|
| `11-rwd.md` | **1** | real rear-wheel drive; the RWD cars' longitudinal numbers are physics now, not fiction |
| `12-lapdriver.md` | **2** | `LapDriver` scaled off the car; all three cars lap inside the ribbon |
| `10-lap-reconclusions.md` | **3** | the lap-based wing conclusions re-run: **the sign flips** |
| (in `README.md` + `CONTRACT.md`) | **4** | the new flags, settings, the mount, the car library, the ML agent |
| (in `render.py`) | **5** | V22 load-normalised against a reference workload measured in the same run |

`salvage/` holds the patch recovered from a worker that was killed mid-flight
early on; it has been reviewed and applied, and is kept only as provenance.

### Wave 3

| note | item | one line |
|---|---|---|
| `06-car-library.md` (engine + brake sections, and the new comparison table at the end) | **1, 2** | per-car engine curves and per-car brakes/steering lock — the library is now three cars, not one with three labels |
| `11-rwd.md` | **1** | 0–100 corrected again: MX-5 10.42 → **9.31 s** |
| `09-ml.md` | **3** | the ML claim **validated and narrowed**: 4.9–9.2 % a lap, and it transfers |

### Wave 4

| note | item | one line |
|---|---|---|
| `09-ml.md` | **1, 2** | anchor fixed and everything retrained (gain **4.9–9.2 % → 3.4–8.7 %**, deliberately); **one policy per car**, and the reason is safety |
| `06-car-library.md` | **3** | suspension and `CdA` grounded in published springs and bars; neither new car's top speed can validate `CdA`, and why |

### Waves 5 and 6

| note | one line |
|---|---|
| `09-ml.md` (wave-5 sections) | the anchor is surface-aware (17 observations), **`MARGIN_FADE` retired by deletion**, the **540i drives `open`**, and the claim that the baseline cannot lap without the device is **withdrawn** |
| `13-curved-flow-and-orientation.md` | the owner's two aero questions: the curved-flow AoA term (**halves the fin's lap benefit**), and why "suction side outwards" is **not answerable** by the current model |

## Waves 5 and 6 in one paragraph

Wave 5 made the ML anchor surface-aware — three new observations
(`kappa_4`, `mu_here`, `mu_ahead`, 14 → 17) and a full retrain of all five
checkpoints — which **retired `MARGIN_FADE` by deleting it** rather than
setting it to zero, made the **540i lap the open map** (63.464 s anchor, where
it used to go off at 443.7 m), and **withdrew a claim carried since wave 1**:
the baseline does *not* fail without the device. Both failures were inside the
arena's wet patch, and wing plate / fin / off now land within 3 ms of each
other. Wave 6 answered the owner directly: the flank panel's angle of attack
was missing its **curved-flow term** (`v + r·x_w`, and the sign is the opposite
of the intuition — a forward-mounted panel sees *less* incidence), which
**more than halves the fin's lap benefit**; and "suction side outwards" is
**unanswerable by the current model**, because the flank panel is solved in
free air with no image of the car body at all.

## Wave 4 in one paragraph

The ML anchor was fixed twice. First the 70 m lookahead it never read — which
is why the hand-written driver drove off the open map — and the whole package
was retrained against it, **shrinking the headline from 4.9–9.2 % to
3.4–8.7 % on purpose**: a smaller gain over a driver that can drive is worth
more than a large one over a broken one, and `arena_plate` now laps **9 cells
of 9** where it managed 6. Then, extending to all three cars, the anchor
spun both rear-driven cars — and the cause turned out **not** to be the driven
axle (my hypothesis) but the arena's 130 m wet patch, which `baseline_action`
cannot see because the observation carries no `mu` term. A throttle cap was
implemented, measured, and **deleted**; an entry-speed margin was kept.
The cross-car answer is **one policy per car, for safety not speed** — two
off-diagonal cells lose the MX-5 on a circuit its own anchor laps — and the
**wing timing genuinely differs per car, monotone in mass and power**: the
540i arms the panel 33.3 m before turn-in and carries it *more*, the Corsa
24.7 m before and *less*, and the MX-5 learns nothing about the device at all.
Separately, the two new cars' suspension and `CdA` were grounded in published
spring rates and bar diameters.

## Wave 3 in one paragraph

`engine_scale` used to scale the *Corsa's* torque curve bodily, so the MX-5
peaked at 4000 rpm and cut at 6200 when a BP-Z3 peaks at 5000 and revs to
7000. `powertrain.engine_curve(car)` now builds each engine's own curve,
hitting **both** published points exactly, and `engine_scale` is retired (it
would double-count) — which also restores reconciliation 9 to meaning what it
originally said. `brake_coeffs(car)` does the same for the brakes from
documented disc and drum geometry, and the steering lock is per-car through
physics, aid and HUD. On the ML side, a policy trained **only on the open map
is +4.86 % on an arena it has never seen**, which settles the memorisation
question, and the residual design is measurably safe: across 27
(policy, cell) pairs the worst regression is −1.39 % and it never puts a car
off the road that the baseline was keeping on.

## Wave 2 in one paragraph

The two RWD cars now actually drive their rear wheels, which made their 0-100
real (540i 7.845 -> 6.8138 s against a real ~6.2 s) and, at first, made their
laps far worse — they stopped understeering off the track and started spinning
off it. `LapDriver` now scales its planned grip (from an open-loop ramp steer
of that car), its gains, its lookahead and its margin off the car itself, and
all three cars lap the arena inside the ribbon with the Corsa bit-for-bit. With
the driver able to complete laps, the lap-based wing conclusions were re-run
for the first time and **the device's lap-time effect changes sign**: it was a
loss with the flank-latch bug present and is a gain without it. V22 no longer
cries wolf under load, and the README and CONTRACT now document everything
both waves added.

## NUMBERS THAT MOVED MORE THAN ONCE — read this before the older notes

Several figures were re-measured two or three times as the measurement got
more honest. The **latest** value is always in the note named above for that
wave; the older notes keep their superseded numbers beside the new ones on
purpose, but if you read only one, read this list.

| claim | where it ended up |
|---|---|
| ML gain over the hand-written driver | 9.22 % → 4.9–9.2 % → 3.4–8.7 % → **6.58 % arena / 3.57 % open / 7.40 % skidpad** (per-car 5.79–6.70 %) |
| "the baseline cannot lap without the device" | **WITHDRAWN.** Both failures were the arena's wet patch. plate/fin/off within 3 ms |
| the device's worth on a lap | +0.159 s (plate) → **+0.147 s** corrected; fin +0.058 → **+0.028 s** |
| the device's per-corner gain | fin +3.86 % → **+3.74 %**, plate +6.05 % → **+5.94 %** corrected |
| MX-5 0–100 km/h | 10.90 (FWD) → 10.42 (RWD) → **9.31 s** (own engine curve) |
| 540i 0–100 km/h | 7.845 → 6.814 → **6.905 s** |
| 540i on the open map | off at 443.7 m → **laps in 63.464 s** |
| "the MX-5 learns nothing about the device" | **CORRECTED** — on the surface-aware anchor it arms the panel 13.1 m before turn-in |
| MX-5 / 540i roll gradient | 5.75 / 6.01 → **5.02 / 5.69 °/g** |

## The three things most worth your attention

1. **`10-lap-reconclusions.md`** — the flank-latch bug (`04-wings-audit.md`
   §3(b), fixed in `ceb5e04`) did not just add noise to the lap-based device
   numbers, it **inverted them**. Fitting the fin used to cost 0.240 s a lap
   and now saves 0.058 s; the plate used to cost 0.157 s and now saves
   0.159 s. Any "the device does not pay for itself over a lap" conclusion you
   hold from before that commit is an artefact. The per-corner and open-loop
   numbers are untouched and remain the figure of merit.
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
