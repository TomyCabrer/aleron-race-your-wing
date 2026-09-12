# 08 — steering / driveability bug report

> "IMPORTANT: steering doesn't work well. No acceleration while steering.
>  No steering while braking."

Owner's settings (`runs/settings.json`): `engine sport` (**power_scale 2.0**),
`abs true`, `tc true`, `steer_aid true`, `gearbox auto`, track `open`.

## Reproduction harness

`python3 -m drive.drive --headless --script drive_probe --engine sport`

New script `drive_probe` in `drive/drive.py`. It is the ONE scripted path that
deliberately switches the aids ON — because the bug is *about* the aids and an
aid that cannot be measured cannot be tuned. It is **not** a measurement:
`validate.py` never calls it, and CONTRACT section 9 item 9 is untouched for
every acceptance path. It runs both input paths:

* `kb`  — real `input.KeyboardInput` fed a synthetic key state through
  `set_keys()`, `steer_limit=True`, `k_us_deg=K_US_DEG_MEASURED`. The human path.
* `raw` — `delta` commanded directly at the same 56.25 deg/s hand rate. No aid.

A difference between the two is an input-layer effect; a difference inside one
is physics.

## Diagnosis (numbers from the baseline run, commit `edffa06`)

### (b) "no acceleration while steering" — REAL, and it is the TC sensor

Full throttle, 15 m/s, gear 3, step steer, `power_scale 2.0`, mean `ax` over
t = 3–5 s:

| steer | ax (TC on) | ax (TC off) | tc_gain | eng_load |
|---|---|---|---|---|
|  3 deg | +2.555 | +2.601 | 0.902 | 0.881 |
|  6 deg | +0.691 | +0.739 | 0.739 | 0.495 |
|  9 deg | +0.467 | +0.298 | 0.367 | 0.367 |
| 14 deg | **−0.206** | +0.199 | **0.256** | 0.256 |

At 14 deg the car **decelerates with the pedal on the floor**. Through the real
keyboard at 20 m/s / gear 4: `up` alone gives `ax = +1.773`; `up + right`
gives `ax = −0.033`.

Mechanism: `Vehicle._tc` reads `max(kx[FL], kx[FR])`. On this FWD car with an
open diff the INSIDE front unloads and spins up — at 6 deg of steer,
`Fz_FL = 991 N` against `Fz_FR = 4846 N` (17% of the axle load) and
`kappa_FL = 0.508` against `kappa_FR = 0.0116`. `max()` therefore reads the
wheel that is light, not the car that is out of traction, and cuts the engine
to a quarter exactly when the driver is cornering.

### (c) "no steering while braking" — REAL, two separate mechanisms

Settled yaw rate, full brake from 30 m/s, ABS on, vs commanded road-wheel angle:

| delta | r (rad/s) | alpha_f (deg) |
|---|---|---|
|  2 | 0.123 |  −1.29 |
|  8 | **0.523** |  −9.29 |
| 14 | 0.473 | −15.88 |
| 32 | 0.404 | −35.96 |

Yaw response **peaks at 8 deg and falls thereafter** — past the tyre peak
(alpha at the MF front-axle peak is 10.35 deg) adding lock subtracts turning.
With the clutch ENGAGED (his automatic box) it is far worse: 0.500 rad/s at
8 deg collapsing to **0.007 rad/s at 32 deg** — full lock, straight ahead.

The aid does not stop him getting there: `steer_limit_deg` adds
`BETA_LOCK_GAIN * |beta_deg|` to the limit **symmetrically**, so the limit at
30 m/s is 9.30 deg at beta = 0 but **32.625 deg (full mechanical lock) at
|beta| = 20 deg**. Sliding raises the limit, more lock deepens the slide:
positive feedback. The contract's stated purpose for that term is *counter*-steer
("so slides can be caught"), which needs it on one side only.

With ABS OFF the wheels lock and the car is a sled: r = 0.0004 rad/s at 2 deg,
0.026 at 32 deg, `Fy_f` 190–2431 N against 5165–5344 N with ABS. That half is
**correct physics** and is left alone.

### (a) "steering doesn't work well" — the aid, not the ramp

Limiter floor with `k_us_deg = 7.82`: 9.30 deg at 30 m/s, 8.31 at 50 —
*plenty* (the car needs 7.97 deg at R = 100, 9.57 at R = 50), so the floor is
NOT the complaint. Ramp to the limit at 30 m/s is 0.165 s, return to centre
0.103 s — brisk, not sluggish. What is wrong is the beta feedback above: the
useful band is 0–9.3 deg and the aid lets the driver sit at 32 deg where the
axle is 26 deg past its peak.

### NOT guilty

* **The keyboard input layer.** `up+right`, `down+right` and `LSHIFT`+both all
  deliver both axes: throttle 1.0 / delta −12.84, brake 1.0 / delta −18.50,
  throttle 0.50 (fine cap) / delta +13.07. No press is lost.
* **The limiter's static floor** (see above).
* **The friction ellipse / `ECAP = 1.05`.** Not reached in any of these cases;
  the combined-slip loss is the MF `Gxa`/`Gyk` weighting, which is the honest
  physics the owner should feel.

---

## Where it landed

The repro harness is commit `a5f36b1`. The fixes themselves were staged when a
concurrent session's `git commit` swept them into **`0b145f6`** ("handoff:
wings audit -- fix P_req line number"), whose message is therefore NOT about
them -- this note is the record. The flank-panel latch fix is its own commit,
`ceb5e04`, and the eased-bound initialisation is the commit carrying this line.

## What was changed

Four changes, all in `drive/input.py`, `drive/vehicle.py`, `drive/drive.py`,
plus `drive/CONTRACT.md` sections 4, 6 and 8 in the same commits.

### 1. `Vehicle._tc`: the cut depth is limited by the slipping wheel's load share

The sensor is still `max(kx[FL], kx[FR])` — a spinning wheel must be seen —
but the gain floor is now `max(TC_GAIN_MIN, 1 - 2*share)` where `share` is
that wheel's fraction of the front-axle vertical load, and `Fz` is passed in
from the same step (`_tc(throttle, Fz, dt)`; no lag, no new algebraic loop).

The `2` is the open diff: both driven wheels carry equal torque, so the axle's
tractive force is twice the force of the wheel with less grip, and a wheel of
load share `s` can still deliver `2s` of the axle's capacity. Cutting below
`1 - 2s` throws away torque the axle could have used. Straight ahead
`share = 0.5`, the floor is 0, and the full `TC_GAIN_MIN` authority is back —
which is why nothing about the launch moves.

Two alternatives were tried and rejected, numbers below: a load-WEIGHTED
sensor (reads 0.03 at 9 deg, so the aid stops existing mid-corner and gives
back the 0.29 m/s² the cut was actually buying) and `min()` (identical to TC
off in a corner, and loses a third of the launch protection).

### 2. `steer_limit_pair_deg`: the beta bonus goes to the counter-steer side only

`steer_limit_deg` is untouched — the contract pins its signature and V16/V17
and `validate.py` quote from it. New `steer_limit_pair_deg(V, beta_deg, ...)
-> (limit on LEFT lock, limit on RIGHT lock)` puts the whole
`beta_gain*|beta_deg|` bonus on the side that counter-steers the slide and
leaves lock into the slide at the floor. Counter-steer has the SAME sign as
`beta` (`beta = atan2(v, |u|)`, `v` leftward → a left-turn slide is
`beta < 0` and the catch is right lock — the same statement CONTRACT section 9
item 7 makes). `KeyboardInput` and `GamepadInput` both clamp against the pair.

### 3. The clamp eases down instead of snapping

Each bound is a state: it rises to the commanded value instantly, and falls at
`_return_rate_deg(V)` — the rate the wheel's own self-aligning torque would
unwind it at. Required by change 2 (a directional bound can collapse 20+ deg
the instant `beta` changes sign) and an improvement in its own right, since
the old hard clamp was a discontinuity in the road-wheel angle.
`delta_lim_deg` now reports whichever bound is binding the direction the wheel
is actually turned.

### 4. Gamepad trigger rest (found while checking the pad path, not a symptom)

* Before the pad's first HID report every axis reads exactly `0.0`, and the old
  `(a+1)/2` turned that into HALF TRAVEL: **a hot-plug handed the car 47%
  throttle and 47% brake simultaneously**, for as many frames as the pad took
  to report. The pedals are now held at 0 until `_rest_checked` is true.
* `_trig_rest` latches whichever rest the hardware actually showed on its first
  report (`-1.0` or `0.0`) and `_trigger()` normalises from it, so both driver
  conventions give 0 at rest and 1 at the stop. With `-1.0` it is the
  historical `(a+1)/2` exactly. Only a genuinely mid-travel rest still gets the
  loud "layout may not match" warning, and `_check_rest` no longer `return`s
  after the first oddity (which meant the second trigger was never examined).

## Before / after

`python3 -m drive.drive --headless --script drive_probe --engine sport`

### (b) acceleration while steering — raw path, 15 m/s, gear 3, WOT, TC ON

mean `ax` (m/s²) over t = 3–5 s:

| steer | OLD (full authority) | NEW (load-limited) | TC off | (tried) Fz-weighted | (tried) min() |
|---|---|---|---|---|---|
|  3 deg | 2.555 | **2.576** | 2.601 | 2.680 | 2.601 |
|  6 deg | 0.691 | **0.762** | 0.739 | 0.789 | 0.739 |
|  9 deg | 0.467 | **0.584** | 0.298 | 0.298 | 0.298 |
| 14 deg | **−0.206** | **+0.207** | 0.199 | 0.199 | 0.199 |

`tc_gain` at 14 deg: 0.256 → 0.718. The car no longer decelerates at full
throttle, and the new law is the best of the five at every angle bar 3 deg.

### (b) through the real keyboard, 20 m/s, gear 4, aid on, 1.5 s hold

| keys | ax OLD | ax NEW | delta OLD | delta NEW | r OLD | r NEW |
|---|---|---|---|---|---|---|
| up only  | +1.773 | +1.773 |   0.00 |   0.00 | −0.008 | −0.008 |
| up+right | **−0.032** | **+0.642** | −12.84 | −11.46 | −0.424 | −0.310 |
| up+left+LSHIFT | −0.161 | +0.565 | +13.07 | +11.38 | +0.401 | +0.321 |
| down+right | −5.968 | −6.010 | −18.63 | −18.92 | −0.332 | −0.307 |
| right only | −1.377 | −1.230 | −13.58 | −12.15 | −0.412 | −0.312 |

Throttle and brake are 1.000 / 0.500 (fine cap) / 1.000 in both columns —
the input layer never lost a simultaneous press, before or after.

### (a) + (c) steering while braking — real keyboard, DOWN+LEFT from 30 m/s

Scored on total heading change, not instantaneous yaw rate: a car dragging
22 deg of lock at `alpha_f = 32 deg` has a high BODY yaw rate because it is
pirouetting, and `r` alone calls that good steering.

| | OLD symmetric | NEW directional |
|---|---|---|
| max lock handed over | 22.32 deg | **14.08 deg** |
| heading change over 3 s | 37.98 deg | **41.10 deg** |
| velocity-vector heading change | 41.41 deg | **43.78 deg** |
| lateral displacement | 13.67 m | **13.77 m** |
| max front slip angle | 31.61 deg | **18.85 deg** |
| body yaw rate at 3 s | 0.362 rad/s | 0.151 rad/s |
| V at 3 s | 15.00 m/s | 14.82 m/s |

**8.2 deg less lock buys 3.1 deg more turn.** Coasting (LEFT only, no brake)
is the same story: 19.82 → 10.21 deg of lock, heading change 45.87 → 46.61 deg,
lateral 20.75 → 21.24 m. With the aid OFF (`--no-steer-limit`) the driver
still gets all 32.625 deg and turns LESS (`dpsi` 37.08 deg) — the aid is now
earning its place instead of getting in the way.

### Launch protection, unchanged (this is what the TC exists for)

WOT from rest, `power_scale 2.0`:

| | kappa_max | 0–100 km/h |
|---|---|---|
| TC off | 1.500 | 8.63 s |
| TC, OLD law | 0.312 | 8.32 s |
| TC, NEW law | 0.312 | 8.32 s |

### The limiter table (`k_us_deg = 7.82`), for reference

| V | floor | symmetric at beta 20 | into the slide | out of it |
|---|---|---|---|---|
|  5 | 32.63 | 32.63 | 32.63 | 32.63 |
| 15 | 13.93 | 32.63 | **13.93** | 32.63 |
| 30 |  9.30 | 32.63 | **9.30** | 32.63 |
| 50 |  8.31 | 32.31 | **8.31** | 32.31 |

## Deliberately left alone

* **The friction ellipse and the MF `Gxa`/`Gyk` combined-slip weighting.** A
  real car does lose longitudinal grip while cornering (`ax` still falls from
  +2.6 to +0.2 m/s² between 3 and 14 deg of steer, TC or no TC) and does
  understeer under braking (peak `ay` 8.55 → 6.61 m/s² at full pedal). That is
  the physics the owner should feel; `ECAP = 1.05` never fired in any of these
  cases.
* **The locked-wheel sled with ABS off.** `r` 0.0004 rad/s at 2 deg, `Fy_f`
  190 N against 5165 N with ABS. Correct physics, and his ABS is on.
* **Front brake bias** (`KBF` 1.6187e-4 vs `KBR` 5.4154e-5). Front-limited
  under braking is what this car is.
* **`K_US_DEG_MEASURED = 7.82` and the limiter floor.** Verified as plenty:
  9.30 deg at 30 m/s against the 9.57 deg the car needs at R = 50, and 8–14 deg
  is exactly where yaw response peaks under braking.
* **The 56.25 deg/s hand rate and the 45–90 deg/s return.** 0.165 s to the
  limit and 0.103 s back at 30 m/s: brisk, not the complaint. At 5 m/s it is
  0.58 s each way, which *is* slow for parking manoeuvres — flagged, not
  changed, because it is a spec constant (`W_HAND_DRIVE = 900 deg/s`) and no
  measurement says the owner meant that.
* **An EDL / brake-vectoring channel on the spinning inside front.** It would
  genuinely recover the tractive effort the open diff throws away and it is
  real hardware on cars of this era, but it is new physics on the brake path,
  and CONTRACT section 4 says the TC stays engine-only.

## Could not verify

* No real DualSense on this machine (`pygame.joystick.get_count() == 0`), so
  every pad change is exercised against `_StubPS` in `input.self_check` only.
  The two new pad assertions cover the pre-report gate, both rest conventions,
  and the directional limit through the stick. `--pad-calib` still reports
  "no gamepad found".
* Everything here is at `--engine sport` (`power_scale 2.0`) on dry tarmac with
  `mu = 1` everywhere, which is the owner's car but not his wet patches.

## One more, unrelated to the three symptoms

`ceb5e04` fixes the flank panel's side latch, which the concurrent wings audit
found. `armed` asked `st.dev_side != 0`, but the latch only moves while
`dep_raw <= DEV_DEP_LOCKOUT`, so the first corner pinned `dep_raw` at 1.0 and
the gate could never reopen: the panel never changed flanks again and never
stowed. `armed` now asks `want != 0 and want == st.dev_side`. Verified on the
latch state machine alone through a left-straight-right chicane (old: sides
{0, +1}, no swap, 57.1% of cornering on the wrong flank, `dep` still 1.000
with the wheel centred; new: sides {-1, 0, +1}, swap at 2.799 s, 0.0%,
`dep` 0.000) and on the real `Vehicle` at 25 m/s with `--wing plate`
(deploys +1 / F_dev +193 N at 0.50 s, stows by 2.00 s, latches -1 and deploys
F_dev -157 N at 3.00 s). `steady_state_corner(100)` with the plate is
unchanged at V = 30.415546 m/s, gain +3.8198%.

## Why nothing was added to validate.py

The task allowed cheap deterministic groups there. Deliberately not taken: the
suite's counts are pinned at 82/82 and 100/100, and adding groups moves them.
Every new assertion lives in `drive/input.py`'s own `self_check()` instead --
the directional pair on both sides, the V17 catch magnitude, the eased
unwind, the pad's pre-report pedal gate, both trigger-rest conventions, and
the pad's directional stick limit. `--modules` runs `python3 -m drive.input`
as group M and checks its exit code, so they are enforced by the suite without
changing a single count.
