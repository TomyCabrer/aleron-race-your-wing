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
