# Item 2 — generalise `LapDriver` so the other cars complete a lap

## The Corsa-specific constants, as found

| constant | was | now |
|---|---|---|
| `AY_MAX_DRY = 8.4608` | the Corsa's `qss.max_ay(V)` at `k = 0`, used as *every* car's planned grip | `driver_scaling(car)['ay']` = `AY_MAX_DRY x (car_ay_peak(car) / AY_PEAK_REF)` |
| `KP_N = 0.06`, `KD_PSI = 0.55` | fixed path-following gains | scaled by wheelbase, `x (car.L / 2.491)` |
| `lookahead = 12.0 m` | fixed | scaled **inversely** with grip, `12.0 / ay_ratio` |
| `car.Vmax`, `atan(car.L*kappa)` | already per-car | unchanged |
| `DELTA_LOCK_DEG` | the Corsa's steering lock, used as the clamp | **left alone** — `CarSpec` has `steer_ratio` but no hand-wheel lock, so there is nothing honest to scale it by |
| `SpeedPI` gains | Corsa-tuned | **left alone** — see "still open" |

Plus a plain bug: **`LapDriver` never passed the car to `speed_profile`**
(`speed_profile(tr, margin, global_wet=...)`, so `car=None` → `CorsaC()`).
A 540i lap therefore planned the Corsa's grip *and* capped its target speed at
the Corsa's 47.2 m/s `Vmax`. Fixed.

`car_ay_peak(car, mu_scale)` makes the same statement `qss.residuals` makes,
minus the device and the yaw split — both axles at capacity, total lateral
transfer `m*a_y*h_cg/t` split 0.74/0.26, bisected on `a_y` — using
**`qss.fy_max` with `qss.TYRE`**, which CONTRACT sections 4 and 7 both require
as the tyre reference. It is not a replacement for `qss.max_ay` (which is the
reference truth and is bound to the Corsa at module scope); it exists only to
form a **ratio** between two cars, and only the ratio is used.

## The Corsa is bit-for-bit

Every scaling is a ratio against the Corsa's own value, so for the Corsa it is
exactly `1.0` and `x * 1.0 == x`:

```
corsa  ay 8.4608  ay_ratio 1.0000  kp 0.06000  kd 0.5500  look 12.00 m
mx5    ay 8.8813  ay_ratio 1.0497  kp 0.05456  kd 0.5001  look 11.43 m
540i   ay 8.6919  ay_ratio 1.0273  kp 0.06817  kd 0.6248  look 11.68 m

driver_scaling(CorsaC()):  ay == AY_MAX_DRY  kp == KP_N  kd == KD_PSI
                           lookahead == 12.0        all True, with ==
```

Scripted arena lap, Corsa: **60.8355 s, max |n| 3.2148 m** — the acceptance
number, unchanged.

## THE FINDING: this was never a driver-tuning problem

Retuning the driver **did not fix the other two cars**, and measuring why
settled it. The failure is **wheelspin from putting a scaled-up engine through
the FRONT axle** — the two library cars are marked `rwd` but
`powertrain.step` is FWD-only, so all their torque goes to the wheels that are
supposed to be steering. At `kappa` 1.5 a front tyre has almost no lateral
force left, so the car simply stops responding to steer.

Arena, 120 s, same driver, engine scale forced to 1.0 to isolate it:

| car | engine | max front &#124;kappa&#124; | max &#124;n&#124; |
|---|---|---|---|
| corsa | x1.00 (stock) | 1.500 | **3.21 m** |
| mx5 | x1.53 as shipped | 1.500 | 24.87 m |
| mx5 | **x1.00 forced** | 0.997 | **3.58 m** |
| 540i | x4.00 as shipped | 1.500 | 44.31 m |
| 540i | **x1.00 forced** | 1.125 | **5.13 m** |

Both cars are **inside the 6.0 m half-width** the moment the front axle is not
being asked to put down four times the Corsa's torque. The Corsa also touches
the `KX_LIM` clamp at 1.500, but with only 110 N.m the spin is brief and costs
it nothing.

**So item 2 is gated on item 1 (real rear-wheel drive), not the other way
round.** The driver generalisation here is still correct and still needed — the
`speed_profile` car bug was real, and the scaling is what stops a 1780 kg car
planning a 1010 kg car's grip — but the lap numbers for the two new cars
cannot be finished until their torque reaches the correct axle. Re-verify then.

## RESOLVED — the final driver, after RWD landed

Real rear-wheel drive (`.handoff/11-rwd.md`) removed the front-axle wheelspin,
and then made the lap **worse**: the two cars stopped understeering off the
track and started *spinning* off it instead (MX-5 390 m, 540i 1371 m). Three
things were tried, in order, and only the third earned its place:

1. **Throttle capped by the driven axle's measured `util_r`.** A step too late
   every time — `util_r` only rises once the rear is already sliding, by which
   point a 210 kW car has gone. MX-5 fixed, 540i still 1371 m off.
2. **The same cap, feedforward on the corner the driver can SEE**
   (`u = V²·|kappa| / ay_plan`, so `room = sqrt(1 - u²)`). This is what
   shipped. MX-5 on the island; 540i still 963 m off.
3. **A counter-steer term, `delta += k_beta·beta`.** Swept k_beta over
   0 / 0.5 / 1.0 / 1.5 / 2.5 / 4.0 on the 540i: it never completed a lap, and
   at the margin that actually works it made `max |n|` **worse** (2.74 m at
   k = 0 against 3.25 at k = 1.0 and 4.46 at k = 2.0). **Deleted.** The 540i
   was not losing the lap to a slide it could have caught — it was entering
   the corner too fast.

What was actually wrong was the entry speed, in two ways:

* **`car_ay_peak` is 4.1 % HIGH on a heavy car** (8.8534 estimated against
  8.5046 measured on the 540i; 0.1 % and 0.3 % on the Corsa and MX-5). It
  omits the scrub-drag and yaw-balance terms `qss.max_ay` carries. So the
  driver planned *more* grip than the car had. The planned grip now comes
  from `car_ay_measured` — an **open-loop ramp steer of that car**, cached,
  which is what CONTRACT §4 says quantitative limits must come from.
  `car_ay_peak` survives only for `power_grip_ratio`, where 4 % is noise.
* **A car far outside the driver's calibration needs a more careful driver.**
  `margin` now fades as `margin * (1 - MARGIN_FADE*(pg - 1))` with
  `MARGIN_FADE = 0.10` and `pg = power_grip_ratio`: exactly `margin` at
  pg = 1 (the Corsa), 0.836 for the MX-5 at pg 1.71, 0.796 for the 540i at
  pg 2.16 — and 0.80 is the value the sweep independently found the 540i
  needs. This is what a human does in an unfamiliar overpowered car.

`POWER_GRIP_MODULATE = 1.25` gates both the throttle cap and the margin fade
on a **physical** property — power per unit grip, measured 1.00 / 1.71 / 2.16
— so the Corsa is inert by physics rather than by a flag.

### Final: `--script lap`, arena, 200 s, ribbon half-width 6.0 m

| car | lap | `max_n` | margin | was (Corsa-tuned driver, FWD) |
|---|---|---|---|---|
| corsa | **60.8355 s** | **3.2148 m** | 0.900 | 60.836 s / 3.21 m |
| mx5 | **60.2900 s** | **2.2577 m** | 0.836 | 65.527 s / 16.27 m |
| 540i | **60.4886 s** | **2.6745 m** | 0.796 | 73.526 s / 42.30 m |

All three inside the ribbon. The Corsa is bit-for-bit: `margin_scale == 1.0`,
`ay == AY_MAX_DRY`, `kp_n == KP_N`, `modulate == False`, all with `==`.

**Read these to compare a change to one car against itself, not to rank the
three cars.** The two new cars are driven deliberately more conservatively, so
their lap times are a property of the driver as much as the car.

`validate` 82/82 and `--modules` 100/100, 0 HARD / 0 soft.

## Still open / not verified

* Only the **arena** has been run for all three cars with the final driver.
  `--script lap` is arena-only by construction (`lap_script` calls
  `trk.make_arena`), and the other three tracks are exercised by the other
  scripts, which are open-loop and were never Corsa-tuned: `skidpad_limit`
  and `brake` run clean on all three cars (`ay_g_openloop_ref` 0.858363 on
  every car, `t_stop` 18.314 / 14.346 / 11.412 s).
* `SpeedPI`'s throttle/brake gains are still the Corsa's. They did not show up
  as the binding constraint (the excursions are lateral, not longitudinal), so
  I left them rather than tune something that is not broken.
* `DELTA_LOCK_DEG` is still the Corsa's lock on every car. `CarSpec` would need
  a hand-wheel lock field to do better; flagged, not invented.
* `car_ay_peak` ignores the device, aero downforce and the scrub-drag term that
  `qss.max_ay` carries. It is a ratio-former, and both sides of the ratio
  ignore them identically.
