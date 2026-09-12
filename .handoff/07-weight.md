# Task 7 — "Be able to change weight"

Done as a **point-mass model**, not a mass slider. `cars.PointMass(m, x, h)`
plus `cars.with_masses(car, pts)` returns a new `CarSpec`; the drive exposes
it as two Settings rows (`Ballast`, `Ballast at`) and two CLI flags
(`--ballast`, `--ballast-at`), and the garage charges the three fitted wings
through the same door.

## Why not a mass slider

Scaling `m` alone makes the car CORNER HARDER, because every tyre is better
loaded and nothing else changed. Measured, 200 kg in the boot of a Corsa C:

| quantity | stock | +200 kg boot |
|---|---|---|
| `m` | 1010 | 1210 kg |
| `wdist_f` | 0.6100 | **0.4926** |
| `h_cg` | 0.5500 | 0.5665 m (+16.5 mm) |
| `Izz` | 1200 | **1722.7** (+43.6 %) |
| `m_s` | 892 | 1092 kg |

A front weight fraction that goes 61 % → 49 % is a different car. `Izz`
+44 % is a different car. Neither follows from `m` alone.

## What moves, and what deliberately does not

`with_masses` recomputes, by the first moments and the parallel-axis theorem:

```
m'      = m + mb
dx      = sum(m_i x_i)/m'          CG moves FORWARD by dx
dh      = sum(m_i (h_i - h_cg))/m'  ... and UP by dh
wdist_f'= (b + dx)/L               (b is CG->REAR axle, wdist_f == b/L)
h_cg'   = h_cg + dh
Izz'    = Izz + m dx^2   + sum(m_i (x_i - dx)^2)
Ixx'    = Ixx + m dh^2   + sum(m_i (h_i - h_cg')^2)
Iyy'    = Iyy + m(dx^2+dh^2) + sum(m_i ((x_i-dx)^2 + (h_i-h_cg')^2))
m_s'    = m_s + mb
```

and leaves untouched: `L`, `t_f`, `t_r`, `Kphi_*`, `k_wheel_*`, `h_rc_*`,
`m_us_*`, `r_roll`, the tyre, the gearing, `CdA`, `P_max`, `T_max`, `Vmax`.
Those are properties of the CAR, not of what is in it — which is exactly why
a ballasted car **rolls more** (same springs, more sprung mass, longer roll
arm through `h_cg`) and **accelerates less** (same engine, more mass) with no
extra code. `Vmax` and `P_max` stay the stock car's published figures because
they are data, not predictions, and `vehicle.py` never reads `Vmax`.

Everything downstream follows because `vehicle.py` already read all of it off
the car: `wheel_positions`, `Fz_f_static / Fz_r_static`, the load-transfer
demand (`c.h_cg`, `self.t_bar`), the roll block through `car_derived` (which
consumes `Ixx`, `m_s`, `wdist_f`, `h_cg`, `t`, `r_roll`), `Izz` in the yaw
equation, and `powertrain.from_car`.

## The ballast is placed, not abstract

Four stations, quoted **from the axles** so they mean the same thing on a
2.265 m MX-5 and a 2.830 m 540i. Heights are `est` and they are the honest
ones — a hatchback's **boot floor is ~0.65 m** above the road, well above the
0.55 m CG, so a sandbag in the boot **raises** the CG. Only ballast bolted to
the floorpan (~0.30 m, which is what a race car does) lowers it.

| key | station `x` | height `h` | what it is |
|---|---|---|---|
| `nose` | `+(a + 0.30)` | 0.32 | on the front subframe, ahead of the axle |
| `seat` | `0.0` | `h_cg` | a passenger, at the CG — **the control case** |
| `floor` | `-b` | 0.30 | floorpan over the rear axle, low |
| `boot` | `-(b + 0.25)` | 0.65 | boot floor, behind the axle, high |

`seat` exists so the model can be falsified: at the CG and at CG height it may
move the mass and **nothing else**, and `cars.self_check` asserts exactly that
with `==` (`wdist_f`, `h_cg`, `Izz`, `Ixx` all unchanged; only `m` and `m_s`).

Range: Settings cycles `0, 25, 50, 75, 100, 150, 200` kg; `--ballast` takes
any value in `[0, 300]` and is clamped. 200 kg on a 1010 kg Corsa is +20 %,
which is a lot and is also entirely real (two passengers and luggage).

## Measured — the whole point of modelling the station

Open-loop ramp steer at 29.0875 m/s and `steady_state_corner(100)`, Corsa C,
`qss_parity`, 200 kg:

| ballast | `m` | %front | `h_cg` | `Izz` | peak `a_y` | R=100 V | `util_f` | roll |
|---|---|---|---|---|---|---|---|---|
| stock | 1010 | 61.0 | 0.550 | 1200 | **0.8550 g** | 28.943 | 0.993 | 4.55° |
| 200 nose | 1210 | 69.4 | 0.512 | 1470 | 0.8389 | 28.688 | 0.987 | 5.20° |
| 200 seat | 1210 | 61.0 | 0.550 | 1200 | 0.8365 | 28.651 | 0.989 | 5.46° |
| 200 floor | 1210 | 50.9 | 0.509 | 1585 | **0.8468** | 28.773 | 0.993 | 4.61° |
| 200 boot | 1210 | 49.3 | 0.567 | 1723 | 0.8353 | 28.595 | 0.992 | 5.31° |

Three things worth reading off that table:

1. **Every** ballast loses grip. That is tyre load sensitivity (`mu` falls
   16.1e-6 per N), and it is the reason mass is never free.
2. `floor` is the **least bad** (−0.96 %) and `seat` the pure penalty
   (−2.16 %): the low station lowers `h_cg` by 41 mm, which takes 7.5 % out
   of the lateral transfer and buys most of the load sensitivity back. Roll
   4.61° against `seat`'s 5.46° says the same thing.
3. `boot` is the worst (−2.30 %) — rearward AND high — but it is the one that
   moves the **balance**, 61 % → 49 % front. On a front-limited car that is
   the interesting knob for a wing study: `util_f/util_r` goes 0.993/0.938
   stock to 0.992/0.927, i.e. the front is still the limit but the rear has
   more margin to give, which is what the flank panel is trading against.

Straight line, WOT from rest, dry (`--script accel`):

| car | 0–100 km/h |
|---|---|
| stock | **14.802 s** |
| +100 kg boot | 16.100 s |
| +200 kg nose | 17.185 s |
| +200 kg floor | 17.383 s |
| +200 kg boot | 17.418 s |

Mass dominates; the station is worth 0.23 s out of 2.6 s, through the front
axle load and hence traction on this FWD car.

## The wings are charged (task 3's loop, closed)

`CarBuild.mass_points(lib)` turns each filled slot into a `PointMass` at that
slot's own `(x, h)`, with `drive/aero/wing.wing_mass(spec, standoff)` — the
bottom-up floor task 3 built (two skins, two tip plates, two pylons of the
mount standoff; no ribs, no fasteners, no body reinforcement). `_apply_design`
puts the tuple on `opts.mass_points` and `Settings.car_spec(extra)` folds it
in. **Both** flanks are charged: a mirrored build carries two panels.

Measured on the seeded library (`flank-e423` both sides + `rear-s1223` top):

```
4.91 + 4.91 + 4.36 = 14.18 kg  ->  m 1024.18   wdist_f 0.6122
h_cg 0.5577 (+7.7 mm)          Izz 1212.7 (+1.1 %)
```

7.7 mm of CG height is ~0.9 % more lateral load transfer, on a car whose
whole device is worth 2.35 %. So it is small and it is not nothing, and it is
now in the ledger rather than out of it. **The default has no wings**, the
tuple is empty, and `with_masses` returns the same object — which is why this
was safe to switch on at all.

## The bit-for-bit chain

`with_masses(car, ())` returns **the same object**, not an equal copy. That is
load-bearing: `vehicle.car_derived` scales the calibrated roll block by
`hat(car)/hat(CORSA_C)`, and a ratio is exactly `1.0` only when the two floats
are identical. Asserted in three places, all with `==`:

* `cars.self_check` — `with_masses(car, ()) is car`, and a 0 kg `PointMass` too
* `vehicle._check_reference()` — at import, **raising** (a bare `assert` would
  vanish under `python3 -O`)
* `vehicle.validate()` T21 and `drive.self_check` V26

`python3 -m drive.validate` **82/82**, `--modules` **100/100**, 0 HARD, 0 soft.

## Not verified / known gaps

* **Lateral offset is not modelled.** `PointMass` has no `y`. Ballast goes on
  the centreline and the flank panels are a symmetric pair, so the only
  y-offset that could exist cancels. A one-sided wing would be mis-modelled
  (its mass would act on the centreline) — but the garage's mirror lock makes
  that hard to reach, and the honest fix is a `y` field, not a fudge.
* **The unsprung masses never change.** Correct for ballast and for wings
  (nothing a driver adds bolts to a hub), wrong for a wheel/tyre change, which
  this feature does not offer.
* **No suspension re-rate.** 200 kg on unchanged springs droops the car in
  reality and this model has no ride height, so the wing's `h_w` and the top
  wing's ground effect do not respond to load. That is a real gap and it is
  the same gap the stock model already has.
* **The tyre does not get bigger.** `r_roll` and the tyre geometry stay the
  car's, which is right for ballast.
* Nothing above is a *measurement* of a real ballasted Corsa. It is the model
  answering consistently, which is all that was asked for.
