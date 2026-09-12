# Item 1 — real rear-wheel drive

The library shipped two cars marked `drive_layout="rwd"` while
`powertrain.step` returned `T_drive=(T_fl, T_fr, 0.0, 0.0)` — **front-wheel
drive and nothing else** — so both drove their front wheels and every
longitudinal number quoted for them was fiction. That is the one thing in the
last batch a user could trip over without knowing it.

## What changed

`PowertrainParams.driven` is `'front'` or `'rear'`, read off
`CarSpec.drive_layout` by `from_car`. Everything that pairs with the torque
follows the driven pair:

| | before | after |
|---|---|---|
| `T_drive` | `(T_fl, T_fr, 0, 0)` always | driven pair carries it, other pair `0.0` |
| reflected inertia | `I_w_front`, into `I_wf = 0.76` | `I_w_driven`, into `I_w_bare(p)` — the **0.73** rear wheels on a RWD car, so the whole curve shifts down 0.03 kg·m² |
| `omega_drv` / `omega_in` | `0.5*(omega[0]+omega[1])` | the driven pair's mean — this is what the clutch and the shift scheduler read |
| `I_w_eff` | front pair got the reflected value | driven pair does |
| `accel_run` traction cap | front axle load, `- a*h_cg/L` | driven axle load, `+ a*h_cg/L` when rear-driven |
| `Vehicle._tc` | `max(kx[FL], kx[FR])` | `max` over the **driven** pair, same open-diff `1 - 2*share` floor |
| `diff_split` | — | already axle-agnostic, checked |

`'awd'` is **refused with a `ValueError`**, not silently treated as one of the
two. This driveline has one clutch, one gearbox and one open diff; a centre
differential with a torque split is physics it does not have, and guessing one
would be a lie in the shape of a feature.

`PowertrainState.I_w_front_eff` keeps its historical name — `specs/powertrain.txt`
quotes it in its golden table and specs are not edited — and now means the
**driven** axle. `I_w_front` survives as an alias of `I_w_driven` for the same
reason.

## The load transfer needed no new code — verified, not assumed

`dFz_x_demand = (sum(Fxb_i) * h_cg) / L` in `vehicle.py` was already general,
so acceleration transferring load rearward **unloads a front-driven car and
loads a rear-driven one** all by itself. Driven-axle normal load, WOT from
10 m/s in gear 2:

| car | driven | at t = 0 | at peak `ax` | delta | peak `ax` |
|---|---|---|---|---|---|
| corsa | front | 6043.9 N | 5539.5 N | **−504.5 N** | 2.90 m/s² |
| mx5 | rear | 5368.0 N | 6133.5 N | **+765.4 N** | 4.13 m/s² |
| 540i | rear | 8556.3 N | 10613.8 N | **+2057.5 N** | 6.15 m/s² |

That is the RWD traction gain the brief asked for, and it emerges rather than
being imposed. No second load-transfer path was added.

## Re-measured, all three cars

0–100 km/h through the real integrator (`--script accel`). **Superseded by
wave 3 item 1 (per-car engine curves) — both columns kept:**

| car | + per-car curve | RWD only | stale FWD | real car |
|---|---|---|---|---|
| corsa (FWD) | **14.8024 s** | 14.8024 s | 14.80 s | 14.4 s (Opel), ~15.5 s quoted |
| mx5 (RWD) | **9.3117 s** | 10.4167 s | 10.90 s | ~8.5 s |
| 540i (RWD) | **6.9047 s** | 6.8138 s | 7.845 s | ~6.2 s |

Rigid rig (`accel_run`), with the per-car curve: corsa 14.686 s (unchanged),
mx5 9.488 s, 540i 6.353 s.

Open-loop ramp steer peak `a_y` (`peak_ay`, the reference truth per CONTRACT §4):
corsa **8.6084**, mx5 **9.0225**, 540i **8.5046** m/s². Unchanged by this work —
they are lateral measurements and the driveline does not enter them.

Braking, `--script brake`, `t_stop`: corsa 18.314 s, mx5 14.346 s, 540i 11.412 s.
Also unaffected: braking is on all four wheels on every car.

## What is still not right, and why

* ~~**The engine, not the driveline.**~~ **FIXED in wave 3 item 1** —
  `powertrain.engine_curve(car)` builds each engine's own curve from its own
  anchors, and `engine_scale` is retired. The MX-5 went 10.42 → 9.31 s
  (against a real ~8.5 s), so the remaining ~0.8 s is launch, driveline loss
  and the still-estimated shape between the two published points, not the
  rev range. The 540i is 6.90 s against a real ~6.2 s.
* The brakes are the Corsa's on every car (`corsa_c.brakes` is `MISSING` and
  `CarSpec` has no brake fields).
* `LOCK_RAD` / `DEV_DEADBAND` are still the Corsa's steering lock.
* Nobody has *driven* a non-Corsa interactively with a pad.

## The Corsa is bit-for-bit

`driven == 'front'`, `I_w_driven(p, 1) == 2.438` exactly, `powertrain` 90/90,
`vehicle` 33/33, `validate` **82/82** and `--modules` **100/100**, 0 HARD /
0 soft. 0–100 km/h 14.8024 s and the scripted arena lap 60.8355 s / `max_n`
3.2148 m are unchanged to every digit previously recorded.
