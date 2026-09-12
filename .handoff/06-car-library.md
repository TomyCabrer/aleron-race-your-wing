# Task 6 — "Add option for different car" (the library; wiring is separate)

`cars.py` at the repo root, next to `corsa_c.py`. `python3 cars.py` is its
self-check: **ALL PASS**.

## The cars

| key | car | m (kg) | L (m) | % front | kW | N·m | CdA | mu_scale | layout | gears |
|---|---|---|---|---|---|---|---|---|---|---|
| `corsa` | Opel Corsa C 1.2 16V (2003) | 1010 | 2.491 | 61 | 55 | 110 | 0.66 | 1.00 | fwd | 5 |
| `mx5` | Mazda MX-5 1.8 (NB2, 2001) | 1140 | 2.265 | 52 | 109 | 168 | 0.61 | 1.05 | rwd | 5 |
| `540i` | BMW 540i (E39, 1998) | 1780 | 2.830 | 51 | 210 | 440 | 0.66 | 1.08 | rwd | 6 |

Deliberately contrasting: the MX-5 is 120 kg heavier than the Corsa but twice
its power, on a 226 mm shorter wheelbase with near-neutral weight distribution
and a much lower CG — and a *worse* Cd (0.36, an open-able roadster). The 540i
is 640 kg heavier and nearly four times the power on a wheelbase 340 mm longer.
`corsa_c.py`'s `est` discipline is followed exactly: published values carry
their source on the same line, estimates say `est` with a band.

Sources: Wikipedia (Mazda MX-5 NB; BMW 5 Series E39), encyCARpedia
`01-mx-5-1-8-roadster`, carfolio `bmw-540i-96305`, auto-data.net. CG heights,
axle splits, inertias and the whole suspension block are estimates — as they
are for the Corsa, because no manufacturer publishes them.

Inertias use the same dynamic index the Corsa's own estimate uses,
`DI = Izz/(m·a·b) = 0.80`, so the three are consistent with each other rather
than three unrelated guesses.

## The default is provably untouched

`CORSA_C` is **not retyped**. `_from_corsa()` reads every shared field off
`corsa_c.CorsaC()` by name, so the only way it can drift is if a field is added
to one dataclass and not the other — which the self-check catches:

```
[ok] CORSA_C is corsa_c.CorsaC() on every shared field: 39 fields compared with ==, 0 differ
[ok] CarSpec is a superset of CorsaC: CorsaC fields absent from CarSpec: none
[ok] derived properties agree: a 0.9715  b 1.5195  t 1.42450  P_wheel 47.3 kW
[ok] the default name resolves to the Corsa
```

`get()` falls back to the Corsa on an unknown name rather than raising,
because it is reached from a persisted settings file and a stale name there
must not stop the sim from starting. `corsa_c.py` is **not modified**.

## THE IMPORTANT FINDING: there is one tyre's worth of data in this repo

`tyre_data/` has twelve `.tir` files. Only **five** load into `drive.tyre`:

```
TNO_car205_60R15    OK      Siemens_car205_60R15  OK      TASS_car205_60R15  OK
car145_70R13        OK      car205_60R19          OK
CS_car225_60R18     FITTYP=61, refused            MagicFormula61_Paramerters  FITTYP=61
PacejkaBook_Defaults FITTYP=61                    Sedan_Pac02Tire   ZeroDivisionError
audi_Pac02Tire      ZeroDivisionError             mf_185_80R14      ZeroDivisionError
pac2002_235_60R16   ZeroDivisionError
```

and **all five carry the IDENTICAL Magic Formula force/moment coefficient
set.** Diffed key by key against `TNO_car205_60R15.tir`, the only differences
are `UNLOADED_RADIUS / WIDTH / ASPECT_RATIO / RIM_RADIUS / VERTICAL_STIFFNESS /
VERTICAL_DAMPING` and file metadata. `car145_70R13` returns bit-identical
`Fx/Fy/Mz` to `TNO_car205_60R15` at the same slip.

So **a car cannot be given different tyre coefficients from the data present**,
and inventing Pacejka data is out of scope. Following CONTRACT section 2's own
rule ("Do NOT rescale FNOMIN or LFZO. Rescale geometry only"), every car reads
the validated `TNO_car205_60R15.tir` and overrides the geometry to its real
tyre size (MX-5 195/50R15 → R0 0.2880; 540i 235/45R17 → R0 0.3217). The grip
difference is carried by `mu_scale` — an existing first-class `VehicleConfig`
field, +5 % for the MX-5 and +8 % for the 540i, **labelled a calibration and
never presented as measured data**. Tyre-test literature puts the
summer-performance vs touring dry-grip gap at 5–12 %; these are the
conservative end. If the owner disagrees with those two numbers they are one
line each.

## Second finding, incidental: the tyre is not odd under combined slip

While writing the tyre sanity check I found that `evaluate(Fz,-k,-a) !=
-evaluate(Fz,k,a)`, which CONTRACT section 2 requires "to 1e-9". **This is
already known** — `drive/tyre.py`'s own docstring says so, and its
`self_check` step 4 decomposes it and raises it as a note, proving no shift was
*missed*: `RHX1 RHY1 RHY2` (the `SHyk`/`SHxa` combined-slip shifts) are kept
deliberately, and keeping them is incompatible with exact oddness. Identical at
the pristine baseline `048716a`, so nothing in this batch caused it.

What was **not** previously quantified, and which I measured:

| condition | asymmetry in the governing force |
|---|---|
| pure longitudinal (k 0.05, a 0) | 2.4e-5 relative — effectively odd |
| pure lateral (k 0, a 0.10) | 23.4 N on 3330 N = **0.70 %** |
| combined (k 0.05, a 0.10) | 248 N = **8.5 %** |
| braking in a corner (k −0.08, a 0.08) | 226 N = **8.6 %** |

and at the **car** level (pristine tree, mirrored left/right step-steer at
30 m/s, 3 s):

| input | `max abs(r_L + r_R)` | `max abs(ay_L + ay_R)` |
|---|---|---|
| coast | 7.3e-4 rad/s | 0.054 m/s² (~0.8 %) |
| throttle 0.5 | 7.3e-4 rad/s | 0.054 m/s² |
| brake 0.6 | 4.6e-4 rad/s | **1.489 m/s² transient** |

So the car is mirror-symmetric to ~0.8 % in steady cornering, but
braking-while-steering produces a transient left/right difference of up to
1.5 m/s². That is second-order and it is *not* the mechanism behind the owner's
"no steering while braking" (task 8 found that separately), but it is real and
it is the one place the asymmetry is big enough to feel. **Not fixed here:**
zeroing `RHY1/RHY2` would make the model exactly odd but would move every
calibrated number in the suite, so this is a report-upward finding in the style
of CONTRACT section 10, not a patch.

## What the wiring agent must plumb (the handover)

Nothing imports `cars.py` yet — it is inert and cannot move the suite. To make
`--car` and the Settings row work, the next agent needs:

1. **`drive/vehicle.py`** — `Vehicle.__init__` already takes `car=CorsaC()`, so
   a `CarSpec` drops straight in. What does **not** come from the car today and
   must, or the other cars will be wrong:
   * `CORSA_TYRE` is a module singleton in `drive/tyre.py` parsed once at
     import. A car with different tyre geometry needs its own `TyreModel(
     car.tyre_file, R0=car.tyre_R0, width=car.tyre_width)`. Keep the singleton
     as the object used when the car IS the Corsa, or the default stops being
     bit-for-bit.
   * `cfg.mu_scale` should default to `car.mu_scale` (it is already a
     `VehicleConfig` field, so this is a default, not new plumbing).
   * The roll block's `I_roll = 342.8`, `Kphi = 36669`, `Cphi = 2482`,
     `h_ra = 0.160`, `t_bar = 1.42450` and `roll_dist_f = 0.74` are **hard-coded
     module constants**, not read from the car. `roll_dist_f` in particular is
     a *calibration* the contract forbids "fixing". For a different car these
     must be derived from its own `Kphi_tot / h_rc_* / m_s / t`, and the Corsa's
     values must be reproduced exactly when the car is the Corsa. **This is the
     single biggest wiring risk in task 6.**
   * `util_f/util_r` must call `qss.fy_max` with `qss.TYRE` per contract — which
     is the *Corsa's* tyre. For another car the utilisation readout is
     referenced to the wrong tyre unless a `TYRE`-shaped dict is derived from
     that car's own `mu_ref/Fz_ref/s`. Extend the contract in the same commit.
2. **`drive/powertrain.py`** — `PowertrainParams.from_car(car, power_scale)`
   already reads `gear / finaldrive / eta_drive / P_max / T_max / r_roll`. The
   19-breakpoint torque curve is built **once at import** from the Corsa's
   curve; a 440 N·m V8 needs its curve scaled to `T_max`/`P_max`, and the
   engine/clutch inertias and `T_CLUTCH_CAP_STOCK` are Corsa numbers.
   The 6-speed 540i also needs the shift scheduler to handle 6 ratios.
3. **`drive/drive.py`** — `--car`, `Settings.KEYS` + a `Car` row on the
   SETTINGS page, persistence in `runs/settings.json`, and `Sim.restart()`
   semantics (changing car must rebuild the session, like `track` and `wet`).
   Fixed defaults for scripts/headless must stay `corsa`, per CONTRACT §8.
4. **`drive/render.py`** — the HUD's `_CAR` is a module-level `CorsaC()` used
   for the `%mg` weight fractions; it should follow the selected car.

## Not verified

* No car other than the Corsa has been *driven* — nothing imports this module
  yet. The MX-5 and 540i parameter sets pass the two cross-checks
  `corsa_c.self_check` applies, and nothing more.
* `CdA` for the two new cars is `Cd × A` with an estimated `A`; unlike the
  Corsa's 0.66 it has **not** been validated against the published top speed.
  The 540i's 250 km/h is a limiter, so it cannot be.
* Both new cars are RWD and will drive their front wheels until someone
  implements the rear split. `drive_layout` records the intent; this is the
  most visible honesty gap in the feature.
