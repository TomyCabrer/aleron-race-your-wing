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
* ~~Both new cars are RWD and will drive their front wheels~~ — **implemented**,
  see `.handoff/11-rwd.md`. `'awd'` is still refused with a `ValueError`
  rather than guessed at, since a centre diff is physics this driveline does
  not have.

---

# WIRED (the follow-up task, same author-role, later session)

Everything in "What the wiring agent must plumb" above is done. What follows
is what was actually built, the calls that were made, and the numbers.

## The interface

```
python3 -m drive.drive [--car corsa|mx5|540i] [--ballast 0..300]
                       [--ballast-at nose|seat|floor|boot]
```

ESC → SETTINGS grew three rows — `Car`, `Ballast`, `Ballast at` — persisted in
`runs/settings.json` (`Settings.KEYS` now has 12 keys). All three return
**True** from `Sim.apply_setting`, i.e. they rebuild the session exactly as
`track` and `wet` do, via `Sim.restart()` / `stop_reason == 'restart'`. They
have to: a different `CarSpec` is a different tyre model, wheel station set,
static load set, derived roll block **and** gearbox, every one of which is
built in `Vehicle.__init__`. Live-patching `veh.car` would leave all of them
stale, which is the exact failure mode the batch was about.

Scripts and `--headless` keep the fixed defaults `corsa` / `0 kg`
(CONTRACT §8). **`_opts_car(opts)` returns `None` there**, and `_build` then
constructs `CorsaC()` exactly as it always did — identical, not merely equal,
which matters because the telemetry sidecar serialises that object. An
*explicit* `--car` / `--ballast` DOES reach a script, the way `--wing` does:
the car is the subject of a measurement, not a driver aid, and measuring
another one with the repo's own rigs is the point of having them. Nothing in
`validate.py` passes either flag; verified by grep that no scripted, headless
or rig path calls `Settings.load()` (only `run_interactive_cli` does; every
other `Settings(...)` in `drive.py` is `path=""` or a tmp path in V26).

## The roll block — how the six constants were made to follow the car

`car_derived(car, cfg) -> CarDerived` scales `h_ra`, `I_roll`, `Cphi`, the
four `lltd_*` shares and `Y_DEV` as

```
value(car) = value(Corsa) * ( hat(car) / hat(Corsa) )
```

`hat` being the cheap bottom-up estimate each field's own comment already
states: roll centres interpolated at the CG station (`h_rc_f + (h_rc_r -
h_rc_f)(1-wdist_f)`, 0.16275 on the Corsa against the calibrated 0.160);
`Ixx - (m_us_f+m_us_r)(t/2)^2 + m_s h_r^2` (340.91 against 342.8);
`sqrt(Kphi*I_roll)` at a fixed `zeta_roll = 0.35`; the instantaneous
geometric+unsprung shares; half the mean track.

**The absolute level stays the Corsa's calibration and only the change
between cars is derived.** That is the honest statement, not a compromise:
the Corsa is the one car in this study with calibrated numbers and the other
two have an entirely `est` suspension block. It also makes the default
bit-for-bit **by construction**: when `car`'s fields equal `CORSA_C`'s,
`hat(car)` and `hat(Corsa)` are the same float, the ratio is exactly `1.0`,
and `x*1.0 == x`.

### `roll_dist_f` stays 0.74 on every car

The call, and the reasons in order:

1. CONTRACT §4 forbids deriving it, and the bottom-up route provably gives
   **0.51** for the one car that has data. Reproduced here: `lltd_geo_f +
   (1 - inst)*Kphi_f/(Kphi_f+Kphi_r)` = `0.101 + 0.678*0.6146` = **0.5176**.
   That is the contract's 0.51 and it breaks qss/crossover/ledger.
2. `Kphi_f / Kphi_r` — the only quantity that would drive a per-car answer —
   is `est` on **all three** cars. A car-specific LLTD would be a guess
   dressed as a measurement.
3. `qss.py`, `crossover.py` and `ledger.py` are pinned to 0.74, and
   `util_f/util_r` is compared against them.

What DOES follow the car is the split of that same 0.74/0.26 between the
**instantaneous** (geometric + unsprung, no spring in the path) and the
**elastic** (roll-lagged) route, because that follows from the roll-centre
heights, which are per-car. It is a transient statement only:
`lltd_geo_* + lltd_roll_*` is held at `roll_dist_f` **to the last bit** by
writing the roll share as `cfg.lltd_roll_f + (cfg.lltd_geo_f - geo_f)`, which
is exactly `cfg.lltd_roll_f` at ratio 1.0 whatever the reference pair is.

The MX-5's and the 540i's roll centres are nowhere near the Corsa's 0.300 m
twist beam, and it shows: instantaneous share 0.183 / 0.186 against 0.315.

| car | `h_ra` | `I_roll` | `Cphi` | `geo_f` | `roll_f` | Σ front | roll grad |
|---|---|---|---|---|---|---|---|
| corsa | 0.1600 | 342.8 | 2482 | 0.1010 | 0.6390 | **0.740** | 5.33 °/g |
| mx5 | 0.0633 | 419.4 | 2871 | 0.0760 | 0.6640 | **0.740** | 5.75 °/g |
| 540i | 0.0831 | 834.8 | 5413 | 0.0702 | 0.6698 | **0.740** | 6.01 °/g |

`zeta_roll` is 0.35 and `f_n` 1.646 / 1.556 / 1.474 Hz; `_det_roll` is
225 208 / 309 505 / 899 498, all well conditioned.

**A real concern to pass on:** 5.75 °/g for an MX-5 and 6.01 °/g for a 540i
are both too soft — a real MX-5 is nearer 4 °/g. The cause is not this
scaling, it is `cars.py`'s `Kphi_tot` estimates (700 and 1250 N·m/deg) being
low against those cars' `m_s * h_r`. One line each in `cars.py` if the owner
wants them stiffer; not changed here, because changing an `est` to make a
derived number look nicer is how provenance rots.

## `qss.TYRE` — no derived dict was needed, and that is now asserted

The handoff above worried that `util_f/util_r` would be referenced to the
wrong tyre. It is not, and the reason is a fact worth pinning:

* `mu_y(Fz)` and `mu_x(Fz)` contain **no geometry** — only `PDY1/PDY2/FNOMIN/
  LFZO/LMUY`. Checked: `width`, `aspect` and `rim_radius` appear nowhere in
  `drive/tyre.py` except their own assignment. `R0` is the only geometric
  term any equation reads (`Mz`, and the rolling moment).
* and there is **one** coefficient set in `tyre_data/` (the finding above).

So every library car has the *same* `mu(Fz)` curve as the Corsa, `qss.TYRE`
is the correct reference for all of them, and the grip difference is carried
by `cfg.mu_scale = car.mu_scale`, which `qss.fy_max` already multiplies in.
`util_f/util_r` still calls `qss.fy_max` with `qss.TYRE` exactly as CONTRACT
§4 requires — **unchanged**, no local `mu(Fz)`, no derived dict.

That is only legitimate while both facts hold, so both are now assertions,
not assumptions: `tyre.mu_curve_matches(tyre, ref)` compares the two mu
curves at five loads with `==`, `Vehicle.__init__` runs it and publishes
`self.tyre_ref_ok`, and T21 / V26 check it. The day somebody adds a genuinely
different `.tir`, it fires and says a `TYRE`-shaped dict must be derived.

## The tyre singleton

`tyre.tyre_for(path, R0, width)` is a cache keyed on `(abspath, R0, width)`
**pre-seeded with `CORSA_TYRE` under the Corsa's own key**, so `tyre_for()`
and `tyre_for(the Corsa's three numbers)` return the singleton itself —
identity, asserted (`v.tyre is CORSA_TYRE`), not equality. A plain `CorsaC`,
which has no `tyre_*` fields, falls through the same defaults and also gets
the singleton. `_tyre_eval` gained a trailing `tyre=CORSA_TYRE` argument so
the rigs in `vehicle.py` read exactly as they did.

MX-5 `R0 = 0.2880`, 540i `R0 = 0.3217`. Nothing else differs.

## `engine_scale` needed one fix, in `drive.py` not `powertrain.py`

`engine_scale` only reaches the physics through `cfg.power_scale`, and
`_build` never passed one — so `--car 540i --script accel` measured a 1780 kg
saloon with a **110 N·m Corsa engine** and reported 0–100 km/h in 24.3 s.
`_build` now applies `getattr(car, "engine_scale", 1.0)`.

`power_scale` therefore carries two factors and only one is a driver aid:

* the **Engine setting** (stock/tuned/sport) — the aid, still 1.0 on every
  scripted path, reconciliation 9 amended to say so explicitly;
* the **car's own engine** — the car, and `1.0` exactly for the Corsa.

The interactive path multiplies both (`Settings.power_scale`), so "Sport" on
a 540i is 2 × 4.0 = 8 × the Corsa's curve. The Engine row and the HUD label
now read the car's own PS through `engine_ps / engine_hud / engine_label`,
which reproduce the hard-coded `75 / 110 / 150 HP` on the Corsa exactly
(asserted in `drive.self_check`). "Sport" on a 540i reads 570 hp; the dicts
would have called it "150 HP".

The 6-speed 540i works: `update_shift` uses `len(p.gear)` and
`n_up_schedule()`, not a fixed table. **One latent bug found and NOT fixed**
(`powertrain.py` is owned elsewhere): `accel_run` at line ~1095 builds
`n_up = {g: ... for g in range(1, 6)}` while line ~1120 lets `gear` reach
`len(p.gear)`, so a 6-speed car `KeyError`s on `n_up[6]`. Only
`powertrain.self_check` and `validate.py` call it, both with the Corsa, so
nothing is broken today. Patch: `range(1, len(p.gear) + 1)`.

## What the two other cars actually measure

Open-loop ramp steer at 29.0875 m/s + `steady_state_corner(100)`, `qss_parity`:

| car | peak `a_y` | at `mu_scale` 1.0 | R=100 V | `util_f`/`util_r` | roll |
|---|---|---|---|---|---|
| corsa | **0.8550 g** | 0.8550 | 28.943 m/s | 0.993 / 0.938 | 4.55° |
| mx5 | **0.8980 g** | 0.8610 | 29.638 m/s | 0.993 / 0.944 | 5.15° |
| 540i | **0.8523 g** | 0.8001 | 28.918 m/s | 0.982 / 0.897 | 5.12° |

Read that middle column: with the grip *calibration* removed, the MX-5 is
worth only +0.7 % over the Corsa (lower CG, neutral balance, wider tyre
geometry — which does nothing here) and the 540i is **−6.4 %**, which is pure
tyre load sensitivity on 1780 kg. All three are still front-limited. The
headline +5 % / +8 % is `mu_scale` and nothing else, exactly as `cars.py`
labels it.

WOT from rest, dry, TC off. **These numbers were re-measured after real
rear-wheel drive landed (`.handoff/11-rwd.md`) and the stale FWD column is
kept beside them, because it is what the first version of this note quoted:**

| car | now: RWD + own curve | RWD only | as shipped FWD (stale) | real car |
|---|---|---|---|---|
| corsa (FWD) | **14.8024 s** | 14.8024 s | 14.80 s | ~15.5 s quoted, 14.4 s Opel's own |
| mx5 (RWD) | **9.3117 s** | 10.4167 s | 10.90 s | ~8.5 s |
| 540i (RWD) | **6.9047 s** | 6.8138 s | 7.845 s | ~6.2 s |

The Corsa is unchanged to four decimals through both changes, as it must be.
See `.handoff/11-rwd.md` for the driveline and this note's engine section for
the curve.

### Which of these are still nonsense, plainly

* **The longitudinal numbers are now real physics, not fiction.** This bullet
  used to say the opposite and it was right at the time: `powertrain.step`
  returned `T_drive` with `RL = RR = 0`, so both RWD cars drove their front
  wheels and spun them (`kappa` 0.906 and 1.500, the latter on `KX_LIM`).
  The tell was that switching TC **on** made both *faster*. That is fixed:
  `PowertrainParams.driven` follows `CarSpec.drive_layout`, and the rearward
  load transfer under acceleration now **loads** the driven axle on these two
  instead of unloading it (+765 N on the MX-5, +2058 N on the 540i, against
  −504 N on the Corsa). What remains off is the **engine**, below, not the
  driveline.
* ~~**The torque curve is the Corsa's shape, scaled.**~~ **FIXED (wave 3
  item 1).** `powertrain.engine_curve(car)` builds each engine's own curve
  from five anchors on the car plus its published `T_max` / `P_max`, and hits
  **both** published points exactly (T 168.000 / 440.000 N·m, P 109.000 /
  210.000 kW). The rev range is now the engine's own — MX-5 cut 7000, 540i
  6400 — and the shift schedule is placed as a fraction of the cut, so the
  MX-5 no longer short-shifts 950 rpm below its own power peak.
  `engine_scale` is **retired and 1.0 on every car**; it would double-count.
  What is still `est` is the curve SHAPE between the two published points
  (the shape family is the Corsa's, re-anchored) — a real per-engine curve
  needs a dyno sheet and none of these have one.
* ~~**Closed-loop lap times do not transfer.**~~ **FIXED** — see
  `.handoff/12-lapdriver.md`. `LapDriver` now scales its planned grip (from an
  open-loop ramp steer of that car, cached), its path gains (by wheelbase),
  its lookahead and its **margin** (faded by how far the car's power-to-grip
  ratio is outside the Corsa's) off the car itself. `--script lap`, arena,
  200 s, ribbon half-width 6.0 m:

  | car | lap, now | `max_n`, now | lap, stale | `max_n`, stale |
  |---|---|---|---|---|
  | corsa | 60.8355 s | 3.2148 m | 60.836 s | 3.21 m |
  | mx5 | **60.2900 s** | **2.2577 m** | 65.527 s | 16.27 m |
  | 540i | **60.4886 s** | **2.6745 m** | 73.526 s | 42.30 m |

  All three are now inside the ribbon and the Corsa is bit-for-bit. They are
  still not *competitive* lap times — the driver is deliberately conservative
  on the two unfamiliar cars (margin 0.836 and 0.796 against the Corsa's
  0.900) — so use them to compare a change to one car against itself, not to
  rank the three cars against each other. CONTRACT §4's rule still holds:
  quantitative limits come from open-loop ramp steer.
* The MX-5's and 540i's roll gradients are too soft — see the `Kphi_tot`
  note above.
* `CdA` for both is `Cd × A` with an estimated `A` and has never been
  validated against a top speed (the 540i's 250 km/h is a limiter).
* ~~**The brakes are the Corsa's on every car**~~ — **FIXED (wave 3 item 2).**
  `CarSpec` carries the documented hardware (MX-5 255 mm front / 251 mm rear
  discs, 540i 325 / 320 mm vented discs, against the Corsa's 236 mm disc and
  200 mm drum) and `powertrain.brake_coeffs` derives `kbf`/`kbr` with the
  BRAKE BLOCK's own formulas. Locked-wheel 100–0: corsa 49.8614 m unchanged,
  mx5 55.35 → 50.3193, 540i **64.00 → 48.0817**. Front still locks before
  rear on every car, dry and wet, by a wider margin than the Corsa's.
  Steering lock is per-car too (`vehicle.car_lock_rad`). The original text
  follows for the record:
  `corsa_c.brakes` is `MISSING`, `CarSpec` has no brake fields, and
  `powertrain.brake_torques` holds `KBF = 1.6187e-4`, `KBR = 5.4154e-5`
  N·m/Pa as module constants. In practice it does not bite for these three,
  because the pedal still locks all four wheels on the heaviest of them
  (`kappa_min` −0.989 on the 540i), and a locked-wheel stop measures `mu`
  and mass, not brake torque. 100 → 0 km/h, full pedal, ABS **off**, dry:
  corsa **49.86 m** (against the contract's 49.74 locked — unchanged),
  mx5 55.35 m, 540i 63.997 m, peak decel 1.087 / 1.162 / 0.831 g. The gap
  would open on a car whose brakes could NOT lock its wheels, and there is
  no way to express that here.
* `LOCK_RAD = radians(32.625)` (= 522° / 16.0) and `DEV_DEADBAND` are still
  the Corsa's steering lock. `steer_ratio` is in `CarSpec` (15.0 / 17.0) but
  the hand-wheel lock is not, so the wing's arming deadband is the Corsa's on
  every car. It is 5 % of lock and moves nothing measured; flagged, not fixed.
* `cfg.h_aero` (0.55, "= h_cg") is **dead code** — it appears nowhere but its
  own definition. Left alone. If it is ever wired up it must follow `h_cg`.
* `drive/plots.py` and `drive/validate.py` both hold their own `CorsaC()`, so
  post-run plots and the acceptance suite are always referenced to the study's
  car. Correct for `validate.py`; for `plots.py` it means a plot of a 540i
  trace draws the Corsa's g-g envelope. Neither file is in this task's
  ownership. The telemetry **sidecar** is right: it serialises `veh.car`,
  which is the full ballasted `CarSpec` (53 fields plus the four derived), so
  the trace is self-describing and a fixed `plots.py` has the data it needs.

## Files touched, and the contract

`cars.py` (extended: `PointMass`, `with_masses`, `ballast_point`, the station
table, 12 new self-checks — the three-car library itself is **unchanged**),
`drive/tyre.py`, `drive/vehicle.py`, `drive/drive.py`, `drive/render.py`,
`drive/garage.py`, `drive/CONTRACT.md` (§1 module map, §2 `tyre_for` +
`mu_curve_matches`, §4 the whole per-car scaling + added-mass block, §7
`set_car` + the two `HudData` fields, §8 the CLI, `Settings`, `car_spec` and
the restart rule, reconciliation 9), `.handoff/07-weight.md`.
`powertrain.py`, `qss.py`, `crossover.py`, `ledger.py`, `corsa_c.py`,
`validate.py`, `input.py`, `drive/aero/*` and `specs/*`: **not touched.**

`python3 -m drive.validate` **82/82**, 0 HARD, 0 soft [152.9 s].
`python3 -m drive.validate --modules` **100/100**, 0 HARD, 0 soft [235.2 s],
against the pristine baseline's 100/100 [230.0 s] measured before any edit.
`python3 cars.py` ALL PASS. `python3 -m drive.vehicle` 33/33 (was 32/32;
T21 is new). `python3 -m drive.drive --self-check` ALL PASS.
`python3 -m drive.tyre` / `--modules garage` / `render` all pass.

One intermediate `--modules` run reported 99/100 with `render.py`'s
`V22 frame budget` as the HARD failure. It is a **wall-clock** check
(`p99 <= 16.0 ms`) and it is load-flaky: 7 consecutive identical runs gave
FAIL/FAIL/ok/FAIL/ok/ok/FAIL with p99 11.19-22.45 ms, at load average 20.7
with 19 concurrent `python3` processes from the other agents. The same
attribution is independently recorded in `.handoff/05-when-automatic.md`.
It is not reachable from this work: the one drawing call added here
(`_draw_minimap`'s car/mass line) is guarded on `aux.car_name`, which the
render self-check never sets, so it draws nothing in V22.

## The patch `powertrain.py` needs (owned elsewhere; NOT applied)

`accel_run` builds its upshift table for five gears only:

```python
# drive/powertrain.py:1119
n_up = {g: p.n_up_a + (p.n_up_k12 if g <= 2 else p.n_up_k34) for g in range(1, 6)}
```

while the loop below lets `gear` reach `len(p.gear)`, and `n_up[gear]` is
evaluated BEFORE the `gear < len(p.gear)` guard. On the 6-speed 540i:

```
accel_run(p540, car540, v_targets=(100/3.6,))  ->  ok   (never leaves 3rd)
accel_run(p540, car540, v_targets=(260/3.6,))  ->  KeyError: 6
```

Latent today -- only `powertrain.self_check` and `validate.py` call it, both
with the 5-speed Corsa. Fix:

```python
n_up = {g: p.n_up_a + (p.n_up_k12 if g <= 2 else p.n_up_k34)
        for g in range(1, len(p.gear) + 1)}
```
