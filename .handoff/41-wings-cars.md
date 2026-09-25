# Task 41 — each car's own span limit, Unlimited wings, builds per car, the Express and the bus

Owner (2026-09-24), verbatim:

> *"Let the max span be different for each car, the actual physical maximum
> the span can be without touching the ground (for the side) and 1.2 of the
> width for the top wing. Also we can let the player have a span larger than
> this but this will be mostly just so that can play with impossible wings on
> a car just for fun and this will have like a separate spot on the challenges
> scores and this won't go towards the public leaderboard.
> Also I want to add 2 cars a Renault express from the 2000s and a bus so the
> players can play with larger wings.
> There has to be an easy way to save and change the wing cars. Also a custom
> default for each car the user wants"*

Asked and answered before building:

| question | owner's answer |
|---|---|
| which bus | a **12 m city bus** (Mercedes-Benz Citaro O530 type) |
| which Renault Express | the **older Renault 5-based Express** (1985-2000), not the 2000s Kangoo Express |
| how to go past the limit | a **Settings switch**: Wing limits Real / Unlimited (3x) |
| do Unlimited runs earn medals / stars | **yes, kept separate**: their own spot, never counted officially |

```
python3 -m drive.validate             ->  82/82  pass  0 HARD  0 soft   [125.3 s]
python3 -m drive.validate --modules   -> 122/123 before the rebuild (drive.medals: the stale table only); see the rebuild line
python3 cars.py / qss.py              ->  ALL PASS (five cars: gearing + power balance, tyre loads inside each tyre's range)
python3 -m drive.audio                ->  ALL PASS  51/51
python3 -m drive.ml                   ->  ALL PASS  36/36
python3 -m drive.medals --build       ->  315 classes (270 with an author time, 45 'no lap on this map'), 2586 runs, 8 workers, 5326 s;
                                          the 189 pre-41 classes bit-identical; medals self-check 8/8
stock-class author times, arena / linden / kestrel / ashdown / skidpad:
  Corsa 58.46 / 58.21 / 77.48 / 64.20 / 16.56   Express 61.01 / 59.81 / 80.13 / 65.87 / 17.16
  Citaro 74.44 / 71.73 / 96.18 / 79.78 / 18.72
```

## Decisions made without asking (the owner may overrule any)

* **The flank's ground margin is the car's own underbody clearance**, not
  zero: the panel's lower tip may come down to the lowest underbody station of
  the car's shell (Corsa 0.15 m, MX-5 0.14, 540i 0.15, Express 0.16, bus
  0.28). "Without touching the ground" with a zero margin would put a legal
  panel on the road the first time the car rolls onto it.
* **The rule is static** (the car standing still). Roll dips the deployed
  outer panel about 0.12 m at 1 g on the Corsa; the physics never tests
  ground contact.
* **Legality is computed, never stored**: a run is Unlimited when
  `bodies.over_limits(build, library, car)` finds any fitted wing past its
  car's limit -- whatever the setting says. The setting only gates the
  garage's editors. A build that cannot be judged (no library, the legacy
  one-panel car) counts as official.
* **Unlimited laps live in `runs/records/unlimited/`** with the same class
  keys, so medals, top 5s, ghosts and the HUD PB separate by construction and
  medal targets / reference ghosts still work. `records.publishable(lap)` is
  the one test the planned public leaderboard (T28) must use.
* **The three stock cars keep the garage's pre-41 slot bands** (station and
  height ranges, default slots) to the bit, so no saved build moves; only the
  Express and the bus take their slot bands from their own bodies. The span
  LIMITS are per car on all five.
* **The garage's default key is `F`** (`D` already opens the designer).
* **Per-map memory is keyed by map AND car**, and a build made for another
  car is never put on a car silently: a car change opens the car's default,
  else a build that car last drove on this map, else an empty car and a hint.
* **Every session drives a fitted copy of the build** (moved into that car's
  slot bands); the build itself never moves, so a trip to another car or into
  a challenge (all eight run in the Corsa) and back returns exactly the build
  you had.
* Express: **E7J 1.4, 55 kW, 1995**; Citaro: **OM 906 hLA 205 kW, ZF 6-speed,
  governed to 80 km/h** (BVG's 2005 cars; the press release's gearbox, which
  fits the sim's stepped box better than BVG's Voith 4-speed). Truck-tyre grip
  `mu_scale` 0.80, a labelled calibration.
* Colours: Express **fleet white**, bus **transit teal** (a yellow bus would
  clash with the Corsa). A bus in a race lines up **behind the painted grid**
  (about 42 m back) rather than repainting bigger boxes on every circuit.

## What it does

### 1. Each car's span limit (`drive/bodies.py`, new)

The body shells moved out of `render.py` into `drive/bodies.py`, a pure module
(no pygame) that the wing module, the garage, challenges, records and AeroBO's
bridge can all read. `Body(car)` is the shell on the car's own axles.

| car | ground clearance | width | flank limit, default slot | flank limit, highest slot | top limit |
|---|---|---|---|---|---|
| Opel Corsa C | 0.15 m | 1.646 m | 1.50 m (h 0.90) | 2.10 m (h 1.20) | 1.975 m |
| Mazda MX-5 | 0.14 m | 1.680 m | 1.52 m (h 0.90) | 2.12 m (h 1.20) | 2.016 m |
| BMW 540i | 0.15 m | 1.800 m | 1.50 m (h 0.90) | 2.10 m (h 1.20) | 2.160 m |
| Renault Express | 0.16 m | 1.566 m | 1.78 m (h 1.05) | 2.75 m (h 1.54) | 1.879 m |
| Mercedes Citaro | 0.28 m | 2.550 m | 2.64 m (h 1.60) | 5.20 m (h 2.88) | 3.060 m |

`wing.span_fit(role, h, car=None, unlimited=False)` delegates to
`bodies.span_ceiling` (the Corsa's old sill 0.28 / roof 1.34 fit is gone: on
the Corsa at h 0.90 the flank's limit went from 0.88 m to 1.50 m).
`WingSpec.clamp`'s span / area / top-ride ceilings rose (`wing.CLAMP_HI`) so
the largest Unlimited wing of any car survives a save and a reload.

### 2. Real / Unlimited, and Unlimited runs

* **Settings > Wing limits** (after Paint). *Real* holds every editor to the
  limit: the designer's span row, its design box and optimiser band, its save
  (no wing past the limit in ANY slot carrying it, mirror off included), `W`,
  the library page, and `↓` (a flank stops where its tip reaches the
  clearance). *Unlimited* opens them to 3x.
* The car page's **SPAN LIMITS** panel: each fitted wing as `span / max`,
  **PAST THE LIMIT** in red.
* An Unlimited run: its own records book, its own medals, its own challenge
  best / stars (`progress.json` section `challenges_unlimited`), UNLIMITED on
  the HUD PB row, the lap note, the results card, the pre-race page, the
  ghosts (`UNL PB`), the pause page and the build lists. Official totals never
  count it.

### 3. Builds per car, and the default

S / SHIFT+S / B / SHIFT+B / F on the car page, D / R1 / R on the library page,
four pad rows on the OPTIONS menu, Settings > Build and Settings > Default in
the drive; `CarBuild.car`, `records.BUILD_META` (no pre-41 PB moves),
`last_builds.json` keyed `track|car`, `Library.rename`. README "Your builds,
per car" has the player's view; CONTRACT section 8 "Task 41" the rules.

### 4. The Renault Express 1.4 and the Mercedes Citaro

| | Express 1.4 (E7J, 1995) | Citaro O530 12 m (2005) |
|---|---|---|
| mass (EU convention) | 915 kg | 11 459 kg |
| wheelbase / % front | 2.580 m / 60 | 5.845 m / 36 |
| engine | 55 kW @ 5600, 109 N·m @ 4000 | 205 kW, 1120 N·m @ 1300 |
| cornering (sim) | 0.802 g @ 15 m/s, 0.826 @ 25 | 0.616-0.682 g @ 10-25 m/s, no wheel lift, no spin |
| 0-50 / 0-100 | 5.57 s / 14.97 s | 10.7 s / governed |
| top speed | 148.6 km/h (150 published), drag-limited | 78.8 km/h, the 80 km/h governor, 6th at 1652 rpm |
| 60-0 km/h | 18.1 m ABS | 20.8 m ABS (0.68 g) |
| scripted arena lap | 62.35 s (Corsa 60.84) | 76.14 s |

The bus needed physics a car does not. On the one tyre in `tyre_data/` it
cornered at 0.225 g and spun (its wheel loads, 23-41 kN, are 2-4x the tyre's
10 kN ceiling); with the Corsa's wheel inertia its ABS chattered; with the
Corsa's 120 rpm soft-limiter band it stuck in 2nd; its brakes saturated at
400 bar. Each fix is a new OPTIONAL CarSpec field whose default is the old
behaviour (`cars.PHYSICS_DEFAULTS`; the stock three set none): per-axle tyre
**load scale** (CONTRACT section 2's one declared exception: `LFZO`, `FZMAX`
and `CFX`/`CFY` scaled together, exactly λ times the file tyre at Fz/λ; λ 6.644
front, 12.233 rear twin pair), roll split 0.45, compliance steer, wheel and
engine inertias, rev-scaled gearbox bands, a road-speed governor, an air-brake
equivalent, the proportioning valve, `vmax_by`, and `own_aids` (the steer aid
and the computer drivers use the car's own wheelbase, grip and lock).
Sources are on the lines in `cars.py` (DaimlerChrysler 2006 press release,
traditionsbus.de BVG fleet data, ZF HP 502 C data sheet, Wikipedia / French
Wikipedia / autotitre / car.info for the Express).

Presentation (`render.py`, `audio.py`, `fx.py`, `race_grid.py`): each car
drawn as itself, a chase camera that scales its framing about the tail by
height / 1.44 (exactly the old camera for the stock three, 84/84 poses), an
engine voice each (E7J four; OM 906 turbo-diesel six), measured smoke onsets
(10.9° / 16.6°), a per-car g-g envelope and rev marks, grid placement for a
car too big for a box, and a longer RACE-page Test for the bus (169 s).

## How it was built

Foundation and plumbing by the lead (d9fa447, 7a3a601), then three parts on
their own branches in parallel worktrees: A limits + Unlimited (`t41-limits`),
B builds + defaults (`t41-builds`), C cars physics then presentation
(`t41-cars`); merged on `task41-wings-cars` with a glue commit (8fc2025). One
review (4 finders, 1 judge): 12 confirmed findings, all fixed (commits
183898d..06b04fe) -- the serious one was a challenge driving another car's
build unfitted, which could file an over-limit wing as an official lap.

## Known gaps

* The DESIGN page's mission lap still scores wings on a Corsa lap
  (`aero.mission`), whichever car is fitted. That page is being replaced by
  the AeroBO engine in `~/Desktop/carsim-aerobo`; its bridge calls
  `wing.span_fit` and must read `bodies` per car when it is ported (its
  `FLANK_SPAN_M 1.30` cap and the Corsa `deck_z`).
* Laps recorded before task 41 are not reclassified (no span snapshot).
* The bus: an automated-manual gearbox with a 0.70 s torque gap (a real ZF
  shifts under power), no air-brake lag, the Corsa's `tau_roll`, twin rears
  as one tyre station (and drawn as one wheel), the Corsa's NA torque shape
  re-anchored to the published points (too much low-end for a turbo diesel),
  no separate turbo whistle / retarder sounds, no text on the destination
  display.
* The Express's final drive, brake sizes, CG, inertias and gear ratios are
  est; sources disagree on the E7J's output (0-100 15.0 s in the sim against
  12.0 s for an "80 ch" 1.4 RT and 14.5 s for the older 50 kW engine).
* No bred ML checkpoints for the new cars (their medals use LapDriver and
  the anchor).
* The HUD's UNLIMITED tag gives way for a few seconds to a medal just earned
  (the results card and the lap note still say UNLIMITED).
* On a pad a build name can only be accepted as offered (rename and typed
  names need the keyboard).
* The MX-5's and the 540i's g-g limit is still the Corsa's (pre-existing).
