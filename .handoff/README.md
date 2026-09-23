# Feature batch of 2026-09-12 — read this first

**125 commits** on `main` from `048716a` ("baseline: carsim before feature batch"),
which is the rollback point. Waves 1-7. 41 files, +19.8k lines.

**The suite is green.** Measured on a quiet machine at the end:

```
python3 -m drive.validate            ->  82/82  pass  0 HARD  0 soft  [116.1 s]
python3 -m drive.validate --modules  -> 104/104 pass  0 HARD  0 soft  [257.8 s]  (21 modules: aero.blend/screen/section/mission added by task 16)
```

`--only` no longer lies: a filter that matches nothing prints what it matched
against and exits 2, instead of `0/0 pass` after 324 s of work.

identical to the baseline recorded in `00-baseline.md` before anything was
touched. Module self-checks: `tyre` pass, `powertrain` **90/90** (was 82),
`vehicle` **33/33** (was 32), `input` pass, `render` **29/29** (was 19; 27 through wave 7),
`garage` pass, `aero.wing` **27/27** (was 21), `cars` pass, `ml` **28/28**
(was 11; 17 through wave 7), `drive --self-check` pass.

The Corsa C is **bit-for-bit unchanged, by identity**: `engine_curve`,
`brake_coeffs` and `tyre_for` return the module constants *themselves*, and
`car_lock_rad(Corsa) == LOCK_RAD` raises at import if that ever stops being
true. 0-100 km/h is still 14.8024 s.

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

### Wave 8

| note | one line |
|---|---|
| `14-urop-parity.md` | the wing design follows **AeroBO's procedure**: mission -> section -> wing, and AeroBO's own design vector (two twist rows, a ride-height row, area-and-span sizing, chord derived). The flank standoff had **three disagreeing values** (0.45 / 0.60 / 0.45) and is now one, the 0.60 the car deploys to: flank `CLa` **-2.5 to -6.1 %**, top wings and all geometry **bit-identical**, study **untouched** (82/82) |
| `15-inner-flank.md` | the owner's fourth cell of the **(flank × section orientation)** matrix, which wave 7 never swept: panel on the **INNER** flank with the section turned over. The outer flank turns out never to have been *chosen* — `crossover`, `qss` and `ledger` contain no `y` at all, so the whole upstream study is blind to the one term that distinguishes the flanks. Cell 4 wins by **+0.02 pp (a designed panel) to +0.48 pp (the plate)**, every bit of it the drag yaw moment changing sign, **exactly zero** change in roll, and it is paid for out of the rear axle's margin: `util_r` rises in every pair, the plate departs at **25 m/s and 6° of lock**, and loses 0.083 s a lap inside `WET_T3`. `VehicleConfig.dev_flank` / `--dev-flank` and `wing.build_lattice(wall_side=)` added; **default unchanged** |
| `17-swarm.md` | **Deploy swarm**: a GENETIC ALGORITHM over `Policy` genomes (`drive/ml/swarm.py`), N cars on the map and car the user drives, elites kept, tournament + BLX + gaussian mutation with a **self-adaptive per-genome step** (one shared sigma collapsed 24 cars onto the centreline), seeded from **a lap the user drove** (`K` arms it; the sim writes 100 Hz JSON with no import of `drive.ml`, `drive/ml/clone.py` fits the residual to it: rmse steer 0.059 of lock on the real lap) or from any checkpoint (a 4-output one is widened). New **free-wings head**: 5 outputs, one per wing, 373 parameters, `theta = 0` still laps; `--legacy-wings` breeds the old one. The window replays a scored generation as ghosts while the pool breeds the next; `K` names and saves the best re-measured at 1 ms. Deliverable `checkpoints/swarm_arena.json`: **62.205 s on the arena, wing off, against the anchor's 65.940 s (-5.7 %)**; 64 cars x 8 generations. Measured **14.4 s a generation** for 32 x 70 s on 12 cores -- the README's "~3 s" was wrong and is corrected. `ml` 17 -> **28/28**, `render` 27 -> **29/29**. Not done: no swarm bred WITH a wing yet, and the user's own 59.70 s lap is still faster than anything bred |
| `16-by-sections.md` | **the wing is designed BY SECTIONS, through the UROP app's own navigator, and the order is enforced**: a gated `MISSION` page, then a DESIGN navigator in the LEFT column carrying `AIRFOIL / ENDPLATE / WING / RESULTS`, four stages each (library screening -> ranking -> section -> shape optimisation). New `drive/aero/mission.py` closes what task 14 section 5 listed as missing — the wing's mission is now a **LAP** of carsim's own circuits (read off `track.py`'s own arc/straight list, 27 ms), not a designer's drag cap. At zero downforce it reproduces `qss` **bit for bit**, and lands on the repo's published numbers (R=130 -> 31.880 m/s power-limited; R=100 -> 29.0875 m/s). New `drive/aero/section.py` DESIGNS the aerofoil in 2-D against it (10 rows: CST(4)x2 + t/c + incidence, box = the library's own hull). **Finding: the published flank panel changes sign** — worth -0.0796 s at `qss`'s scrub angle, **+0.0275 s** at the 10.34 deg `drive.validate` measures. **The END PLATE becomes a design problem**: `vlm.Lattice` always took `plate_a`/`plate_L0` and `build_lattice` hardcoded them flat, so `WingSpec.plate_airfoil` lifts a default rather than adding physics -- a cambered plate is worth **+10.5 % of CL0** at `plate_h` 0.14 m, and it is a CAMBER problem (the plate's slope moves CL0 by 0.004). The library screen scores every section on the **mission**, not on the AIRFOIL page's weighted composite -- AeroBO's *chosen on the map it is judged on*. Fixes **two** pre-existing bugs: the wing optimiser's start vector sat OUTSIDE the bounds it searched (and the page blamed a cap that was not involved), and the wing search flew **flat end plates** however they had been designed. Single element, as asked; **no slotted section yet**. **Sections 8-11**: the WEIGHTS were being asked in the wrong step -- AeroBO's are the SCREEN's seven CRITERION weights, not the search's CST vector, so new `drive/aero/screen.py` ports `airfoil_select.score_candidates` (seven criteria, its two presets, its dead criteria for a panel at zero lift, `GOAL_PENALTY` 3.0, `RHO_ASF` 0.05) and the screening step now asks them, with gates and floors. The three composite objectives maximise the SAME frozen 0-100 band the shortlist was chosen on, and are **BLOCKED** until a screen has measured it -- a composite candidate costs 3.6 ms against a lap's 31.0 ms because it never flies the lattice. **AeroBO's gates admit ONE of carsim's 34 sections** (measured: |cm| median 0.0684 here, ceiling 0.08 there, and it deletes S1223/S1210/CH10/E423), so they are off by default at this library's own quartiles -- the one number deliberately not copied. On the wing: the **design box is a BAND table** now (it was showing values while the bands sat unreachable in `wing.BOUNDS`), the **reference area** became a design variable (`design_table(area=True)` had carried the row since the port and no form ever set it), and the plate's section became a wing-type choice. Fixes a third pre-existing bug: `design_bounds`'s `area` switch **collided with the `area` design variable's own name**, so the AREA row could never be overridden. **Sections 12**: the gate was a MARK, not a gate -- `Nav`'s own docstring said "a blocked step can still be looked at", where AeroBO's `select()` **returns without moving** and notifies the stored reason. Now enforced: the section groups' four steps are a sequence, the wing's and the results' four are views of one problem and open together, and a group is finished by **fitting its section to the wing** -- not by optimising, which AeroBO is explicit is optional. BLOCKED (*not yet*) and LOCKED (*not here* -- plates at zero height) are different marks and different sentences. The ENDPLATE group carries the way past it (`or fly FLAT plates and move on`), so no gate makes a decision compulsory. **The mouse works on every page now** -- `Nav`/`ParamList`/`ListBox` record what they drew and hit-test it; a click on a value's `< >` halves IS a LEFT or a RIGHT, an action row selects then fires, the wheel moves the selection, hover lights the row, and the navigator's gate refuses a click exactly as it refuses an arrow key. A full chain walk, mouse only, is in the note. **Section 13**: the optimiser defaults are AeroBO's MEASURED ones now, read off `data/search_budget.json` -- the budget law `9.61 + 3.08 d` at 95 % (RMS 10.7 over 13 cases) with its three effort settings, and the Sobol split `0.5 x d` clamped [4, 16], which that study ranks 1st against 1x/2x/4x over 15 cases. carsim was spending **16 of 48** on a 10-row section (1.6 x d, between the two worst arms); now 5 of 40, and measured on carsim's own problem it is **better on 17 % fewer evaluations** and tighter across seeds. **The default weights ranked a car wing BACKWARDS**: |cm| is lower-better so it rewards reflex, and alone it scores rho -0.97 against the lap -- AeroBO's own `wing-trimmed` job already moves that weight for a surface whose moment something else carries, and a car wing's mount carries it. Removing it halves the lap cost of the section the screen picks. The PLATE preset is keyed on the role, because |camber| correlates **+0.96** with the lap on a flank plate and **-0.96** on a top one, and the shipped AeroBO set picked the worst plate in the library on the flank. The screening lift is AeroBO's `REFERENCE_CL = 1.0` -- reading it off the wing censored **30 of 39** sections at CL 1.718. Fixes three more defects: a COMPOSITE winner was bolting the wing on at an unpriced angle (-5.933 deg, the band edge); **`apply_design` was not the inverse of `design_x0`** it is documented as (the `area = 0.0` sentinel decoded the chord to zero and the lattice refused every candidate); and a `cd0 = 6.6e7` candidate overflowed the straight integrator. Mount offers pylon/endplate only, and the duplicate plate-section row is gone from WING TYPE. **Section 14**: the wing and the plate MEET now. `vlm.Lattice` bolted the plates on at a right angle and changed the surface -- its direction, its zero-lift angle, its twist -- in **one step** at the junction panel; new `drive/aero/blend.py` ports AeroBO's transition geometry (`geometry.winglet_turn_angle` / `winglet_path`, `vlm.transition_ramp`) and all of `junction.py`, reproducing **12 of its own tip positions to the bit** (3 turn laws x 4 blends) along with its projection, tip height and minimum blend radius. `plate_h` becomes the plate's DEVELOPED ARC, so `plate_blend = 0` is the published right-angle corner bit-for-bit across the whole library; the plate's section and toe ramp on `psi/phi` (E423's -2.30 deg to the plate's -1.00, monotone). **The wing pays for the reach**: a blended plate leans outboard and the span row is what the car is allowed to be wide, and that accounting is load-bearing -- unpaid, the lap improves monotonically to -0.032 s at full blend, which is AeroBO's own fixed free-lunch bug reproduced. Hoerner's junction charge is an ADD-ON, OFF by default, its fillet credit labelled a calibrated shape, reported beside the uncredited number, and switched on by raising the blend (32.3 ct sharp -> 5.5 ct at blend 0.6 on a NACA0012 plate; a plate with no section is bare sheet and charges nothing). **Measured over 2 roles x 3 circuits x 3 shapes x 6 blends: on this car it never pays** -- the curve turns over near blend 0.10 as the credit saturates (+0.0013 s at best) and reaches +0.022 s at full blend, because the span the reach costs is worth more than the corner it smooths. Asked on WING TYPE as a SETTING, as AeroBO carries its own; `Param.enabled` takes a callable so a row can lock on a value one panel away; the planform is drawn at the FLOWN span and the loading stage gained *wing and plate, seen from behind* |

Also in wave 8: `13-curved-flow-and-orientation.md` is **corrected in place** —
its "keep the suction side inboard" conclusion is right but is conditional on
the panel staying on the outer flank, and its "there is no configuration in
which the suction surface faces outboard and the useful force is still inboard"
is true of the two cells it swept and false of the two it did not.

### Steam plan, Phase A (`PLAN-steam-engagement.md`, tasks 19-22)

| note | task | one line |
|---|---|---|
| `19-records.md` | **19** | every lap recorded; the valid ones are the class's **top 5** in `runs/records/` (class = `track\|car\|engine\|surface`, file name `__`-joined for Windows). A record keeps a 50 Hz trace, the controls log and the lap's exact start state, and `records.resimulate` drives it again **bit for bit** (V31). The continuous controls are **quantised** (2^-24 rad, 2^-20 pedal) while a lap is recorded so the log is exact and 22-174 KB. An adversarial review found a 12 s "lap" (reverse over the line) filed as the PB, the engine change filing into the wrong class, and a 0.45 s stall at the line -- all fixed. Dragstrip **excluded** |
| `20-prerace.md` | **20** | a timed session on a lap map opens on the **TIME TRIAL** page: class, build, top 5, medal targets; **RACE** is one press, **PICK** any saved build (with its best in this class, filed by the car's content), **EDIT** through the garage and back. Each map opens with the build last **used** on it (in memory: the garage's file is untouched; a pick autosaves an unsaved car). Every menu page takes the **mouse** now (press arms, release fires; a double-click guard; long lists scroll). Review: a map switch could delete the garage's car, a double-click ran the next page's row -- fixed |
| `21-medals.md` | **21** | author / gold / silver / bronze for all **81** lap classes, DERIVED from 816 headless reference runs (LapDriver at four margins, the anchor, the bundled bots; aids off and on), author = the best full, spin-free lap; 1.02 / 1.06 / 1.12. The first build had five far-too-easy medals (a spun 43 s lap on a 20 s skidpad) and three classes with none: careful margins and the spin rule fixed both. Each valid lap says its medal; the best is kept; staleness hashes the tracks, cars, engines AND the reference drivers |

### Wave 7

| note | one line |
|---|---|
| `13-curved-flow-and-orientation.md` | the flank panel gets the **car's body as a rigid-wall image** (+5.20 % lift, e 0.9996 → 1.0802), and the orientation question is **answered**: keep the suction side inboard, a 5.4 pp swing |
| `09-ml.md` (wave-7 sections) | the 540i's skidpad regression characterised and the safety guarantee **narrowed**; the wing timing re-measured under the corrected AoA and **unchanged** |

Also in wave 7: the corrected device AoA is now the **DEFAULT**, re-baselined
deliberately — physics that is only correct behind a flag is a trap.

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

> **Superseded, twice.** These are wave-3 numbers, measured on the Corsa only
> and correct for it. The worst regression later improved to −1.20 % and then
> −1.08 %, and the "never puts a car off the road" half is **now false for one
> (car, track) pair** — the 540i on `skidpad`. See the table in *NUMBERS THAT
> MOVED MORE THAN ONCE* below, and the wave-7 sections of `09-ml.md`.

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
| the device's per-corner gain, again | the **body image** raises the panel's lift 5.2 %; with it and the corrected AoA a designed flank panel is **+2.79 %** at R = 100 |
| "it never puts a car off the road that the baseline was keeping on" | **narrowed** — false for the 540i on `skidpad`; the guarantee is conditional on a varying-radius circuit |
| "keep the suction side inboard" (wave 7) | **narrowed** — true with the panel on the OUTER flank, which is the only flank wave 7 swept. On the inner flank outboard *is* toward the turn centre, and that cell wins on the rigid-wall model (`15-inner-flank.md`) |
| the flank panel's lift from the body image | +5.20 % (wave 7, standoff 0.45) → **+3.50 %** (standoff 0.60, `14-urop-parity.md`), and **+2.31 %** with the body on the panel's pressure side |

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
