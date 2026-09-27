# Task 43 — AeroBO's own engine inside the garage's design pages

Two asks, one task. First (2026-09-23): *"For designing the wing and airfoil
make it more clear. Follow the steps of AeroBo, make clear which airfoil is
chosen in every section, what evaluation is running. Ideally it would de
identical in every stage (except mission) from AeroBo in how every section
looks and how the optimiser is ran end everything."* That built AeroBO's
LOOK: the shell, the tree, the views (the first half of this note's history,
kept below where it still holds). Then the PIVOT (2026-09-24): *"I basically
want AeroBo (only for car) to exist inside the game ... The only thing that
could be different (doesn't have to) is the mission and the looks"* -- plus:
the endplates symmetric; a live graph of every evaluation; running and
finished unmistakable; AeroBO's budgets (164 for a section); the wing
objective chosen (maximise downforce, efficiency, ...). So the design page
no longer runs carsim's own search: **AeroBO v1.0.0 is vendored unmodified
at `aerobo/` and every stage calls it**, as AeroBO's own V3 window does.

```
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy CARSIM_HEADLESS=1   # from the worktree's root
python3 -m drive.aerobo_bridge   ->  26/26  ALL PASS  (new; 22 + 4 XFOIL rows, ~12 s warm, ~51 s cold)
python3 -m drive.design_jobs     ->  20/20  ALL PASS  (was 24: the engine's own jobs; 0.2 s)
python3 -m drive.aerobo_models   ->  32/32  ALL PASS  (new; every run a replayed fixture, ~5 s)
python3 -m drive.garage          ->  88/88  ALL PASS  (was 133: the carsim-engine rows replaced; the new pin)
python3 -m drive.wing_tutorial   ->  24/24  PASS      (prose only; the predicates unchanged)
python3 -m drive.design_shell    ->  63/63  ALL PASS  (was 52; N11 draws 19 views pre / running / post)
python3 -m drive.design_shots    ->   7/7   ALL PASS  (209 PNGs, 5 walks x 2 sizes, every run replayed, ~27 s)
python3 -m drive.cae.theme / cae.widgets / cae.plot / cae.chrome -> 24 / 42 / 12 / 18 (unchanged)
python3 -m drive.aero.wing / aero.library / aero.airfoil / aero.section / aero.screen / aero.optimize
                                 -> 37 / 10 / 8 / 32 / 25 / 27 (unchanged)
python3 -m drive.validate --modules  -> 130/130 pass, 0 HARD, 0 soft (500 s; MODULES + aerobo_bridge, aerobo_models)
```

All on the worktree, 2026-09-24 23:25-23:40, after every fix below.

## Rebased onto main `82eba99` (tasks 38-41), 2026-09-25

The work was stashed, the branch fast-forwarded to `82eba99` and the stash
popped; this note was `40-aerobo-design.md` and is renumbered **43** (main
has 40 and 41, another session is on 42). Conflicts, and what each side kept:

* `drive/garage.py` (23 hunks). The AeroBO side wherever main edited the
  carsim-engine designer it replaced (`Designer`, `SectionModel`,
  `DesignPage._save` / `draw`, the dark mission draw, their self-check
  rows); main's everywhere else: `bodies` imported, `CarBuild.car` /
  `clamp(lib, car)` / `reset(car)` / `for_car`, `from_json` robustness,
  `default_build_name` / `new_build`, `_could_not_save`, the per-car preview
  and paint, `Garage(car=, settings=)` / `unlimited`, `_fit_in` /
  `handed_back`, the build keys (S / SHIFT+S / B / F, rename, default), W /
  library / DOWN held to the limit, the first-wing tutorial offer, R1 on the
  library page, `_check_builds`. Ours kept inside them: `rederive_stale()`
  now runs in `_write_build` (every library save of a build) and on drive;
  the XFOIL-arrival re-analysis skips AeroBO wings and has main's save
  guard; `self.designer` is gone (no such object since the pivot).
* `drive/aero/wing.py`: `clamp` leaves an AeroBO wing's rows alone (ours)
  and raises the carsim wing's ceilings with `CLAMP_HI` (main).
* `README.md`: our design text, with main's circuit list carried into
  Stage 1; `drive/CONTRACT.md`: our module rows, main's `menu` and
  `challenges` rows; `.handoff/README.md`: main's rows 38-41, ours as 43.
  Every "task 40" of ours in the docs now reads 43.
* `views_mission.BRAKING` takes task 39's plain words ("Braking uses tyre
  grip only").

**Task 41 wired into the designer** (`aerobo_bridge` reads `drive/bodies.py`,
pure, for the host's `car` and `unlimited`; `aerobo_models.DesignSession`
takes them off the garage):

| car | flank limit at the default slot (Design box `b_m` / `S_m2`) | top: limit (= `b_m` max), deck, ride band at the default station |
|---|---|---|
| Corsa | 1.50 m at h 0.90 (b 1.142-1.50, S 0.12-0.75) | 1.9752 m, deck 1.434, ride 1.574-1.85 |
| MX-5 | 1.52 m at h 0.90 (b 1.156-1.52, S 0.12-0.770) | 2.016 m, deck 1.434 (the stock bands' Corsa deck), 1.574-1.85 |
| 540i | 1.50 m at h 0.90 (b 1.142-1.50, S 0.12-0.75) | 2.16 m, deck 1.434, 1.574-1.85 |
| Express | 1.78 m at h 1.05 (b 1.328-1.78, S 0.12-1.056) | 1.8792 m, deck 1.776, 1.916-2.186 |
| Citaro | 2.64 m at h 1.60 (b 1.618-2.64, S 0.12-1.625) | 3.06 m, deck 2.94, 3.08-3.53 |

The flank's span row IS the car's limit at the slot's h (`span_ceiling`,
the sill / roof fit and its 1.30 m cap are gone: 0.88 m -> 1.50 m on the
Corsa) and its area row stops at `area_ceiling`, both cut to AeroBO's
AR >= 3; the top's span row is 1.2 x the width, its area row kept inside
AR 3-40. Unlimited leaves the rows and opens the ceilings to 3x
(`OperatingPoint.size_caps`: a typed band is held there, and clipped again
at every build). `WingModel.commit` refuses in Real mode a wing past the
limit in any slot of its role that would carry it (the garage's old rule),
names the slot and where Unlimited is; in Unlimited it saves and logs
UNLIMITED. The designer's library writes have task 39's save guard.
3 Wing ▸ Design box and 1 Mission ▸ "Where the wing flies" print the
slot's limit (`bridge.limit_words`). The lap (mission and lap-time
objective) is still the Corsa's on every car, as on main.

The fixtures were recaptured (the Corsa's rows moved): 12 fixtures, 880 kB
(`FIXTURE_MAX_BYTES` 800 -> 900 kB: two section searches beat their seed,
so their reports carry the seed's drawing, 42 kB each). The replayed flank
winner is now 1.17 m.

```
python3 -m drive.aerobo_bridge   ->  27/27  ALL PASS  (+B27: every car's rows from bodies)
python3 -m drive.aerobo_models   ->  33/33  ALL PASS  (+M33: the session is the host car's)
python3 -m drive.garage          -> 118/118 ALL PASS  (main's task-41 rows + the designer's limits)
python3 -m drive.design_jobs / design_shell / design_shots -> ALL PASS / 63 / 7
python3 -m drive.cae.theme / widgets / plot / chrome -> PASS / 42 / 12 / 18
python3 -m drive.wing_tutorial, aero.wing, aero.library, bodies, challenges -> PASS
python3 -m drive.validate --modules -> 132/132 pass, 0 HARD, 0 soft (536 s)
```

iCloud struck again during the rebase: ~2000 `"X 2.*"` copies of the
untracked files appeared (20:40-20:53) and were moved aside; a bridge run
made while they were there screened them into
`aerobo/results/airfoil_screen_checkpoint.json` (restored from the seed).

## Where it was built, and the merge

Built in the git worktree **`/Users/tolomeelsabio/Desktop/carsim-aerobo`**,
branch **`aerobo-design`**, off `d244938` (task 37). Nothing is committed;
the lead merges it into `main`, which moved on meanwhile (tasks 38, 39, and
task 41 on `task41-wings-cars`). Where the edits meet:

* **`drive/garage.py`** is still the hotspot, but the pivot is a net
  DELETION there: `SectionModel`, `Designer`, `SectionObjective`,
  `WingObjective` and the carsim-engine self-check rows are gone; the models
  live in `drive/aerobo_models.py`. Task 39's save guards and plain criterion
  labels on those classes fall away with them (the section weights are
  AeroBO's criteria now, labelled in `aerobo_models.CRITERIA`); its
  `Garage`-level hunks still have to be re-applied by hand.
* **Task 41 (`MERGE_NOTES.md` of the building session, verbatim in
  substance):** `wing.span_fit(role, h, car=None, unlimited=False)` delegates
  to the new `drive/bodies.py` (span limit per car: a flank's span <=
  2 (h − ground clearance), a top wing <= 1.2 x the body width; *Wing limits:
  Real / Unlimited* allows 3x); the Corsa's sill / roof fit is removed (at
  h 0.90 its flank limit is 1.50 m, was 0.88); new cars `express`, `bus`;
  `Garage` gains `.car` / `.settings` / `.unlimited`, `CarBuild` a `car`
  field. **ACTION AT PORT:** `aerobo_bridge.flank_size_rows` calls
  `span_fit` -- pass the car and `unlimited`; `FLANK_SPAN_M` (1.30) and the
  Corsa `deck_z` must read `drive/bodies.py` per car (`g.car`,
  `g.unlimited`); the operating point's top ride band and flank band come
  from `bodies.top_h_band` / `flank_h_band`. Then recapture the fixtures
  (`python3 -m drive.aerobo_bridge --capture`, a few minutes with XFOIL).
* `drive/CONTRACT.md` §1: rows for `aerobo/`, `aerobo_bridge`,
  `aerobo_models`, `data/aerobo_fixtures`; the `garage`, `design_jobs`,
  `design_shell`, views, `design_shots`, `drive/aero/` rows rewritten; the
  new import rule ("AeroBO's engine has ONE door"); the design page and
  keys paragraphs. `main` edited the `menu` row and added `paint`: take both.
* `README.md`: a new *Install* section; "Design the wings" rewritten;
  Layout lines. `main` edited the mission paragraph (new circuits) -- the new
  text says "one of carsim's circuits", keep this side.
* `drive/validate.py` `MODULES` += `aerobo_bridge`, `aerobo_models`;
  `drive/challenges.py:264` (the refusal names the Design box's area and
  span rows).
* New tracked content: `aerobo/` (about 20 MB tracked, the 4.7 MB seed included; PLAN2 Q1),
  `requirements.txt`, `.gitignore` += `aerobo/results/`. The Desktop is
  iCloud-synced: a re-extraction once left 2085 `"X 2.py"` duplicates under
  `aerobo/` (removed, byte-checked); `aerobo/sync.sh` now rsyncs changed
  files only and refuses a tree with strays.

The plan (`specs/PLAN2.md`), the probes (`pivot/*.py`), the agents'
progress files and reports (`pivot/A` .. `pivot/F`) and the screenshots
(`pivot/F/final_shots`, `pivot/F/shots`) are in the building session's
scratchpad,
`/private/tmp/claude-501/-Users-tolomeelsabio-Desktop-carsim/7510d2b1-92a0-4ab3-87fa-1e02f35af8ce/scratchpad/`
-- a temporary directory: copy what should outlive the session.

## The owner's asks, and where each is answered

| owner | answered by | checked by |
|---|---|---|
| AeroBO (only for car) inside the game | `aerobo/` vendored unmodified (68 src, 2260 data, 1 record, sha in `VENDORED.md`); every stage calls `aerobo.api` with V3's arguments, through `drive/aerobo_bridge.py` only | bridge B1-B6, B11, B26 (the vendored tree unchanged by a whole check) |
| the mission and the looks may differ | Mission = carsim's circuit, surface and slot -> AeroBO's operating point (design speed, reference CZ, carsim's air, the slot's deck / ride band); the look = task 43's pygame shell | models M3, M4 |
| endplates symmetric | 2.8 screens only `api.symmetric_section_names()` (229 of 2174), shape-optimises with `symmetric=True` (`w_lower == -w_upper`) at cl 0; the engine refuses a cambered plate polar itself | bridge B14, B21; models M6, M9 |
| a live graph of every evaluation | `views_common.convergence_fig`: one point per evaluation (feasible / infeasible / refused), best-so-far line, x = 1..N, Sobol / BO and BO -> SLSQP marked, a Keep going's inherited evaluations shaded; on Shape optimisation, Convergence and Evaluations | shell rows, shots `af.opt running_a / running_b / continued`, `w.conv` |
| running vs finished unmistakable | one EngineJob contract: `RUNNING · k/N` + spinner + bar + status `k/N · phase · elapsed · ≈ left` + the tree's running glyph + the action turned into Stop; after: the STORED `DONE` / `CONVERGED` / `STOPPED` / `FAILED` tag with its wall time, a toast and an Output line | design_jobs J1-J8; shell RUNNING -> DONE / STOPPED rows |
| AeroBO's budgets (164 for a section) | `api.recommended_search` over AeroBO's `search_budget.json`: sections 109 / **164** / 240, the wing 42 / 53 / 87 (quick / balanced / thorough); carsim's budget law is gone from the page | bridge B6; models M2 |
| choose the wing objective | Wing type: AeroBO's car menu -- efficiency (default), downforce (side force on a flank), drag, downforce + drag, lap time (top wing, plain fences) -- with a drag ceiling and a downforce floor in newtons | models M15-M17 |
| no "cd at the design cl" on the wing (D9, kept) | the wing's screen has no cdcr weight, column or comparison row (`aerobo_models.REDUNDANT`, `aerobo_bridge.WING_WEIGHTS`); the endplate keeps it | models M7 |
| "One more circuit should be added 'Stopping'. Left flank and right flank, should be side." (2026-09-25) | the mission is the slot's JOB. Top wing: the circuits or **stopping** (`aero.mission.STOPPING`, a straight stop from `MissionSpec.v_stop_kmh`, default 100 = the Stop from 100 challenge), design speed = its start speed, 3 Wing opens on downforce + drag (exactly the stop's ranking at mu = 1), Results shows `aero.mission.stop`'s distance with vs without. Flanks: job **side** whatever the top's (`DesignSession.job`), design speed `V_REF["flank"]` (R 100 m limit speed), opens on side force, not reset by a top-circuit change; Results shows the R 100 m corner's limit speed. Player words say "side" (`garage.SLOT_LABEL`); keys and the "flank" role unchanged. The objective menu stays: a job only sets the speed and the objective it opens on (`aerobo_models.JOB_OBJECTIVE`). Replay fixtures were captured at efficiency, so the harnesses pick efficiency before a replayed side-wing run | mission self-check 7 (stop rows); models M3, M34; garage "the jobs" and "KEEPS a side wing's session" rows; real engine: stop 53/53, -0.37 m; side 53/53, +0.31 m/s at R 100 |
| "endplate t/c shouldn't be given as an option" (2026-09-25) | 2.8: no t/c gate rows and no t/c weight on the plate (AeroBO's own gate 0.10/0.15 and weight 0.20 stay underneath); 3 Wing: `endplate_tc` has no box controls (`NO_BOX_CONTROLS`), pinned to 2.8's section, never releasable; the family's own plate: AeroBO searches it, hidden | models M7b, M18b; garage box row |
| wing mount: "Carried by" as AeroBO's GUI asks it (2026-09-25, built by another session and merged in) | `WingModel.choices["plates"]` is now the one Carried-by answer (True endplates at the tips, False inboard pylons; the Param key stays "plates"). Endplates: root blend / leaning, each "I state it / optimise it" (free = `api.plate_freedom_name` variants, seeded with the upright plate, `bridge.plate_seed`). Pylons: AeroBO's continuum pylon flags plus a tip device (none / vertical / canted / blended); the plate follows the wing chord. Lap time stays top wing + pylons, real circuits only. The garage and Results previews stop drawing pylons under an endplate mount (`garage.wing_polys` / `hud_kwargs`, `render.wing_mesh3`, `aero/blend.py`, draw-only `WingSpec` fields). `derive_law` takes the span from the breakdown's `b_m` (it used to subtract the plate projection twice) | models M35-M37; garage 130; render 67; bridge 32 |
| "why no longer circuits?" (2026-09-26) | supersedes the flanks' fixed **side** job above: a side wing flies a circuit of its OWN (`MissionSpec.side_track`, `for_side()`: same surface and stated flag; circuits only, `aero.mission.SIDE_JOBS`, never the stop). Design speed = that lap's mean speed (as before the jobs), 3 Wing still opens on side force (`aerobo_models.SIDE_OBJECTIVE`), Results shows that circuit's lap with vs without. The top wing's job never moves it (`DesignPage.signature` per role). `ms.SIDE`, the R 100 m corner card and `SIDE_R_M` are gone. Old saves load with side_track arena | mission self-check 8; models M3, M34; garage "the jobs", Results lap row, "the top wing's changed circuit KEEPS a side wing's session"; real engine: side wing on arena 53/53, 240.1 N, game force = AeroBO, lap -0.134 s |
| "tip devices should be given option to follow chord distribution (for wing design both)" (2026-09-26) | under the pylons, "Its chord": follows the wing's chord / holds the tip chord (AeroBO v3's `car_endplate_chord_follows` switch), `WingModel.choices["tip_chord"]`, default follows (the owner's earlier ask), sent as `endplate_chord_follows` only when on; shown for every device but none, top and side wings alike. A record re-opens on its own answer (`choices_of_record`) | bridge B29 (new "pylons, vertical, tip chord held" row), B30/B31; models M35 |
| "change right and left are given independent name (just side wing)" (2026-09-26) | `SLOT_LABEL` left/right both "SIDE WING"; the designer's slot toggle offers "side wing" once while the pair is mirrored (both sides only after M splits them, then with ", left"/", right"); the wing page tag, slot words, build summary and car-page header say side wing | garage 130; design_shell 64 |

## What changed for the player

The shell, the tree, the tabs, the Output log and the keys are task 43's
(below, unchanged in shape). What runs behind them is now AeroBO:

* **Stage 1 Mission** hands AeroBO its operating point: the lap's mean speed
  (or a typed one), the reference CZ, carsim's air (ρ 1.2, ν 1.5e-5, so
  AeroBO's forces ARE the game's), the slot. *Search & budget*: AeroBO's
  recommended plan (164 / 164 / 53 at balanced) or own budgets, and AeroBO's
  convergence stop (the wing only).
* **Stages 2 / 2.8** are AeroBO's section workspace: screen AeroBO's 2174
  sections at the surface's own Re (a live XFOIL sweep of the shortlist,
  ~20 s the first time) or at the cached library point; rank; take a
  section; or run AeroBO's CST shape search (164 live-XFOIL evaluations at
  balanced, ~13 min at the measured 4.8 s an evaluation; Stop keeps the
  best, Keep going resumes).
* **Stage 3 Wing** is AeroBO's car rear wing, `car rear wing + endplates +
  free chord law` (14-D), searched with `bo_slsqp`; Wing type switches
  designed endplates / plain fences and free chord law / straight taper and
  picks the objective; the Design box has AeroBO's constrain / fix per row.
* **Stage 4 Results** is the run that landed: AeroBO's design report, every
  evaluation, the law the car flies, and carsim's own QSS lap as a labelled
  cross-check.
* **Onto the car**: the winner is flown through a law SAMPLED from AeroBO's
  evaluator at the winning design (exact at it); `S` / *Put it on the car*
  saves the sections and the wing, sets the slot's incidence (and the top's
  height) and mirrors the right flank.
* The top wing flies AeroBO's rear wing WITH ground effect over the car's
  deck; the flanks fly it WITHOUT (the image plane 100 m off).

## What changed in the code

New:

| file | lines | what |
|---|---|---|
| `aerobo/` | -- | AeroBO v1.0.0 (`3f1b07d`) vendored unmodified + `seed/` (the warm library checkpoint, 4.7 MB) + `VENDORED.md` + `sync.sh`; `results/` its runtime cache (gitignored) |
| `requirements.txt` | 14 | numpy, scipy, pygame; pymoo; torch, botorch, gpytorch (optional, noted) |
| `drive/aerobo_bridge.py` | 2764 | the one importer of `aerobo`: path / env / seed, slot families, the operating point, V3's builders, the runners, the resumes, the law and the `WingSpec` / `AirfoilSpec` mapping, carsim's circuits for the lap, fixture capture, the smoke self-check (B1-B26) |
| `drive/aerobo_models.py` | ~3100 | `DesignSession`, `SearchPolicy`, `SurfaceModel`, `WingModel`, `ResultsModel`; `replaying()` / `use_fixtures()`; the self-check (M1-M32) |
| `drive/data/aerobo_fixtures/*.json` | 12 files, 722 kB | real runs captured once (`--capture`): three screens, section main / plate (+ stopped at 5, + continued +6), wing top / flank (+ stopped at 8, + continued +6), a lap-time wing |

Rewritten on the new models: `drive/design_jobs.py` (`EngineJob`,
`ReplayRunner`, the warnings router; `OptJob` / `ControlJob` / `ScreenJob`
and every engine import deleted -- standard library only), the five
`views_*.py`, `design_shell.py`'s reads (chips, Properties, stage states,
Run / Stop routing, the lock), `design_shots.py` (every run replayed,
"running" frozen by the replay's `pause_at`). Small edits:
`drive/aero/wing.py` (`WingSpec.design`, provenance, persisted; `clamp`
leaves AeroBO's rows alone), `drive/aero/library.py` (`analyse_wing` never
re-analyses an AeroBO wing), `drive/wing_tutorial.py` (prose), `garage.py`
(the pages hold the sessions; `_top_aero` / `_dev_aero` / `mission_aero`
read AeroBO's laws; the self-check). Untouched: `drive/aero/{section,
screen,optimize,...}` (Q9), `drive.py`, `prerace.py`, `menu.py`,
`render.py`, `results.py`, `tutorial.py`.

## The decisions (PLAN2 §1.2)

* **D1 The family.** AeroBO's own car session: `car rear wing + endplates +
  free chord law` (14-D), `bo_slsqp`, budget 53, flags `{chord_trend:
  root_largest, mount: tips, section: shaped, bo_refusal: worst,
  bo_feasibility: guide}`. Wing type's two switches change the family.
* **D2 Slot families.** The only ground-proximity switch in the engine is
  the class constant `RIDE_HEIGHT_BOUNDS_M`, so the bridge registers carsim
  SLOT FAMILIES: AeroBO's builder, then the built problem re-instantiated as
  a carsim-side subclass that moves only that band (and the deck, carsim's
  air, the lap's track / car). Bit-for-bit AeroBO at the family's own band
  (B7). TOP: ride band `[deck_z(x) + 0.14, 1.85]` over the deck, ground
  effect on. FLANKS: the image plane 100 m off (ground term < 5e-6), the
  ride row = the plate's reach to the car's side, 0.25-0.70 m; the right
  flank mirrored.
* **D3 carsim's air** (ρ 1.2, μ 1.8e-5) in the slot families.
* **D4 Lap time** on carsim's circuits (`cartrack.TrackSpec` from the
  track's segments; the Corsa as `cartrack.CarSpec`), top wing with plain
  fences only, as AeroBO's menu; carsim's QSS lap on Results as the
  labelled cross-check.
* **D5 The thread.** One daemon worker, `runner(emit, stop)`, drained by the
  frame; one engine thread at a time; an abandoned run keeps the manager
  busy until its thread ends.
* **D6 Resume** = AeroBO's (`continue_run_config` / `continue_section`):
  nothing re-flies, the counter continues at k + 1.
* **D7 Budgets** only AeroBO's.
* **D8 The warm checkpoint** vendored as a seed (Q1).
* **D9 Fit to the car**: the law derived by sampling AeroBO's evaluator at
  the winner (incidence sweep on a sampling twin): CZ exactly affine
  (residual 0), exact at the design point, stall clamps at AeroBO's refusal
  edges, drag quadratic (exact at the design, <= 2 % off it). `slot.inc_deg
  := alpha*`, top `slot.h := ride_height*`.
* **D10 Gating = AeroBO's** (Q3).
* **D11 No XFOIL** (or `CARSIM_NO_XFOIL=1`): screening at the library point
  only, shape optimisation refused with the reason.
* **D12 Deterministic UI tests use FIXTURES**, replayed through the same
  EngineJob path; the engine itself is exercised by the bridge's smoke rows.
* **D9 of task 43 (owner): no "cd at the design cl" on the wing** -- kept:
  at one design cl, L/D = cl / cd ranks the library identically; the wing
  neither weights nor shows it, the endplate keeps it (its drag at cl 0).
  AeroBO's own wing weights already give it 0.

## The open questions, answered (PLAN2 §11.2 -- the lead accepted every recommended answer)

1. **Q1** The 4.7 MB warm screen checkpoint is tracked (`aerobo/seed/`):
   without it the first screen costs about an hour per Reynolds number.
2. **Q2** Flank design speed = the lap's mean speed, like the top; a typed
   override per slot (29 m/s is the R 100 m limit speed).
3. **Q3** AeroBO's gating: the wing is ready once the mission is stated and
   flies the family's own section until one is chosen; the tutorial keeps
   its order by guidance.
4. **Q4** The plate's t/c row is fixed to the section chosen in 2.8, as a
   visible *fixed from 2.8* row the player can release.
5. **Q5** The top's ride height is searched in the slot's band and written
   back to the slot, like the incidence.
6. **Q6** No section `pareto` objective (a front run cannot be stopped, and
   returns a set, not a section).
7. **Q7** The default car objective is AeroBO's `efficiency`.
8. **Q8** Leaving the garage while a run is live: refused, "stop it first"
   (drive and quit abandon it).
9. **Q9** carsim's `drive/aero/{section,screen,optimize}.py` stay (self-
   checked, imported by other checks); a later clean-up.
10. **Q10** No torch: AeroBO's `torch_optimiser_note()` and its non-torch
    optimisers; the budgets still apply.
11. **Q11** No carsim-QSS objective for the flanks: AeroBO as-is; the QSS lap
    stays a Results read-out.

## Deviations from PLAN2 the agents recorded

* **Flank size rows** (A): the plan's rule gave 47/200 feasible Sobol draws;
  now `S [min(0.12, S_hi/2), b_hi²/3]`, `b_lo = sqrt(3 S_mid)` (the narrowest
  span that carries the middle area at AR 3): 76/200 at the default flank
  (b 0.753-0.88 m, S 0.12-0.258 m²).
* **B14** asserts the gate reaches the engine at V3's screen gate t/c >= 0.15
  (122 of 2174 eligible) as well as the plan's 583 at t/c 0.10.
* **Fixtures** 722 kB (limit 800, not 600); run records stay exact (AeroBO
  refuses to resume evaluations a rounding moved off the box), only drawing
  arrays are rounded.
* **Screen tags**: a library-point screen is ONE read, so its tag says
  `DONE · 2174 screened` (the stored outcome k = n = the report's
  `n_screened`).
* **Shape optimisation** puts the run first once a search is live or has
  run (at 1280x800 the graph was below the fold otherwise).
* **`design_shell` has no torch warm-up job**: `Garage.__init__` calls
  `aerobo_models.warm_up()` (0.5-0.9 s at garage entry; the first mission
  frame 550 -> 41 ms).
* **Output lines** (F, an integration patch to `aerobo_models`): the models
  now hand the job its design point, seed, optimiser, Sobol block, AeroBO
  problem and the sections a wing run flies, so the start lines say what
  runs (they read "the family's own section" and "1 sections" before).
* **Integration fixes** (F, exact-match patches, reported): `design_jobs`
  `JobInfo.handoff` and an `SLSQP` phase (the status bar said `BO` past
  AeroBO's BO -> SLSQP handoff); a resumed run's CARRIED best is no longer
  claimed as "evaluation 13" with the current candidate's design
  (`_take_eval` checks `f == best`, `now_evaluating` says "inherited");
  `views_common.convergence_fig` names both splits on one label when they
  are a step apart (12 | 1 | 40 at 53 overprinted); `garage.open_mission`
  makes the slot's `DesignSession` at key time, beside the mission's own
  laps (its operating point flies a lap, 40-60 ms, which the first mission
  FRAME used to pay: `garage`'s "mission page draws" row failed about one
  run in six at 64 ms against 60, and `validate --modules` once read
  129/130; now 19-47 ms).
* **The screenshot harness** (F): the left-flank chain, own budgets set to
  the fixtures' (12 / 12 / 20) right after the mission is stated (m.search
  "pre" still shows AeroBO's 164 / 164 / 53); a stop lands on the captured
  stop point (section 5, wing 8), so the replay hands back AeroBO's own
  stopped record; the plate's library-point screen and a stopped screen
  have no captures (no running moment / no captured stopped screen); new
  state `continued` (Keep going, inherited band).

## The real engine, end to end (F, headless, 2026-09-24)

`pivot/F/e2e_real.py` drives a headless Garage the way the frame does
(pump, draw) on the REAL engine: the left flank, arena, dry; own budgets;
screen the wing's sections at their own Re, take rank 1, shape-optimise,
take the optimised section, screen the plate's (symmetric) sections at
their own Re, take rank 1, run the wing, derive the law, put it on the
car, read the build's `cfg_kwargs`. Every run ended DONE, no view raised.

| run | what it measured |
|---|---|
| seed 0; sections 12, wing 20; XFOIL cold at these Re | main screen 29 sections 18.2 s (hg40 first); section search 12/12 in 44.9 s -- **4.79 s an evaluation**, so AeroBO's 164 is ~13 min; plate screen at Re 5.36e5, 26 sections 23.4 s, all 14 ranked symmetric (mi-vawt1 first); wing `bo_slsqp` 20/20 in 5.2 s with the law and the report: **13.21 CZ/CD**, α* 8.04°; the game's force **45.688 N == AeroBO's 45.688 N**; `e2e-flank` in the left slot and mirrored right at 8.04°; `cfg_kwargs` `dev_left` / `dev_right` = `DevAero(S 0.1332, CL0 0.2597, CLa 5.165/rad, ...)`, `top` None. Frames while it flew: median 2.6 ms, p95 5.2 ms, max 241 ms |
| the same again | 17 s in all: AeroBO's XFOIL cache and evaluation cache answer every design already flown (the screens read "answered from this point's XFOIL cache") |
| seed 7, wing 20 | the 4 Sobol draws infeasible, one BO step, 15 SLSQP steps from a poor start: **2.67 CZ/CD** at α 0°, AeroBO's verdict "cannot be said yet". The same seed at AeroBO's recommended **53: 18.95 CZ/CD**. 20 is the plan's quick check, not a design budget for a 13-D wing |
| seed 3; sections 12, wing 53 (the final shots) | section search 32.9 s cold, 41.60 against the seed's 39.25 (t/c 0.151 -> 0.131, cl max +13.6 %); wing 53/53 in 5.4 s: **16.69 CZ/CD**, α* 6.93°, game force 36.324 N == AeroBO's; frames median 2.7 ms, p95 12.2 ms, max 148 ms |

Screenshots (1600x1000) in `pivot/F/final_shots/`: `af.opt_running.png`
(RUNNING · 5/12, Sobol -> BO), `af.opt_done.png` (DONE · 12/12, seed vs
optimised), `ep.rank_done.png`, `w.conv_running.png` (RUNNING · 9/53,
the Sobol block), `w.conv_done.png` (DONE · 53/53, 12 Sobol -> 1 BO -> 40
SLSQP), `r.summary_done.png`. The replayed set, every view in every state
at both sizes: `python3 -m drive.design_shots --out DIR` (a copy in
`pivot/F/shots/`).

## Self-check assertions that changed

* `drive.garage` 133 -> **88** (C): rows 1-34 kept; the navigator and draw
  rows kept; the gate rows adapted to AeroBO's gate; ~36 new rows on
  fixtures through the garage's own entry points (keys, forms, the mission's
  V rows): operating point, 164 / 164 / 53, the live screen, the ranking
  without cd@cl on the wing, taking and declining sections, the symmetric
  plate screen, the plate t/c fixed from 2.8, section run / Keep going /
  ESC = Stop, the flank objective menu, fences locking 2.8, the Design box,
  Solver config (13-D, budget 50 with the pin), wing run -> law -> report,
  Stop at 8 then +6, the car flying AeroBO's force exactly, the stale law
  re-derived after a slot move, the mirrored session, own budgets, the
  no-XFOIL refusal, the live-run lock, abandon on re-open, the live-run
  frame budget, restating keeps the session, a changed circuit clears it.
  The old N1-N22 (carsim's engine: sync == live, random control, exports,
  stop early) are gone; N14 (the 1600x1000 rects) is the last row.
* `drive.design_jobs` 24 -> **20** (B): J1-J20 on EngineJob / ReplayRunner /
  RunManager / the templates.
* `drive.design_shell` 52 -> **63** (E): N11 on fixtures; RUNNING -> DONE,
  Stop -> STOPPED, the live graph drawn, Keep going's `n_prior`, the chips
  and Properties on the new models, AeroBO's gate.
* `drive.design_shots` 7 -> **7** (F): the same seven rows; the "frozen"
  row also proves every run was a replay.
* New: `drive.aerobo_bridge` 26, `drive.aerobo_models` 32.

## Known leftovers

* **Fixture artefacts** (only in the checks and the screenshots): the
  sections were captured at the TOP slot's Reynolds numbers (2.91e5 main,
  4.20e5 plate) and the top wing over a 0.90 m deck (the game's top deck at
  x = −0.90 is 1.434 m). The flank chain therefore shows the report's Re
  beside the flank's own 3.72e5 in two places. `--capture --deck 1.434`
  and a flank section capture would align them (the 2026-09-25 recapture for
  task 41 kept the 0.90 m reference deck).
* A replayed Keep going after a FULL wing run shows 20/20 (the captured
  continuation resumes from the stopped-at-8 record); the harness only keeps
  going after a stop.
* On a live Keep going, *now evaluating*'s "best so far (evaluation 13)"
  names the first evaluation of the continuation that reported the carried
  best, not the evaluation that found it.
* The Output log shows AeroBO's own warnings on every section run (torch.jit
  `DeprecationWarning`, `bo_feasibility='guide'` `RuntimeWarning`): honest,
  noisy; filter or demote them if the owner prefers.
* Garage entry pays ~0.5-0.9 s for `aerobo_models.warm_up()` (torch and
  AeroBO's first plan) on top of the ~1 s bridge import; the first
  af.section draw ~185 ms (AeroBO's 2 MB branch sidecar, read once). In
  `garage`'s own check the first mission frame still sometimes pays ~50 ms
  more than in a fresh process (not garbage collection; not reproduced
  outside the check) -- within its budget since the fix above.
* The status bar calls a resumed `bo_slsqp` run's new points `BO` (AeroBO's
  resume report was not read for the phase); fresh runs say Sobol / BO /
  SLSQP.
* `aerobo/sync.sh` was syntax-checked only after its rewrite (re-syncing
  would touch the vendored tree).
* Task 43's look leftovers (SF word spacing, MONO advance, bold width) still
  stand -- listed at the end of this note.


## The shell, from task 43's first build (still true)

The look, the tree, the focus model and the keys were built before the
pivot and kept by it. Its decisions:

* **D1 Look.** AeroBO's light CAE theme frames the garage pages `mission` and
  `section`; the car, airfoil and library pages stay carsim's dark ones.
* **D2 Fonts.** SF Pro / SF Mono (macOS), Segoe UI / Consolas (Windows),
  DejaVu Sans / Mono (Linux), by explicit path first, then `match_font`, then
  Menlo / Arial, then pygame's default; Material Icons bundled; the +1 px
  SANS rule is SF's only.
* **D3 (superseded by the pivot's D5)** Live optimisation without threads. Ask / tell steppers stepped from
  `Garage.frame` under a per-frame time budget; `maximise()` /
  `random_search()` are thin loops over the same steppers and reproduce the
  old results bit for bit; `SectionModel.screen_library()`,
  `SectionModel.optimise()` and `Designer.optimise()` stay synchronous and
  run the same job code. A run is ONE BO phase of exactly N evaluations
  (the seed is the first); it ends early only by Stop or, with *stop early*
  on, by AeroBO's convergence rule. The random control is off by default
  and runs as a separate job. While a run is live, the controls that would
  change ITS objective are locked, a fixed list of global hazards is
  refused, and everything else stays live (each job flies a frozen
  snapshot).
* **D4 Pins.** Every pin of the garage's self-check is kept unless parity
  needed a change, and each change is listed below with the behaviour it
  still tests. The 16 step keys are the view keys; page ids stay. Mouse is
  primary; keyboard and pad reach every control. The derived pin "all 16 step
  rows drawn at once" gives way to the expand-on-select tree; every tutorial
  anchor still resolves.
* **D5 Which airfoil** is visible in every view (list above); "what the wing
  flies" is read from the lattice's own polar (`spec.aero`), not the search's
  polar bank.
* **D6 What evaluation** is visible: the `RUNNING · k/N` tag, the status bar
  and its bar, the tree's running glyph, the log lines, the live convergence
  plot, the live-only now-evaluating block.
* **D7 Screenshots.** `drive/design_shots.py` renders every stage / view in
  pre / running / post (and stopped) at 1280x800 and 1600x1000, as scroll
  series; "pre" is the chain walked up to the view's own action.
* **D8 A concurrent session.** New code in new files; every edit to an
  existing file an exact-match patch right after re-reading it; never
  commit; never touch `drive.py`, `prerace.py`, `menu.py`, `render.py`,
  `results.py`, `tutorial.py`. Then the work moved into the worktree above.

Its leftovers, still open:

Theme (`drive/cae/theme.py`, W5-F4), all measured against AeroBO 03:

* **SF's word spacing** is 2 px at 11-12 px (3 at 13) against AeroBO's
  4-5: *AR estimate* reads *ARestimate*, *lap as it stands* reads
  *lapasitstands* in Properties, the status line *12/34sections*. The most
  visible defect left.
* **MONO advance**: SF Mono 11 is 7 px a glyph in pygame against the
  browser's 6.63 (9.5 -> 6 against 5.72): status cells start 18 px left,
  the log wraps early, long tree chips (`NACA 0010 (its own default)`) are
  cut where AeroBO fits them.
* **Bold is no wider than regular** in `pygame.font`: the active tab, the
  tabs after it (408 / 479 / 547 against 412 / 481 / 548), the selected tree
  label and pane titles run narrow; `pygame.freetype` with `strong=True`
  measures right (103 px against AeroBO's 102).
* The +1 px SF rule overshoots at 11 and 11.5 css (inactive tabs and
  Properties keys 1-4 px wide); *Stop* 2 px narrow moves two tool-bar
  separators by -2; Properties rows drift +1-2 px (AeroBO's pitches are
  fractional, the plan pins 21 / 23).
* KPI digits render lighter than AeroBO's; SF lacks `⟺` (the Results
  margins hint shows `<=>`).

Kit and views:

* Garage Param help strings print ASCII ` -- ` in the "?" popups; `views_wing`
  swaps them locally (`_help`), the kit should do it once.
* `ui.table` has no per-cell bold (the ▲ / ▼ verdict cells are colour only).
* `now_evaluating` uses the 26 px stack pitch (six rows); a tighter pitch
  needs a `WorkUI` option.
* `v1_row` labels are SANS 12 in a 60 px column and wrap (*Blend shape*,
  *Reference area*); AeroBO's look nearer 10 px.
* `Form.nav` on a view that drew no focusable control falls back to the
  whole ParamList and could focus hidden actions; `views_results` works round
  it with one link per view.
* At 1280x800 on the lap objective, the Ranking's `lap Δ [s]` column sits
  behind the table's horizontal scroll.
* Tier 2, small: no per-parameter tooltip on Results' *Best design* grid.
