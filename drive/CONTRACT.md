# drive/ — interface contract (AUTHORITATIVE)

Real-time drivable simulator for the Opel Corsa C 1.2 16V flank-wing study.
Every module below is written against THIS file. Where a subsystem spec in
`../specs/*.txt` disagrees with this file, THIS FILE WINS. The reconciliations
are recorded at the bottom.

Package root: `/Users/tolomeelsabio/Desktop/carsim`, package `drive/`.
Run as `python3 -m drive.<module>`. `corsa_c.py`, `qss.py`, `crossover.py`,
`ledger.py` live one level up and are importable because the package is run
from the repo root.

## 0. Global conventions — non-negotiable

* **Frame**: ISO body frame. `x` forward, `y` **LEFT**, `z` up. Yaw `psi` and
  yaw rate `r` positive counter-clockwise seen from above = **left turn**.
  `a_y` positive = leftward. Roll `phi` positive about `+x` (right-hand rule)
  = body leans to the RIGHT = outside of a left turn.
* **Wheel order is always `FL, FR, RL, RR` = index 0,1,2,3.** Every 4-tuple
  and every `(4,)` array in every module uses this order. No exceptions.
* **Steer**: `delta > 0` steers LEFT. Road-wheel angle in **radians**.
  Only the front wheels steer.
* **Slip**: `kappa = Vsx / |Vx|` with `Vsx = omega*R_e - Vx`. Positive kappa
  drives the car forward (`Fx > 0`). `alpha = atan2(Vy, |Vx|)`, positive alpha
  gives `Fy < 0` (restoring). Both are computed as **transient (relaxed)**
  states, never formed directly as a quotient — see §4.
* **Angles**: radians internally, degrees only at display/CLI boundaries.
* **Units**: SI. N, m, s, kg, rad, N·m. rpm only at the HUD/CLI boundary.
* **No `Date.now()`-style nondeterminism in the physics path.** `t = n_steps*dt`,
  never accumulated. No wall clock, no unseeded RNG, no dict-iteration-order
  dependence inside `step()`.

## 1. Module map and ownership

| file | owns | may import |
|---|---|---|
| `drive/tyre.py` | Magic Formula 6.2 combined slip from `.tir` | `corsa_c` |
| `drive/powertrain.py` | engine, clutch, gearbox, diff, brakes | `corsa_c`, scipy |
| `drive/vehicle.py` | EOM, load transfer, roll, aero+wing, integrator | `tyre`, `powertrain`, `corsa_c`, `cars` |
| `drive/track.py` | track geometry, projection, surfaces | numpy |
| `drive/input.py` | keyboard/gamepad → `Controls` | pygame |
| `drive/bodies.py` | task 41: the body SHELLS per style (hatch / roadster / saloon / van / bus; moved out of `render.py`), `style_of(car)`, `Body(car)` on the car's own axles (`x_front`, `x_rear`, `width`, `height`, `ground`, `deck_z`), each car's slot bands and default slots, and the owner's SPAN RULE: `span_limit` (flank `2 (h - ground)`, top `1.2 x width`), `span_ceiling` (x `UNLIMITED_FACTOR` 3 when Settings' Wing limits is Unlimited), `area_ceiling`, `over_limits(build, lib, car)` (computed per run, never stored). Read by `aero.wing`, `garage`, `records`, `challenges`, the drive, and `aerobo_bridge` (the AeroBO designer's per-car rows, task 43) | numpy; `cars` and `aero.wing` lazily. Never pygame |
| `drive/render.py` | pygame drawing + HUD; the chase camera and the 3-D car (a body STYLE per fitted car: hatch / roadster / saloon / van / bus, generic shapes, the shells read from `bodies`); `look_config` / `Renderer.set_look` (the Graphics setting); `set_paint` / `Renderer.set_paint` / `factory_colour` (the Paint setting) | `track`, `qss`, pygame; `world`, `props`, `fx` inside `Renderer` (built when `ViewConfig.scenery` / `effects`) |
| `drive/paint.py` | the player car's paint: `PAINT_ORDER`, `PAINT_LABELS`, `PAINTS` (name -> RGB; `factory` -> None, the car's own colour), `PAINT_DEFAULT`, `rgb(name)`. A fixed palette, every tone of it checked against the colours the self-checks count. Cosmetic: never in the class key, a ranking, a medal, a ghost, the CarSpec or the build JSON (a lap record's `settings` snapshot lists it, as it lists Graphics) | nothing (pure data); `race_grid`, `ghosts` in its self-check only. Never pygame |
| `drive/scenery.py` | the ground beside the road, laid out from the track's own geometry (a generated track gets it too): verges, gravel traps (slow corners), painted run-off (fast ones), apex + exit kerbs, the racing line, paint (grid boxes at `race_grid`'s slots, sector bars, the dragstrip's lane / numerals), all within 24 m of the edge; `wheel_surfaces(track, pts, on_track4, mu4, scenery=)` -> per wheel `'tarmac' \| 'wet' \| 'kerb' \| 'grass' \| 'gravel'`, read off the SAME tables the drawing uses. Never read by the physics | `track`, numpy |
| `drive/world.py` | the look's backdrop and ground: sky + a per-map panorama (clouds, ridges, a tree line; scrolled by the chase camera's VIEW heading), grass with mowing stripes, the dressing from `scenery`, detail ON the tarmac (rubber, repairs, water, inset lines, raised kerbs), the horizon haze; the chase view's track layers batched (one projection a layer). The SHARED look: `SUN_DIR`, `HAZE_RGB`, `haze_factor`, `hazed`, `HAZE_LAND` | numpy, pygame, `scenery`; `render` only lazily (it is handed the renderer) |
| `drive/props.py` | the solid things round the road: trees, tyre walls / armco with fictional boards, catch fences, stands, the pit building and tower, the start gantry, brake boards, marshal posts, masts; per-map themes, generic from the geometry; every solid prop >= 30 m from the edge (the car has no collision), thin exceptions justified in its docstring. Plan footprints and a pre-lit chase soup + tree sprites, split at the car's depth | numpy, pygame, `world`; `render` only lazily |
| `drive/fx.py` | the particles -- tyre smoke past the grip peak (a per-car onset), dust off the road, spray on wet -- in one fixed pool (`POOL_N`), and the chase view's surface-aware camera jolt; a consumer of `HudData` (`surf4`, `wheels_xy`, `car_key`), never an input | numpy, pygame, `world`; `render` only lazily |
| `drive/telemetry.py` | CSV logging | csv |
| `drive/plots.py` | matplotlib post-run plots (Agg) | matplotlib, numpy |
| `drive/garage.py` | 3D garage: `CarBuild` (three wing slots) -> `VehicleConfig` kwargs and the fitted wings' mass; the MISSION page (`MissionPage`: circuit, surface, Search & budget, the slot's design speed) and the DESIGN page (`DesignPage`: `DESIGN_TREE` over one `aerobo_models.DesignSession` per slot, `sessions`; a mirrored right flank is the left one's) -- the page designs nothing itself (task 43's pivot): every stage is a view of the session's models, which call AeroBO's own engine. `_top_aero` / `_dev_aero` / `mission_aero` fly an AeroBO wing's STORED law (`WingSpec.aero`, re-derived when the slot has moved: `aerobo_models.law_stale` / `rederive`), never a re-analysis. Both pages are drawn and driven by `design_shell`. The navigator GATES by STAGE with AeroBO's rule (`DesignSession.state`): once the mission is stated 2 Airfoil, 2.8 Endplate and 3 Wing are open, 4 Results waits for a completed run, 2.8 is LOCKED on plain fences. Every page takes the MOUSE as well as the keyboard. Plus the airfoil and library pages | `corsa_c`, `cars`, `crossover`, `input`, `menu`, `garage_ui`, `bodies` (task 41), `cae` (`form`), `design_jobs`, `design_shell`, `aero`, `track` (`make_track`, for the mission's circuit), `vehicle` (the two aero dataclasses only), pygame; `aerobo_models` LAZILY (`_am()`, so importing the garage does not import AeroBO before it is needed). Never `aerobo` or `aerobo_bridge` directly |
| `drive/garage_ui.py` | widget kit for the garage pages (params, lists, plots, prompt) | pygame, numpy |
| `drive/cae/` | the AeroBO-look widget kit (light CAE theme, D1/D2): `theme` (tokens, fonts, icons, text), `form` (`Form`, the ParamList-compatible object a shell view binds its controls to), `widgets` (`WorkUI`, the immediate-mode work area), `plot` (`Figure`), `chrome` (menu bar, tool bar, tree, properties, tabs, output log, status bar, toasts, dialogs). Pure UI: no physics, no file IO beyond its bundled fonts | pygame, numpy, `garage_ui` (`form` subclasses `ParamList`) |
| `drive/design_jobs.py` | the live-run machinery (PLAN2 §6): `EngineJob` (one AeroBO call on ONE daemon worker thread, `runner(emit, stop)`; the frame drains its queue; `finish` and `on_finish` run on the main thread only), `ReplayRunner` (a captured run replayed through the same path, `pause_at` / `release`, the fixture's own stopped variant), the warnings router (an engine warning on the worker becomes an Output line), `RunManager` (one live run; `cancel` abandons a run and stays `busy` until its thread has ended), `Notices`, `JobInfo`, the chips (`RUNNING · k/N`, `terminal_tag`) and the log / status templates | the standard library only. Never pygame, numpy, `aero`, `aerobo` or `aerobo_bridge` |
| `drive/design_shell.py` | the AeroBO shell framing the garage's MISSION and DESIGN pages: layout, tree (`CaeTree`, a `garage_ui.Nav`), the stage chips and Properties (read off the slot's `DesignSession`), stage states (`DesignSession.state` + the live `EngineJob`'s stage), tabs, Output, status bar, focus model, keyboard / pad / mouse routing, Run / Stop (routed to the models' job starters), the live-run lock (`LAUNCH_KEYS`, scoped forms) | pygame, `garage_ui`, `cae`, `design_jobs`, the `views_*` modules (by name, `VIEWS`); `garage`, `aerobo_models` and `aero.library` only in its self-check |
| `drive/views_common.py`, `views_mission.py`, `views_section.py`, `views_wing.py`, `views_results.py` | the stage views drawn in the shell's work area, one module per stage group, and the builders they share (`views_common`: the designing row, the live evaluation graph `convergence_fig` / `convergence_card`, `run_banner`, `now_evaluating`, the section outline and polar figures, the terminal tags) | numpy, `cae`, `design_jobs`, `views_common`; `views_mission`, `views_section`, `views_wing`, `views_results` also `aerobo_models` (the models' API, and AeroBO's READ-ONLY helpers through `am.bridge`); `views_mission` also `aero.mission`, `track` (`make_track`); `views_results` also `views_wing`, pygame lazily. Never `garage` (garage constants arrive in `ViewCtx.consts`), never `aerobo` itself |
| `drive/design_shots.py` | the design pages' screenshot harness (every view x pre / running / post / stopped / continued x 1280x800 / 1600x1000, as scroll series), every run REPLAYED from the captured fixtures (`aerobo_models.replaying`) and frozen "running" by the replay's `pause_at` | `garage` and `aerobo_models` lazily, `design_jobs`, `aero.library`, pygame |
| `aerobo/` | AeroBO v1.0.0 (commit `3f1b07d`) VENDORED UNMODIFIED: `src/aerobo/` (68 files), `data/` (2260), `records/airfoil_screen_branch_v2.json`, `LICENSE` (MIT); `seed/airfoil_screen_checkpoint.json` (the warm library checkpoint, sha256 in `VENDORED.md`); `VENDORED.md`, `sync.sh` (the re-sync). `results/` is AeroBO's runtime cache (XFOIL polars, eval cache, screen checkpoints), gitignored | -- (a package of its own; its requirements are `requirements.txt`'s torch / botorch / gpytorch / pymoo) |
| `drive/aerobo_bridge.py` | the ONLY importer of `aerobo`: `sys.path` / env set-up and the seed install; the carsim SLOT FAMILIES (AeroBO's car family re-instantiated as a subclass that moves only `RIDE_HEIGHT_BOUNDS_M`, the deck and carsim's air: top = ground effect on over the deck, flanks = the image plane 100 m off); the operating point (`OperatingPoint`, from the stated lap); V3's configuration builders (weights, gates, section conditions, shape kwargs, wing flags, searches from `api.recommended_search`); the job RUNNERS (`screen_runner`, `section_runner`, `wing_runner`, `score_runner`, `polar_runner`, `design_report_runner`, `law_runner`) and the resumes (`continue_section`, `continue_wing`); the law derived by sampling AeroBO's evaluator (`derive_law`) and the mapping onto carsim's `WingSpec` / `AirfoilSpec`; carsim's circuits as `cartrack.TrackSpec`; fixture capture (`--capture`); the engine smoke self-check. Task 41's per-car numbers, from `bodies` for the build's car and Settings' Wing limits: `car_deck`, `top_ride_band`, `flank_h`, `span_limit`, `size_rows` / `flank_size_rows(h, car, unlimited)`, `size_caps`, `over_limits`, `limit_words`; `OperatingPoint.car / unlimited / limit / size_caps` | `aerobo`, numpy, scipy (`qmc`, the self-check), `aero` (`xfoil`, `polar`, `wing`, `airfoil`, `mission`), `bodies`, `track`, `qss`; `vehicle` (`TopAero`) in its self-check. Never pygame, never `garage` |
| `drive/aerobo_models.py` | the DESIGN page's models (PLAN2 §7): `DesignSession` (one per slot: `op`, `policy`, `af`, `ep`, `wing`, `results`, `state(stage)`; `car` / `unlimited` / `deck_z` read off the host -- the garage's `car` and `unlimited`, task 41 -- and `past_limit(span, keys)`), `SearchPolicy` (Mission > Search & budget: AeroBO's plan or own budgets), `SurfaceModel` (a section surface, main or the symmetric plate: screen, ranking, take, shape search, Keep going, polar), `WingModel` (Wing type, Design box, Solver, run, Keep going, verdict, law, fit, commit), `ResultsModel` (the design report, the summary, carsim's QSS lap as the cross-check); every engine call an `EngineJob` whose runner freezes its arguments at launch; `replaying()` / `use_fixtures()` switch every job to the captured fixtures (the checks) | `aerobo_bridge`, `design_jobs`, `garage_ui` (`Param`), `cae.form`, `aero.mission`, `track`, numpy; `garage` and `aero.library` only in its self-check. Never pygame |
| `drive/data/aerobo_fixtures/` | twelve REAL engine runs captured by `python3 -m drive.aerobo_bridge --capture` (screens, section and wing searches with their stopped and continued variants, a lap-time wing run; 722 kB), replayed by the deterministic checks and the screenshot harness | -- |
| `drive/aero/` | wing-design physics: sections, panel method, polars (XFOIL / estimate), vortex lattice, GP-BO, the library, and carsim's own three-step design procedure -- `mission.py` (the lap a wing is for), `screen.py` (the seven weighted criteria the library is ranked on), `section.py` (the aerofoil designed in 2-D against it), `wing.py` (the planform; `WingSpec.design` carries an AeroBO wing's provenance, and `clamp` leaves its rows alone), `blend.py` (how the wing and its end plates meet); the ask/tell steppers (`optimize.BOStepper`, `RandomStepper`). Since the pivot the DESIGN page calls none of `screen` / `section` / `optimize` (AeroBO's engine designs; PLAN2 §11 Q9 keeps them, self-checked, for a later clean-up); `library.analyse_wing` never re-analyses an AeroBO wing (its law is AeroBO's) | numpy, scipy, the `xfoil` binary if present; `mission.py` alone also imports `corsa_c` and `qss` |
| `drive/menu.py` | pause / help menu overlay (ESC, OPTIONS); a row's colour swatch through `show(swatches={action: rgb})` (rows stay 2-tuples); pure UI | pygame only |
| `drive/title.py` | task 44: the title screen an interactive launch opens on (`Title.run()` -> one of `ACTIONS`: drive / challenges / garage / tutorial / settings / quit); the menu column and the pause menu's command vocabulary from `input.BlendedInput` in menu mode; the live background -- `pick_scene` (a `track.CIRCUITS` map, 1..5 cars of distinct `cars.CAR_ORDER` types with a reference lap), `Scene` (the reference laps replayed as a train, re-spaced at each chase-camera cut every `CAM_SWITCH_S` s), `_SceneRenderer` (a `render.Renderer` whose car pass draws every scene car in its own body and paint, restoring `render`'s module car and paint); the panorama fallback (`world.build_panorama`). No physics, no player file | pygame, numpy, `render`, `menu`; `track`, `cars`, `bodies`, `records`, `medals`, `world`, `input` lazily. Never `drive.drive` |
| `drive/audio.py` | procedural car sound: `Synth` (numpy) + `CarSound` (one pygame.mixer channel, stereo when the mixer grants it; task 27's chime on channel 1); an engine PROFILE per car (`HudData.car_key`); a render-loop consumer of `HudData`, never an input | numpy, pygame; `scipy.signal` optional, imported off-frame by `warm_up` (its fast path uses scipy's private `_sigtools._linear_filter`, checked for exact equality with `lfilter` at import, else the public one); its self-check imports `render.frame_budget_verdict` lazily (the `garage` exception) |
| `drive/records.py` | lap records: the class key `track\|car\|engine\|surface`, `RecordBook` (top 5 per class, `runs/records/<class>.json`, best sectors, best medal, `last_builds.json`), `LapRecorder` (the `Sim` hooks: controls log, 50 Hz trace, the lap's exact start state), `resimulate` (a lap re-driven from its log, bit for bit) | numpy; `vehicle`, `powertrain`, `cars`, `corsa_c` (dataclass registry only); `drive.drive` / `track` lazily inside `resimulate` and the self-check. Never pygame, never `drive.ml` |
| `drive/prerace.py` | the pre-race (TIME TRIAL) page's content: `PreRace` rows and help sections from a `RecordBook`, the medal table and the library's builds; `wanted(opts, settings)` (never a script, headless, `--ml-drive`, offscreen, the dragstrip); the PICK page rows. Pure UI logic: the `Sim` owns the menu and dispatches | `records`; `medals` lazily. Never pygame |
| `drive/medals.py` | medal times per class (plan D4): author = the best valid, spin-free, full lap a reference driver sets headless in the class's STOCK car (`LapDriver` at margins 0.90 / 0.80 / 0.70 / 0.60, the `drive.ml` anchor, every bundled checkpoint for that car + track; aids off and on), gold / silver / bronze = author x 1.02 / 1.06 / 1.12. Owns `drive/data/medals.json` (`--build`) and `drive/data/reference_laps.json` (the author laps' 20 Hz traces, task 22's reference ghost). `targets`, `medal_for`, `reference_trace`; staleness by a hash of the track definitions, car specs and engine modes | numpy, `records` at module level; `drive.drive`, `track`, `cars` lazily; `drive.ml` only in the build's workers. Never pygame |
| `drive/ghosts.py` | the time trial's two ghosts (the class PB; the D2 slot: reference bot / none / your P2..P5), clocked from the line (`sim.t - LapTimer.t_lap_start`); the live delta `tau - t_pb(p)`, `p` the centreline progress counted from the crossing on the ribbon only (the race gap's trail approach); the sector flash (purple / green / red, none for a lap that cannot count) | numpy, `records`; `drive.drive._Replay` and `medals` lazily. Never pygame |
| `drive/progress.py` | `runs/progress.json` (kind `carsim-progress-1`): one section per feature (`tutorial`, task 25's `challenges`); `save(section)` merges with the file; a corrupt / foreign file is ignored with a note and moved aside as `.bad-<stamp>`, an unreadable one never written over | `records._atomic_json` lazily. Never pygame |
| `drive/tutorial.py` | the driving tutorial: 13 data-driven `Step`s (`id, map, kind, title, text, hint, check(frame, mem), status, reset, setup, wing, gearbox, group`; task 31's optional manual-gearbox pair), `frame_of(sim)` (what a predicate sees), the `Tutorial` state machine (`tick(sim)` -> 'restart' / 'page' / 'done'), the overlay payload, the page / menu rows, `wing_car` (the published plate for a car without a flank wing) | `corsa_c.G`; `records`, `medals` lazily. Never pygame, never `drive.drive` |
| `drive/wing_tutorial.py` | the wing-design tutorial: 10 `WStep`s (`check(garage, mem, action)`, an `anchor` naming the garage widget the hint points at), `WingTutor` (`update` / `draw` / the garage menu's rows; the box at `BOX_CAR` on the dark pages, at the shell's `tutor_box` on the mission and design pages), `anchor_rect` (from the widgets' own `_rect` / `_hits`, the shell's tool buttons and tree fallbacks, and `Form.rect_of`) | `garage_ui`, `prerace._same_build`, `progress` (via the object handed in); `design_shell` (lazily, for the tutor box and the tool-button anchors); `garage` and pygame only lazily / in the self-check. Never `drive.drive` |
| `drive/challenges.py` | challenges (`drive/data/challenges/*.json`, kind `carsim-challenge-1`; no numbers in them since task 44): `validate`, `load_all`, the combos (task 44: `CONFIGS`, `config_build` / `config_parts` -- a config applied to the fitted copy of a build --, `resolve` -- a challenge in a car + config, thresholds derived from `refs.json` (kind `carsim-challenge-refs-1`, `load_refs` / `validate_refs` / `write_refs`) --, `combos`, `not_for_car`, `g_modes`), the constraint checker (`build_stats`, `refusals`), `stars_for`, the per-step `Meter` (lap_time, stop_distance, skid_ay; drag_time and trap_speed stay measurable, no challenge uses them since task 36) and `ChallengeRun` (the Sim's hooks, best + stars into `runs/progress.json` under the combo's key; a stop challenge starts rolling at its goal's `start_kmh`, the pose `ChallengeRun.rolling` gives every `Sim.reset`, task 40; at v0 a `StopHold` keeps it there on rails, no physics, until the brake is in, task 42, its top wing as it rides at v0 -- a FIXED one out -- `Sim._hold_top`, task 44), the reference runs (`measure`, `measure_combo`; `--measure` / `--write [--only S] [--jobs N]` measure the 160 combos in a process pool into refs.json, every threshold ref x 1.12 / 1.06 / 1.02), the page rows (`pick_rows`, `wings_line`, `wings_section`, `list_items`, `detail`) | `records`, `progress` (via the object handed in); `drive.drive`, `garage`, `track`, `vehicle`, `aero.library` lazily. Never pygame |
| `drive/race_grid.py` | the race grid (plan D3): `GRID_MAX` (5, measured), `grid_slot(i)` (rows of two, 7 m apart), the slot colours, `bred_meta` (what a swarm writes into a checkpoint about the car it bred in), `own_car(meta, cfg, lib)` (that car rebuilt: stock or yours with ballast and wing masses, the garage build's aero, the engine) | `cars`, `vehicle`; `garage` lazily. Never pygame, never `drive.ml` |
| `drive/swarm_panel.py` | the swarm window's progress panel (`panel_lines`: the best lap per generation with a bar, the class's medal lines and the owner's PB with the swarm's gap to each) and the Deploy-swarm page's free values (`clamp_pop` 4-128, `clamp_T` 20-240 s; task 34: `POP_PRESETS` / `T_PRESETS`, `step_value` / `cycle_value` over them, `Typed` digits, `row_value`) | nothing (pure; the caller hands in the PB). Never pygame, never `drive.ml` |
| `drive/controls_page.py` | the CONTROLS page (task 37): `CONTROLS` (each DualSense control, its anchor on the drawing, its label, what it does), `MENU_PAD`, `pad_rows`, `kb_rows` (input.MENU_HELP_KB), `draw_pad(screen, rect)` (the controller and leader-lined labels, fitted to the rect, the labels' font shrunk to their column) | `input` (the bindings), `render.FONT_NAMES`; pygame lazily |
| `drive/airbrake.py` | the G key's wing mode (tasks 35, 44): `AUTO / AIR / TOP / TOP_FIXED / TOP_FIX_SIDE / LEFT / RIGHT` (0 / 3 / 4 / 5 / 6 / 1 / -1; 2 was ALL 3, removed, not reused), `CYCLE`, `LABELS`, `WHAT`, `TOP_MODES`, `STOWS_FLANKS`, `FLANK_LAW`, `usable(mode, cfg)` / `next_mode(mode, cfg=None)` (a TOP mode only with a top wing), `flanks_hidden(mode)`, `cruise_top(mode)` (the top wing a mode commands on a straight with no brake: True / False / None = the slot's mode; a stop held at v0), `pair(cfg)` (a matching pair), `top_fitted(cfg)`, `AirBrake.command(mode, ctl, V, veh, dt)` -> `Controls.wing_cmd` or None (AIR BRAKE: all three while the pedal is >= 0.30, in to 0.15, above 5 m/s, a flank that is not a matching pair left out; TOP `(False, False, active law)`; TOP FIXED `(False, False, True)`; TOP FIX+SIDE `(None, None, True)`), `AirBrake.g_changed(old, new, veh)` on every G press (the latches start afresh; `handing`: an active top wing leaving a TOP mode stays on the free path until its law triggers afresh) | `vehicle.TOP_HOLD / TOP_BRAKE_ON` lazily (TOP's law). Never pygame, never the physics |
| `drive/results.py` | the lap's results card (task 27): `card(res, book)` from the recorder's lap result (time, delta to the PB it was driven against, the sectors coloured purple / green / red, place, medal, `new_pb`, the class `key`), `view(card, age)` (None after `SHOW_S`), `drop` / `pulse` (the animation); task 32: `summary(card)` (the Settings row / the page's list), `page_rows(log)`, `LOG_N` cards kept a session | `records` lazily. Never pygame |
| `drive/drive.py` | main loop, CLI, scripted runs | everything |
| `drive/validate.py` | the whole acceptance suite | everything |

`garage.py`'s SELF-CHECK (and only its self-check) imports
`render.frame_budget_verdict`, to normalise its three page-draw budgets
against the machine's current speed rather than reimplementing the
normaliser. No cycle (`render` never imports `garage`) and nothing on the
interactive or acceptance path reaches it.

**AeroBO's engine has ONE door** (task 43's pivot, PLAN2 §3). The vendored
package `aerobo` is imported by `drive/aerobo_bridge.py` and by nothing else;
the bridge never imports pygame or `garage`. `design_jobs` imports neither the
bridge nor `aerobo` (it runs whatever `runner(emit, stop)` it is handed, and
imports only the standard library). `aerobo_models` imports the bridge and
`design_jobs` and never `garage`. The views reach the engine only through
the models (`am.bridge` for read-only helpers: the symmetric names, the
library point, the plans, the handoff split). `garage` reaches the models
lazily. Nothing in carsim edits a file under `aerobo/`: every adaptation
(slot families, carsim's air, the law) is a carsim-side subclass or a
function of AeroBO's outputs, and `aerobo_bridge`'s B26 row proves the
vendored tree unchanged by a whole check. The engine runs on a worker
thread, one at a time (`RunManager`); nothing on the worker touches pygame,
the models or the library -- it emits events, and the frame applies them.

**The designer is held to the car's limits** (task 41 wired into task 43's
designer). A slot's session reads the host's `car` (the garage's
`Garage.car`) and `unlimited` (`Settings.wing_limits`), and every number that
depends on the car comes from `bodies` through the bridge: the flank's span
row opens at the car's physical limit at the slot's height (`bodies.
span_ceiling`: the lower tip at its own ground clearance -- 1.50 m on the
Corsa at h 0.90, where the sill / roof fit gave 0.88) with its area row under
`area_ceiling`, both cut to AeroBO's AR >= 3; the top's span row is 1.2 x the
body's width, its deck and ride band `bodies.top_h_band`'s (the three stock
cars keep the garage's Corsa deck, as their slot bands do). Unlimited leaves
those default rows where they are and lets the player open them to 3x
(`OperatingPoint.size_caps`); a typed band past the ceiling is held at it and
`WingModel.bounds_overrides` clips it again at every build (the slot may have
moved). In Real mode `WingModel.commit` puts no wing past the limit on the
car -- this slot, or any other slot of its role carrying a wing of that name
unless the mirror overwrites it -- and says which slot and where Unlimited is;
in Unlimited it is saved and logged as filed apart (`bodies.over_limits`).
Its library writes carry task 39's save guard (a failed write is said, the
garage stays up). A law re-derived after the car page moved a slot
(`aerobo_models.rederive(spec, slot, car=)`) flies the car's own deck and band.
The lap AeroBO's lap-time objective and the mission's design speed are timed
on stays the Corsa's (`aero.mission` and `aerobo_bridge.car_spec`, as on
main).

**The END PLATE carries a section.** `vlm.Lattice` has always taken `plate_a`
and `plate_L0` and `wing.build_lattice` hardcoded them to a flat plate;
`WingSpec.plate_airfoil` (`""` = flat) now names a library section and
`build_lattice` / `analyse` / `spanwise` take `plate_polar=`. `plate_polar=None`
is **bit-identical** to the path every shipped wing was analysed on, and a wing
saved before the row decodes to `""`, so nothing published moves. Anything that
solves the lattice for a DESIGNED wing must pass it: `library.analyse_wing`
does, and the garage's wing optimiser does -- it did not, and the search flew
flat plates while the page drew cambered ones.

*The four paragraphs below describe `drive/aero/`'s own search and screen,
which the DESIGN page no longer calls (the pivot): its budgets are AeroBO's
`api.recommended_search` (sections 109 / 164 / 240, the wing 42 / 53 / 87),
its weights and gates AeroBO's V3 ones (`aerobo_bridge.WING_WEIGHTS` /
`PLATE_WEIGHTS` / `SCREEN_GATES`, the wing without `cdcr`, D9), and its gate
AeroBO's (below). They stay true of the modules, which keep their
self-checks.*

The SEARCH DEFAULTS are AeroBO's measured ones, not hand-picked numbers.
`drive/aero/optimize.py` carries its frozen `data/search_budget.json` payload:
the budget law `evals = 9.61 + 3.08 d` at the 95 % convergence target (with
its 10.7-evaluation residual RMS, because a law with that spread is a sizing
rule and not a prediction) and its two companions at 90 % and 99 %; and the
Sobol split `n_init = clamp(round(0.5 d), 4, 16)`, which that study ranks
first against 1x, 2x and 4x the dimension over 15 cases. `budget_for`,
`n_init_for` and `split_for` are what both the section and the wing pages
size themselves with. The STOP RULE is not ported: the payload gives the wing
class `patience 40, tol 0.002` and records that the airfoil class ships none.
The CONSTRAINED seed rule is not used either, and that is measured: a uniform
draw over carsim's section box is refused 42 % of the time on the top role and
36 % on the flank, where the study's constrained arm was written for a box in
which 4.7 % of draws fly.

THE DEFAULT CRITERION WEIGHTS DIVERGE FROM AeroBO's, with the reason measured.
Its `gdp-sweep` preset weights |cm| at 0.20; |cm| is lower-better, so it
rewards REFLEX -- AeroBO's own argument, which is why its `wing-trimmed` job
moves that weight for a surface whose moment something else carries. A car
wing's MOUNT carries it. Measured over the 39 shipped sections on all three
circuits, |cm| alone ranks the library backwards (Spearman -0.97 against the
lap), and removing it halves the lap cost of the section the screen picks.
The END PLATE's preset is keyed on the ROLE, because |camber| correlates
+0.96 with the lap on a flank plate and -0.96 on a top one; and |cm| is NOT
retired on a carsim plate, unlike AeroBO's, because carsim's plates are
lifting panels and may be cambered. The screening lift itself is AeroBO's
`REFERENCE_CL = 1.0`, a plain editable reference: reading it off the wing
instead censored 30 of the 39 sections.

`drive/garage_ui.py`'s `Nav` is a GATE, not a mark. `state(key)` returns
`done | ready | blocked | locked`, and `select()` REFUSES the last two and
keeps the reason -- which is what `gui/v3/app.py`'s own `select()` does
("a greyed-out node with no explanation is the thing this shell exists to
avoid"). `blocked` is *not yet* and opens when the step before it is
finished; `locked` is *not here* and no upstream work opens it. A section
group is finished by FITTING its section to the wing, never by optimising:
AeroBO's stage 2 docstring says the search is optional, and the endplate
group carries an explicit "fly FLAT plates" answer so that no gate makes a
design decision compulsory. (Since the pivot the DESIGN page's gate is
AeroBO's own, `DesignSession.state`: a section stage is finished by a
DECISION -- a ranked section taken, the search's winner taken, or the
family's own section kept (`SurfaceModel.decline`) -- and no section stage
gates the wing, which flies the family's own sections until one is chosen.)

On the MISSION and DESIGN pages the gate is AeroBO's, per STAGE
(`design_shell`, task 43). The navigator is `design_shell.CaeTree`, still a
`Nav` (every call above is kept, and the plain `select` still refuses a shut
step with its reason). The tree shows five stages -- `1 Mission`,
`2 Airfoil`, `2.8 Endplate`, `3 Wing`, `4 Results` -- each with a state glyph
decided in this order: the base state (`check_circle` done, `radio_button_unchecked`
ready, `lock` locked, `cancel` error when the mission's lap does not close),
then `pending` (running) while a screen, search, law or report of that
stage is live on AeroBO's worker (over done), then `play_circle` (active)
on the selected unfinished stage. A locked stage refuses a click or a key
with an info toast quoting its reason; INSIDE an unlocked stage every view
is reachable (`nav.select(view, force=True)`, what the tree, tabs, crumbs,
links and prev / next all go through) and a view with nothing to show draws
its empty state. Each stage row carries a chip naming what it holds -- the
circuit and surface, the wing's section and its t/c (also before the mission
is stated, from the slot's library wing; after it, `NACA 2412 (the family's
own)` until a section is taken), the plates' section (or `fences — no plate
section`), the wing's searched dimension and objective (`14-D · efficiency`),
the result's best score in AeroBO's units (`15.43 CZ/CD`, or `lap 50.289 s`)
-- so the chosen airfoil is on screen in every view. The tree EXPANDS ON
SELECT: a fresh garage shows `1 Mission` open and
the rest collapsed; selecting a stage opens it for good, the twisty only
toggles, and nothing collapses a stage by itself. A collapsed stage's steps
are not drawn and not hit-recorded: `nav._hits` holds the step rows drawn
this frame (so `_hits[2]` is `af.section` once `2 Airfoil` is open and
revealed), and stage rows, twisties and the mission's rows are in
`nav._node_hits`.

`Nav`, `ParamList` and `ListBox` record the geometry they drew and hit-test
it, so every page is driven by the mouse as well as the keyboard. On the car,
airfoil and library pages the mouse does what it always did: a click on the
left or right half of a `< value >` is a LEFT or a RIGHT, an action row
selects on the first click and fires on the second, the wheel moves the
selection, and the navigator's gate refuses a click exactly as it refuses an
arrow key. On the MISSION and DESIGN pages every control is a `cae.form.Form`
widget and behaves as AeroBO's: a BUTTON FIRES ON THE FIRST CLICK; a weight
or a fraction is a SLIDER (a click sets the snapped value under it, a drag
applies once a frame, and a `costly` row -- anything that re-flies the
lattice -- only on release); a number field shows its step arrows on hover
or keyboard focus and takes typed digits; a choice is a toggle group or a
drop-down; a flag is a switch. Only button 1 clicks (2 and 3 do nothing;
4 / 5 are the wheel's echo), the wheel (`MOUSEWHEEL` only) scrolls the pane
under the cursor and never moves a selection, SHIFT+wheel scrolls a wide
table sideways. The `Form` records every control of the view in draw order
(`order`, the keyboard's path, off-screen ones included: focusing one
scrolls it into view) and hit-records only the visible ones (`_hits`). A
control the live run's lock covers draws disabled and says why on hover.

`drive/aero/screen.py` is AeroBO's `airfoil_select.score_candidates` in
carsim's units, and it owns the one thing the garage was asking in the wrong
place. The seven criterion weights (`ldcr`, `clmax`, `cm`, `ldmax`, `cdcr`,
`thick`, `astall`) are the LIBRARY SCREEN's question, not the shape
optimiser's: a designer has no opinion about a CST weight before a search has
run. The screen measures a FROZEN 0-100 band over the sections that survived
its gates, and the three composite objectives -- `composite`,
`composite, none below the seed`, `lift the weakest criterion` -- maximise the
same number the shortlist was chosen on. A composite run is refused until the
screen has produced that band, because a live min-max would move with the
population and the score would not be a fixed function of the shape.

THE GATES ARE carsim's OWN NUMBERS and are OFF by default. AeroBO screens an
aircraft library at `t/c >= 0.15`, `|cm| <= 0.08`; measured over carsim's 34
race sections that pair admits ONE, and the |cm| ceiling deletes exactly the
high-lift sections a downforce wing exists for (S1223 0.348, S1210 0.306,
CH10 0.275, E423 0.247). The values a gate comes back ON at are this library's
own quartiles.

`drive/aero/blend.py` is how the WING AND THE END PLATE MEET, ported from
urop-bo-aero's `geometry.winglet_turn_angle` / `winglet_path`,
`vlm.transition_ramp` and `junction.py`, and reproducing their geometry TO THE
BIT (`blend.self_check` pins twelve of AeroBO's own tip positions, three turn
laws by four blends). `vlm.Lattice` used to bolt the plates on at a right
angle and change everything about the surface in ONE STEP at the junction
panel -- the wing's cambered `alpha_L0` on one side of an edge and the plate's
on the other, the wing's tip twist likewise. A blend spends `plate_blend` of
the plate's arc turning out of the wing plane, and the plate's section and toe
ramp on `psi/phi`, the turn angle normalised by the cant, so the surface
finishes becoming the plate exactly where it finishes turning into it.

ARC LENGTH IS THE INVARIANT: every (blend, shape) plate of the same `plate_h`
has the same developed length and the same wetted area, so the row compares
SHAPE and not size. What a blend trades is TIP HEIGHT for OUTBOARD REACH, and
THE WING PAYS FOR THAT REACH OUT OF ITS OWN SPAN (`wing.plate_flown`): the
span row is what the car is allowed to be wide -- a flank panel's span is its
vertical extent, down to the car's own ground clearance (task 41,
`bodies.span_limit`), a top wing's is 1.2 x the car's width -- and
a plate that leans out of it has to come from somewhere. That accounting is
load-bearing, not bookkeeping. Without it CZ/CD climbs monotonically to full
blend against a junction credit that saturated at about 0.1, so nothing in the
model opposes curling the plate into a quarter-round winglet; measured here,
the unpaid variant reads -0.032 s at full blend and the paid one +0.022 s.

`blend.junction_report` is the other half, and it is an ADD-ON that is OFF by
default. A lifting-surface method values a corner only through the wake line
it draws, so the interference drag of two surfaces meeting at an angle -- the
whole reason to blend -- is invisible to it. Hoerner's unfilleted-junction
correlation is charged at the two corners with a fillet credit for the blend
radius; the credit is a CALIBRATED SHAPE and not a measurement, which is why
it is a row, why it is reported beside the uncredited number, and why raising
the blend off zero is what switches it on. The junction member is the PLATE,
so a plate with no section is bare sheet, below the correlation's own root,
and charges nothing rather than being extrapolated.

MEASURED, 2 roles x 3 circuits x 3 shapes x 6 blends: on this car a blend
never pays at equal car width. The curve does turn over inside the range --
the credit saturates near blend 0.1, which is where the cost is least (+0.0013
s on the flank) -- but the span it costs is worth more than the corner it
smooths, and it reaches +0.022 s at full blend. That is an answer about this
car at these plate depths, not a defect: the same rows on AeroBO's 3x-chord
endplate turn over below zero, because its junction charge is far larger.

NOT PORTED, and stated: the WING-SIDE ARC (AeroBO can start the turn inboard
of the tip and bend the wing's own panels into it). The law takes the
argument, `vlm.Lattice` passes 0. AeroBO wants one because a device-side blend
has R <= h/cant, "a fraction of a tip chord", so its fillet credit saturates
immediately -- but that is about ITS scale: a carsim flank plate at h 0.12 m
and blend 0.6 already has R/c 0.102, past `FILLET_FULL_R_OVER_C`, so there is
nothing for a wing-side arc to buy. Also not ported: a variable or signed
CANT (carsim's plates stand normal to the wing), and AeroBO's chord ramp
(carsim's plate carries the wing's tip chord, so there is no chord step to
ramp over).

`drive/aero/mission.py` imports `corsa_c` and `qss` at module level, and
nothing else from the simulator. That is the one widening of the aero
package's import rule and it is deliberate: the mission is a lap of THIS car,
so the car's parameters and the published cornering model are the thing being
extended, and at zero downforce `mission.residuals` and `mission.max_ay`
reproduce `qss`'s bit for bit (`mission.self_check`). The package still never
imports `drive.track`: a circuit reaches it as a `TrackProfile` built from a
`Track`'s own `Seg` list, and the only `from .. import track` is deferred
inside `mission.self_check` and `section.reference_problem`, neither of which
is on the interactive or acceptance path. `drive/aero` still NEVER imports
pygame.

`vehicle.py` NEVER imports `track`, `render`, `input`, `aero` or pygame: the
designed wings reach it as two frozen dataclasses of numbers (`DevAero`,
`TopAero`, section 4) that the garage builds. `drive/aero` NEVER imports
pygame; the 1 kHz step never calls a solver.
`render.py` NEVER touches physics state except read-only.

## 2. `drive/tyre.py`

Source data: `tyre_data/TNO_car205_60R15.tir` (FITTYP=62, MF-Tyre/MF-Swift 6.2).
This file **is** the provenance of `qss.TYRE`: its `PDY1=0.8784`,
`PDY2=-0.06445`, `FNOMIN=4000` give `mu_y(2477) = 0.90294`, slope `-16.11e-6/N`,
which is `qss.TYRE = dict(mu_ref=0.903, Fz_ref=2477.0, s=-16.1e-6)` rounded.

**Do NOT rescale FNOMIN or LFZO.** `LFZO = LMUX = LMUY = LKX = LKY = 1.0`.
Rescale geometry only: `R0 = 0.2915`, width `0.175`, aspect `0.65`,
rim radius `0.1778` (175/65R14).

*Amended by task 41 -- ONE declared exception.* FNOMIN is never rescaled, and
LFZO is not rescaled on any car tyre: every CarSpec with
`tyre_lfzo_f == tyre_lfzo_r == 1.0` (the Corsa, the MX-5, the 540i, the
Express) reads the file tyre with only its geometry overridden, and
`cars.self_check`, `vehicle` T41a and the drive self-check's per-car loop
assert that the three stock cars run the unscaled file tyre and `qss.TYRE`
itself. A car whose wheel loads lie outside the file's load range (`FZMAX`
10 kN) because it runs on TRUCK tyres may DECLARE a per-axle load scale
`lambda = published rated load x 0.86 / FNOMIN`; a twin pair is one tyre at
twice a single's dual-rated lambda. `tyre.tyre_for(..., lfzo=lambda)` then
scales `LFZO`, `FZMAX` and `CFX` / `CFY` together, which makes the tyre
exactly `lambda x` the file tyre at `Fz / lambda` for Fx, Fy, Mz and the
stiffnesses (tyre self-check step 13, to 1e-12). Its utilisation reference is
`qss.tyre_ref(lambda) = {mu_ref, lambda Fz_ref, s / lambda}`, the same law with
the load axis stretched. The only car that declares a scale is the Citaro
bus: 6.644 front, 12.233 rear pair. Its truck-tyre grip is `mu_scale` 0.80, a
labelled calibration like the MX-5's and the 540i's.

**Symmetrisation is mandatory.** Zero these 13 camber-EVEN shifts:
`PHY1 PHY2 PVY1 PVY2 PHX1 PHX2 PVX1 PVX2 QHZ1 QHZ2 QDZ6 QDZ7 QSX1`.
Keep every camber-ODD shift (`PVY3 PVY4 QHZ3 QHZ4 QDZ8..QDZ11`) and keep
`RHX1 RHY1 RHY2`. After this, `peak|Fy|(Fz) == mu_y(Fz)*Fz` to 5 s.f. and the
model is exactly odd in `(kappa, alpha)`.

```python
def load_tir(path: str) -> dict[str, float]
class TyreModel:
    def __init__(self, tir_path, *, symmetrise=True, R0=0.2915, width=0.175,
                 LFZO=1.0, LMUX=1.0, LMUY=1.0, LKY=1.0, LKX=1.0,
                 CFX=381913.4, CFY=157632.5): ...
    def evaluate(self, Fz, kappa, alpha, gamma=0.0, mu_scale=1.0) -> tuple[float,float,float]
        # -> (Fx, Fy, Mz) in the WHEEL-CARRIER frame, N, N, N.m
    def stiffnesses(self, Fz, gamma=0.0) -> tuple[float, float]      # (Kxk, Kya); Kya < 0
    def relax_lengths(self, Fz, gamma=0.0) -> tuple[float, float]    # (sigma_kappa, sigma_alpha)
    def mu_y(self, Fz, mu_scale=1.0) -> float
    def mu_x(self, Fz, mu_scale=1.0) -> float
    def peak_fy(self, Fz, mu_scale=1.0) -> float                     # analytic mu_y*Fz
CORSA_TYRE: TyreModel      # module-level singleton, parsed ONCE at import
def tyre_for(tir_path=None, R0=0.2915, width=0.175) -> TyreModel   # cached per size
def mu_curve_matches(tyre, ref=None, loads=(...)) -> bool
def self_check() -> None   # python3 -m drive.tyre
```

* **`tyre_for` is how a different car gets a different tyre size.** It is a
  cache keyed on `(abspath, R0, width)` pre-seeded with `CORSA_TYRE` under the
  Corsa's own key, so `tyre_for()` and `tyre_for(the Corsa's three numbers)`
  return the SAME OBJECT and the default stays bit-for-bit. Call it at
  construction time; never from `step()`. `R0` is the only geometric term any
  equation reads — `width`, `aspect` and `rim_radius` are provenance.
* `mu_curve_matches(tyre)` is the assertion that licenses §4's rule that
  `util_f/util_r` come from `qss.fy_max` with `qss.TYRE`: those are the
  *Corsa's* `mu_ref/Fz_ref/s`, and referencing another car to them is only
  legitimate because there is **one coefficient set** in `tyre_data/` (all
  five loadable `.tir` files are identical except for geometry) and `mu(Fz)`
  contains no geometry. `Vehicle.__init__` asserts it per car. The day
  somebody adds a genuinely different `.tir`, that assertion fires and a
  `TYRE`-shaped dict must be derived from the new tyre instead.

* `evaluate` must be pure and allocation-light. Unpack coefficients into
  `__slots__` attributes; no dict lookups in the hot path. Target < 3 µs/call.
* `Fz <= 50 N` → return `(0.0, 0.0, 0.0)`.
* `Fz` clamped to `[0, 10000]`.
* `evaluate(Fz, 0, 0, 0, 1)` must be **exactly** `(0.0, 0.0, 0.0)`.
* Odd symmetry: `evaluate(Fz, -k, -a) == -evaluate(Fz, k, a)` to 1e-9.
* **Relaxation is NOT done here.** `vehicle.py` owns the transient slip states
  because the integrator ordering is load-bearing (§4). `tyre.py` exposes
  `relax_lengths()` and `stiffnesses()` so `vehicle.py` can do it. Do not
  export a `step_contact` that integrates state.
* `Mz` sign: positive (restoring) for positive alpha; `vehicle.py` flips it in
  reverse, not `tyre.py`.
* The `sigma` floors are mandatory: `sigma_kappa >= 0.05`, `sigma_alpha >= 0.10`.
* `mu_scale` must be fused into `LMUX`/`LMUY` at the top of `evaluate` and used
  in **every** one of `Dx, Dy, SVyg, DVyk, Bt, Br, Dr` — `Bt` and `Br` DIVIDE
  by it.

Full equation list (E1–E39, R1) and every golden number: `../specs/tyre.txt`.

## 3. `drive/powertrain.py`

```python
@dataclass
class PowertrainParams: ...        # classmethod from_car(car: CorsaC, power_scale=1.0)
@dataclass
class PowertrainState: ...         # omega_e, theta_slip, gear, stalled, shift_*
@dataclass
class PtInput:                     # named PtInput here, NOT DriverInput
    throttle: float = 0.0; brake: float = 0.0; clutch: float = 0.0
    handbrake: float = 0.0
    shift_up: bool = False; shift_dn: bool = False    # edges, consumed
    starter: bool = False; auto_gearbox: bool = True
    auto_clutch: bool = True       # the box works the clutch (launch / anti-stall
                                   # assist, downshift blip, self-restart); False
                                   # = H-pattern, the pedal is the only clutch
    tc_scale: float = 1.0          # vehicle._tc's gain on the ENGINE LOAD only;
                                   # `throttle` stays the driver's pedal
@dataclass
class PowertrainOutput:
    T_drive: tuple    # (4,) N.m at the wheels, FL FR RL RR. The DRIVEN pair
                      # carries it and the other pair is 0.0 -- front-driven
                      # (RL=RR=0) on the Corsa and every acceptance number,
                      # rear-driven (FL=FR=0) on a drive_layout='rwd' car.
    T_brake: tuple    # (4,) N.m MAGNITUDE, always >= 0
    I_w_eff: tuple    # (4,) kg m^2, gear-dependent on the front
    rpm: float; gear: int; T_eng: float; T_clutch: float
    clutch_slip: float; P_wheel: float; stalled: bool; on_limiter: bool
    load: float = 0.0 # the engine load fraction applied this step (HUD, sound)

def step(p, s, inp: PtInput, omega_w, Fz, Fx, v_x, dt) -> PowertrainOutput
def brake_torques(p, brake, handbrake) -> tuple[float, float]   # (front/wheel, rear/wheel)
def driven_pair(p) -> tuple[int, int]        # (0,1) front | (2,3) rear
def I_w_bare(p, driven=True) -> float        # one wheel's own inertia
def I_w_driven(p, g) -> float                # ... with the driveline reflected
def diff_split(p, T_axle, omega_l, omega_r) -> tuple[float, float]
def wot_torque(p, n_e) -> float
def wot_power(p, n_e) -> float
def kmh_per_1000rpm(p, g) -> float
def self_check() -> None
```

* `step()` **returns** wheel torques and effective inertias. It does **not**
  integrate wheel speeds — `vehicle.py` does, because `Fx` couples the wheels
  to the chassis.
* Clutch is a **saturated PD** (`K_c=800`, `C_c=15`, cap `200 N.m`,
  anti-windup on `theta_slip`), never a discrete lock/unlock state machine.
* `eta_eff = 0.86` when `T_c >= 0`, `1/0.86` when `T_c < 0` (losses always
  oppose power flow).
* Engine torque: 19-breakpoint PCHIP (`scipy.interpolate.PchipInterpolator`),
  built **once at import**. Passes exactly through 110 N·m @ 4000 rpm and
  55.0 kW @ 5600 rpm.
* Reflected inertia architecture (a): integrate `omega_e` separately and put
  **only** `0.5*I_TRANS*N_TOT^2*eta` into each front wheel. Never also add
  `I_ENG` to the wheels.
* `C_RR = 0.012` charged as `C_RR*Fz_i*R_ROLL` per wheel. No separate chassis
  rolling-drag term anywhere in the codebase.
* Brakes: `KBF=1.6187e-4`, `KBR=5.4154e-5` N·m/Pa, `P_MAX_LINE=110e5`,
  proportioning knee `30e5` Pa, slope `0.30`. Front must lock before rear in
  both dry and wet — this is a hard test.
* **Three driver models share one shift machine** (`update_shift`, which now
  returns `(clutch_pedal, throttle_scale, blip_load)`):

  | mode | `auto_gearbox` | `auto_clutch` | the driver |
  |---|---|---|---|
  | auto | True | True | steers and pedals |
  | manual | False | True | + shifts (edges); cannot stall |
  | clutch | False | False | + launches on the pedal; can stall |

  The launch / anti-stall assist follows `auto_clutch`, **not** `auto_gearbox`.
  The gear change itself always runs declutch → gate → engage (0.70 s): that
  is the driver's own foot during a shift, in every mode.
* **Launch assist band.** The assist is proportional on engine speed over
  `[max(n_stall + 100, n_tgt − n_launch_band), n_tgt]`, `e = 1` at the target
  (continuous with the locked branch `n_input > n_tgt`), `n_launch_band =
  400 rpm`. The clutch's torque balance fixes the slip equilibrium at
  `e = (T_eng / T_cap)^(2/3) ≈ 0.63`, so the band sets the droop: the old band
  (550 rpm .. target, 1850 wide at WOT) parked the engine at ~1700 rpm
  transmitting ~85 N·m, 0–50 km/h 5.64 s against the rig's 5.12; 400 holds it
  ~150 under 2400 (5.46 s). Narrower acts as a stiff damper on the engine DOF
  (`1.5·T_cap·√e / band`, 5.7 N·m·s/rad here) and couples into the 11.3 Hz
  driveline mode. The zero-throttle anti-stall band (550..850) is unchanged.
* **Per-car brakes.** `brake_coeffs(car) -> (kbf, kbr, p_max_line, t_hb_max)`.
  `corsa_c.brakes` still says MISSING and still is — nobody publishes pad mu
  or a torque split — but the HARDWARE is documented, so `CarSpec` carries
  disc/drum diameters, whether the rear is a disc or a drum, and the piston
  diameters, and this turns them into N·m/Pa with **the same formulas the
  BRAKE BLOCK uses** (`BRK_MU_PAD`, `BRK_CSTAR`, `BRK_PAD_H` stay estimated in
  one place). A car matching the Corsa's geometry gets the BRAKE BLOCK's own
  constants back, by identity. `p_max_line` is solved to hold
  `BRK_AUTHORITY = 1.479/1.041` — the same 42 % lock-up over-authority on
  every car, which is what the lock-order tests and the locked-wheel sled
  depend on — and `t_hb_max` scales with the rear axle's own locking torque.
  **Front still locks before rear on every car, dry and wet**, with a larger
  margin than the Corsa's +7.7 % (MX-5 +102 % dry / +155 % wet, 540i +94 % /
  +116 %).
* **Per-car steering lock.** `vehicle.car_lock_rad(car)` =
  `steer_turns * 180 / steer_ratio` in rad, and `Vehicle.lock_rad` /
  `Vehicle.dev_deadband` (5 % of it) come from the car. The Corsa's documented
  2.9 turns at 16.0:1 gives exactly `LOCK_RAD`, asserted at import in
  `_check_reference` — so `LOCK_RAD` is a special case of the function, not a
  rival truth. `input.py`'s `DELTA_LOCK_DEG` is unchanged: the input layer is
  handed a lock by the caller and every validation path bypasses the aid.
* **Per-car engine curve.** `engine_curve(car) -> (rpm_bp, nm_bp, orpm_bp,
  onm_bp)` builds THIS engine's WOT and overrun curves from five anchors on
  the car (`n_peak_torque`, `n_peak_power`, `n_idle`, `n_cut`,
  `displacement`) plus its published `T_max` / `P_max`. **A car whose anchors
  match the Corsa's gets the module constants themselves** — the same tuple
  objects — so the Z12XE curve is identical to the last digit.
  The built curve passes through **both** published points exactly: the rpm
  axis is warped through the knots (0, peak torque, peak power, cut),
  piecewise linear and monotone, and the torque is scaled to `T_max` then
  corrected by a factor ramping 1 → c between the peaks so that
  `T(n_peak_power) == P_max / omega_peak_power`. Overrun scales with
  displacement (`FMEP*Vd/(4*pi)`). `from_car` also takes the rev cut, idle,
  `n_overrev`, the whole shift schedule (as a FRACTION of the cut — leaving
  the Corsa's 6050 upshift on an engine that revs to 7000 short-shifts it 950
  rpm below its own power peak) and `T_clutch_cap` (held at the Corsa's
  1.82× ratio to its own peak) from the car.
  **`CarSpec.engine_scale` is RETIRED and is 1.0 on every car** — it was a
  bodily multiplier on the Corsa's curve, and with a real per-car curve it
  would double-count.
* **The driven axle.** `PowertrainParams.driven` is `'front'` or `'rear'` and
  `from_car` reads it off `CarSpec.drive_layout`. **`'awd'` is REFUSED with a
  `ValueError`**, not silently treated as one of the two: this driveline has
  one clutch, one gearbox and one open diff, and a centre differential with a
  torque split is physics it does not have. Everything that pairs with the
  torque follows the driven pair — `T_drive`, the reflected transmission
  inertia (`I_w_driven`, which reflects into the 0.73 rear wheels instead of
  the 0.76 fronts on a RWD car), the input speed `omega_drv`/`omega_in` the
  clutch and the shift scheduler read, `P_wheel`, `I_w_eff`, and
  `accel_run`'s traction cap. `diff_split` was already axle-agnostic.
  `PowertrainState.I_w_front_eff` keeps its historical name (the spec's golden
  table quotes it) and means the DRIVEN axle.
  **The load transfer needs no new code**: `dFz_x_demand = (sum(Fxb_i)*h_cg)/L`
  is already general, so acceleration transferring load rearward UNLOADS a
  front-driven car and LOADS a rear-driven one on its own. Measured, WOT from
  10 m/s: driven-axle load −504 N (Corsa, FWD), +765 N (MX-5), +2058 N (540i).
  `Vehicle._tc` reads the DRIVEN pair's transient slip, with the same
  open-diff `1 - 2*share` cut floor.
* **`power_scale`** (`from_car(car, power_scale)`, the drive's Engine setting):
  `nm_bp` scaled as a whole and `T_clutch_cap = T_CLUTCH_CAP_STOCK ×
  power_scale` (a 2× engine on the stock 200 N·m clutch would slip at its own
  peak); overrun curve, idle governor, limiter and ratios untouched. 1.0 in
  every rig and every script. `PtInput.tc_scale` multiplies the mapped load
  (`load = throttle_map(pedal × thr_scale) × tc_scale`) so the scheduler's
  `N_UP = n_up_a + k·throttle` and the assist's `n_tgt` still see the pedal —
  a TC that cut the pedal made the auto box upshift at 3600 rpm and hunt.
* **The automatic and the limiter / the stop (task 45, owner-approved physics
  change).** `n_up_schedule` is capped at `n_cut - (1 + N_UP_SOFT_MARGIN) ·
  n_soft` (`N_UP_SOFT_MARGIN = 0.25`; the Corsa's 1-2 WOT line 6150 -> 6050,
  the bus's 2480 -> 2439.5), so the line never sits in the soft limiter's fade;
  `accel_run` uses the same schedule. Inside the 0.8 s lockout an UPSHIFT is
  still taken when the engine is in the fade band at throttle > 0.9 (road
  speed agreeing as always). `_assist_pedal(p, inp, g, v_x, n_e)` is the
  launch / anti-stall assist plus, on the automatic only, a **brake hold**
  (brake > 0.3, throttle <= 0.02, the gear's input speed under idle: clutch
  fully open, the engine idles in gear) and a **locked-wheel release** (road
  speed above the target but the engine under idle: the clutch opens through
  the anti-stall band). On the automatic the engage ramp never closes the
  clutch further than `_assist_pedal` would. Manual and clutch modes are
  unchanged. V20's trace does not reach an upshift line, the limiter or a
  stop: its sha is unchanged (`ed41b7f781e959ea`).
* **Rev-match blip** (`auto_clutch` only): through the gate and engage phases of
  a DOWNSHIFT the engine is fuelled to `min(n_input(target gear), n_cut − 150)`,
  proportional over `n_blip_band = 800 rpm`, dead inside `n_blip_min = 150`.
  An upshift never blips. Measured 3rd→2nd at 15 m/s: engine 3960 rpm at
  engagement against a 4274 target (2595 without), peak clutch slip 269 rpm
  against 1644.
* **Stall → restart.** A stalled engine is cranked by `starter`, by
  `auto_clutch`, or by the clutch pedal fully in (`clutch_engagement <= 0`).
  While cranking it is **fuelled** (load ≥ 0.30, `fuel = 1`) plus `t_start`
  below `n_crank = 400`; it is running again above `n_fire = 500`. The
  previous rule (starter torque only, fuel off while stalled) left the engine
  at 400 rpm for ever — every stall was permanent.

Full equations, all 26 validation numbers: `../specs/powertrain.txt`
(the module self-check now carries 82).

## 4. `drive/vehicle.py` — integrator ordering is load-bearing

State (integrated): `X, Y, psi, u, v, r, phi, p, omega[4], kx[4], ky[4],
dFz_f, dFz_r, dFz_x`, plus `PowertrainState` held by composition.
`kx` is the transient slip ratio, `ky` the transient **tan** of slip angle.

**The order inside one physics step is fixed and must not be "tidied up":**

1. contact-point velocities from the **current** state
2. per-corner `Fz` from the **lagged `dFz_*` states** (no algebraic loop)
3. tyre coefficients (`sigma_x`, `sigma_y`, stiffnesses) from `Fz`
4. **advance `kx`, `ky`** from the **current** `omega` — exact exponential:
   `a = -|Vx|/sigma`, `c = Vs/sigma`,
   `k <- k*exp(a*dt) + (c/a)*(exp(a*dt) - 1)`, falling back to `k + c*dt`
   when `|a*dt| < 1e-6`. **Never form `kappa = Vsx/Vx`.** Clamp
   `kx∈[-1.5,1.5]`, `ky∈[-3,3]`.
5. tyre forces from the **new** `kx`, `ky` via `CORSA_TYRE.evaluate(...)`
   with `kappa = kx`, `alpha = atan(ky)`
6. Besselink low-speed damper, then the friction-ellipse cap at `ECAP = 1.05`
7. **implicit** wheel-spin update (damper term on the LHS):
   `omega <- (omega + (dt/I)*(T_drive - T_brake*tanh(omega/1.0) - R_e*Fx_t
   - C_RR*Fz*R_e*tanh(omega/1.0) + kv*R_e*Vx)) / (1 + dt*kv*R_e^2/I)`
8. body forces/moments → `u, v, r` (semi-implicit Euler, **keep `+v*r` and
   `-u*r`**), then `phi, p`, then pose `X, Y, psi` from the NEW velocities
9. load-transfer states, deploy state

Reversing steps 4 and 5/7 diverges at every dt below 8 ms at 3 m/s — measured.

**Load transfer** (must be algebraically identical to `qss.py`):

```
dFz_tot_demand = (SFy_tyre*h_cg + F_dev*(h_cg - h_w)) / t_bar     t_bar = 1.42450
dFz_x_demand   = (sum(Fxb_i) * h_cg) / L
two-rate split (LT = LPF(dFz_tot_demand, tau_roll=0.09)):
  ddFz_f/dt = (0.101*demand + 0.639*LT - dFz_f) / tau_LT           tau_LT = 0.02
  ddFz_r/dt = (0.221*demand + 0.039*LT - dFz_r) / tau_LT
  steady state -> 0.740 front / 0.260 rear == roll_dist_f
ddFz_x/dt = (dFz_x_demand - dFz_x) / tau_pitch                     tau_pitch = 0.12
Fz_FL = max(m*g*wdist_f/2 - dFz_x/2 - dFz_f, 0)      # + to the RIGHT wheels
Fz_FR = max(m*g*wdist_f/2 - dFz_x/2 + dFz_f, 0)
Fz_RL = max(m*g*(1-wdist_f)/2 + dFz_x/2 - dFz_r, 0)
Fz_RR = max(m*g*(1-wdist_f)/2 + dFz_x/2 + dFz_r, 0)
```

`SFy_tyre` is the sum of **tyre** body-frame lateral forces only — it must NOT
already contain `F_dev`, or the `h_w` sensitivity is doubled.
`roll_dist_f = 0.74` is a **calibration constant**, not a derived one. Do not
"fix" it from the roll centres (that gives 0.51 and breaks everything
downstream in `qss`/`crossover`/`ledger`).

**Per-car scaling of the calibrated blocks** (`CarDerived`, `car_derived(car,
cfg)`). Six numbers used to be Corsa C module/config constants and now follow
the fitted car: `h_ra`, `I_roll`, `Cphi`, the four `lltd_*` shares and
`Y_DEV`. The rule is

```
value(car) = value(Corsa) * ( hat(car) / hat(Corsa) )
```

`hat` being the cheap bottom-up estimate of that quantity (roll centres
interpolated at the CG station; `Ixx - m_us*(t/2)^2 + m_s*h_r^2`;
`sqrt(Kphi*I_roll)` at a fixed `zeta_roll`; the instantaneous geometric +
unsprung shares; half the mean track). The **absolute level stays the Corsa's
calibration and only the change between cars is derived** — which is the
honest statement, because the two other cars' whole suspension block is `est`.
It is also bit-for-bit by construction, not by luck: when `car`'s fields equal
`cars.CORSA_C`'s, `hat(car)` and `hat(Corsa)` are the same float, the ratio is
exactly `1.0`, and `x*1.0 == x`. `vehicle._check_reference()` raises at import
if that ever stops being true (a bare `assert` would vanish under `-O`), and
`validate()`'s **T21** prints it.

`roll_dist_f` is **NOT** scaled — it stays `0.74` on every car, for the reason
above plus two more: `Kphi_f/Kphi_r` is `est` on all three cars, so a per-car
LLTD would be a guess dressed as a measurement; and the bottom-up route
provably gives 0.51 for the one car that has data. What does follow the car is
the split of that same 0.74/0.26 between the instantaneous and the elastic
path (the roll centres are per-car), and `lltd_geo_* + lltd_roll_*` is held at
`roll_dist_f` **to the last bit** by writing the roll share as a difference
from the reference pair.

`Vehicle.__init__` also builds `self.tyre = tyre.tyre_for(car.tyre_file,
car.tyre_R0, car.tyre_width)` (§2) — `CORSA_TYRE` itself for the Corsa and for
a plain `CorsaC`, which has no `tyre_*` fields — and sets
`self.tyre_ref_ok = tyre.mu_curve_matches(self.tyre)`, the assertion that
licenses the `qss.TYRE` rule below for a non-Corsa car. `_tyre_eval` takes the
tyre as a trailing argument defaulting to the singleton.

**Added mass** (`cars.PointMass`, `cars.with_masses(car, pts)`,
`cars.ballast_point(car, kg, where)`). Ballast and the garage's fitted wings
reach the physics as a **new `CarSpec`**, never as a patched `m`: the first
moments move `wdist_f` (hence `a`, `b` and every static wheel load) and
`h_cg`, and the parallel-axis theorem moves `Izz / Ixx / Iyy`; `m_s` carries
all of it (nothing a driver adds bolts to an unsprung hub). Springs, roll
centres, wheelbase, track, tyres, gearing and `CdA` are properties of the car
and are untouched, so a ballasted car rolls more and accelerates less on its
own. `with_masses` returns **the same object** at zero added mass, which is
what makes the ratio above exactly 1.0. Stations are quoted from the axles
(`nose` = `a + 0.30`, `seat` = the CG at `h_cg`, `floor` = `-b` at 0.30 m,
`boot` = `-(b + 0.25)` at 0.65 m) so they mean the same thing on every
wheelbase; `seat` is the control case and moves the mass and provably nothing
else.

**Roll** (visual + roll-camber only; the TOTAL transfer above is already a
ground-plane statement and must not be rebuilt from spring forces):
`I_roll = 342.8`, `Kphi = 36669 N.m/rad`, `Cphi = 2482 N.m/(rad/s)` (ζ=0.35),
`h_ra = 0.160`, `h_r = h_cg - h_ra = 0.390`. Solve the coupled 2×2 with `a_y`:

```
Mx_ext = m_s*g*h_r*sin(phi) - Kphi*phi - Cphi*p - F_dev*(h_w - h_ra)
[ m        -m_s*h_r ] [a_y ]   [SFy   ]
[ -m_s*h_r  I_roll  ] [pdot] = [Mx_ext]
```

**Steering** (`vehicle.py` receives an already-ramped, already-limited road-wheel
`delta` from `input.py` and applies only the physics):
* Ackermann fraction `A_ack = 0.60` about the bicycle angle.
* Front compliance steer: `delta_F* -= eps_f * Fy_f_prev`, `eps_f = 5.40e-6 rad/N`.
  This is what makes the linear-range understeer real (`V_char ≈ 25 m/s`).
* Rear roll steer `c_rs = 0.0` by default. `corsa_c.rollsteer_r = 1.0` is wrong
  in sign and magnitude — never implement it literally.

**Aero and the flank wing**:
```
q = 0.5*rho*V^2 ;  beta = atan2(v, max(u, 1.0))
D_aero  = q * CdA                                    # CdA = 0.66, at h_aero = h_cg
Fy_body = -q * A * Cs_psi * beta ; Mz_body = Fy_body * x_cp   # A=2.01, Cs_psi=2.2, x_cp=+0.30
sgn_dev = sign of the STEERING COMMAND (deadband 5% of lock, 0.3 s hold)
          — NEVER sign(beta) and NEVER sign(v)
alpha_dev = delta_dev_geom + sgn_dev*(-beta_dev)
beta_dev  = atan2(v + r*x_w, max(u, 1.0))            # cfg.dev_curved_flow True (DEFAULT)
          = beta                                     # False: degrade to qss's model
CL_dev  = clamp(CL0 + dCLda*alpha_dev, 0, 1.6)       # dCLda = 2.47 /rad; 0 in parity mode
dep     = smoothstepped deploy fraction over 0.45 s
F_dev   = sgn_dev * dep * q * S_DEV * CL_dev         # +y = inward for a LEFT turn
D_dev   = dep * q * S_DEV * CL_dev / 3.2
y_dev   = -flank*sgn_dev*Y_DEV                       # Y_DEV = 0.72; flank = +1 OUTER (default), -1 INNER
Mz_dev  = F_dev*x_w + y_dev*D_dev                    # = F_dev*x_w - sgn_dev*Y_DEV*D_dev on the outer flank
```
The old form `Mz_dev = F_dev*x_w - D_dev*y_dev*sgn_dev` squared the sign and
gave `+0.72*D_dev` in BOTH directions — a term that does not mirror, so the
car was not left/right symmetric on paper even though the code is. `Mz` from
the panel is `x*Fy - y*Fx` with `Fx = -D_dev` at `y = -sgn_dev*Y_DEV`, i.e.
`F_dev*x_w - sgn_dev*Y_DEV*D_dev`, which is what `vehicle.py` implements and
has always logged as its own DEVIATION. Checked: left-turn fin
`177.389*0.97 - 0.72*55.434 = 132.155` == the reported `Mz_dev`, and the drag
term yaws the car OUT of the turn, which a force on the outer flank must.

**The curved-flow term.** A body-fixed point at `(x_w, y_dev)` moves at
`V + omega x r`, and with `omega = (0,0,r)` that is `(u - r*y_dev, v + r*x_w)`,
so the panel's OWN flow angle is `atan2(v + r*x_w, u)` and not the body's
`beta`. The `y_dev` term only perturbs the axial component and is second order
in the angle. The sign is the opposite of the intuition: in a left turn
`r > 0` and `v < 0`, so `v + r*x_w` is LESS negative for a panel AHEAD of the
reference point, `|beta|` falls, and **a forward-mounted panel sees LESS
incidence in a corner, not more**.

`cfg.dev_curved_flow` defaults **True** — the physically correct model — and
`False` is the diagnostic that degrades to `qss`'s. That is the direction
`vehicle.py`'s own precedent runs in (`force_cos_delta` defaults True and
includes a projection qss omits; the module docstring says "if this model ever
MATCHES qss, something has been reimplemented that should not have been"), and
physics that is only correct behind a flag is a trap. The acceptance numbers
were **re-baselined deliberately**: the bands absorb it, W 13/13, D 9/9, suite
82/82 and `--modules` 100/100 unchanged. Measured at the R = 100 limit,
`x_w = 0.97`:

| | beta -> beta_dev | CL | ramp-steer gain |
|---|---|---|---|
| fin | -5.807 -> -5.200 deg | 0.9503 -> 0.9215 (-2.76 %) | +3.8587 -> +3.7412 % |
| plate | -7.631 -> -6.944 deg | 1.5789 -> 1.5448 (-1.88 %) | +6.0473 -> +5.9449 % |

and on the scripted arena lap the device's worth falls +0.0582 -> **+0.0279 s**
(fin, more than halved) and +0.1594 -> **+0.1472 s** (plate, -7.7 %). The sign
of the benefit survives in both. `qss_parity` freezes it exactly as it zeroes
`dCLda`, so the parity path is unaffected either way.

**The flank the panel deploys on** (`cfg.dev_flank`, `'outer'` the DEFAULT |
`'inner'`). The flank reaches the physics through **one term only**, and the
shortness of that list is the point: a pure LATERAL force has no lateral moment
arm, so `F_dev`'s yaw moment is `x_w*F_dev` and its roll moment is
`-F_dev*(h_w - h_ra)` wherever across the width of the car it acts, and the
lateral load transfer reads `F_dev*(h_cg - h_w)` with no `y` in it either. What
the flank changes is the **drag's** yaw moment, `y_dev*D_dev`: out of the corner
from the outer flank (−40 N·m against +172 from the lift at the R = 100 limit,
i.e. 23 % of the device's yaw authority), into it from the inner one. Everything
else is bookkeeping — `dev_left` / `dev_right` swap roles, since the slot that is
outer in a left turn is the inner one on the other setting.

The side force does **not** flip with the flank. Keeping it pointed at the turn
centre from the inner flank means turning the section over so the suction
surface faces OUTBOARD, away from the body — a different aerodynamic problem,
which is why `wing.build_lattice` takes `wall_side` (§7). `vehicle.py` never
learns what a wall is: the orientation reaches it inside a `DevAero`'s CL/CD
laws, exactly as a mount does. Measured both ways in
`.handoff/15-inner-flank.md`; **the default stays `'outer'`**.

`S_DEV = 0.35 m²` is **one panel**. Exactly one panel is active at a time.

**Public API**:
```python
@dataclass
class Controls:
    delta: float = 0.0        # rad at the ROAD WHEEL, already ramped/limited
    throttle: float = 0.0; brake: float = 0.0; clutch: float = 0.0
    handbrake: float = 0.0
    gear_req: int = 0         # -1 down, 0 none, +1 up  (edge, consumed)
    auto_gearbox: bool = True
    auto_clutch: bool = True  # see section 3: launch assist, blip, restart
    wing_on: bool = False     # driver's toggle; the actuator lag lives in vehicle
    starter: bool = False
    wing_cmd: tuple | None = None   # FREE WINGS: (left, right, top) bools, gated by
                                    # wing_on; None (every published caller) = the
                                    # automatic law below. Both flanks = air brake.

@dataclass
class VehicleConfig:
    qss_parity: bool = False        # Cs_psi=0, dCLda=0 -> reproduces qss exactly
    force_cos_delta: bool = True    # False = diagnostic that removes the cos(delta)
                                    # projection qss omits (0.855 g -> 0.861 g)
    wing: str = 'off'               # 'off' | 'fin' (CL0 0.70) | 'plate' (CL0 1.25)
    x_w: float = 0.97; h_w: float = 0.90
    dev_flank: str = 'outer'        # 'outer' (shipped) | 'inner'; see above
    mu_scale: float = 1.0
    guards: bool = True
    power_scale: float = 1.0        # the Engine setting -> powertrain.from_car; 1.0 in every rig
    tc_on: bool = False             # driver aid; DEFAULT OFF in every rig (below)
    abs_on: bool = False            # driver aid; DEFAULT OFF in every rig (below)
    dev_left: DevAero | None = None   # the DESIGNED panel on the left flank (y > 0)
    dev_right: DevAero | None = None  # ... and on the right; None = the closed form above
    top: TopAero | None = None        # the top (rear / roof) wing; None = none

@dataclass(frozen=True)
class DevAero:      # what a designed flank panel is to the physics
    name; S; CL0; CLa; CL_min; CL_max; cd0; cd1; cd2; x_w; h_w; inc   # inc in rad
    # CL = clamp(CL0 + CLa*alpha_dev, CL_min, CL_max); CD = cd0 + cd1*CL + cd2*CL^2
    # DevAero.legacy('fin'|'plate', x_w, h_w, inc_deg) states the published panel
    # in the same terms (CLa 2.47, CL in [0, 1.6], cd1 = 1/3.2): it reproduces
    # the closed form to round-off (validate W2, 1.4e-14).
@dataclass(frozen=True)
class TopAero:      # the top wing at its mounted incidence
    name; S; CZ; CD; CD_stowed; x_t; h_t; mode ('fixed'|'active'); t_ext 0.45; t_ret 0.30

class Vehicle:
    state: VehicleState
    tel: dict            # the diagnostics dict, rebuilt every step
    def __init__(self, car=CorsaC(), cfg=VehicleConfig()): ...
        # `car` is a corsa_c.CorsaC or a cars.CarSpec (a field-for-field
        # superset of it, optionally already carrying added mass)
    def reset(self, x=0.0, y=0.0, psi=0.0, V=0.0, gear=1) -> None: ...
    def step(self, ctl: Controls, mu: Sequence[float], crr: Sequence[float],
             dt: float) -> None: ...
```

**Three wings.** The car carries up to three: a panel on each flank and a
wing on top. The FLANK physics is the device above, unchanged: exactly one
panel is active, the one on the OUTER flank of the turn (`dev_right` for a
left turn, `sgn_dev = +1`), armed by the driver's toggle and the steering
sign. A `DevAero` replaces `CL0 / dCLda / S_DEV / LD_DEV` for its side with
the affine lift law and quadratic drag law `drive/aero/wing.analyse` fitted
from the lattice; `qss_parity` freezes its slip term exactly as it zeroes
`dCLda`. With both sides `None` the code path is the published one, and the
suite holds it bit-for-bit.

**Free wings** (`Controls.wing_cmd`, additive). With a 3-tuple the flank
panels are commanded DIRECTLY: each flank has its own deploy state
(`dep_raw_l / dep_raw_r`, the same `t_ext / t_ret`), the panel on flank f
(+1 left) is the one the automatic law deploys there, so its force law is
the branch above with `sgn = -f * flank_sgn` and `y_dev = f * Y_DEV`; two
panels out sum (side forces cancel, drags add: the air brake). The roll and
load-transfer terms read one `F_dev` at the force-weighted `h_w`. A `None`
top entry leaves the top wing on its own law. `wing_cmd is None` is the
published path bit-for-bit (the suite, 33/33, is unmoved). Users reach it
with `G` (`drive/airbrake.py`'s modes: AIR BRAKE, TOP, TOP FIXED, TOP FIX+SIDE,
LEFT, RIGHT; AUTO is `None`); `drive.ml`'s free-wings policy head
(5 outputs: steer, pedal, wing_l, wing_r, wing_top; 373 parameters) emits
it, and is what `--swarm` breeds by default. The anchor's wing rule enters
that head as a ±`policy.WING_PRIOR` prior per wing, composed at
`policy.WING_GAIN` (full authority), so the head really can command any of
the eight deployment patterns; through task 17 the rule's ±1 verdicts at
the 0.55 residual gain could not be overruled and the head was free in
name only. 4-output checkpoints load and drive unchanged: that head's wing
output is still the rule's, to the bit.

The TOP wing is new physics, all of it exactly 0.0 when `top is None`:

```
top_cmd = wing_on                                          (mode 'fixed')
        = wing_on and (brake > TOP_BRAKE_ON = 0.05 or |delta| > DEV_DEADBAND), held TOP_HOLD = 0.8 s   ('active')
dep_top = smoothstep(top_raw), actuator t_ext / t_ret as the flank
F_top   = dep_top * q * S * CZ                             # DOWN, positive
D_top   = q * S * (dep_top * CD + (1 - dep_top) * CD_stowed)
SFx    -= D_top
normal_loads(): Fz_front += F_top_prev * (x_t + b)/L / 2 per wheel, Fz_rear += F_top_prev * (a - x_t)/L / 2
demand_x = (SFx_t * h_cg + D_top * (h_t - h_cg)) / L      # the drag's pitch arm
P_req   += V * D_top
```

`F_top_prev` is LAGGED one step into `normal_loads()` exactly as the `dFz_*`
states are (no algebraic loop; `VehicleState.F_top_prev`, non-integrated,
not in `as_array`). The station matters: behind the rear axle
`(x_t + b)/L < 0` and the wing UNLOADS the front of this front-limited car
(measured: peak a_y 0.8550 g -> 0.8488 at x_t = -1.6, 0.8804 at +0.3, group
W). After every `step()` `F_top D_top top_deploy` are current alongside the
flank names above.

`mu` and `crr` are per-wheel `(4,)` surface scales supplied by the caller from
`track.surface_at()`. `Vehicle.step` advances **exactly one physics step of
`dt`** — the accumulator lives in `drive.py`.

After every `step()` these read-only attributes must be current (the renderer,
HUD and telemetry depend on the exact names):

```
x y psi u v r ax ay phi p beta          floats  (ax, ay body-frame at the CG)
Fz Fx Fy alpha kappa delta_wheel omega  (4,) numpy float arrays, FL FR RL RR
rpm gear engaged stalled on_limiter
wing_deploy_l wing_deploy_r             (per FLANK, +y left / -y right; both > 0
                                         only under `wing_cmd`; the renderer's
                                         `flank_deps()` reads them, falling back
                                         to the pair below when absent)
F_wing D_wing wing_deploy wing_side     (wing_side is the TURN sign = sgn_dev:
                                         +1 left turn, so the RIGHT panel is
                                         deployed; -1 right turn, so the LEFT
                                         panel is deployed; 0 never armed.
                                         It is NOT the flank index.)
util_f util_r limited_by                ('FRONT' | 'REAR' | 'POWER')
wheel_lift                              (4,) bool
abs_active                              (4,) bool  (ABS gain < 1 on that wheel)
tc_active tc_gain                       bool, float (TC load gain < 1)
eng_load                                float  PowertrainOutput.load
```

`util_f/util_r` MUST be computed by calling `qss.fy_max` with `qss.TYRE`, not
by a local reimplementation of `mu(Fz)`.

**ABS** (`cfg.abs_on`, `Vehicle._abs`): a 4-channel slip-threshold controller
on the HYDRAULIC brake torque between the powertrain call and the wheel ODE;
the handbrake passes through unmodulated. It reads the transient `kx` (never a
`Vsx/Vx` quotient), dumps the gain at `|kx| > 0.16` (10 ms), rebuilds at
`|kx| < 0.13` — fast (20 ms) up to `0.85 ×` the gain that last provoked a
release, slowly (0.30 s) beyond it, which is what lets one controller stop dry
(learned level ≈ 0.65) and wet (≈ 0.35) without locking either. Floor 0.10,
off below 2 m/s. Pure in (state, dt): the replay is bit-identical. Measured
100→0 km/h at full pedal: dry 43.24 m against 44.64 (best fixed pedal) and
49.74 (locked); wet 62.03 m against 80.45 locked. **It is OFF in every rig
and every acceptance number** (the lock-order tests need the wheels to lock);
the interactive drive switches it on from its settings.

**TC** (`cfg.tc_on`, `Vehicle._tc`): engine-only traction control on the
driven axle, applied BEFORE the powertrain call as `PtInput.tc_scale` (a gain
on the engine load; the pedal itself is untouched — see section 3). Reads
`max(kx[FL], kx[FR])`: the target gain is 1 below `TC_SLIP_RESTORE = 0.12`,
falling linearly to `TC_GAIN_MIN = 0.15` at `TC_SLIP_CUT = 0.20`; the gain
slews to it at `1/0.03` per s down and `1/0.12` per s up; gain 1 and inactive
below `TC_V_MIN = 1 m/s` or with the pedal up.
**The cut depth is limited to `1 - 2*share`**, `share` being that wheel's
fraction of the front-axle vertical load, floored at `TC_GAIN_MIN`. The 2 is
the open diff: both driven wheels carry equal torque, so the axle's tractive
force is twice the force of the wheel with less grip and a wheel of load
share `s` can still deliver `2s` of the axle's capacity. Straight ahead
`share = 0.5`, the floor is 0 and the full `TC_GAIN_MIN` authority is back,
so every launch number below is unchanged. This is not cosmetic: with full
authority everywhere the unloaded INSIDE front of this FWD car (991 N against
4846 N, `kx` 0.508 against 0.012 at 6 deg of steer) held the gain at 0.256
mid-corner and the car DECELERATED at full throttle, `ax = -0.206 m/s²`
(`.handoff/08-steering.md`, symptom b). Pure in (state, dt). Measured
on the 2× Engine setting, WOT from rest on the open map: front slip ratio
1.50 (limiter-bouncing through all of 1st) without it, 0.31 with it, 0–100
km/h 8.42 s; it never acts on the stock car at WOT on dry tarmac (a
bang-bang cut/restore at 0.14/0.12 was tried first: 0.80 mean load in 1st,
8.78 s). **OFF in every rig**, like the ABS.

**Also in `vehicle.py`** (validation entry points, sharing the same force code):
```python
def steady_state_corner(R, *, car, cfg, power_cap=False, tol=1e-6) -> dict
def ramp_steer(V, rate_deg_s=2.2, T=16.0, *, car, cfg) -> dict   # THE grip-limit driver
def validate(verbose=True) -> bool                               # python3 -m drive.vehicle
```
Quantitative limits come from **open-loop ramp steer**, never from a
closed-loop skidpad controller (it saturates ~12% below the car).

## 5. `drive/track.py`

```python
@dataclass(frozen=True)
class Seg: kind: str; length: float; radius: float = 0.0; turn_deg: float = 0.0
@dataclass(frozen=True)
class SurfacePatch: s0,s1,n0,n1: float; mu_scale=1.0; crr_scale=1.0; label=''; colour=(43,58,74)
@dataclass
class Track: name,width,segs,origin,heading0,closed,surfaces,sector_s,guide_radii,gates
            # build() fills s, xy, psi, kappa, left, right, length, bbox

@dataclass(frozen=True)
class Area: kind: str; params: tuple; mu_scale=1.0; crr_scale=1.0; drivable=True; label=''; colour
            # 'rrect' (cx,cy,hx,hy,r) | 'rect' (x0,y0,x1,y1) | 'circle' (cx,cy,r)
            # contains(x, y) -> bool  (O(1), analytic);  polygon(n_arc) -> (M,2);  bbox()
Track also carries: areas: list[Area]   # open map: world-space tarmac / surface regions
                    features: list      # painted decorations, drawn only:
                                        # ('circle',cx,cy,r,colour,w) ('cone',x,y)
                                        # ('line',x0,y0,x1,y1,colour,w) ('box',x0,y0,x1,y1,colour)
                    title: str          # human name (HUD minimap, menus)

def build(tr) -> Track
def project(tr, x, y) -> tuple[float,float,float,float,int]   # (s, n, kappa, psi_c, i); n>0 = LEFT
def surface_at(tr, x, y, global_wet=1.0) -> tuple[float,float,bool]   # (mu_scale, crr_scale, on_track)
def on_tarmac(tr, x, y, n=None) -> bool          # ribbon OR a drivable area
def start_pose(tr, offset_n=0.0) -> tuple[float,float,float]
def point_at(tr, s, n=0.0) -> tuple[float,float]
def make_arena(surfaces=True); make_linden(surfaces=True); make_kestrel(surfaces=True)
def make_ashdown(surfaces=True); make_open(); make_skidpad(radius=50.0, cw=False); make_dragstrip()
def make_track(name, radius=50.0, cw=False, surfaces=True) -> Track   # the one builder;
                                        # an unknown name builds the arena
def solve_closure(segs, ia, ib, heading0=0.0) -> tuple[float,float]   # the two straights
                                        # that close a loop (self-check only)
TRACKS = {'arena':..., 'linden':..., 'kestrel':..., 'ashdown':...,
          'open':..., 'skidpad':..., 'dragstrip':...}
TRACK_ORDER = ('arena', 'linden', 'kestrel', 'ashdown', 'open', 'skidpad', 'dragstrip')
                                        # TAB / the Map setting cycle this
CIRCUITS = ('arena', 'linden', 'kestrel', 'ashdown')         # the race circuits
TRACK_TITLES = {...}                                          # menu labels
CLOSURE_FREE = {'arena': (0, 8), 'linden': (2, 6), 'kestrel': (10, 14), 'ashdown': (0, 12)}
```
`DS = 0.5 m` sampling, 8.0 m grid hash for projection with 3×3 → 5×5 → full
argmin fallback. Literal geometry (segments, node coordinates, arc centres,
closure to 3.8e-8 m, length 1249.2022 m) is in `../specs/harness.txt` and must
be used verbatim.

**The circuits** (`CIRCUITS`). Three more closed circuits are built the
arena's way -- straights and constant-radius arcs, 12 m wide, heading 0 at
s = 0 -- with every quantity round except two straights solved for closure by
`solve_closure` (`CLOSURE_FREE` names them) and stored as literals at 7 dp
(closure 2e-8..6e-8 m). A map name is a plain lowercase identifier with no
`_`, `,` or `|` and never `<another map>_...` (records class keys and file
names, a checkpoint's comma-listed meta, the `seed_<map>_` glob).

| name | title | turn | length m | corners (R m) | straights | sector_s | patches (s, n) |
|---|---|---|---|---|---|---|---|
| `linden` | Linden park | CCW | 1110.4021 | 7, R 30..60 (T2 R30 hairpin) | 284 m back, 150 m pit | 0 / 370 / 690 | WET_T5 850..905 full; WET_T2_ENTRY 400..435, n -6..0 |
| `kestrel` | Kestrel ring | CCW | 1913.3440 | 7, R 55..100 | 390 m pit, 320 m back | 0 / 610 / 1140 | WET_T2 466..544 full; WET_T4_ENTRY 830..870, n -6..0 |
| `ashdown` | Ashdown circuit | CW (sum -360) | 1390.0362 | 6 right + 1 left, R 30..120 | 211 m pit, 200 and 190 m | 0 / 505 / 930 | WET_T5 830..886 full; WET_T4_ENTRY 495..530, n 0..6 |

What a circuit must satisfy, and `self_check` measures on each: closure < 1e-4 m
and |sum(turn_deg)| = 360; the stored straights re-solve to 1e-6 m; the V3
round trip at |n| <= half-width; centreline separation >= 60 m (width + two
24 m run-off bands) over pairs > 90 m apart in s; R 30..130 m; three sector
lines, `sector_s[0] == 0.0`, ascending, each on a straight; |kappa| <= 1/80
over s -37.4..+2.1 (the six painted grid rows); every surface patch covers
n = 0 (drive.ml reads the grip on the centreline only) and every centreline
sample inside it reads its mu. Measured, not asserted here: `LapDriver` at
0.90 laps each cleanly in all three cars on the stock engine, surfaces none
and patch, aids on and off (the classes it laps on the arena), and the
drive.ml anchor laps each twice (`python3 -m drive.ml`). Scenery and props dress them from their
geometry under their own names (the 'circuit' theme); every name-keyed table
(`records.LAP_TRACKS`, `medals.T_MAX`, `challenges.METRICS`, `props.REQUIRED`,
`aero.mission.TRACKS`) lists them, and adding one makes the medal table stale.
In `drive.drive` every `CIRCUITS` name gets the `LapDriver` as the
`--headless` default driver (planned on the run's surfaces, global wet and
car; the arena with no flags is built exactly as before; the test maps keep
the 20 m/s `PathFollower`), and `--script lap --track <circuit>` laps it
(any other map laps the arena, as before). The RACE page's Test shows
`bot_test_T(track, T)` seconds and hands `bot_lap` the arena budget; the
Deploy-swarm page's default Sim time is `swarm_T(track, car=)` (70 s, pro
rata on a longer circuit, and -- task 41 review -- in proportion for a car
whose standing arena lap under the swarm's anchor is slower than the
Corsa's 68.24 s: `SWARM_T_SLOW_LAPS`, the Express 71 s, the bus 82 s on the
arena; a stock car or None changes nothing), as is `--swarm-T` when not
given (the car the swarm breeds in); what the page keeps of a Deploy
(`_swarm_menu_kept(launch, tr, car=)`) drops a Sim time still at that map's
default, so the next map's page shows its own, and keeps one the player set;
the page's Car row moves a Sim time still at the old car's default.

**The open map** (`make_open`, name `'open'`): a closed perimeter loop
(straights 420 / 260 m, corners R = 45 m, 12 m wide, length 1642.743 m, sector
lines at 0 / L/3 / 2L/3) whose rounded-rectangle outer edge + 6 m is one
drivable `rrect` Area (522 × 362 m), plus a `rect` Area (250..370, 95..175)
with `mu_scale = MU_WET_SCALE`. `surface_at` on a track with areas: `on` =
ribbon OR any drivable area containing the point; every containing area
multiplies its scales in; off everything = grass (`MU_OFF_TRACK`,
`CRR_OFF_SCALE`). Features (skidpad circles R 50 / 30.48 at (105, 150), a
300 m drag lane at y = 40..46 with 100 m boards, nine slalom cones at 18 m,
a cone gate) are never read by the physics. Geometry is not a spec number.

## 6. `drive/input.py`

```python
class InputSource(Protocol):
    def poll_events(self) -> list[str]                       # once per RENDER frame
    def update(self, dt, V, beta_deg, rpm, gear) -> Controls  # once per PHYSICS step
class KeyboardInput(InputSource): def __init__(self, steer_limit=True, ...)
class GamepadInput(InputSource):  @staticmethod def available() -> bool
class BlendedInput(InputSource)
class ScriptedInput(InputSource): def __init__(self, fn(t, veh, track) -> Controls)
def steer_limit_deg(V, beta_deg, ay_max=8.4608, L=2.491, k_us_deg=3.2,
                    lock_deg=32.625, beta_gain=1.2) -> float
def steer_limit_pair_deg(V, beta_deg, ...same kwargs...) -> tuple[float, float]
```
* Ramp the **road-wheel angle** at a hand rate (`900 deg/s` at the wheel ÷ 16.0
  = `56.25 deg/s` at the road wheel), never a normalised axis.
* Return-to-centre `45 → 90 deg/s` road wheel with speed; counter-steer rate =
  drive + return.
* Speed limiter `delta_lim(V)` opened up by `+1.2*|beta_deg|` so slides can be
  caught. `steer_limit=False` bypasses the whole aid — every validation script
  uses that path.
* **The `beta` bonus is DIRECTIONAL and the clamp is a PAIR.**
  `steer_limit_deg` keeps the symmetric form (the signature is pinned here and
  V16/V17 are quoted from it, and it is still what `validate.py` checks), but
  what `KeyboardInput` and `GamepadInput` clamp against is
  `steer_limit_pair_deg(V, beta_deg) -> (limit on LEFT lock, limit on RIGHT
  lock)`, which spends the whole bonus on the **counter-steer** side and
  leaves lock *into* the slide at the floor. Counter-steer has the SAME sign
  as `beta` (`beta = atan2(v, |u|)` with `v` leftward, so a left-turn slide is
  `beta < 0` and the catch is right lock — the same statement as section 9
  item 7). Symmetric, the term was positive feedback: more lock → more slide →
  bigger `|beta|` → a higher limit → more lock, and at `|beta| = 20 deg` the
  aid handed over full mechanical lock at any speed. Measured, keyboard,
  DOWN+LEFT held from 30 m/s: symmetric `delta_max 22.32 deg` for `dpsi 37.98
  deg` of heading change; directional `delta_max 14.08 deg` for `dpsi 41.10
  deg` — 8.2 deg less lock, 3.1 deg more turn (`.handoff/08-steering.md`).
  The magnitude of the catch is untouched: a 12 deg slide still opens 19.12 deg
  of opposite lock at 30 m/s.
* **The clamp eases down, it does not snap.** Each of the two bounds is a
  state: it rises to the commanded value instantly (a wider limit is never a
  surprise) and falls at `_return_rate_deg(V)`, which is the rate the wheel's
  own self-aligning torque would unwind it at. Without this, a directional
  bound collapsing 20+ deg the instant `beta` changes sign would teleport the
  road wheel. `KeyboardInput.delta_lim_deg` reports whichever bound is binding
  the direction the wheel is actually turned.
* **Trigger rest.** A pad trigger is normalised from the rest value its FIRST
  HID report showed, `-1.0` or `0.0` (`GamepadInput._trig_rest`), so both
  driver conventions give 0 at rest and 1 at the stop; with `-1.0` that is the
  historical `(a+1)/2` exactly. Before that first report every axis reads
  `0.0`, which `(a+1)/2` turns into HALF TRAVEL — the pedals are held at 0
  until `_rest_checked` is true, or a hot-plug hands the car 47% throttle and
  47% brake at once.
* Pedal ramps: throttle 3.5/6.0 s⁻¹, brake 5.0/8.0, clutch 8.0/3.0,
  handbrake 8.0/10.0. Integrated at `DT_PHYS`, key state sampled at 60 Hz.
* Shift and wing keys are **edge-triggered** from the event queue only.
* `ScriptedInput` never reads pygame or the wall clock.
* **Gamepad layouts.** `GamepadInput` picks `'ps'` (DualSense / DualShock,
  by name: SDL HIDAPI order, R2 = axis 5 throttle, L2 = axis 4 brake, cross
  0 handbrake, square 2 clutch, circle 1 wing, triangle 3 wing side, L1/R1
  9/10 shift, options 6 menu, create 4 reset, touchpad 15 garage, d-pad
  11-14 hud/vectors/slowmo/normal, L3/R3 zoom_auto/camera) or `'generic'`
  (the SDL Xbox order that was here before). `~/.carsim_pad.json` overrides
  either. The pad does NOT fold `wing` / `wing_side` into its own state —
  `Sim` owns that toggle (a second copy left the panel armed with both
  readouts saying OFF).
* `BlendedInput` **hot-plugs**: it recounts joysticks every 30 render frames
  and attaches / drops the pad. `pad=None` is the steady state without one.
* `feedback(HudData)` drives rumble at 20 Hz. It reads `time.monotonic()` and
  is therefore called from the **render loop only**, never from `update()`.
* `BACKSPACE` -> `'garage'`; `Sim` ends the session with
  `stop_reason = 'garage'` and `run_garage_cli` re-enters the editor.
* **Pause menu.** `ESC` / OPTIONS / START -> `'menu'`. `set_menu(True)` puts
  every input in menu mode: the keyboard emits only `MENU_KEYS`
  (`nav_up`/`nav_down`/`nav_left`/`nav_right`/`select`/`menu`/`reset`/
  `full_reset`/`garage`/`ghosts`, the digits, and since task 45 `track_next`
  on TAB -- read only by the pause page and the TIME TRIAL page) and reads
  all held keys as released; the pad emits
  `MENU_PAD_NAMES` edges plus left-stick up/down (and left/right) through
  `menu.StickNav`. Entering menu mode SEEDS the
  pad's edge state from the live buttons (the OPTIONS press that opened the
  menu must not close it), and menu mode keeps refreshing the driving edge
  state silently (a circle held across the close must not toggle the wing).
  `menu_help(layout)` is the single source of the on-screen key tables.
  In menu mode the keyboard half also turns the MOUSE into menu commands
  (`_menu_mouse`): motion -> `hover:X:Y`, left click -> `click:X:Y`, right
  click -> `menu` (back), the wheel -> `nav_up` / `nav_down`; with the menu
  closed the mouse is ignored.
* **Gearbox modes.** `GEARBOX_MODES = ('auto', 'manual', 'clutch')`,
  `gearbox_flags(mode) -> (auto_gearbox, auto_clutch)`, `GEARBOX_LABELS`,
  `GEARBOX_HUD` (`AUTO` / `MAN` / `MAN+CL`). `KeyboardInput`, `GamepadInput`
  and `BlendedInput` carry `set_gearbox(mode)` and put both flags on every
  `Controls`; `BlendedInput.gearbox` reads the mode back, `set_steer_limit`
  reaches both halves.
* **Session boundaries.** `GamepadInput.seed_edges()` takes the live button
  state as already-pressed; the interactive session calls it on a pad it was
  handed (garage → drive), and the garage seeds its own edge table on the
  first poll — the ✕ that selected *Garage* in the drive's menu must not be
  the ✕ that drives straight back, and vice versa.
* **The drive's limiter runs on `K_US_DEG_MEASURED = 7.82`**, the module's own
  calibration finding; `K_US_DEG = 3.2` stays the function default because
  V16/V17 are quoted from it. Every validation path bypasses the aid anyway.

## 7. `drive/render.py`, `drive/telemetry.py`, `drive/plots.py`

* `Renderer(cfg: ViewConfig, track, headless=False)` with `update_camera`,
  `draw_frame`, `present`, `screenshot`, `frame_ms`.
* The tarmac ribbon is **ONE concave polygon per visible run** (0.58 ms) —
  never per-quad tessellation (20.4 ms, misses 60 fps). With the world on
  (`drive/world.py`) a run's verge is one more polygon under it, a long run
  is decimated where the curve and the distance allow, and the chase view
  cuts it at a few camera depths into pieces that overlap by one sample (so
  no seam shows); every chase track layer is projected in one call, and only
  what crosses the near plane meets `Chase3D`'s clipper. That batching made
  the chase frame FASTER with all of the look on (6.1 ms against 9.5 over the
  standard scenes, worst 8.3 against 17.6).
* Skid marks capped at 600 visible segments by strided subsampling.
* Tyre-force and wing-force arrows drawn at the **same** px/N. The wing is
  63–227 N against a 9908 N car and must look that small.
* HUD utilisation must call `qss.fy_max` with `qss.TYRE`.
* `render.set_car(car)` points the module's `_CAR` at the fitted car and drops
  the g-g cache (whose curve is built from `_CAR.CdA / Crr / m / P_wheel`).
  It defaults to `CorsaC()`, so every module self-check and every headless
  renderer is unchanged; `drive.drive` calls it once per session.
  `HudData.car_name` / `mass_kg` are drawn under the minimap — the Ballast
  setting is otherwise invisible, and 200 kg is 20 % of a Corsa.
* `render.set_paint(rgb)` paints the fitted car (`drive/paint.py`'s palette;
  None, the default, is its factory colour `render.factory_colour(car)`,
  `C_CAR_STYLE` by body style; `car` is a CarSpec or a `cars.py` key, the
  two give the same colour). Module-wide like `set_car`: called before a
  Renderer is built, and live through `Renderer.set_paint(rgb)`, which shows
  on the very next frame in the chase view, the plan views and the minimap
  dot (the car's colour, 1 px light ring). The colour is in `CarGeom.key`,
  so every mesh and tone built on the old one is not looked up again; the
  plan tones are guarded off the counted colours and the HUD's text / dim
  grey. Cosmetic only; headless and scripted runs keep it None.
* g-g envelope calls `qss.max_ay` — cache, recompute only when `|ΔV| > 1 m/s`.
* `HudData` carries `x_w, h_w, wing_type, inc_deg` (defaults 0.97 / 0.90 /
  '' / 0): the plan-view panel is drawn at `aux.x_w`, not at a constant, so a
  garage-built car shows its panel where the physics has it.
* `HudData.ghosts` (default `[]`, tuples `(x, y, psi, (r, g, b))`) are other
  cars drawn as flat GROUND silhouettes (`_gpoly`) in the plan views and, in
  the chase view, as translucent low-poly 3-D cars (the nearest few; further
  ones cross-fade to the silhouette, one level with the hero is drawn before
  it with its outline, one between the eye and the hero fades but keeps its
  outline and label), after the skid layer and BEFORE the car; `HudData.overlay`
  (default `[]`, strings; a leading `!` draws the line in the warning
  colour) is a top-left text panel drawn after the HUD and before the menu.
  A ghost tuple may carry a fifth element, a label string, drawn above the
  silhouette in its colour. Ghosts are the swarm's replay and the race's bot
  (`Sim.rival`, one tuple per frame); empty, nothing is drawn and every other
  session's frame is bit-identical.
* `HudData.menu` (default None) is drawn LAST by `draw_frame` when
  `menu.open`, duck-typed (`render.py` does not import `menu.py`).
  `HudData.gearbox` (`'AUTO' | 'MAN' | 'MAN+CL'`), `abs_active` and
  `track_name` are drawn under the gear glyph, in the flag row and on the
  minimap respectively; a stalled engine puts the restart hint in the warn bar.
  `HudData.engine` (`'75 HP' | '110 HP' | '150 HP'`) sits under the gearbox
  label, `tc_active` joins `abs_active` in the flag row (`TC ABS`), and
  `eng_load` (`PowertrainOutput.load`) is carried for the sound.
* **Sound** (`drive/audio.py`). `CarSound(volume)` owns one `pygame.mixer`
  channel (16-bit, stereo when the mixer grants it, else mono; `CHUNK =
  2048` samples, `BUFFER = 512`) and a `Synth` that renders phase- and
  filter-state-continuous chunks in numpy: the ENGINE as per-cylinder firing
  pulses (sub-sample placed, cycle-to-cycle jitter) through a pipe echo and
  exhaust resonances with a load-opened low-pass, per-firing flow noise,
  intake roar, a valve tick; a PROFILE per car from `HudData.car_key`
  (`car_name` as the fallback): the Corsa's small I4, the MX-5's rorty I4,
  the 540i's cross-plane V8 (its banks' uneven trains, one per side in
  stereo, asymmetric so a mono fold keeps the burble); pops on a genuine
  lift from high rpm (a per-lift budget, never on a shift's declutch),
  an ignition-cut limiter, the starter; ONE clunk per shift (the physics'
  g -> 0 -> g' is one shift) with an upshift cut; a faint final-drive whine
  (louder in reverse); tyres -- scrub from 80 % of the grip, squeal past it
  (panned to the sliding side), a harsher lock-up, ABS chatter; the road by
  `HudData.surf4` per wheel (tarmac hum, wet spray, a kerb's rumble strip at
  V / 0.35 m on its side, grass swish and thumps, gravel crunch); wind with
  gusts; the active wing's servo whine only while a panel MOVES; sector
  chimes on the rising edge of `sector_flash` (the lap's own chime is task
  27's `CarSound.chime`, called once by `Sim._rec_lap`); a bus compressor,
  a DC blocker and a soft clip under `CEIL`. Every HudData read is guarded
  (non-finite -> default, clamped) and a non-finite chunk resets the synth
  (one silent chunk, then sound). Synthesis IS frame time: <= 1.5 ms a chunk
  (the worst case, V8 stereo, everything on; ~0.65 ms typical). `Sim.run_
  interactive` calls `audio.update(hud)` right after the pad `feedback(hud)`
  -- the render loop, never `step_physics` -- and keeps the channel one chunk
  ahead (`play` + `queue`), so the sound trails the physics by 50-90 ms and
  `underruns` counts the frames that ran the queue dry. The interactive
  session calls `audio.warm_up()` at its start (the scipy.signal import and
  the chimes in a thread, so turning the sound on later never stalls a
  frame). `CarSound.ok` False = no device; the sim drives silently.
  `python3 -m drive.audio` is its self-check (50 checks: continuity, each
  profile, the spectra of every layer, cue edges, the lap chime, junk HUD
  values, silence, speed, streaming mono and stereo on SDL's dummy driver,
  one demo WAV per car).
* **The look** (`drive/world.py`, `drive/props.py`, `drive/fx.py`,
  `drive/scenery.py`; renderer config only, no physics read or written that
  the flat modes did not already read). `ViewConfig.scenery / effects /
  shake / detail` ('high' | 'low'); `render.look_config(mode)` maps the
  Graphics setting ('full' | 'low' | 'classic' = the original plain look)
  onto them and `Renderer.set_look(mode)` applies it live. `draw_frame`:
  `fx.update` -> the jolt -> `world.draw_backdrop` (else `_draw_sky3` /
  `C_BG`) -> `world.draw_ground` -> the track layers (with
  `world.draw_surface` after the ribbon) -> skid -> `world.draw_atmosphere`
  (the haze band, ground layers only: props and particles haze themselves by
  depth) -> `props.draw(far=True)` -> `fx.draw(far=True)` -> ghosts -> the
  car -> `props.draw(far=False)` -> `fx.draw(far=False)` -> HUD. far / near =
  deeper / nearer than the car (`Renderer._car_depth`); a plan view draws
  everything far, under the car. `HudData.surf4` (drive.hud_data, from
  `scenery.wheel_surfaces`) and `HudData.car_key` feed the particles, the
  jolt and the sound. `Renderer.smoke` (task 27's handle) is a view of fx's
  pool. The renderer reads two more state fields, `phi` (the body rolls)
  and `omega` (the rims spin; V/R when absent), read-only (DEVIATION 9).
  The chase camera is a spring (it trails under acceleration, closes under
  braking, swings out and looks into a corner, never rolls the horizon) at a
  CONSTANT focal length; `Chase3D.view_psi` is the heading it actually looks
  along, which leans off `psi_cam` in a corner by up to ~3 deg, and the
  panorama scrolls by that. `Renderer.__init__` pre-builds the meshes and
  panels and ends with `gc.unfreeze(); gc.collect(); gc.freeze()` (the
  session's object graph out of the full collections: the 3-4 ms gen-2
  pauses; at most one stale session is held across restarts). Frame cost,
  standard scenes, 1280x800: chase 5.7-6.1 ms mean of scene means (worst
  scene ~7.7-8.3), plan ~3.1 ms (worst ~4.0).
* **Open-map drawing.** `Renderer._prep_track` precomputes each Area's polygon
  and bbox and the feature list. `draw_frame` draws the areas (filled, bbox-
  culled) after the grid and before the ribbon, and the features after the
  marks. `_visible_indices` returns `(runs, s_car, windows)`: on a circuit one
  arclength run + window as before; on a track with areas the centreline
  samples are culled by DISTANCE from the camera and split into contiguous
  runs (wrapping at the seam), each with its own window, because the
  perimeter road is in view on both sides of the pad and an arclength window
  draws only one. Ribbon / edges take runs, kerbs / dashes / marks / patches
  take windows. Measured 2.1–2.4 ms a frame.
* `drive/menu.py`: `Menu(title, items, sections)` with `show / hide /
  handle(cmd) -> action / draw(screen)`. `show(art=f, art_h=px)` (task 32)
  gives a page its own drawing: `f(screen, rect)` gets `art_h` px at the top
  of the first help column, the sections go below it, every `show()` resets
  it, and one that raises is dropped with a note (the LAP RESULTS page draws
  the card there with `Renderer.draw_card`; the CONTROLS page the DualSense,
  `controls_page.draw_pad`). `show(help_for=f)` (task 37): `f(idx)` gives the
  help sections for the highlighted row (the settings page's
  `SETTINGS_ROW_HELP`). Items are `(label, action)`;
  `select` returns the action and closes, `back`/`menu` return `'resume'`,
  `nav_left`/`nav_right` return `'prev:<action>'`/`'next:<action>'` and keep
  it open. On the drive's settings page these step the row's value:
  `Sim.preview_setting(key, d)` applies a live setting at once but only
  BROWSES a `RESTART_KEYS` one (`track`, `wet`, `car`, `ballast`,
  `ballast_at`), holding the running value in `Sim._pending` so ESC
  (`revert_pending`) restores it and a save meanwhile writes the running
  value, not the preview; `select` on a browsed row is `commit_pending` +
  restart.
  `Menu.draw` records each item row's rectangle; `handle('hover:X:Y')`
  moves the cursor to the row under the pointer and `handle('click:X:Y')`
  runs it exactly as `select` would (`hit`, `row_centre`), so every page is
  driven by the mouse as the garage's are.
  `show(..., title=, idx=, columns=)`: `idx` keeps the cursor when a page
  re-shows itself, `columns=1` stacks every help section (and the note) in
  one column to the right of the items, and the highlight / column origin
  follow the widest item label. Overlay + panel surfaces are cached per size
  (a fresh full-screen SRCALPHA fill was 4 ms); warm draw ≈ 1.3 ms. Fonts are
  SysFont'd lazily: the first draw in a cold process pays the ~100 ms system
  font scan once.
* `HudData` also carries the three wings the renderer draws: `dev_left /
  dev_right` (present), `x_w_left / x_w_right`, `dev_chord / dev_span /
  dev_plate`, `wing_left_name / wing_right_name`, and `top_on top_deploy
  F_top D_top top_x top_span top_chord top_plate top_mode wing_top_name`
  (`CarBuild.hud_kwargs` + the vehicle's `F_top D_top top_deploy`). Both
  flank panels are always drawn -- stowed on the flank, the OUTER one slid
  out and lit while deployed, with its force arrow at the tyre scale -- and
  the top wing as a span-wide bar with plates, outline when stowed, filled
  when out, its drag as the backward arrow. The HUD's `ACTIVE AERO` panel
  has a flank row (L/R, side, %, F, D) and a top row (%, mode, Fz, D).
* `drive/garage.py`: `WingDesign(wing, x_w, h_w, inc_deg)` is still the
  published panel with `cfg_kwargs()` -> exactly the `VehicleConfig` fields
  `_aero` reads (`delta_dev_geom = radians(inc_deg)`); its ranges (x_w keeps
  the 0.45 m chord on the body, h_w 0.40-1.20 m, incidence -10..+15 deg) and
  closed-form readouts (`crossover.gain`) are unchanged. Around it:
  `CarBuild(name, left, right, top: Slot(wing, x, h, inc_deg, mode), mirror)`
  -- three slots holding LIBRARY wings by name. `cfg_kwargs(lib)` returns the
  closed-form kwargs (bit-identical to `WingDesign.cfg_kwargs`) whenever both
  flanks carry the same published panel and there is no top wing, else
  `dev_left / dev_right / top` built by `DevAero.from_aero` / `TopAero.from_aero`
  (a library top wing is re-analysed at its slot height first: ground effect;
  an AeroBO wing flies its stored law, derived from AeroBO's own evaluator
  at the slot's height and incidence and re-derived if the slot moved).
  `hud_kwargs(lib)` -> the HudData fields above. Persisted to
  `runs/garage_design.json` as `{"version": 2, "slots": ...}`; a v1
  `WingDesign` file upgrades to the same panel on both flanks. Pages: CAR
  (the 3-D view, three slots, `1 2 3` / TAB select, arrows place, `W` cycles
  the slot's library wing, `M` mirror, `T` top mode, `SPACE` deploy preview,
  `ENTER` drive), MISSION and DESIGN (task 43: both drawn and driven by
  `design_shell`, AeroBO's light CAE shell -- menu bar, tool bar (Start the
  design over, Save, **Run**, **Stop**, previous / next stage, the crumbs, the
  slot / circuit / surface chips), the Simulation tree, Properties, the tab
  strip, the work area, the Output log and the status bar. MISSION = stage
  `1 Mission` (`D` / `L3` / the pause menu), three views -- Operating point,
  Design point, Search & budget -- for the selected slot: circuit and
  surface, the lap of the car as it stands against the bare car, where the
  search settings come from (AeroBO's recommended plan: 164 / 164 / 53 at
  balanced -- or own budgets), *stop when it stops improving*; `State the
  mission` / `ENTER` / Run STATES it through `Garage.state_mission()`
  (re-stating an unchanged mission keeps the design stages; a changed
  circuit, surface or car clears the slot's session; `Garage.open_section`
  refuses until it is stated). DESIGN = `CaeTree` over `DESIGN_TREE`:
  `2 Airfoil` / `2.8 Endplate` / `3 Wing` / `4 Results`, four views each (the
  16 step keys), every one a view of the slot's `aerobo_models.DesignSession`.
  Every screen and search is AeroBO's engine on a worker thread
  (`design_jobs.EngineJob`), drained by `Garage.frame` (`RUNNING · k/N`, the
  status bar's `<evaluator> evaluation k/N · phase · elapsed · ≈ left` and
  its bar); Stop is AeroBO's stop rule (the evaluation in flight finishes,
  the best is kept: `STOPPED · k/N`); Keep going is AeroBO's resume (nothing
  re-flies, the counter continues at k + 1). `S` puts the fitted wing on the
  car (`WingModel.commit`: the wing and its sections saved to the library, the
  slot's incidence -- and on the top its height -- written back, the right
  flank mirrored); the keys are the table below),
  AIRFOIL (`A`: the section library ranked by the
  AeroBO screen weights at the design cl / Re, section + polar plots, `X`
  queues XFOIL in a worker thread, `N` a NACA-4 code), LIBRARY (`L`: wings and
  builds; ENTER puts a wing in the selected slot if its role matches, or
  loads a build; `S` saves the car as a build). Wing meshes are lofts of the
  chosen section (flank: vertical, suction side to the car; top: inverted,
  stowed on the deck, raised to its slot on deploy), painter's-sorted,
  backface-culled; the CAR page draws three wings in ~7 ms.

  **The MISSION and DESIGN pages' keys** (`design_shell`, `DesignShell._keydown`
  / `pad`). A KEYDOWN goes first to an open dialog, menu or drop-down, then to
  a number field being typed into (every key), then to this table. The focus
  is one of three REGIONS, `tree` -> `tabs` -> `work` (`shell.region`,
  mirrored into the two-valued `dp.focus`: tree / tabs = `"nav"`, work =
  `"rows"`); its ring shows only after a key or the pad, and a click moves the
  region and hides the ring. The pad is the keys it stands for (PS names):
  d-pad and left stick = arrows, CROSS = ENTER, CIRCLE = ESC, SQUARE = F5,
  TRIANGLE = TAB, R1 = `]`, L1 held = fine; OPTIONS is the garage's pause menu.

| input | tree | tabs | work |
|---|---|---|---|
| `↑` `↓` | walk the VISIBLE rows, selecting as it goes (a view row shows that view, a stage row its remembered view; a locked stage is stepped over); on the mission page a design stage's row leaves it | -- | `form.nav(∓1)` in draw order, off-screen controls included (focusing one scrolls it in); on a Ranking view the table cursor |
| `←` `→` | `nav.nav_group(∓1)` (design page, pinned); fold / unfold the stage (mission page) | previous / next view of the stage | `form.adjust(∓1)`, SHIFT / L1 fine; a locked control toasts why |
| `ENTER` | `dp.act()`, the step itself; mission page: `g.state_mission()` | as tree | the focused control (a button fires, a switch flips); on a Ranking view takes the row (`dp.act()`); mission page: a control that is not a button states the mission (pinned) |
| digits `.` `-` | | | on a focused number field: start typing (`ENTER` commits, `ESC` cancels) |

| input (any region) | does |
|---|---|
| `TAB` / `SHIFT+TAB` | next / previous region |
| `[` / `]` | previous / next view of the current stage |
| `PAGE UP` / `PAGE DOWN`, `HOME` / `END` | scroll the work area (a focused table pages first) |
| `F5` | Run the current stage: state the mission; screen the library, or on Shape optimisation optimise the shape; launch the wing (then show Convergence); on Results fetch AeroBO's design report if there is none yet (else a toast). Busy: a toast |
| `ESC` | a run in state `running`: Stop (AeroBO's stop rule: the evaluation in flight completes, nothing more flies, the best is kept); `stopping`: a toast, not Back; otherwise `g.close_page()` (design -> mission -> car, committing a dirty wing) |
| `L` `O` `K` `F` `S` `X` `N` `A` | design page only (pinned): screen / optimise / keep going / take the section (a ranked one, the optimised one on Shape optimisation, the family's own plate on 2.8's Section before a screen) / put the wing on the car / a line saying the sections' polars are AeroBO's XFOIL sweeps / rename / the airfoil page; every one refused with a toast while a run is live |
| `F1`, `?` | the *Keys and controller* dialog; `?` / `F1` on a focused control with help opens its help popup |
| `H` | the wing tutor's box (the garage's, before the shell) |

* `drive/aero/` (no pygame): `airfoil` (NACA-4, UIUC .dat, CST), `panel2d`
  (Hess-Smith, validated: NACA 0012 a = 6.91/rad, 4412 alpha_L0 = -4.26 deg),
  `polar` (`Polar` table; `estimate_polar` = panel slope + friction/form-
  factor drag + camber/thickness stall correlation, LABELLED estimate),
  `xfoil` (subprocess + JSON cache, split sweep from 0), `blend` (the
  wing/plate transition: three turn laws, the ramp the plate's section and
  toe meet the wing on, and Hoerner's junction charge -- AeroBO's geometry to
  the bit, and `plate_blend = 0` is the published right-angle corner),
  `vlm` (horseshoe
  lattice, cosine edges / interlaced stations, tip plates, rigid-wall image,
  Trefftz CDi -- reproduces `aerobo.vlm.VLM` to 1e-12: CL_alpha
  4.55942959749662, e 0.9742625474419787 on the AR 8 rectangle), `wing`
  (`WingSpec` -> `analyse` -> the laws; critical-section stall; `design_point`
  with `crossover.gain`; `DESIGN_VARS` is the ONE ordered table of design
  variables and it is the garage DESIGNER page's row order -- `BOUNDS`,
  `design_bounds`, `design_x0`, `design_labels` and `apply_design` all read
  it, so the optimiser's vector cannot drift out of step with the page;
  `span_fit(role, h, car=None, unlimited=False)` is the car's span ceiling --
  `bodies.span_ceiling`, the owner's rule of task 41 (the Corsa's old
  sill 0.28 / roof 1.34 fit is gone) -- that BOTH the page's span row and the
  optimiser's span band must use; `CLAMP_HI` is `WingSpec.clamp`'s sanity
  ceiling on span / area / top ride, wide enough for every car's Unlimited
  wing and pinned by `bodies.self_check`), `optimize` (numpy GP,
  Matern 5/2, EI, Sobol init; `labels` ride through into the result and a
  label list out of step with `bounds` raises),
  `library` (`runs/library/{airfoils,wings,builds,polars}`; 39 seeded
  sections, the published `fin` / `plate` as legacy wings, `flank-e423`,
  `rear-s1223`; XFOIL worker + `poll()`). The Re bank snaps to
  {1e5 ... 3e6}. Estimate vs XFOIL is printed on every read-out.
* **`WingSpec.mount`** (`wing.MOUNTS = ('pylon', 'endplate', 'none')`) is a
  real aerodynamic choice, not a label, and it applies to either role:
  * `'pylon'` -- **the default, and what every library wing was analysed
    with**. Charges `strut_cd`: two mounts of `standoff` length, wetted-area
    friction on `S_ref`, times a **1.3 form factor** which is the pylon/wing
    junction interference allowance (Hoerner, *Fluid-Dynamic Drag* ch.8).
    Tip plates stay an independent continuous knob.
  * `'endplate'` -- carried by structural tip plates instead. `strut_cd` is
    NOT charged, but `WingSpec.plate_h_flown` raises `plate_h` to
    `MOUNT_PLATE_H` (0.06 m flank / 0.12 m top, `est`: enough flange for two
    fasteners plus edge distance) and **the lattice then produces the reduced
    tip loss itself** -- it is not a correlation. The plates' wetted area is
    charged by the existing `cd_pl` term.
  * `'none'` -- nothing charged. The idealisation to compare against, and the
    parity setting for a legacy panel whose published L/D already includes
    its mounts.
  **`build_lattice(..., wall_side=+1)`** is which side of the panel the body
  wall is on: `+1` (the default) the SUCTION side, which is where the body is
  when the suction surface faces inboard and the panel is on the OUTER flank;
  `-1` the PRESSURE side, which is the same panel turned over — what it takes
  to deploy on the INNER flank and still point the side force at the turn
  centre (§4, `cfg.dev_flank`). Rotating the panel 180° about its span carries
  the tip plates with it and this lattice builds them on the lift (+z) side, so
  they still point +z and now face away from the body: no clip. At the standoff
  the car deploys to the clip does not bind on any library flank wing (0.566 m
  at `RIDE_H0['flank'] = 0.60`, against plates of 0.06–0.16 m), so the two
  orientations differ **only** in which side of the panel the wall is on.
  Measured on `flank-e423` at 5°:
  CL 0.7783 → 0.7694, CDi 0.08111 → 0.07975, e 1.2366 → 1.2291 — slightly
  **less** lift for slightly **less** induced drag with the body on the
  pressure side, both above free air, and the free-air limit comes back
  monotonically as the wall is moved away. `analyse` and `spanwise` pass it
  through and `analyse` returns it, so a stored aero says which orientation it
  was flown in.

  `analyse()` returns `mount` and `mass`; `wing_mass(spec, standoff)` is a
  bottom-up floor (two skins + two plates at `SKIN_KG_M2 = 4.17`, 1.5 mm
  2024-T3, plus `PYLON_KG_M` when pylon-mounted) and is what the garage
  charges to the car through `CarBuild.mass_points`. An unknown mount name
  clamps to `'pylon'`. `plate_h_flown` is what `build_lattice` and
  `hud_kwargs` both read, so the renderer draws the plates the aero flew.
  The mount's entire effect reaches the physics inside `CZ` / `CD` via
  `TopAero.from_aero` -- **`vehicle.py` never learns what a mount is.**
* Telemetry: fixed 63-column schema (`../specs/harness.txt`), 100 Hz, buffered
  200 rows, `newline=''`, `f'{v:.6g}'` (`repr` under `--telem-precision full`),
  sidecar `.json` with every constant.
* `plots.py` calls `matplotlib.use('Agg')` **before** importing pyplot. No pandas.

## 8. `drive/drive.py`

Fixed-timestep accumulator, `DT_PHYS = 0.001`, `FPS = 60`,
`MAX_SUBSTEPS = 40`, `MAX_FRAME_DT = 0.10`. Force `dt_wall = DT_PHYS` and
`acc = 0` on the first frame, after any reset and after unpause. Slow-mo
multiplies `dt_wall`, never `DT_PHYS`.

`--headless` sets `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy` **before
pygame is imported anywhere in the package** (do it in `drive/__init__.py`
guarded by `CARSIM_HEADLESS`).

CLI:
```
python3 -m drive.drive [--track arena|linden|kestrel|ashdown|open|skidpad|dragstrip]
  [--radius 50] [--cw]
  [--car corsa|mx5|540i] [--ballast 0..300] [--ballast-at nose|seat|floor|boot]
  [--wet none|patch|all] [--wing off|fin|plate] [--wing-x 0.97] [--wing-h 0.90]
  [--dt 0.001] [--fps 60] [--size 1280x800] [--camera car_up|world_up|chase]
  [--headless] [--render off|offscreen|window] [--script NAME] [--duration 60]
  [--telemetry PATH] [--telem-hz 100] [--telem-precision 6g|full]
  [--no-steer-limit] [--gearbox auto|manual|clutch] [--manual] [--auto-gearbox]
  [--abs|--no-abs] [--tc|--no-tc] [--engine stock|tuned|sport]
  [--sound off|low|mid|high] [--wing-inc 0.0] [--dev-flank outer|inner]
  [--garage] [--build NAME]
  [--ml-drive CHECKPOINT] [--seed-lap] [--race anchor|best|CHECKPOINT]
  [--race-car own|same|corsa|mx5|540i]
  [--swarm N] [--swarm-seed none|latest|FILE] [--swarm-gens G] [--swarm-T S]
  [--swarm-name NAME] [--swarm-resume STATE] [--swarm-car same|corsa|mx5|540i]
  [--swarm-fast] [--swarm-save ask|always|never]
```
`--script drive_probe` is the one scripted entry that deliberately switches
the driver aids ON, because it exists to measure them (section 9 item 9 is
about MEASUREMENTS, and this is not one: `validate.py` never calls it, every
figure it prints is labelled driveability, and it takes the aid state as
arguments rather than reading `runs/settings.json`). It runs both input paths
— `KeyboardInput` fed a synthetic key state through `set_keys()`, and `delta`
commanded directly at the same 56.25 deg/s hand rate — so an input-layer
effect can be told from a physics one. See `.handoff/08-steering.md`.
`--track`, `--wet`, `--camera`, `--gearbox`, `--engine`, `--abs`, `--tc`,
`--sound`, `--no-steer-limit`, `--car`, `--ballast`, `--ballast-at` default to
**None**: an interactive launch fills them from `runs/settings.json` (an
explicit flag wins and is saved back); scripts and `--headless` runs fill them
with the fixed defaults (arena, patch, car_up, auto, **stock**, **corsa**,
**0 kg**, ABS off, TC off, sound off, aid on) and never read the file. An
*explicit* `--car` / `--ballast` does reach a script, the way `--wing` does —
the car is the subject of a measurement, not a driver aid, and measuring
another one with the repo's own rigs is the point of having them. Nothing in
`validate.py` passes either flag, so every acceptance number is the stock
Corsa C with no ballast.

**Settings** (`drive.Settings`: `track`, `car`, `ballast`, `ballast_at`,
`engine`, `gearbox`, `abs`, `tc`, `steer_aid`, `wet`, `camera`, `sound`,
`shake`, `graphics`, `paint` — `Settings.KEYS`; properties `power_scale`,
`volume`, `car_base`; methods `car_spec(extra=())`, `ballast_text()`,
`paint_of(car=None)`, `load / save / clamp / apply_cli / to_opts /
cycle(key, d, car=None)`).

`paint` is a table, car key -> a `drive/paint.py` name (a car with no entry
is `'factory'`); `clamp` keeps only known cars wearing palette names.
`cycle('paint', d, car)` steps ONE car's entry: the settings page hands in
the RUNNING car (`Sim._pending['car']` while another is only browsed), so
the row always paints the car on the road. It is not in `RESTART_KEYS` and
not in the class: `apply_setting('paint')` neither discards nor retargets
the lap being recorded, and hands `drive.paint_rgb(settings, car)` (None =
factory) to `renderer.set_paint` if the renderer has one. The row's swatch
and the garage's preview take `paint_rgb(..., concrete=True)` (factory
resolved by `render.factory_colour`). `_interactive_session` calls
`render.set_paint` after `set_car`, before the Renderer is built, on every
session (a factory car sets None); the loop's garage is `_painted_garage`
(`Garage.set_paint` on construction); the swarm viewer's `_swarm_view` sets
None. V41 builds real sessions and checks all three. `as_dict()` carries the
table, so the `settings` snapshot a lap record, the seed lap and a
telemetry header keep lists it, like Graphics; nothing reads it back.

`Settings.power_scale` is `ENGINE_SCALE[engine]` and nothing else. It briefly
also carried `cars.get(car).engine_scale = T_max/110`, which was how another
car's torque peak rode the existing `power_scale` path with no change in
`powertrain.py`; `engine_curve(car)` (§3) now builds each engine's own curve,
so that multiplier is retired and would double-count. The
Engine row and the HUD label read the car's own PS through `engine_ps /
engine_hud / engine_label`, which reproduce the hard-coded `75 / 110 / 150 HP`
on the Corsa (asserted in `self_check`).

`Settings.car_spec(extra)` is the `CarSpec` a session is built on: the named
car, plus the ballast, plus `extra` (the garage's fitted wings, from
`CarBuild.mass_points(lib)` via `opts.mass_points`). With the stock car, no
ballast and no wings it returns `cars.CORSA_C` **itself**.

`car`, `ballast` and `ballast_at` return **True** from `Sim.apply_setting`, the
same as `track` and `wet`: a different `CarSpec` is a different tyre model,
wheel station set, static load set, derived roll block and gearbox, all of
which are built in `Vehicle.__init__`, so the session is rebuilt rather than
live-patched. `_opts_car(opts)` resolves the scripted/headless car and returns
**None** for the stock Corsa with no ballast, so `_build` constructs `CorsaC()`
exactly as it always did; `_build` gained `car=None` and `mu_scale=None` (the
car's own `mu_scale`, an explicit value still winning, which is what the wet
rigs pass).
Defaults `ENGINE_DEFAULT = 'sport'` (`ENGINE_SCALE` stock 1.0 / tuned 1.5 /
sport 2.0) and `SOUND_DEFAULT = 'mid'` (`SOUND_VOLUME` 0 / 0.3 / 0.6 / 1.0).
The ESC menu has two pages: PAUSED (Resume / Settings / Reset to sector /
Full reset / Garage / Quit) and SETTINGS (one row per setting with its value —
Map, Engine, Gearbox, ABS, TC, Steer aid, Surface, Camera, Sound — then
Garage, Back). `Sim.apply_setting(key)` cycles, applies live and saves;
`track` and `wet` return True = the session must be rebuilt, which
`Sim.restart()` does by ending it with `stop_reason == 'restart'`. `TAB`
(`'track_next'`) is the same path. `Sim.set_gearbox(mode)` puts the mode on
the input layer and hands a stationary car in gear to the clutch-pedal mode
in NEUTRAL (it would stall on the spot); `Sim.reset` and the session start do
the same. `Sim.set_engine(mode)` swaps `veh.pt_p` for
`from_car(car, power_scale)` live (state untouched). `Sim._audio_apply()`
builds, re-levels or drops `Sim.audio` from `settings.volume`; it does nothing
unless `Sim.sound_enabled`, which only `_interactive_session` sets (a headless
Sim, or V26's duck-typed renderer, never touches the mixer).

**The session loop** is `run_interactive_cli` (`run_garage_cli` = the same
with `opts.garage = True`): one pygame session, `mode` in {garage, drive}.
garage → ENTER / ✕ saves the design, applies it to opts, mode = drive; Quit
ends. drive → `stop_reason` `'garage'` (BACKSPACE, touchpad, the menu or the
settings page) → mode = garage; `'restart'` → a new session; else exit. The
pad object survives every transition and is re-seeded at each boundary
(`GamepadInput.seed_edges`, `Garage._pad_seeded`); the garage also ignores
`'drive'` for its first 0.35 s. `_interactive_session(opts, pad, garage,
settings)` calls `settings.to_opts(opts)` first — the settings are the truth,
opts is the carrier — builds the track with `trk.make_track`, the car with
`abs_on = settings.abs`, the inputs with `steer_limit = settings.steer_aid`
and `k_us_deg = K_US_DEG_MEASURED`, `tc_on = settings.tc`, `power_scale =
settings.power_scale`, and `Sim(..., settings=settings)` with `has_garage`
True whenever `drive.garage` imports; with a real window it sets
`sound_enabled` and calls `_audio_apply()`. The garage design saved in
`runs/garage_design.json` is the car every plain launch drives unless a
`--wing…` flag is explicit -- or, on a map that has one, the build last used
on that map (`last_builds.json`, the pre-race section below), which replaces
it IN MEMORY: only the garage writes `garage_design.json`. `Sim.handle_event` also owns the view toggles
(`camera`, `zoom_*`, `hud`, `vectors`, `gg`, `skid`) — renderer config only,
never physics.

**The build in the loop.** garage → ENTER / ✕ saves the `CarBuild`
(`runs/garage_design.json`, v2) and applies it to opts (`_apply_design`: the
four legacy fields plus `opts.wing_cfg` = `CarBuild.cfg_kwargs(lib)` and
`opts.hud_cfg` = `CarBuild.hud_kwargs(lib)`). The session builds
`VehicleConfig(**legacy, **wing_cfg)`, hands `hud_cfg` to `Sim.hud_cfg`
(merged into `hud_data()`), starts a designed build ARMED (`wing_on = True`)
The pause menu has a third page, `'swarm'` (**Deploy swarm**): rows
`set:sw_pop` / `set:sw_seed` / `set:sw_gens` / `set:sw_T` cycle
`Sim.swarm_opts` through `SWARM_MENU_CHOICES` (LEFT / RIGHT or ENTER),
`swarm_arm` toggles `seed_armed`, `swarm_go` closes the menu, copies the page
to `Sim.swarm_launch`, sets `stop_reason = "swarm"` and quits the session.
`run_interactive_cli` routes that to `run_swarm_cli(opts, settings=,
embedded=True)` -- no `pygame.init` / `quit`, the settings object it already
has, the same `_session_car` -- and then starts a new drive session with the
page's values kept on `opts.swarm_menu`. Nothing on the swarm page restarts
or rebuilds anything until Deploy.

`--swarm N` (and `--swarm-resume STATE`) hands the launch to `run_swarm_cli`
instead of a session: a `drive.ml.swarm.Swarm` -- a genetic algorithm over
`Policy` genomes, N per generation, elites kept, the rest bred by crossover and
mutation, fitness = `env.rollout`'s reward then lap time -- on the SAME track,
car and `VehicleConfig` a session would drive (`_session_car` is the one place
that assembles those, and both paths call it). The window is a REPLAY: the pool
scores a generation and returns each car's `(t, x, y, psi, u, deploy, side)`
trace; the viewer draws them as `HudData.ghosts` at playback time while the
next generation is already computing. No `Vehicle` is stepped in the window
and no physics runs in the render loop. `--swarm-seed` is `none` (the anchor),
a seed lap, or a `Policy` checkpoint; the seed is individual 0 of generation 0,
unmutated. `K` / ESC prompt for the bot's name (empty = the swarm's; letters,
digits, `-` `_`; `meta['name']`) and save the best as a plain `Policy` checkpoint
(`drive/ml/checkpoints/swarm_<name>.json`, lap re-measured at DT_EVAL) that
`--ml-drive` drives and a later swarm seeds from. `drive.ml` is imported
inside `run_swarm_cli` only; the import-guard check in `drive.ml.__main__`
(import `drive.drive`, assert no `drive.ml` module loaded) still holds.

**The seed lap** (`K`, the swarm page's *Seed lap*, or `--seed-lap`) is the
user deciding BEFORE a lap that it is the swarm's base. `K` sets
`Sim.seed_armed`; at the next `start`/`lap` event a recording opens
(`_seed_open`). The page's *Seed lap* does a full `reset()` to the line and
opens the recording there, from the standing start. Every `SEED_LAP_HZ`
(100 Hz) step appends the published state `(t, x, y, psi, u, v, r, beta, ay,
util_f, util_r, wing_deploy)` and the controls `(delta, throttle, brake,
wing_on)`; a step with all four wheels off clears `_seed_valid`. The next
line crossing (`start` or `lap`, more than 5 s after the opening) closes it:
valid (and `LapTimer.lap_valid` for a `lap`) → with a window the
`garage_ui.TextPrompt` asks the lap's name (`KeyboardInput.key_sink` takes
the keys, the sim pauses); ENTER writes
`runs/swarm/seed_<track>_<car>_<name>.json` (`name` = letters, digits, `-`
`_`, empty = a time stamp; also in the file), sets the page's seed to
`latest` and re-opens it on *Deploy*; ESC discards. Headless, the stamp is
used. `latest` is the newest file on this map by mtime. An invalid lap or a
`reset()` discards the rows; `K`'s arm stays. The sim writes plain JSON and
imports nothing from `drive.ml`; `drive/ml/clone.py` reads it, rebuilds the
observation from the rows with `env.observe` on a stand-in object and the
track, and fits the residual net to `clip((user - baseline) / RESID_GAIN)`.
`drive.ml.__main__` asserts the two column lists agree by name.

`--ml-drive CHECKPOINT` puts a trained `drive.ml` policy in the driver's seat
instead of the keyboard/pad, through the SAME local `ScriptedInput` closure the
acceptance scripts use — so no new code reaches the physics path. `drive/ml` is
an OPTIONAL sub-package: **nothing imports it unless this flag is given**, and
a missing checkpoint, a missing numpy or a shape mismatch prints why and hands
the session back to the keyboard rather than stopping it. `drive/ml` never
imports `pygame`.

`--race BOT` (and the pause menu's RACE VS BOT page: `Sim.race_opts['bot']`,
`start_race` / `stop_race`) puts the ML driver in a SECOND car -- `Sim.rival`,
a `Rival`: its own `Vehicle` built from the session's car and `VehicleConfig`,
its own per-wheel surface sample, its own `LapTimer`, the policy through the
same `observe -> Policy.controls` call `--ml-drive` uses. It is stepped ONCE
per physics step, in `Sim.step_physics` AFTER the user's car and reading
nothing of it, at the session's `dt`; the user's `Vehicle` is bit-identical
with and without a rival (V30 asserts it). There is no collision: the renderer
draws it as a labelled `HudData.ghosts` entry. `bot` is `'none'`, `'anchor'`
(`Policy()`, theta = 0 -- the hand-written driver, needs only numpy), `'best'`
(the newest `swarm_*.json`, command line only) or a checkpoint path; the same
lazy try/except as `--ml-drive` -- a bot that cannot be loaded prints why and
the session runs without one. It lines up `RACE_START_OFFSET_M` LEFT of the
line, standing start in 1st like the user; off the map or spun for
`RACE_RESPAWN_S` it rejoins at its last sector line. Any reset (R, SHIFT+R,
the page's Start) restarts the race from the line for both. The HUD's `msg`
carries the gap: `Rival.gap_to` reads the leader's `(progress, t)` trail
(`RACE_GAP_HZ`, unwrapped centreline metres, jumps > 10 m ignored) at the
follower's progress, so `+` is the user behind, in seconds and metres, plus
the bot's lap count / last / best. `opts.race_menu` carries the choice and
whether a race was on across a restart (a map change keeps racing). Adding the
rival costs one more `Vehicle.step` and one `observe` per physics step; V21's
RTF is measured without one.

The page's *Test* (`Sim.start_bot_test`) is the one bot path NOT stepped in
lockstep with the session: bot 1 alone in each `RACE_BOT_CARS` entry, built by
`Sim._race_car`, driven by `drive.ml.evaluate.bot_lap` (the `lap_time`
rollout at `DT_EVAL` for `bot_test_T(track, T)` s -- `BOT_TEST_T` on the arena
and any shorter lap, pro rata on a longer one: Ashdown 167 s, Kestrel 230 s;
whatever the page and terminal print as the Test's length must come from the
same function -- the session's Track, the global wet
folded into `mu_scale`) in a `multiprocessing` pool of its own, collected by
`_bot_test_poll` from the render loop. Nothing of it reaches the session's
physics; its output is text (the page, the HUD `msg`, the terminal). The
swarm's car (`--swarm-car`, the swarm page's *Car*) is built by the same rule,
`_swarm_car`.

exactly as `--wing` does, and `--build NAME` loads a car saved in the garage
library (`runs/library/builds/NAME.json`) instead of the last one built.

**Lap records** (`drive/records.py`, task 19). `Sim.recorder` is None on
every scripted and headless Sim; only `_interactive_session` attaches one
(`records.session_recorder`: a lap map, the standard skidpad, no
`--ml-drive`, not headless), so no acceptance run reads or writes
`runs/records/`. It is fed from `step_physics` at three points and nowhere
else: `recorder.controls(sim, ctl)` right before `Vehicle.step` -- it
QUANTISES the five continuous `Controls` fields in place (road wheel to
2^-24 rad, pedals to 2^-20) and logs the row, so the log is integers and
exact; `recorder.event(sim, e)` for each `LapTimer` event; and
`recorder.after_step(sim)` after `n` / `t` advance (the lap opens at the
end of the crossing step with a snapshot of every dynamic `Vehicle`
attribute, the harness's surface samples and the `LapTimer`, and closes at
the end of the next crossing step, after its sector event). `reset()`, a
live `apply_setting` (anything but sound / camera / shake / graphics /
paint) and the `T` wet toggle
`discard` the open lap: the physics could not replay it; the live change
also `retarget`s the recorder (the engine is in the key, the assists go
with the lap). A lap is filed only when `LapTimer.lap_valid` holds, it went
ROUND (every sector event in order, centreline progress >= 0.95 of the
length: the timer counts a crossing after its lockout however the car got
back to the line), and no step of it ran in slow motion or single-stepped.
A failed write keeps the book in memory and never raises into the step. `records.resimulate(rec)` rebuilds
the car and `VehicleConfig` from the record, restores the snapshot, and
drives the log through a headless `Sim`: V31 asserts the lap time and the
sector times come back BIT FOR BIT, and that a 3-lap scripted run leaves
the right top 5. The class key is D1's; its file name writes `|` as `__`
(Windows). `HudData.pb_lap` / `lap_rank` carry the class PB and the top-5
place of a lap that just landed.

**Medals** (`drive/medals.py`, task 21). `Sim._rec_lap` (the recorder's
lap callback) asks `medals.medal_for(key, time)` for every VALID lap, puts it
on the lap's HUD note and `HudData.lap_medal`, and keeps the class's best in
its records file (`RecordBook.set_best_medal(..., save=False)`, written by
the recorder's filing thread: `LapRecorder.save_later`). The pre-race page
shows the class's targets and your best medal. V31 checks the medals of its
three laps and the best kept. The table is generated, never typed:
`python3 -m drive.medals --build` (816 runs in 26 min on 6 workers for the
first three maps; the three newer circuits make it 1626 runs), and
its self-check soft-fails with that command when a track, a car or the
engine modes changed since.

**Ghosts and the live delta** (`drive/ghosts.py`, task 22). A session with
a recorder gets `Sim.ghosts`, a `GhostSet` on the recorder's book and class
(`opts.ghost_slot` / `ghosts_on` carry the slot and the `J` toggle across a
restart). `step_physics` hands it every `LapTimer` event (a crossing reloads,
a sector flashes); `hud_data` calls `sync(recorder.key)` (a live engine
change, a lap filed, a trace landed), prepends `ghost_tuples` to
`HudData.ghosts` and sets `delta_s` and `sector_flash` / `flash_col`, which
`render._draw_delta` draws at the top centre when the HUD is on. The ghosts
are `_Replay`s, drawn by the existing ground-silhouette path. `records.
resimulate(on_step=, on_event=)` lets V33 read the delta against a lap's own
trace while it is re-driven: tolerance 5 ms, measured 0.17 ms. `J` ->
`'ghosts'`. Nothing reaches the physics.

**The driving tutorial** (`drive/tutorial.py`, `drive/progress.py`, task 23).
A player session (`_player_session`) opens `runs/progress.json` once per
launch in `run_interactive_cli` (`opts.progress`) and, when its `tutorial`
section was never offered, the first session opens the WELCOME page (menu
page `'tutorial_offer'`; `Sim.open_tutorial_offer`) instead of the pre-race
page. `Sim.progress_file` is None in any other run, and with it the pause page's
`tutorial` row. A running `Tutorial` rides on `opts.tutorial` across
restarts and is `Sim.tutorial`; `Sim._tutorial_tick()` runs once per frame in
`run_interactive` after the physics (skipped while a menu is up) and acts on
`Tutorial.tick(sim)`: `'restart'` (the step's map is another one:
`run_interactive_cli` sets `settings.track` to it before the next session,
so TAB cannot leave it), `'page'` (menu page `'tutorial_step'`, paused;
Continue / ESC advance, End stops). Its hooks: `Sim.reset` passes
`command('reset')`; `hud_data` sets `HudData.tutorial` (the overlay dict,
`render._draw_tutorial`, drawn in `R_TUTOR` whether the HUD is on or not).
A step that starts on the line calls `sim.reset(to_checkpoint=False)` once,
and that reset is not the player's. The wing steps need a flank wing:
`run_interactive_cli` gives a car without one `tutorial.wing_car` (the
library's `plate` on both flanks, in memory, `opts.tutorial_car`; no
per-map build is recorded for it) and re-applies the player's design at the
next session; ending the tutorial on it restarts the session. The pre-race
page is not opened while a tutorial runs. V34 drives it end to end headless.
Task 31: steps 11-12 are an optional group (`Step.group`; the page's *Skip
it*, action `'tut_skip_group'`, skips both). Step 12 has `Step.gearbox =
'manual'`: `Sim._tutorial_gearbox(tut)`, first thing in `_tutorial_tick`,
switches an AUTOMATIC to it as the settings page would
(`Sim._gearbox_live`: saved, the lap discarded, the recorder retargeted) and
records `Tutorial.gearbox_prev = (prev, set)`; when no step wants it (passed,
skipped, ended, started over -- `release=True` -- a challenge pick, or a quit:
`_tutorial_gearbox_restore` in `_challenge_switch` and in
`run_interactive_cli`'s `finally`) the player's box comes back. It is
looked at once a step (`Tutorial.gearbox_seen`), and a player's own change
(`apply_setting('gearbox')`) clears `gearbox_prev`: that box stays. The pair
is also saved in the progress file's `tutorial` section, and
`run_interactive_cli` undoes a switch left there by a run that died
(`tutorial.gearbox_left`). The done page's `'tut_wing'` and the WELCOME page's
`'wt_garage'` (both only with a garage) finish / leave and go to the garage
with `Sim.wing_tutor_start` (`Sim._wing_tutor_go`).

**The wing-design tutorial** (`drive/wing_tutorial.py`, task 24). The
garage's hooks: `Garage.tutor` (a `WingTutor` or None) and
`Garage.progress` (the player's progress file, None in a script);
`Garage.frame` calls `tutor.update(self, action)` after the events and
`tutor.draw(self)` after the page (before the menu); `_menu_open` adds its
rows (`wt_start` / `wt_resume`, or `wt_skip` / `wt_hide` / `wt_end` while it
runs) and `_menu_action` runs them; `H` toggles the box (not while a number
field on the shell pages is being typed into: every key is the field's then);
a click or a wheel notch on the box (`WingTutor.hit`) is the box's, not the
page's under it -- checked before the shell sees the event. The garage menu
takes the mouse through `input._menu_mouse`. A tutor inside the design chain
whose state is gone (a new garage: no stated mission; a re-opened design
page: its section gates re-locked) walks back to the first step that is
open (`WingTutor._rewind`). `run_interactive_cli` hands the garage
`opts.progress` and `opts.wing_tutor` and keeps a running one on
`opts.wing_tutor` across the garage <-> drive trips; the drive's Tutorial
page's `wt_garage` row sets `Sim.wing_tutor_start` and goes to the garage,
where it starts (or continues). The last step passes on the garage's
`'drive'`.

On the MISSION and DESIGN pages (task 43) the hooks are the same and the
geometry is the shell's. The box has two places, picked by `g.page` in
`WingTutor.layout` / `draw`: `BOX_CAR` (748, 468, 520, 212 at 1280x800, x the
page's scale; `BOX` is its alias) on the car and library pages, dark as
before; and `g.shell.geom(W, H).tutor_box` on the shell pages -- the work
area's lower right, above the Output pane, light (the CAE theme's text) --
each growing upward by at most `BOX_GROW` (312 px tall). `anchor_rect(g,
anchor)` reads what the last draw recorded: `('bar',)` is the TOOL BAR on
the shell pages (the key bar on the airfoil and library pages, nothing on
the car); `('tool', id)` a tool button (`g.shell.tool_rect(id)`: the `fit`
step points at `('tool', 'save')`); `('nav', key)` the tree's step row from
`nav._hits`, else -- its stage folded or scrolled out -- the stage's row from
`nav._node_hits`, else the Simulation pane's body, never None on the design
page; `('row', key)` a control of the page's `Form` while it is visible
(`Form.rect_of`); `('list', name)` a list of the library page. A control laid
out below the fold is scrolled in ONCE, when its step first finds it laid out
(`WingTutor._reveal` sets `form.reveal`), so the player can scroll away
again. The prose names the shell's labels (*2 Airfoil ▸ Library screening*,
the *'Fit it and choose the endplate's section'* button on *2 Airfoil ▸
Section*, *3 Wing ▸ Design box*, *4 Results ▸ Summary*); the dark pages print
`▸` as `>`. Screening is live, so the `screen` step passes on a frame after
the screen finishes. The self-check replays AeroBO's captured runs
(`aerobo_models.use_fixtures()`: the same job path, no engine, no XFOIL),
pumps them to their end (`g.runs.run_all()`), and asserts that no resolved
anchor lies under the
box at its full height at 1280x800 and 1600x1000 (T4) and that every anchor
resolves on the 1600x1000 shell (T5).

**Challenges** (`drive/challenges.py`, task 25). A player session with a
garage library has `Sim.challenge_build = (opts.design_json -- the WORKING
build, task 44 -- or opts.build_json, lib)`, and with it the pause page's
`challenges` row (pages `'challenges'` and `'challenge'`,
`_challenge_event`). Task 44: the detail page opens with the rows
`set:ch_car` / `set:ch_cfg` (LEFT / RIGHT -> `prev:` / `next:`, ENTER or a
click -> next; `Sim._ch_step`) -- task 45: no `ch_info` row any more, the
whose-wings line (`challenges.wings_line`, the garage's player names) is
the WINGS section's first row, and the page opens on Start; the pick is
`Sim.ch_pick = (car, config)` (None: `_ch_combo`'s default, the car being
driven with 'full'), kept on `opts.ch_pick` by `_challenge_switch` and
restored at every session's start. The page and the list show
`challenges.resolve(file, car, config)`; its stats are `build_stats` of
`config_build(working build, lib, car, config)`. A combo that is not
`available` has no Start, and `ch_go:` refuses it. Start sets
`Sim.challenge_pick` (and `ch_pick`) and restarts;
`run_interactive_cli` (`_challenge_switch`) resolves the combo (no pick:
the file's car, 'full'; unavailable: nothing starts), remembers the
player's track / car / engine / surface (and radius / cw) in
`opts.challenge_prev`, moves the settings to the combo's class (the car
swapped), and `_drive_design` drives `config_build(design, lib, car,
config)` instead of the plain fitted copy while `opts.challenge` has a
config (`opts.design_json` stays the working build). `_interactive_session`
attaches a `ChallengeRun` when that copy (`opts.build_json`) passes
`refusals` (else a HUD note), and `_challenge_wings(sim)` starts it armed
in AUTO (AIR BRAKE kept on a config with side wings). In a run, G
(`handle_event('wing_side')`) steps only `challenges.g_modes(config)`:
AUTO / AIR BRAKE with side wings; a top-only config leaves the mode and
notes "the challenge sets the wings". A stop held at v0 (`StopHold`, task
42) is a reset's pose every held step, which starts every wing stowed;
`Sim._hold_top(ctl)` then puts the top wing as it rides at v0 -- out
(`state.top_raw = top_deploy = 1.0`) when armed and `airbrake.cruise_top
(mode)` says so, or, where that is None (AUTO, AIR BRAKE), when the slot is
'fixed'; else in -- on every held step and on the step it lets go (a
reference's pedal is at full on the first step, so it is never held). Not
the step's `wing_cmd`: a held step may carry a pedal on its way down, and
AIR BRAKE's top wing comes out on the brake. The flanks stay stowed.
Its hooks: `step_physics` calls `challenge.event(self, e)` per LapTimer
event and `challenge.step(self)` after every step (after `n` / `t`);
`reset` calls `challenge.reset()`; `hud_data` puts `challenge.overlay(self)`
in `HudData.tutorial` when no tutorial has the box. Every step,
`ChallengeRun._void` drops the attempt in progress when the session is out
of the class (a live engine change, the T toggle -- `handle_event('wet')`
also resets it), in slow motion or single-stepped. A build refused at the
session start still gets its run (`refused`: listed, endable, never
counted). A class changed elsewhere (TAB, the settings page) ends it and
`_challenge_restore` puts back every class field the player did not change;
`ch_end` and quitting (the `finally` of `run_interactive_cli`) put all of
them back. A challenge pick ends the driving tutorial and a tutorial
started during a challenge ends the challenge. No per-map build switch
while a challenge runs. The drag area counts EACH fitted flank panel's own
D/q (G can put both out). V35 re-runs a subset of the references (every
challenge on corsa / full, plus one other car + config each, rotating: 3
stars, refs.json's value exactly) and the page flow; V44p drives the Car /
Wings rows, the list, a combo not for its car, G in a run and the next
sessions' cars and copies through `_loop_run` (a step's `combo`), and a
held stop's top wing at the let-go (`Sim._hold_top`: ONLY TOP, FIXED's out,
ONLY TOP's in, the reference's and a player's alike).

**The race grid and the swarm panel** (task 26). `RACE_GRID_MAX =
race_grid.GRID_MAX` (5) slots, `RACE_GRID` / `C_RIVALS` from `race_grid`;
`RACE_BOT_CARS = ('own', 'same', corsa, mx5, 540i)`, default `'own'`:
`Sim._race_car(name, meta)` rebuilds the bot's bred car from its checkpoint's
`meta['bred']` (`race_grid.own_car`, the library on `Sim.garage_lib`), or an
old checkpoint's `meta['car']` + `meta['wing']`; `start_race` and
`start_bot_test` hand the meta in (`_bot_meta`). `drive.ml.swarm.Swarm.bred`
(set by `run_swarm_cli` from `race_grid.bred_meta`) goes into the saved
bot's meta and the state file. With three or more bots `_race_hud` prints
the slot number and the gap. The Deploy-swarm page's `pop` / `T` are free
values (`swarm_panel.step_value`; ENTER a coarse step), and the command line
is clamped the same way. `run_swarm_cli` adds `swarm_panel.panel_lines` to
its text panel (the PB read only in a player session). `render.V22_GRID_BOTS`
(5) is the grid the frame budget is held to; V36 asserts it equals
`RACE_GRID_MAX`, races the full grid and measures its cost.

**Polish** (task 27). `Sim._rec_lap` builds `results.card(res, book)` (and,
with sound, `CarSound.chime('pb')` for a new PB, `'medal'` for a new best
medal: `audio.chime_wave`, synthesised, played on mixer channel 1 over the
engine's channel 0); `hud_data` sets `HudData.results` (`results.view`, the
card's age in sim time), `HudData.wheels_xy` (`Sim.wheel_world()`) and
`HudData.shake` (0.35 x speed with a wheel on a kerb -- the corners' inner
0.8 m of the ribbon, `_draw_kerbs`' rule, found by `trk.project` of each
wheel -- or off the ribbon, 1.0 x speed with all four off, speed as V / 20
m/s capped at 1; nothing while paused);
`run_interactive` passes `shake` to `Renderer.update_camera(st, dt, shake=)`
only while `Settings.shake` is on (a live setting, saved; not in the class,
so it never discards a lap). With the look merged the kerb test reads
`HudData.surf4` (the DRAWN kerbs, apex and exit) and falls back to the
inner-0.8 m rule without it. The renderer: `shake_offset` (at most
`SHAKE_PX`, deterministic) moves a plan view's anchor; the chase view takes
`drive/fx.py`'s surface-aware jolt (a kerb's fine vertical buzz, grass /
gravel bumps) while the strength is above 0, and falls back to
`shake_offset`'s eye offset only without fx; the HUD stays put.
`_draw_results` (`R_RESULTS`). The tyre smoke is `drive/fx.py`'s particle
pool (smoke / dust / spray), which replaced task 27's `SmokePool`;
`Renderer.smoke` is a view of it (`live()`, `clear()`), so a full reset
still clears the smoke with the skid marks and V22's busy frame still holds
the pool full (a real lock-up, past fx's hold).

**The pre-race page** (`drive/prerace.py`, task 20). `_interactive_session`
builds `Sim.prerace` (a `PreRace` on the recorder's book and class, the
build on opts, the library's builds via `opts.garage_lib`) whenever it has a
recorder and a renderer, and when `prerace.wanted` holds opens it
(`Sim.open_prerace`: menu page `'prerace'`, paused) before the loop starts;
the pause menu's `timetrial` row reaches it any time. Its rows: `pr_race`
(`Sim.start_timed`: `RecordBook.set_last_build(track, ...)`, then
`reset(to_checkpoint=False)`), `pr_pick` (page `'prerace_pick'`,
`pr_build:<name>` rows: a different build sets `Sim.prerace_pick = (name,
json)` and `restart()`s; `run_interactive_cli` first saves a car that is in
no library file as `'<name> (autosave)'` (`_autosave_build`), then makes the
pick the design), `pr_edit` (only with a garage: `stop_reason = 'garage'`;
the garage's ENTER starts a new session, which opens on the page).
Task 33: `prerace.session_start(opts, key, build_name, build_json,
wanted)` decides it: the page opens when this session's `seen_key` (class,
build name, build content) is not the previous session's (`opts.prerace_seen`,
None for a map with no page) or `opts.prerace_force` is set (the garage's
return, `run_interactive_cli`); it records this session's key and spends
`prerace_force` / `prerace_skip`. The page's `set:pr_ghosts` row toggles
`GhostSet.enabled` (J, for a pad); J itself (`MENU_KEYS`) works on the page. ESC on the
page is the pause page (laps still count there); the hotkeys R / SHIFT+R /
BACKSPACE fall through. The build a map opens with: every player session
(`_player_session`: not `--ml-drive`, headless, `--render off/offscreen`)
records the build it drives as its map's (`_track_build_used`,
`last_builds.json`), and `run_interactive_cli` asks `_track_build` for that
entry at launch (unless `--build` / `--wing` named one) and on a map change,
never after the garage; the switch is in memory, `garage_design.json` is
untouched. PICK rows show `RecordBook.build_best`, filed by
`records.build_id` -- the build's CONTENT (a hash of its JSON without name
and library flag), so an edited car that kept a saved build's name is not
credited to it. The menu fires a mouse row on RELEASE over the row the press
armed, and ignores a press within `CLICK_GUARD_DRAWS` frames of a page
change (a double-click cannot run a row of the page its first click
opened); a list longer than the window scrolls. After a swarm the next session skips the page
(`opts.prerace_skip`). V32 drives all of it by events with no window, and a
click through `Menu.draw` on an offscreen surface.

`Sim` pause menu: `'menu'` -> `_menu_open()` sets `paused = True`, remembers
whether `P` had paused already, and calls `inp.set_menu(True)`; while
`menu.open` every command goes to `_menu_event` (hotkeys `reset` /
`full_reset` / `garage` act directly, the rest through `Menu.handle`). On the
settings page `set:<key>` actions re-show the page with the cursor kept
(`Menu.show(idx=)`), `resume` / `settings_back` return to the PAUSED page.
`_menu_close()` restores the `P` pause if it was there, else `unpause()` so
the first-frame guard swallows the wall-clock gap. Without a renderer
`'menu'` is a plain pause toggle. `HudData.menu` carries it to the renderer.

`LapTimer.update(..., active=True)`: `active=False` (the car is more than 2 m
outside the ribbon — the middle of the open map's pad, where the projection
flips between the two sides of the loop and `s` jumps by hundreds of metres)
suspends crossing detection; a jump of more than 10 m in one step is ignored
regardless (a real 1 kHz step is < 0.06 m). `Sim.on_track` is the ribbon on a
circuit and `trk.on_tarmac` on a track with areas.

Keys: `↑` throttle, `↓` brake, `←/→` steer, `LSHIFT` fine, `Z` clutch,
`SPACE` handbrake, `S` starter, `E`/`Q` shift up/down, `F` flank wing, `G`
wing mode next (`SHIFT+G` back), `R` back to the last sector line, `SHIFT+R` restart the lap, `P` pause, `O` single step,
`[`/`]` slow-mo, `C` camera, `-`/`=`/`0` zoom, `H` HUD, `V` force arrows (`Settings.vectors`), `B` g-g,
`N`/`X` skid, `T` wet, `M` marker, `L` record, `K` arm a seed lap, `J`
ghosts, `TAB` next map, `BACKSPACE` garage, `ESC` pause menu / settings. PS5
pad map: section 6.

### Task 39 (player audit) -- interface additions

* `Settings.hud` (`'minimal'` default; `full` / `minimal` / `off`) and `Settings.camera` are
  written by `_view_event` (H / C) and passed to `ViewConfig`. `clamp` accepts only a str in
  each choice's allowed set (else the default) and a finite ballast. `Settings.load` moves an
  unparsable / non-table file to `<path>.bad` and sets `load_note`. `save` is tmp + `os.replace`
  and sets `save_note` on `OSError`. Both notes are shown once on the first frame.
* `LapTimer.restart()` resets the running lap and keeps `best_lap`, `last_lap`, `sector_best`.
  `reset()` is still the full wipe. `Sim.reset(to_checkpoint=False)` uses `restart()`. A
  checkpoint reset with the clock running sets `lap._valid_run = False`.
* `Sim.reset(..., standing=False)`: a full reset uses `Sim._rolling_pose()` -> `(s0, V0, gear)`,
  else a standing start at `s = 0`. The seed lap passes `standing=True`. *Superseded in
  task 45* (where it rolls, and the pose: see that section's "Rolling starts"). The task-39
  rule, for the record: only on `trk.CIRCUITS` in a time-trial session (prerace page or
  recorder) with no rivals, challenge or active tutorial, `s0 = L - min(150, 0.15 L)`,
  `V0 = min(22, sqrt(0.8 G / max|kappa| on [s0, L]))`.
* `hud_data()`: `lap_valid` is now the RUNNING lap's (`True` on the out-lap). Plus
  `lap_void_why` (`''` when valid) and `out_lap_m` (metres to the line before the first
  crossing on a closed track, outside challenges; else `None`), set as attributes after
  `HudData` is built. `lap.lap_valid` (LAST, results, challenges) is unchanged.
* `render.ViewConfig.hud` defaults to `'minimal'`. `results.card()` always carries `next`
  (from `prerace.next_medal(key, t)` -> `(name, target, gap)` or None).
* During an active tutorial ESC opens the TUTORIAL page. On a page step only `tut_next`
  advances. R / SHIFT+R / BACKSPACE are inert on WELCOME and page steps.
* `input`: `WINDOWFOCUSLOST` and pad removal emit `menu` once (never while a menu is up).
  Hot-plugged and launch pads get `lock_deg`. `aero.library` compares build names folded;
  `unique_name` gives `-2`.

### Task 41 (wing limits, builds per car, two new cars) -- interface additions

**The span rule.** `bodies.span_limit(role, car, h)` is the PHYSICAL maximum:
a flank panel's lower tip may come down to the car's own ground clearance (the
lowest underbody station of its shell), `span <= 2 (h - ground)`; a top wing
may be `1.2 x` the body's width, wherever it is mounted. The rule is static
(roll is not charged). `span_ceiling(role, car, h, unlimited)` is what an
EDITOR may offer: the limit, times `UNLIMITED_FACTOR` (3) when
`Settings.wing_limits == 'unlimited'`, floored at the packaging band's lower
end + 0.05 (a floor `flank_h_band` keeps from ever binding on a real slot).
The garage's span row, the design box and the optimiser's band all read it
through `wing.span_fit`. `area_ceiling` grows the reference-area band in
proportion when the span ceiling passes the packaging band's span.

* `CarBuild.clamp(lib, car=None)` holds every slot inside the fitted car's
  bands (`bodies.flank_x_band / flank_h_band / top_x_band / top_h_band`).
  The three STOCK styles (`bodies.STOCK_STYLES`: hatch, roadster, saloon)
  share the garage's pre-41 bands exactly -- flank x from the Corsa's
  `CAR_X_REAR` / `CAR_X_FRONT` half a chord in, flank h 0.40-1.20, top x
  (-1.9175, 0.55), top h from the garage's own STATIONS deck
  (`bodies.LEGACY_DECK`, a copy garage's self-check proves) + 0.06 + 0.08 up
  to 1.85 -- so no build saved before task 41 moves on the Corsa, the MX-5 or
  the 540i (the MX-5's own tail and the E39's own deck would have moved one).
  A new style (van, bus) reads its bands off its own shell: its bumpers, its
  deck + 0.14, its roof. The span LIMITS stay per car on every style, and the
  preview draws each car's real body. `car=None` means the build's own `car` tag,
  else the Corsa. The SPAN is never clamped by a build. `CarBuild.reset(car)`
  and `CarBuild.for_car(car)` use `bodies.slot_defaults`. Every caller that
  knows the car it will drive passes it (garage: `Garage.car`; drive:
  `settings.car`, or `--car` on a settings-less launch).
* The three stock cars keep the garage's old slot bands and default slots to
  the bit: `bodies.self_check` pins every band edge of all three against the
  pre-41 formulas, and garage's self-check loads a pre-41 build at the old
  extremes on each with its build_id unchanged; the Corsa's clamp and preview
  mesh were checked bit-identical on 3000 random builds.
* `garage_ui.Param` lo / hi may be callables (`Param.band()`): the span row's
  ceiling follows the slot height, the car and the Wing limits setting.

**Unlimited runs.** A player session whose build has any entry in
`bodies.over_limits(opts.build_json, opts.garage_lib, settings.car)` is an
UNLIMITED session (`Sim.over_limits`, `Sim.unlimited`), whatever the setting
says -- the setting gates the editors only. `over_limits` judges a build AS
FITTED to the car (each flank at its h held in `flank_h_band(car)`, the
mirror lock applied, a wrong-role wing skipped -- `CarBuild.clamp`'s rules;
the top limit does not depend on h), so the raw library JSON a page tags
UNLIMITED, the fitted copy a session drives and `challenges.build_stats`
(which clamps first) always agree. It is computed at every session
start, never stored on a build (a wing can be re-saved at another span, a
build driven on another car). A build that cannot be judged (no library, a
legacy one-panel `WingDesign`) counts as official.
* Records: `records.session_recorder(..., over=)` files it in
  `records.unlimited_book(root)` = `<root>/unlimited/<same class file>`;
  `last_builds.json` stays in `<root>`. Its laps carry `unlimited: true` and
  `over_limits: [{slot, wing, span, limit}]`. An official book REFUSES such a
  lap (ValueError). Medals, top-5s, `build_bests`, ghosts and the HUD PB follow
  `recorder.book`, so they separate by construction.
* **`records.publishable(lap)` is the ONLY test the public leaderboard (plan
  T28, not built) may use before submitting a lap.** False for any Unlimited
  lap.
* Challenges: `build_stats` returns `unlimited` and `over_limits`. An
  Unlimited run is judged by the challenge's own rules and counted in
  `progress.json`'s section `challenges_unlimited` (`{key: {best, stars}}`,
  the key `'<id>|<car>|<config>'` since task 44, as in `challenges`;
  never nested in `challenges`); the box, the list and the detail show an
  Unlimited best / stars beside the official one; `total_stars` and
  `menu_row` are official only.
* Tags: `LapRecorder` / `lap_note` result dicts, `results.card` and
  `render.HudData` gain `unlimited`; the pre-race page shows an UNLIMITED
  section and headings, and each PICK row's best is read from that build's
  own book (`PreRace(over=, books=, judge=)`, `drive._prerace_books`).
* `Settings.wing_limits` ('real' | 'unlimited', row `set:wing_limits` after
  Paint, `WING_LIMIT_MODES` / `WING_LIMIT_LABELS`) never discards or
  retargets a lap.

**Builds know their car.**
* `CarBuild` JSON v2 gains `car` (a cars.py key; "" or missing = a legacy,
  any-car build). `records.BUILD_META = ("name", "builtin", "car")` is the ONLY
  list of build labels: `records.build_id` and `prerace._same_build` strip
  exactly it (so no pre-41 PB moves), and anything new that compares builds
  must use one of the two.
* `last_builds.json` keys are `track|car`; a bare `track` key is a pre-41
  entry, read for any car until that car has its own. `RecordBook.last_build(
  track, car)` never returns another car's build (`records.build_fits`), and
  `set_last_build(track, name, build, car)` never files one (False, nothing
  written: it would only wipe that car's own entry).
* `Settings.car_build` maps car -> library build NAME. `drive._car_build` is
  the only resolver, run on a car change (`seen_car` in `run_interactive_cli`
  is the PLAYER's car: it does not advance while a challenge runs, so the
  challenge handing the car back is no car change; not right after the
  garage) and at launch in `_resolve_design(opts, settings)` only when
  `garage_design.json` belongs to another car and no `--build` the library
  holds / `--wing` was given. Order: the default if it is still in the
  library; else keep an own or any-car build; else this car's per-map build;
  else `garage.new_build(car)` plus a note of 120 characters or less. A car
  change that replaces a build with wings that is in no library file
  autosaves it first (`_autosave_build`, as a PICK does). A launch that took
  the default sets `opts.build_from_car`, and the loop's first pass then
  skips the per-map switch.
* **Fitted copies (review of task 41).** The player's working build -- the
  loop's `design`, the garage's build -- is never moved by fitting it to a
  car it is merely driven on. `drive._drive_design(opts, design, lib, car)`
  runs right before EVERY session (the first, after a challenge switch, a car
  change, a PICK, the garage): `opts.build_json` is a copy fitted to `car`
  (`_fitted`: what the physics drives and a lap is filed under) and
  `opts.design_json` the working build (what the library and the per-map
  memory hold: `_track_build_used` and `Sim.start_timed` store it). `_car_build`,
  `_track_build` and a PICK return builds as held, unclamped. `PreRace(...,
  design_json=, fit=)`: `saved()` looks up the held build, and a PICK row's
  best is read under its fitted id (`drive._fit_json`). The garage edits its
  own car's build in place, but opens an any-car / other-car build (handed in
  or loaded) as a fitted copy, `Garage.handed_back()` returning the original
  when the copy was only looked at; back in the loop `_stamp_car` compares
  that with the build as it came in. `race_grid.own_car` fits a bred bot's
  build to the car it was bred in, `tutorial.wing_car(..., car)` to the
  session's.
* Stamping: every garage library save stamps `Garage.car`; back from the
  garage, `_stamp_car` tags the working build with `settings.car` only if its
  content changed.
* `Library.rename('builds', old, new)` writes the new file before removing
  the old one (folding guard kept; a case-only rename rewrites its own file);
  the caller moves `Settings.car_build` references and the per-map memory
  (`RecordBook.rename_last_build(old, new)`: every `last_builds.json` entry
  whose name or build name is `old`, one atomic rewrite, in the garage's
  `records_root`). Wings cannot be renamed.
* `Garage._own_build(name)` (what `S` / SQUARE / F may write over): this
  car's own or an any-car user build -- except an any-car build that ANOTHER
  car's `Settings.car_build` names (review of task 41): that one is saved
  beside (the prompt's hint says whose default it is), so another car's
  default is never edited or re-tagged from here.
* Garage keys: car page `S` (save in place / prompt), `SHIFT+S` (save as),
  `B` / `SHIFT+B` (this car's builds, in place), `F` (this car's default;
  `D` stays "design"); library page `D` / pad `R1` (default), `R` (rename);
  pause-menu actions `build_save`, `build_save_as`, `build_load`,
  `build_default`; `_poll_pad` answers a TextPrompt with CROSS / CIRCLE. The
  library page and W both refuse, in Real mode, a wing past the slot's limit,
  and so does `Designer.commit` for every slot the saved wing would sit in
  (`_past_limit_slots`: this slot, and each other slot of its role carrying
  that wing name unless the mirror overwrites it); the designer's span is
  re-capped to `_span_band()` on every `update()` (the slot's h moved) and
  when `open_designer` reopens it (`_recap_span`, never marking it edited).
  Unlimited mode is unaffected.
* The PICK page: `Sim._picker()` is the pre-race page's `PreRace`, else the
  class-less `sim.build_pick = PreRace(None, None, ...)` made whenever there
  is a garage and a library, so Settings' **Build** row works on every map.
  `prerace.pick_order(builds, car)` is the one order (this car's, any-car,
  other cars' tagged); `Settings`' **Default** row reuses a library build
  with the same content (`prerace._same_build`, this car's first), else
  saves the held build (`PreRace.design_json`, not the fitted copy) under a
  free name, never over a library build; the name goes back to the loop
  (`Sim.build_saved_as`), whose working build takes it, so the next session
  reads as saved.

**Two new cars** (`cars.CAR_ORDER = corsa, mx5, 540i, express, bus`): the
Renault Express 1.4 (R5-based van, E7J, 1995) and a 12 m Mercedes-Benz Citaro
O530 (2005, OM 906 hLA, ZF 6-speed, governed 80 km/h). New OPTIONAL CarSpec
fields, every default the old behaviour (`cars.PHYSICS_DEFAULTS`; the stock
three set none): `tyre_lfzo_f / tyre_lfzo_r` (section 2's exception),
`roll_dist_f`, `eps_f`, `I_eng / I_wf / I_wr` (clutch `K_c / C_c` scale with
`I_eng`), `rev_scaled` (the soft limiter, brake downshift, launch / blip,
stall / crank / fire speeds scale with the rev range), `v_governor`,
`brk_air / brk_lever` (an air-brake equivalent), `brk_valve`
('fixed' | 'scaled' | 'none'), `vmax_by` (replaces `key == "540i"` in
`cars.self_check`; its rpm band is `1.5 n_idle < rpm at Vmax < 1.03 n_cut`),
`own_aids` (the steer aid, PathFollower's lock and LapDriver use the car's own
wheelbase / grip / understeer, `input.aid_for_car`). `CarDerived.tyres` holds
one tyre per wheel (on `der`, not the Vehicle: records snapshot the Vehicle);
`util_f / util_r` read `der.tyre_refs[i]`, which IS `qss.TYRE` on every
unscaled tyre. `powertrain.brake_coeffs` raises ValueError for brakes that
cannot reach full authority inside 400 bar (hydraulic) / 10 bar (air) instead
of saturating silently. `roll_dist_f` stays the 0.74 calibration on every car
that declares none (section 4).

**The new cars' presentation.**
* `render.car_mesh3` draws five styles: hatch, roadster, saloon, van (band
  kind `'box'` = a blind panel; the LAST station is the rear-door face) and bus
  (`'win'` = a painted window band with decal panes and doors,
  `render.side_decal`; the FIRST station is the windscreen face, the last the
  engine bay). Factory colours: Express (226, 226, 220), bus (22, 128, 132),
  both palette colours checked at every tone.
* `Chase3D`: a body taller than `CHASE_TOP_REF` 1.44 m has the Corsa's framing
  scaled about its tail by `k = height / 1.44`; `k = max(1, .)`, so the three
  stock bodies take the pre-41 path bit for bit (render self-check: 84/84
  targets). `CHASE_REAR_CAP` is 4.0 m (was 1.0; every stock car sits under 1.0).
* `gg_envelope` uses `max_ay_car` (the car's own mass, axles, roll split and
  tyre references) on `own_aids` cars and `qss.max_ay` on the stock cars.
  `rev_marks` scales the shift point and amber span by `min(1, n_cut / 6000)`.
* `audio`: an EngineProfile per new car (E7J four; OM 906 turbo-diesel six,
  six firings a cycle, idle 600, cut 2500); `EngineProfile.idle_hi` (1800 on
  every petrol, 950 on the bus) and `rpms` (the self-check's sweep, per
  profile).
* `fx.SLIP_ALPHA0_CAR`: Express 10.9 deg, bus 16.6 deg, the same ramp-steer rig.
  The wheelspin assert follows the smoke gate: a spin held past
  `SLIP_KAPPA_HOLD` for `T_LOCK` smokes; one that never gets there (a bus's
  twin rears) makes none.
* `race_grid.grid_layout`: a car that fits a painted box takes `grid_slot(i)`;
  one that does not lines up behind the painted rows on the centreline; a
  user whose body covers boxes pushes the bots to the next clear boxes.
  `Sim.start_race` places rivals with it. Stock grids are unchanged.
* `ml.evaluate.bot_test_scale`: own_aids cars get `max(1, sqrt(ay_Corsa /
  ay_car))` on the Test budget; the stock cars 1.0.

### Task 44 part A (the wing modes) -- interface additions

The owner (2026-09-25): "Mode wing always on (with and without side)", "No All 3",
and "Top wing would represent the normal wing a car has. Just hide and no use for
side wing." The G / TRIANGLE cycle is now

| mode | int | label | flanks | top wing | `wing_cmd` |
|---|---|---|---|---|---|
| AUTO | 0 | `AUTO` | published law | its slot's mode | `None` |
| AIR BRAKE | 3 | `AIR BRAKE` | law; both out braking | out braking | as task 35 |
| TOP | 4 | `TOP` | stowed, hidden | ACTIVE law, whatever the slot says | `(False, False, law)` |
| TOP FIXED | 5 | `TOP FIXED` | stowed, hidden | always out | `(False, False, True)` |
| TOP FIX+SIDE | 6 | `TOP FIX+SIDE` | published law | always out | `(None, None, True)` |
| LEFT / RIGHT | 1 / -1 | as before | one panel | its slot's mode | as before |

* ALL 3 (2) is gone: `airbrake.ALL`, its cycle entry, `challenges.REF_WING_MODES["all"]`.
  2 is not reused, so an old in-process `opts.wing_mode == 2` is not restored.
  `REF_WING_MODES` gains `top` / `top_fixed` / `top_fixed_side` (V38 drives them).
* TOP's law is the physics' active top wing on the free path, the same arithmetic
  (`vehicle.TOP_BRAKE_ON`, new, the 0.05 that was a literal in `_aero`; `TOP_HOLD`;
  `veh.dev_deadband`), its hold in `AirBrake.top_hold`, seeded from
  `veh.state.top_hold` on entering the mode: on an 'active' slot it deploys the top
  wing exactly as AUTO does (V38: to 0.0, the same stop to the bit).
* `next_mode(m, cfg)` skips the TOP modes without a top wing (`usable`); the Sim's G
  passes `self.veh.cfg`, and a session restores `opts.wing_mode` only when `usable` on
  its car. The refusal on a car with no wing at all is unchanged.
* `hud_data()`: in TOP / TOP FIXED (`flanks_hidden`), once the flanks are in
  (`wing_deploy <= 0`), `dev_left = dev_right = False` and `wing_type = 'off'`: the
  car is drawn without flank panels and the HUD's FLANK line shows none.
* `hud_data()`: the aero panel's TOP line reads the MODE's top law in the TOP modes
  (`top_mode = 'active'` in TOP, `'fixed'` in TOP FIXED / TOP FIX+SIDE), over the garage
  slot's `hud_cfg['top_mode']` (the task 44 review).
* G calls `AirBrake.g_changed(old, new, veh)` after the mode changes (the task 44
  review). The next `command()` takes the car's side latch and top hold afresh and the
  pedal state starts off: AUTO never calls `command()`, so a challenge run's AIR BRAKE ->
  AUTO -> AIR BRAKE kept the first stint's latch. Leaving a TOP mode for one where the
  physics' ACTIVE top law runs again, the car's `state.top_hold` is stale (frozen while
  the top wing was commanded); it is NOT written on the key press (a recorded lap
  re-simulates from its controls, and a G press is not a control), so `handing` keeps
  the top wing on the free path, on the same law with `AirBrake.top_hold` (TOP's own; 0
  after a FIXED mode), until a step where the law triggers afresh; from there the car's
  hold is fresh and the command is the mode's own again. `Sim.step_physics` merges
  `command()` in AUTO only while `handing` (never on a scripted path: V20); the Sim's
  reset clears it. V38 pins both (and that such a run replays bit for bit).
* The tutorial's wing ON lap accepts `airbrake.FLANK_LAW` (AUTO, AIR BRAKE, TOP
  FIX+SIDE) and refuses LEFT / RIGHT / TOP / TOP FIXED ("G (TRIANGLE) steps it back
  to AUTO").
* Help texts: `input.MENU_HELP_KB` / `MENU_HELP_PAD` / `KEY_HELP`, `drive.KEYS_HELP`,
  `controls_page.CONTROLS` ("wing mode (air brake, top)"). `render`'s self-check holds
  the widest label (`TOP FIX+SIDE`) whole on the aero panel and the minimal chip at
  eight UI scales.

### Task 44 part C (the title screen) -- interface additions

* `run_interactive_cli` shows the title (`_title_screen`, drive/title.py) once, after the
  settings, the build and the progress are loaded and before the loop, when `_title_wanted(opts)`:
  `_player_session(opts)` and none of `TITLE_SKIP` (`garage`, `race`, `challenge`, `tutorial`,
  `ml_drive`, `swarm`, `swarm_resume`, `replay`, `script`, `seed_lap`, `headless`,
  `pad_calib`, `self_check`; read with getattr) is set. Its pick: `quit` returns 0 (through the
  loop's `finally`); `garage` sets `mode = "garage"` (with a garage); `challenges` /
  `tutorial` / `settings` (`TITLE_PAGES` == `title.PAGES`) set `opts.open_page`; `drive`
  changes nothing. The title never stops a launch: one that cannot open is `drive`. The pad
  it ends with is the loop's `pad`.
* `_interactive_session` reads `opts.open_page` once (and clears it): `Sim.open_page(page)`
  opens the pause menu on the Settings / Challenges / Tutorial page (the pause page when the
  session cannot list that page: no progress file, no garage library; False without a
  renderer) in place of WELCOME and the pre-race page. WELCOME is then not marked offered.
* `_title_bottom(...)` -> `[("CAR", ..), ("MAP", ..), ("BUILD", ..)]`: the build the loop's
  first pass drives (the map's memory by the first pass's own test, `_track_build`, quietly).
* `_loop_run(..., title="drive", titles=None)`: the title is stubbed to pick `title` (each
  call's bottom line appended to `titles`), the garage (`_painted_garage`) to hand the build
  back and drive (a `{'garage': True}` row); each session row carries `page`. V44.
* `title._SceneRenderer` overrides `Renderer._draw_car3d` (the car pass) and reads / restores
  `render._CAR`, `render._PAINT`, `_st`, `_ctl`, `_spin`, `_spin_t`, `_spin_w`: a change to
  those in `render.py` must keep `title.self_check` row 7 passing.
* The title quits at once only on `quit`: ESC (its own key map, `title._title_keys()`:
  `input.MENU_KEYS` with ESC -> `quit` and no P) and the window's close. `menu` (a right
  click, a pad's OPTIONS) and `back` (CIRCLE) move the cursor to Quit (the task 44 review:
  a right click or P closed the game). `input.KeyboardInput.menu_keys` (default
  `MENU_KEYS`) is the map read while `menu` is set; a screen with its own keys sets it.

### Task 45 (player audit, rounds 2 and 3) -- interface additions

Round 2 (31 fixes) and round 3 (the owner's four answers). `.handoff/45-player-audit.md`.

* **Rolling starts.** `Sim._rolling_pose()` -> `(s0, V0, gear)` covers every closed map in
  a time-trial session (circuits, the open map's perimeter, the skidpad), lap and circle
  challenges (with a window), and the tutorial's `TUTORIAL_ROLLING` steps (with a window).
  None -- a standing start -- on the dragstrip, in a race, a stop or strip challenge, the
  tutorial's other steps and a scripted Sim (no recorder, no pre-race page). The pose is
  on a straight (`|kappa| < ROLL_STRAIGHT_KAPPA`, 0.005 1/m), the first case that applies:
  1. the straight INTO the line, where it is at least `ROLL_RUNIN_MIN_M` (45 m) long and
     leaves `ROLL_CLEAR_S` (3 s) at V0 before the first corner past the line: at its
     start, at most `ROLL_RUNIN_MAX_M` (250 m, and `ROLL_BACK_MAX_FRAC` 0.35 L) out; V0
     22 m/s;
  2. else the nearest straight back from the line that has `ROLL_CLEAR_S` at V0 before
     its corner: the further out of `ROLL_BACK_M` (150 m, 0.15 L) and `ROLL_CLEAR_S` at
     V0 before that corner, never before the straight's start; V0 the run-in's tightest
     corner at 0.8 g, capped at 22 m/s;
  3. else the nearest straight at least `ROLL_STRAIGHT_M` (15 m) long, at its start, V0
     cut to its length / `ROLL_CLEAR_S`;
  4. no straight (the skidpad): `L - min(150, 0.15 L)`, in the circle at `ROLL_CORNER_G`
     0.5 g x the wet (the tutorial's circle at `ROLL_TUTORIAL_G` 0.3 g, round 3).

  Cases 1-3 never start behind the last split line nor more than `ROLL_BACK_MAX_M`
  (400 m, 0.35 L) out, so the pose is inside the last sector. `gear` is the first the
  automatic holds at V0 on full throttle. `s0` per map (V45 pins them): arena 1106.2 m
  (case 3: out of the R = 35 m T6 hairpin, 143 m to the line, 30 m of straight to T7
  in 3 s at 10 m/s = 36 km/h), linden 1060.4, kestrel 1843.3, ashdown 1330.5 (case 1:
  50 / 70 / 59.5 m out at 22 m/s), open 1492.7 (case 2: 150 m out, 18.8 m/s), skidpad
  267.0 (case 4: 47 m back). The pose before this rule put the arena's car 16 m before
  that hairpin: holding UP was the gravel in 2 s (the round-3 review).

  `R` with no lap running or in the first sector is that rolling start with a fresh
  clock; past a split it is the sector line and voids the lap. Round 3:
  `Sim.roll_time_trial()` is called once in
  `_interactive_session` after the race block, so a time trial opens rolling with or
  without the pre-race page (not a challenge, the tutorial, rivals, or where
  `_rolling_pose()` is None); it posts RACE's note unless one is already up.
* **Timing (round 3).** `LapTimer.update(..., counts=True)`: `counts=False` (the recorder
  is not recording and no tutorial practice lap) closes the lap as LAST but gives no BEST
  and no sector best, and gives back the sector bests the lap set (`_sec_best0`, taken at
  each lap start). `LapTimer.void_last()` puts back BEST and the sector bests from
  `_undo` when `step_physics` sees `recorder.last['valid']` False for a lap closed this
  step ('not a full lap'). `lap_valid` still means off track only. A run with no recorder:
  an off-track lap no longer sets sector bests. Records re-simulation and V20 unchanged.
* **HUD.** `hud_data()` adds `last_valid` (`lap.lap_valid` AND the recorder's verdict on
  the last lap; a tutorial practice lap ignored), `global_wet` and (round 3) `stop_board`
  (`ChallengeRun.marks(sim)`: (marker s, board s) or None). `render._delta_text(d)`: `|d| <
  0.005` reads `'0.00'` in `C_HUD_TEXT`. The bottom bar carries only short flags; game
  notes are a toast above it. A stall reads `render.STALL_AUTO` (`'engine stalled: S to
  restart'`) on the automatic and as the short fallback, `STALL_MANUAL` on the manual boxes.
* **Keys and settings.** `H` cycles minimal -> full -> off; `Settings.vectors` (default
  False) is `V`, saved; Settings has a HUD row. `SHIFT+G` -> `'wing_side_prev'`. TAB is
  `MENU_KEYS['track_next']` on the pause and TIME TRIAL pages. The pause page gains
  `title` (Main menu) and a quit that asks twice; during a challenge it is a short page
  (`ch_this`, `challenges`, `full_reset` as Retry, `reset` on laps). The pause footer reads
  `R sector line   SHIFT+R restart lap` (round 3).
* **Garage.** `Garage.run()` may return `'title'`. `R` / the menu's reset and the
  library's `DEL` need a second press within `ARM_S` (2.5 s); `U` undoes a reset; BACKSPACE
  never deletes. Round 3: `GARAGE_HELP_KB` / `GARAGE_HELP_PAD` are 17 + 8 rows of at most
  `HELP_TEXT_MAX` (32) characters; build names in the menu rows are cut to `MENU_NAME_MAX`
  (10, `_menu_name`); the wing tutorial's skip row has no step title. The default row
  reads `Set as <car> default  (now: <name>)`, so the menu, wing tutorial on, fits
  1280x720, 1280x800, 1440x900 and 1600x900 with no help line wrapped
  (`_check_menu_fits`).
  `Garage.try_ready_made()` selects the first empty slot (left, right, top) on the car page
  and presses `W` once; background 'wing data' notes do not cover its hint.
* **Pre-race (round 3).** `prerace.no_wings(build_json)`; `PreRace.offers_wings()` (a
  garage and no wing in any slot) adds `TRY_WINGS` -> `'pr_wings'` under Edit and its help
  line (`PreRace.help()`). `pr_wings` ends the session to the garage with
  `Sim.garage_try_wing`, which `run_interactive_cli` carries on `opts.garage_try_wing`
  into `Garage.try_ready_made()`. `PreRace.unsaved()` (a library to miss from, and not
  saved) is the one '(not saved)' rule for the page, the pause subtitle and Settings.
  `drive.NO_WINGS_NOTE` points at `W`.
* **Challenges, round 2.** The box's `flash` is the result alone, `warn` what it lacks
  (`result_warn`); `combo_best` reads a pre-task-44 per-challenge entry as the Corsa /
  'full' combo's; the list has the Car / Wings rows.
* **Challenges, the pause row (round 3).** `pick_stars(progress, car='corsa',
  config='full', allc=None, refs=None)` -> (got, of): one pick's official stars, 3 per
  challenge it can drive, as the list's rows count them. `menu_row(progress, run=None,
  car='corsa', config='full')` says that pick's ('Challenges: 3 of 24 stars (this car +
  wings)'; with a run, its title); the pause page passes `Sim._ch_combo()`, the list's
  subtitle uses the same `pick_stars`, and `total_stars` is only its '(all: ...)'.
* **Challenges, the stop board (round 3).** `MARKER_T = 3.0` s of v0, `BOARD_TOL = 1.0` m.
  `board_marks(ch)` -> (marker s, board s) = (`START_S + MARKER_T·v0`, marker + the
  3-star distance) on a measured stop, else None. `stars_for(ch, value, stats, miss=None)`:
  on a stop with a board the 3rd star is `|miss| <= BOARD_TOL` (the nose past the board,
  short < 0) plus the 3-star build rule, on top of the 2-star distance. `nose_x(car)` is
  `bodies.body(car).x_front`. `ChallengeRun` gains `board`, `miss`, `board_line`,
  `nose_s(sim)`, `marks(sim)`, `aim(sim)` (the overlay's new `aim` key, drawn amber /
  green by `render._draw_tutorial`). `measure(..., brake_at=None)` holds the reference on
  until its nose reaches `brake_at` (`_BrakeAt`; the references themselves brake at once,
  refs.json unchanged by the board) and returns `miss`; `board_brake_s(ch)` is where the
  reference brakes to stop on the board (V35). `scenery.stop_board_rects(tr, marks)`,
  `PAINT_MARKER` / `PAINT_BOARD_DARK` (world.PAINT_RGB keys 3, 4); `World.marks` paints
  them and `_stop_uprights` stands cones and checker boards in chase.
* **Challenges, the player's setup (round 3).** A challenge keeps the player's ABS, TC,
  gearbox, ballast and wings (unchanged); the page and the result say what differs.
  `ref_setup(ch)` (ABS / TC from `ref`, `REF_GEARBOX = 'auto'`; `measure` reads it),
  `player_setup(settings)`, `setup_diffs(ch, setup, parts, stats)` (TC not flagged on a
  stop), `setup_note`, `setup_section` (YOUR SETUP, `menu.Warn` rows),
  `result_warn(ch, value, n, stats, setup=None, parts=None)`, `detail(..., setup=None)`,
  `ChallengeRun(..., parts=None)`. `wings_line` names built-in wings by
  `garage.wing_shown`. `menu.Warn(str)` draws a help row amber (`C_WARN`);
  `Menu.show(note_under=False)` puts the note under the items' key legend.
* **Tutorial (round 3).** `frame_of` carries `push_g` (|`Vehicle.F_wing`| / m g) and
  `mass`; the ON lap's result has `push_g` / `push` (N); the wing page's last row and the
  done page show it (`PUSH_MIN_G` = 0.002: below it, "hardly out"). `TUTORIAL_WING` stays
  `'plate'` (the bigger ready-made side wing). A drive step started in a new session or a
  resumed tutorial starts from its own start (`Tutorial.tick`'s `fresh`); the circle's
  roll-in before the first crossing is not judged.

## 9. Reconciliations (where the subsystem specs disagreed)

1. **Tyre model.** `numerics.txt` proposed a simplified single-shape-function
   tyre; `tyre.txt` proposed full MF6.2 from the `.tir`. **Full MF6.2 wins** —
   it was verified to reproduce `qss.corner_speed` to 0.005% by monkey-patching
   `qss.fy_max`, which is stronger evidence than any refit. Combined slip uses
   the MF `Gxa`/`Gyk` weighting functions, not an elliptical projection.
2. **Relaxation integration.** `tyre.txt` wanted backward Euler inside
   `step_contact`; `numerics.txt` wanted an exact exponential owned by the
   integrator. **Exact exponential in `vehicle.py`** — the staggered ordering is
   the measured difference between stable and divergent at low speed, and it
   must be visible in one place.
3. **Low-speed damper.** `tyre.txt`: raised cosine, `VXLOW = 1.0`, ellipse cap
   at 1.05. `numerics.txt`: Besselink, `V_low = 2.5`, implicit on `omega`.
   **Both**: Besselink form at `V_low = 2.5` with the **implicit** `omega`
   update (`kv*R_e²/I` reaches 2300 s⁻¹ at standstill — explicit blows up), and
   the ellipse cap at `ECAP = 1.05` on the reported force so the damper cannot
   manufacture grip.
4. **Wheel inertia.** `chassis.txt` 0.65, `numerics.txt` 0.85/0.75,
   `powertrain.txt` 0.76/0.73. **`powertrain.txt` wins** (0.76 front / 0.73
   rear) — it is the only bottom-up build from the actual 5.5J×14 + 175/65R14 +
   236 mm disc / 200 mm drum. `powertrain.py` owns these numbers and hands
   `I_w_eff` to `vehicle.py`; `vehicle.py` never hard-codes a wheel inertia.
5. **Engine inertia.** `powertrain.txt` `I_ENG = 0.16`, `I_TRANS = 0.020`, which
   is the pair that produced a verified 15.50 s 0–100 km/h. **Use those.**
   `numerics.txt`'s `Ie = 0.12` came with a different (rigid-clutch) driveline.
6. **Steering filter location.** `chassis.txt` put the ramp+limiter inside
   `vehicle.py`; `harness.txt` put it in the input layer. **Input layer wins** —
   validation scripts must be able to command `delta` directly. `vehicle.py`
   keeps only Ackermann and compliance steer, which are physics.
7. **Wing side selection.** `numerics.txt` `tanh(ay_filt)`, `chassis.txt`
   `sign(steer)`. **`sign(steer)` with a 5% deadband and a 0.3 s hold** — at the
   limit `beta` is negative in a left turn, so anything derived from `beta` or
   `v` deploys the panel on the wrong side exactly when it matters (measured:
   −2.14% instead of +2.35%).
8. **0–100 km/h target.** `powertrain.txt` reproduces the commonly quoted
   ~15.5 s; `numerics.txt` notes Opel's own figure is 14.4 s. Accept
   **14.5–16.0 s** as passing and report the actual number. Never tune the
   torque curve or `eta_drive` to move it — those are pinned by `Vmax`.
9. **Driver aids never enter a measurement.** ABS (`VehicleConfig.abs_on`),
   TC (`tc_on`), the **Engine setting** (`ENGINE_SCALE`), the steering limiter,
   the sound and the settings file are all OFF / 1.0 / bypassed / unread on
   every scripted, headless and rig path; only the interactive session
   switches them on. `cfg.power_scale` is the driver's Engine setting **and
   nothing else**, and `_build` sets it to exactly `1.0` on every scripted
   path. It briefly also carried the car's own `engine_scale = T_max/110`,
   because otherwise `--car 540i --script accel` measured a 1780 kg car with
   a 110 N·m Corsa engine and reported 0–100 km/h in 24.3 s; `engine_curve`
   (§3) makes that unnecessary, and this reconciliation is back to meaning
   what it originally said. The blip, the restart and the launch assist's band are
   part of the driver model (`auto_clutch`) and are on in the scripts that
   drive with the automatic box, which is what the accel and lap acceptance
   numbers already assumed (a driver who blips and holds the launch rpm);
   they move no number outside its band (0–100 km/h 14.97 s → 14.80 s with
   the launch band, scripted arena lap 61.1 s).

## 10. Two findings to report upward (do not silently absorb)

* **`qss.residuals` has an algebra bug in `Y_r`.** Line ~88 has
  `Y_r = (a*m*a_y + F*x_w)/L`; solving `qss`'s own two stated equations gives
  `Y_r = (a*m*a_y - F*(a - x_w))/L`. With the wing OFF (`F = 0`) they agree, so
  every baseline number stands; with the wing ON the rear goes spuriously
  limiting, which is why `qss.sweep()` reports the sealed plate (+1.18%) as
  *worse* than the clean fin (+1.21%) at R=100 — impossible for a front-limited
  car. Do not patch `qss.py` as part of this build; surface it.
* **`alpha_peak_deg = 7.0` in `qss.py`/`crossover.py` is too low.** The MF
  front-axle peak at the R=100 limit load split is 10.35°. Scrub drag goes as
  `sin(alpha_peak)`, so this moves every power-limited answer
  (`corner_speed(130)` 31.88 → 29.03 m/s). Report it; do not change `qss.py`.
