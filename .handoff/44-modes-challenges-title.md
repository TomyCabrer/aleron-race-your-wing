# Task 44 — the TOP wing modes, challenges in any car and wing config, a title screen

Owner (2026-09-25), verbatim:

> *"Mode wing always on (with and without side). When doing the challenges on
> top we decide full wing, only top, only top fixed, top fix + side. ... We
> need a proper title screen, with a background."*

Asked and answered before building:

| question | owner's answer |
|---|---|
| G modes | add the configs as modes; **"No All 3 -- [all three] only out only when braking"**: ALL 3 goes, AIR BRAKE stays |
| challenges | **the player picks the config at the top of the challenge page, and also picks the car** |
| what FULL WING means | **"full wing = top + side (make it clear for the user)"** |
| a build with no top wing, the side wings in a top-only config | **"Top wing would represent the normal wing a car has. Just hide and no use for side wing."** -- and, of the questions: *"Why is this complicated."* So: the stock top wing stands in, the side wings are hidden and unused, nothing more |
| the title's background | **"Randomised the type and number of cars (1 per type of car)"** |
| saving airfoils / wings, designing a wing by hand (chord, span, airfoil) | **not in this task**: they wait for task 43 (the AeroBO designer) |

```
# after the review's fixes
python3 -m drive.airbrake / tutorial / input / controls_page / menu / challenges  ->  PASS each
python3 -m drive.render               ->  PASS 60/60 (TOP FIX+SIDE whole at eight UI scales)
python3 -m drive.title                ->  PASS (12 rows; 1280x800, 5 cars: mean 6.31 ms, p99 7.65 ms)
python3 -m drive.drive --self-check   ->  ALL PASS  (V38 wing modes, V35 16 combos = refs.json, V44p the picker,
                                          V44 the title; V20 sha256 ed41b7f781e959ea IDENTICAL)
python3 -m drive.validate             ->  82/82   pass  0 HARD  0 soft   [125.4 s]
python3 -m drive.validate --modules   -> 124/124  pass  0 HARD  0 soft   [537.2 s]  (123 + title)
python3 -m drive.drive --render offscreen      ->  session line, no traceback (no title: not a player session)
main([]) under SDL dummy, ENTER on the title   ->  title -> drive, first session on WELCOME, exit 0
main([]) under SDL dummy, DOWN + ENTER         ->  title -> challenges, the CHALLENGES list, exit 0
main([]), a right click + P, then DOWN + ENTER  ->  title -> drive (the click only moved to Quit), exit 0

# after the fix of the held stops' fixed top wing (below)
python3 -m drive.airbrake / challenges  ->  PASS each (the cruising row; no FIXED stop = its twin)
python3 -m drive.drive --self-check   ->  ALL PASS  (V44p (d): ONLY TOP, FIXED 1.00 at the let-go,
                                          ONLY TOP 0.00, a player's the same; V35 16 combos = refs.json;
                                          V20 sha256 ed41b7f781e959ea IDENTICAL)
python3 -m drive.validate             ->  82/82   pass  0 HARD  0 soft   [126.7 s]
python3 -m drive.validate --modules   -> 124/124  pass  0 HARD  0 soft   [538.3 s]
```

## Decisions made without asking (the owner may overrule any)

* **The TOP modes' numbers are 4, 5, 6; 2 (ALL 3) is not reused**, so an old
  in-process `opts.wing_mode == 2` is simply not restored.
* **TOP's "out when you brake or turn" is the physics' own active top-wing
  law** (`vehicle.TOP_BRAKE_ON`, `TOP_HOLD`, the car's deadband), run in
  `AirBrake` whatever the garage slot says, so on an 'active' slot TOP deploys
  the top wing exactly as AUTO does (V38: to 0.0, the same stop to the bit).
* **"Just hide" is visual once the flanks are in**: in TOP / TOP FIXED the HUD
  and the car drawing show no flank panel from the step the flanks read fully
  stowed. Switching from AUTO with a panel out goes to the free path, whose
  per-flank deploy reads 0 at once, so the panel vanishes rather than folding
  away (the free path's existing behaviour; AIR BRAKE avoids it by staying on
  the free path).
* **The HUD keeps the short label `TOP FIX+SIDE`**: "TOP FIXED+SIDE" and
  "TOP FIXED + SIDE" were measured and get cut on the aero panel at several of
  the eight UI scales. The challenge page's config row has room for the full
  `TOP FIXED + SIDE`.
* **The challenge's reference car is the config's stock wings** (`rear-s1223`
  on top, the published plate pair on the flanks), fitted to the chosen car;
  Corsa + FULL WING is exactly the old 'tall' reference (`airbrake_150` still
  84.52898970471306 m).
* **The class keeps its map, engine and surface; only the car is swapped.** A
  combo whose reference cannot be driven is *not for this car*: 8 of 160, the
  Citaro's stops from 100 and 150 (it is governed to 80 km/h). Its wet stop
  from 80 is allowed with a 0.5 km/h tolerance.
* **A lap reference that fails backs its driver's margin off by 0.05 per try
  (to 0.70)** and records the margin that worked (`drove`): the two MX-5 arena
  top-only combos drove at 0.85.
* **Challenge rules changed** because every config now has a top wing and the
  bus's drag area is 4.7 m^2 (the blurbs say so): *Stop from 100*'s third star
  went from "no wings" to at most 15 kg of wing; *Wet circle*'s from 12 to
  15 kg; the two dry laps' limit from 15 to 20 kg of wing and their third star
  from 10.5 to 15 kg; *Wet arena*'s third star from a drag area of at most
  0.70 m^2 to at most 0.40 m^2 of wing a slot; *Hold the circle* lost its
  "flank wings only" rule (the config sets the slots now).
* **Stars saved before task 44 (one per challenge) no longer count.** They stay
  in `runs/progress.json`, unread, so the owner's existing stars show as zero.
* **The title's cars run as a train, 28-70 m apart**, re-spaced at every
  camera cut (the cut hides it), not spread evenly round the lap: evenly spread,
  the nearest other car was 220-380 m away and never on screen (`title.GAP_M`).
* **The title is shown once, at launch.** Garage from the title opens the
  garage exactly as `--garage` does, so quitting the garage quits the game.
* **The title's Challenges hint** says the new thing: "three stars each: you
  pick the car and the wings".

## What it does

### 1. The wing modes (`drive/airbrake.py`; G / TRIANGLE)

| mode | int | HUD (`AERO <label>`) | side wings | top wing | `wing_cmd` (free path) |
|---|---|---|---|---|---|
| **AUTO** | 0 | `AUTO` | the outer one in corners (published law) | its garage slot's mode | none (the published path) |
| **AIR BRAKE** | 3 | `AIR BRAKE` | as AUTO, and both out while you brake | out while you brake | as task 35 |
| **TOP** | 4 | `TOP` | stowed and hidden | **out when you brake or turn** (the active law, whatever the slot says) | `(False, False, law)` |
| **TOP FIXED** | 5 | `TOP FIXED` | stowed and hidden | **always out** | `(False, False, True)` |
| **TOP FIX+SIDE** | 6 | `TOP FIX+SIDE` | the outer one in corners | **always out** | `(None, None, True)` |
| LEFT / RIGHT | 1 / -1 | as before | one panel | its slot's mode | as before |

* **ALL 3 is gone** everywhere: `airbrake.ALL`, the cycle, the labels, the
  texts, `challenges.REF_WING_MODES["all"]`, V38's rows, the tutorial's check,
  the help texts, README and CONTRACT.
* **G skips the TOP modes on a car with no top wing** (`next_mode(m, cfg)`,
  `usable`), and a restart restores the kept mode only if this car can use it.
  The refusal on a car with no wing at all is unchanged.
* Everything stays gated by F / CIRCLE (armed), as every wing always was. The
  AUTO path is untouched: V20's sha256 is still `ed41b7f781e959ea`.
* The tutorial's wing ON lap accepts the modes that keep the side wings on the
  corner law (AUTO, AIR BRAKE, TOP FIX+SIDE) and refuses LEFT / RIGHT / TOP /
  TOP FIXED with the existing "G (TRIANGLE) steps it back to AUTO".
* Help texts: `input.py` (the menu help columns, the key help and its self-check
  claims), `drive.KEYS_HELP`, `controls_page` ("wing mode (air brake, top)"),
  README's key table and wing-mode section.

### 2. Challenges: the car and the wings, picked at the top of the page (`drive/challenges.py`)

```
Car     < Opel Corsa C 1.2 >
Wings   < FULL WING: top + side >
top: rear-s1223 (stock) - side: plate (stock)
```

| key | Wings row | top wing | side wings | G in a run |
|---|---|---|---|---|
| `full` | **FULL WING: top + side** | moves: out when you brake or turn | the outer one out in corners | AUTO / AIR BRAKE |
| `top` | **ONLY TOP** | moves: out when you brake or turn | hidden, unused | "the challenge sets the wings" |
| `top_fixed` | **ONLY TOP, FIXED** | always out | hidden, unused | "the challenge sets the wings" |
| `top_fixed_side` | **TOP FIXED + SIDE** | always out | the outer one out in corners | AUTO / AIR BRAKE |

* LEFT / RIGHT (d-pad) or a click / ENTER change the rows; the page starts on
  the car being driven with FULL WING and remembers the pick for the launch
  (`opts.ch_pick`). A WINGS section says what each wing does and what G does.
* **Applied to the fitted copy** of the player's build for the chosen car (the
  build on disk never changes): the top slot is the player's own top wing, else
  the stock `rear-s1223` lent where that car's default top wing goes, its mode
  forced to active or fixed by the config; the side slots are the player's
  flanks, else the stock plate pair, and removed on the two top-only configs.
  The constraints (wing area, wing mass, ballast, drag area) judge that copy.
* **G in a challenge run** steps only `challenges.g_modes(config)`: AUTO and
  AIR BRAKE on a config with side wings; a top-only config keeps AUTO and says
  "the challenge sets the wings: ONLY TOP (G changes nothing here)". Outside a
  challenge G has the whole cycle of section 1. The wings are armed at the start
  of every run.
* **References and stars per (challenge, car, config)**: 8 x 5 x 4 = 160
  combos, measured into ONE file, `drive/data/challenges/refs.json` (152 ok,
  8 unavailable with a why); the thresholds are derived at load with `STAR_X`
  (1.12 / 1.06 / 1.02). `validate()` rejects a number typed into a challenge
  file, `constraints.slots` and `ref.build`; a refs.json entry measured with
  another class, driver or G mode reads as out of date. `--measure` / `--write
  [--only S] [--jobs N]` measure them in a process pool (130 s with 10 jobs).
* Progress is keyed `<id>|<car>|<config>`; the pause row counts 3 x the 152
  available combos; the list shows the stars of the current pick; a combo that
  is not available has no Start and `ch_go:` refuses it.

### 3. The title screen (`drive/title.py`, new)

A launch opens on **CARSIM** (SIM in the accent orange) over a live 3-D scene:
a random dressed circuit and **one to five cars, never two of one type**, each
replaying its reference lap (`medals.reference_trace`, the first of sport /
tuned / stock with a lap) in its own body and paint, with body roll, steered
wheels, spinning rims and brake lamps from its trace. A chase camera cuts to
the next car every 10 s (the scene fades, the menu never does). No HUD, no
physics.

* Menu: **Drive · Challenges · Garage · Tutorial · Settings · Quit**, a hint
  line for the highlighted row, the bottom line `CAR / MAP / BUILD` of what
  Drive starts, a key-hint footer, and the camera car's name. The pause menu's
  input stack: arrows, ENTER, the mouse (hover, click, the click guard), a pad,
  hot-plug. ESC quits; a right click, OPTIONS or CIRCLE only move to Quit
  (ENTER / CROSS there quits); BACKSPACE is the garage.
* Drive = today's first session (WELCOME on a first launch, TIME TRIAL on a
  timed map, exactly as before); Challenges / Tutorial / Settings = a session
  that opens on that pause-menu page (`Sim.open_page`); Garage = the garage;
  Quit = exit 0. A title that cannot open never stops a launch: it drives.
* Skipped by any launch that says what it is for (`TITLE_SKIP`: `--garage`,
  `--race`, `--challenge`, `--tutorial`, `--ml-drive`, a swarm, a replay, a
  script, `--seed-lap`, `--headless`, `--render offscreen`, ...).
* Fallback: a slowly panned panorama (`world.build_panorama`) when there is no
  reference lap, the renderer fails, or a real window's median frame is over
  40 ms for 60 frames.

## What it measures

### The wing modes (V38: the tuned Corsa, plate x2 + rear-s1223, ABS, stop from 150)

| mode | stop | what V38 holds |
|---|---|---|
| AUTO | 90.28 m | no wing command reaches the physics |
| AIR BRAKE | 89.69 m (-0.7 %) | all three out (1.00) in the stop, no flank out before the brake |
| TOP | 90.28 m | the top wing follows AUTO's active law to 0.0 (the same stop to the bit); on a 'fixed' slot it is in on the way up (0.00) and out braking (1.00); no flank ever out, on the skidpad too |
| TOP FIXED | 90.33 m | out on the way up (1.00), so slower to 150: it brakes at 23.87 s against 23.32 s |
| TOP FIX+SIDE | 90.33 m | flank commands `None`: AUTO's outer flank on the skidpad, exactly; the top out |

Also: one flank under the air brake keeps its heading (0.11 deg); trail-braking
on the skidpad, the largest flank step 0.0050; G without a top wing cycles
AIR BRAKE, LEFT, RIGHT, AUTO, and all 7 with one; TOP hides the flanks on the
HUD, and the HUD's TOP line says the mode's ACT / FIX; after the review, AIR BRAKE
-> AUTO -> AIR BRAKE in the other bend puts the outer flank out from the first
step, and TOP -> AUTO keeps an active top wing in on the straight (0.00), out on
the brake (1.00), in after, replaying bit for bit. The differences stay small
because the tyres do the stopping (task 35).

### The challenge references (refs.json; value per car, FULL / ONLY TOP / ONLY TOP FIXED / TOP FIXED + SIDE)

| Corsa | FULL WING | ONLY TOP | ONLY TOP, FIXED | TOP FIXED + SIDE |
|---|---|---|---|---|
| Stop from 100 | 39.43 m | 39.18 m | 39.02 m | 39.07 m |
| Wet stop from 80 | 39.68 m | 39.63 m | 39.59 m | 39.80 m |
| Hold the circle | 0.732 g | 0.733 g | 0.733 g | 0.732 g |
| Wet circle | 0.462 g | 0.463 g | 0.463 g | 0.462 g |
| Air brake from 150 | 84.53 m | 86.08 m | 85.13 m | 84.43 m |
| Proving-ground lap | 1:06.552 | 1:07.001 | 1:07.255 | 1:06.557 |
| Arena, sport engine | 59.813 | 1:00.159 | 1:00.286 | 59.815 |
| Wet arena | 1:12.537 | 1:12.942 | 1:12.950 | 1:12.545 |

| FULL WING | Corsa | MX-5 | 540i | Express | Citaro |
|---|---|---|---|---|---|
| Stop from 100 | 39.43 m | 46.31 m | 47.31 m | 40.94 m | not for this car |
| Wet stop from 80 | 39.68 m | 40.31 m | 40.25 m | 40.94 m | 51.64 m |
| Hold the circle | 0.732 g | 0.712 g | 0.639 g | 0.682 g | 0.478 g |
| Wet circle | 0.462 g | 0.450 g | 0.404 g | 0.433 g | 0.302 g |
| Air brake from 150 | 84.53 m | 93.55 m | 96.36 m | 87.53 m | not for this car |
| Proving-ground lap | 1:06.552 | 1:02.488 | 1:01.676 | 1:08.110 | 1:27.669 |
| Arena, sport engine | 59.813 | 58.212 | 59.981 | 1:01.589 | 1:14.764 |
| Wet arena | 1:12.537 | 1:11.312 | 1:14.266 | 1:14.924 | 1:31.002 |

The threshold for each star is the value x 1.12 / 1.06 / 1.02 (a time or a
distance; divided for a g). The two MX-5 arena top-only laps drove at a 0.85
margin (`drove`). On a straight stop the side wings are only weight: ONLY TOP
stops the Corsa 0.25 m shorter from 100 than FULL WING. The air brake is what
the side wings are for: FULL WING (on AIR BRAKE) stops 1.55 m shorter from 150
than ONLY TOP.

* **On the three stops a fixed top wing is out when the brake goes in** (the
  fix after the integration, below): ONLY TOP, FIXED stops the Corsa 0.16 m
  shorter than ONLY TOP from 100, 0.04 m in the wet, 0.95 m from 150. Before
  the fix every FIXED stop measured its moving twin to the bit. The
  differences are small and not all one way (the tyres do the stopping): of
  the 26 FIXED stop combos (13 car / stop pairs x 2), 7 are 0.002-0.13 m
  longer than the moving twin, 19 shorter (Express, wet, ONLY TOP, FIXED
  40.83 m against 40.80 m; Corsa, wet, TOP FIXED + SIDE 39.80 m against
  FULL WING's 39.68 m).

### The title (title self-check, row 12)

1280 x 800, five cars, the full look, the overlay: mean 6.31 ms, p99 7.65 ms (budget 12.0 / 16.0 ms at machine x1.00). The builder's screenshots are in `~/Desktop/carsim-t44c/runs/title_shots/`.

## How it was built

A base commit of main's working tree (task 42 + the garage_ui arrow fix,
0780e13), then three parts in parallel worktrees: A the wing modes (`t44-a`,
fcfda77), B the challenge picker and refs (`t44-b`, d61a08b), C the title
(`t44-c`, 51194e9); merged on `t44` with `--no-ff`. Two textual conflicts:
the G handler (B's challenge branch first, A's `next_mode(mode, cfg)` in its
`else`) and `_loop_run`'s fake session row (B's `key` / `pick` and C's `page`);
CONTRACT keeps both new sections (part A, part C) and B's edits in place.

## What the review found

One reviewer agent over the merged branch and a judge: four findings, all
low, all fixed. V38 and the title's row 11 pin them; V38 with the new G hook
switched off fails on the first two.

* **AIR BRAKE came back with a stale side latch.** In a challenge run G steps
  only AUTO and AIR BRAKE, and AUTO never runs the air brake's command, so its
  side latch and pedal state were those of the last AIR BRAKE stint. AIR BRAKE
  in a left-hand bend, AUTO through a right-hand one, AIR BRAKE again
  mid-corner: both side wings stayed in for 0.30 s before the outer one came
  out (before task 44 the cycle always went through the other modes, so this
  could not happen). Every G press now tells the air brake
  (`AirBrake.g_changed`), whose next command takes the car's latches afresh:
  the outer flank from the first step.
* **An active top wing popped out on a straight after a TOP mode.** The
  physics counts its top-wing hold down only while nothing commands the top
  wing, so leaving TOP / TOP FIXED / TOP FIX+SIDE for AUTO (or LEFT, RIGHT,
  AIR BRAKE off the brake) found the hold from before the TOP mode: brake in
  AUTO, G into TOP, then back to AUTO on a straight, and the wing came out for
  up to 0.8 s, drag and downforce included. The review's fix, writing the hold
  into the car on the key press, would break a recorded lap's replay (a lap is
  re-simulated from its logged controls, and a G press is not one: the replay's
  car would keep the stale hold). So the top wing stays on the free path, on
  the same law with the air brake's own hold (TOP's; 0 after the two FIXED
  modes), until the law triggers afresh (a brake or a turn), which restarts the
  car's own hold; from there the two are the same bits. Only after a G press:
  a script, a replay and every acceptance run are untouched (V20).
* **A right click or P on the title closed the game.** The menu input calls
  ESC, P, a right click and a pad's OPTIONS all `menu` (and CIRCLE `back`), and
  the title quit on each. Now only ESC (the title's own key map) and the
  window's close quit; a right click, OPTIONS and CIRCLE move the cursor to
  Quit, where ENTER / CROSS quits; P does nothing.
* **The HUD's TOP line showed the garage slot's mode, not what G's mode
  does.** TOP on a fixed slot said FIX while the wing moved; TOP FIXED and TOP
  FIX+SIDE on an active slot said ACT while it stayed out. The line now reads
  the mode in the three TOP modes.

## The fix after the integration: a fixed top wing rides the held stop out

The integrator measured ONLY TOP, FIXED bit-identical to ONLY TOP, and TOP
FIXED + SIDE to FULL WING, on the three stops. `Sim._hold_step` resets the
car every held step (task 42), the reset zeroes the top wing's deploy
(`state.top_raw`), and the reference's pedal is at full on the very first
step, so it lets go straight from `Sim.reset`'s stowed state: a FIXED wing
spent its whole extension lag (0.45 s) in the stop. A fixed wing on a car
running at v0 would already be out.

* `Sim._hold_top(ctl)` puts the top wing as it rides at v0 -- straight, no
  brake, under the step's controls: out (`top_raw = top_deploy = 1.0`) when
  armed and `airbrake.cruise_top(mode)` says so (TOP FIXED, TOP FIX+SIDE),
  or, where that is None (AUTO, AIR BRAKE, LEFT, RIGHT), when the slot is
  'fixed'; in otherwise (TOP; an 'active' slot). It runs after the reset on
  every held step and on the step the hold lets go, which is the reference's
  first step. `measure()` drives the stops through this same
  `Sim.step_physics`, so references and players match (V44p: a player held
  0.5 s then DOWN on the keyboard's ramp stops as the reference, to the bit).
* **Not the step's `wing_cmd`**, which the first draft of the fix read: a held
  step may carry a pedal on its way down, and on the let-go step AIR BRAKE's
  command is `(True, True, True)`; read literally, it put FULL WING's active
  top wing out before the brake (it changed the FULL WING reference from 150,
  and a player's ramp would have had it out while held). The flanks are left
  as the reset has them: in, as the law has them on a straight; AIR BRAKE's
  all-three-out happens on the brake.
* Challenge runs are armed from the start (`_challenge_wings` sets
  `sim.wing_on`, OR'd into every step's `ctl.wing_on` before the hold); F
  disarmed in a run stows it, as anywhere.
* V20 (sha256 `ed41b7f781e959ea`) is unchanged: only a challenge's stop has a
  hold.
* refs.json: the three stops x the four cars that can drive them (five in the
  wet) x `top_fixed` / `top_fixed_side` re-measured, `--measure --write --only
  'brake_100|'` (then `brake_wet|`, `airbrake_150|`), 20 combos each: 26
  entries changed, all FIXED stops; FULL WING, ONLY TOP and the bus's
  unavailable stops came back identical.

| Corsa, before -> after | ONLY TOP, FIXED | TOP FIXED + SIDE |
|---|---|---|
| Stop from 100 | 39.18 -> **39.02 m** | 39.43 -> **39.07 m** |
| Wet stop from 80 | 39.63 -> **39.59 m** | 39.68 -> **39.80 m** |
| Air brake from 150 | 86.08 -> **85.13 m** | 84.53 -> **84.43 m** |

Pinned by V44p (d) (the top wing at the let-go: ONLY TOP, FIXED 1.00, ONLY
TOP 0.00, the reference's and a player's; AIR BRAKE's 0.00 through a ramp;
each stop = refs.json's), airbrake's self-check (`cruise_top` = `command()`'s
top element with no brake, mode by mode) and challenges' (no FIXED stop equals
its moving twin; the Corsa's ONLY TOP, FIXED shorter than ONLY TOP). With the
fix switched off V44p fails; with the literal `wing_cmd` rule it fails too.

## Shape of it

| file | what |
|---|---|
| `drive/airbrake.py` | `TOP` / `TOP_FIXED` / `TOP_FIX_SIDE` (4 / 5 / 6), ALL 3 removed; `TOP_MODES`, `STOWS_FLANKS`, `FLANK_LAW`, `usable`, `next_mode(m, cfg)`, `flanks_hidden`, `AirBrake._top_law` / `top_hold`, `AirBrake.g_changed` / `handing` (the review), `cruise_top` (the held stop's top wing); the texts; self-check rows |
| `drive/vehicle.py` | `TOP_BRAKE_ON = 0.05` (was a literal in `_aero`; same bits) |
| `drive/challenges.py` | `CONFIGS`, `CONFIG_LABELS`, `config_build` / `config_parts`, `resolve`, `combos`, `not_for_car`, `g_modes`, `load_refs` / `validate_refs` / `write_refs`, `measure_combo`, the pool `--measure`; progress per combo; the page rows (`pick_rows`, `wings_line`, `wings_section`); `REF_WING_MODES` without `all`, with the three top keys |
| `drive/data/challenges/*.json`, `refs.json` (new) | no numbers in the challenge files; the 160 measured references |
| `drive/title.py` (new) | the title: layout, input (`_title_keys`: only ESC quits at once), `pick_scene`, `_SceneRenderer` (each car in its own body and paint), the chase camera, the panorama fallback; self-check |
| `drive/drive.py` | G: the challenge's `g_modes`, else `next_mode(mode, cfg)`, then `AirBrake.g_changed` (and `step_physics` merging in AUTO while `handing`); the flanks hidden in `hud_data`, its TOP line the mode's; `usable` on restore; `KEYS_HELP`; the challenge page's Car / Wings rows (`Sim.ch_pick`, `_ch_step`, `_ch_combo`), `_challenge_switch` resolving a combo, `_drive_design` driving the config's copy, `_challenge_wings`; `Sim._hold_top` (a fixed top wing out on a held stop); `TITLE_SKIP`, `TITLE_PAGES`, `_title_wanted`, `_title_bottom`, `_title_screen`, `Sim.open_page`, `run_interactive_cli` showing the title; `_loop_run(title=, titles=)` and a step's `combo`; V38 rewritten, V35 on a rotating subset of combos, V44p (the picker; (d) the held stop's top wing), V44 (the title) |
| `drive/tutorial.py` | the wing ON lap accepts `FLANK_LAW` |
| `drive/input.py`, `drive/controls_page.py`, `drive/render.py` | help texts; `KeyboardInput.menu_keys` (the title's own keys, the review); render's self-check holds `TOP FIX+SIDE` whole at eight UI scales |
| `drive/validate.py` | `title` in `MODULES` |
| `README.md`, `drive/CONTRACT.md` | the modes, the picker, refs.json, the title |

## Deferred (the owner's call: after task 43)

* **Saving airfoils and wings** (asks 2-3) and **designing a wing by hand**
  (chord, span, airfoil): they wait for task 43, the AeroBO designer in
  `~/Desktop/carsim-aerobo`. The garage's wing designer was not touched.

## Known gaps

* **G from AUTO to TOP / TOP FIXED with a side panel out** makes the panel
  vanish in one frame instead of folding away (the free path's per-flank deploy
  starts at 0; the side force goes with it). AIR BRAKE does not do this.
* **The owner's challenge stars show as zero** (the pre-44 per-challenge
  entries are left in `runs/progress.json`, unread) and six challenges' rules
  changed (above).
* **Garage from the title** is the `--garage` launch: leaving the garage by
  quitting quits the game, not back to the title (the title shows only at
  launch).
* **refs.json is measured, not live**: a change to the AUTO / AIR BRAKE physics
  paths, a car, a map or a challenge file needs `python3 -m drive.challenges
  --measure --write` (about 2 min with 10 jobs); `validate_refs` reports an
  entry measured in another class / driver / G mode as out of date, and V35
  re-drives 16 of the 160 combos (it takes about 90 s now, was 44 s: the
  rotation includes bus laps).
* **The branch is not on main.** `t44` sits on `t44-base` (0780e13), a
  temporary commit of main's uncommitted working tree (task 42 + the garage_ui
  arrow fix) that another session still has uncommitted in
  `~/Desktop/carsim`; landing it means committing that work first and then
  merging `t44`.
* `$TMPDIR` holds about 1870 `carsim_*` dirs from this and other sessions' runs
  (43 GiB free). The build removed only its own; the review-fix pass's gate
  runs left theirs (about 40), since another session's gates were writing
  there at the same time and a sweep by time was not allowed.
