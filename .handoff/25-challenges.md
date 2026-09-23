# Task 25 — challenges

The brief (`PLAN-steam-engagement.md`, T25): `drive/challenges.py` +
`drive/data/challenges/*.json`, schema `id, title, blurb, class, constraints,
goal {metric, threshold}, stars` (1 = pass, 2 = a tighter metric, 3 = the
metric plus an efficiency constraint); constraints (max wing area per slot,
max fitted wing mass, max ballast, allowed slots, max CdA) checked before the
start, a refusal quoting the reason in the garage's gating style; metrics
lap_time, stop_distance from V0, skidpad mean a_y over a lap, dragstrip time
and trap speed, reusing the brake / skidpad / accel scripts' measuring;
a Challenges page in the pause menu with star counts; progress in
`runs/progress.json`; **8 challenges** covering braking, skidpad, drag, lap and
wet; every threshold from a measured headless run (recorded here), every
3-star result shown achievable by a scripted or bot run. Acceptance: the
self-check validates every JSON file and the constraint checker; **V35** runs
the achievability checks.

```
python3 -m drive.challenges           ->  24/24   (new: every file, the constraint checker, the meters, the guards)
python3 -m drive.drive --self-check   ->  ALL PASS  (V35 new; V20 sha256 ed41b7f781e959ea unchanged; V21 RTF 11.81)
python3 -m drive.validate             ->  82/82  pass  0 HARD  0 soft   [116.2 s]
python3 -m drive.validate --modules   -> 112/112 pass  0 HARD  0 soft   [362.5 s]  (111 + challenges)
python3 -m drive.ml                   ->  ALL PASS  32/32
```

## The eight, and where every number comes from

Every threshold is DERIVED from one measured headless run -- the reference:
a scripted driver, on a reference build, in the challenge's class, built
exactly as a session builds the car (`drive._session_car`, the start pose,
the default assists), measured by the same per-step meter a player's attempt
is. 1 / 2 / 3 stars are the reference x **1.12 / 1.06 / 1.02** (lower is
better) or / 1.12 / 1.06 / 1.02 (higher is better) -- the medals'
multipliers (D4), so the reference earns three stars by 2 % by construction.
`python3 -m drive.challenges --measure` runs them; `--write` writes the
value and the thresholds into the files. Measured:

| challenge | class | metric | reference driver, build | measured | 1 / 2 / 3 stars | 3-star rule |
|---|---|---|---|---|---|---|
| Stop from 100 | dragstrip / Corsa / stock / dry | stop from 100 km/h | `BrakeDriver` pedal 1.0, no wings | **42.70 m** | 47.82 / 45.26 / 43.55 m | no wings |
| Wet stop from 80 | dragstrip / Corsa / stock / wet | stop from 80 km/h | `BrakeDriver` pedal 1.0, no wings | **40.39 m** | 45.23 / 42.81 / 41.20 m | no ballast |
| Hold the circle | skidpad / Corsa / sport / dry | mean lateral g, flying lap | `LapDriver` 0.85, the plate on both flanks | **0.733 g** | 0.655 / 0.692 / 0.719 g | <= 0.40 m^2 a slot |
| Wet circle | skidpad / Corsa / stock / wet | mean lateral g, flying lap | `LapDriver` 0.85, the plate | **0.462 g** | 0.413 / 0.436 / 0.453 g | <= 12 kg of wing |
| Quarter mile | dragstrip / Corsa / sport / dry | time to 402.3 m | `StraightDriver` WOT, no wings | **16.198 s** | 18.142 / 17.170 / 16.522 s | drag area <= 0.67 m^2 |
| Speed at 1000 m | dragstrip / Corsa / tuned / dry | speed at 1000 m | `StraightDriver` WOT, no wings | **168.8 km/h** | 150.7 / 159.3 / 165.5 km/h | drag area <= 0.665 m^2 |
| Arena, sport engine | arena / Corsa / sport / patches | a lap | `LapDriver` 0.90, the plate | **59.783 s** | 1:06.957 / 1:03.370 / 1:00.979 | <= 10.5 kg of wing |
| Wet arena | arena / Corsa / stock / wet | a lap | `LapDriver` 0.90, no wings | **1:12.935** | 1:21.687 / 1:17.311 / 1:14.394 | drag area <= 0.70 m^2 |

Rules to start (all checked against the build you drive, in the challenge's
car): ballast <= 100 kg (the two stops), flank slots only (Hold the circle),
no ballast (Wet circle, Quarter mile, Wet arena), <= 15 kg of wing (Arena).

The first `--measure` showed one bound that the reference itself broke: the
Arena's 3-star rule was 10 kg of wing and the plate pair weighs 10.1 kg; it
is 10.5 kg now (the rule still refuses a heavier set, e.g. a top wing too).

## How a result is measured

`Meter` watches the Sim through three hooks and is the same object live and
headless: `step_physics` calls it after every physics step (after `n` / `t`,
next to the recorder) and for every LapTimer event, `reset` restarts the
attempt. At 1 kHz, with the crossing interpolated inside the step:

* **lap_time**: a valid lap that went ROUND (95 % of the length counted from
  the crossing, the records' rule) -- the LapTimer's own time.
* **skid_ay**: the skidpad's flying lap, line to line, on the road
  throughout; the time-weighted mean of |a_y|, in g.
* **stop_distance**: armed once the car has been at v0 or faster; from the
  moment it slows through v0 to under 0.1 m/s (`BrakeDriver`'s own stop),
  the distance integrated from the speed. Accelerating back past v0 re-arms.
* **drag_time / trap_speed**: a standing run: armed when the car stands
  (< 0.05 m/s), the clock starts when it moves (0.1 m/s), stops at
  `distance_m` down the strip; the speed there is the trap speed.

A result counts only in the challenge's own class, checked every step: an
attempt with a live engine change or the T toggle in it -- even undone
before the finish -- or run in slow motion or single steps is dropped. The best value and the most
stars are kept per challenge in `runs/progress.json` (`challenges` section,
merged with the tutorial's).

## In the game

ESC > *Challenges: N of 24 stars* -> the list (title, stars, best) -> one
challenge's page: the goal and the three thresholds with the 3-star rule,
the RULES and YOUR BUILD's numbers, your best; **Start**, or "CANNOT START:"
with the reasons ("the top wing is not allowed here: take it off in the
garage (BACKSPACE), or pick a build without"; "the wings weigh 14.2 kg, over
10.0 kg: fewer or smaller wings"; ...). Start restarts the session in the
challenge's class (the player's own map / car / engine / surface are kept and
come back on *End the challenge*); the pre-race page is skipped; the box on
the road (the tutorial's) shows the goal, the stars, the live status ("brake
now", "stopping: 23.1 m", "lap 64 %   mean 0.71 g", "GO: the clock starts when
the car moves") and the last attempt with its stars. Keyboard, pad and mouse,
like every menu page.

## V35

`_v35_challenges`: every challenge's reference driven again through
`challenges.measure` (the Sim's own hooks, a temporary library): each must
earn THREE stars on a build that meets its rules and its 3-star bound, and
give back the file's `ref.value` exactly (so the thresholds are the ones a
measured run derives). Then the page flow by events with no window: the
pause row, the list of 8, a build with a top wing refused on *Hold the
circle* with the reason and no Start, ESC back, the plate build starts it (a
restart with `challenge_pick`).
Measured: **435 s of sim in 36.9 s**; all eight at three stars, every value the
file's to the last bit.

## Integration (scratch smoke test, not a gate)

`run_interactive_cli` with `Sim.run_interactive` scripted, in a temporary
working directory: on the arena (Corsa / sport / patches) ESC > Challenges >
*Stop from 100* > Start -> the next session is **dragstrip / Corsa / stock /
dry** with the challenge attached -> a `BrakeDriver` in the seat stops in
**42.696 m, three stars** (the reference's own number: the live session
builds the same car) -> the box reads "CHALLENGE  Stop from 100   best 42.70
m [***]" -> ESC > Challenges > End -> the next session is **arena / Corsa /
sport / patches** again, `runs/settings.json` restored, `runs/progress.json`
holds `challenges: {brake_100: {best: 42.696..., stars: 3}}` next to the
tutorial's section.

## What the review found

Two lenses (measurement honesty and exploits; the session flow, rules and
UI), 2 finders and 10 verifiers: ten confirmed, all fixed in this commit.

* **The T wet toggle, or a live engine change, inside an attempt counted
  if undone before the finish** (high, two findings): reproduced -- a "wet"
  stop of 28.5 m (the honest one is 40.4), a sport-engine run to the trap
  (183.6 km/h) filed in the tuned class. The class was only checked when the result
  landed. Now `ChallengeRun` checks it EVERY step and drops the attempt in
  progress (and T resets it at once, as it does the recorder's lap).
* **Slow motion and single steps counted** (medium): dropped, the records'
  rule.
* **A trap run could reverse back for a longer run-up** (medium): a standing
  run that goes backwards is void; the next stand-still re-arms there.
* **The drag-area rule read only the LEFT flank, and one panel** (medium):
  an unmirrored build with the plate on the right passed the Wet arena's
  3-star bound at "0.678" (really 0.797). Now each fitted flank panel's own
  D/q counts (G can deploy both).
* **Ending a challenge any other way lost the player's settings** (high):
  TAB or a settings change ended it and left the challenge's class in
  `runs/settings.json`; quitting mid-challenge did the same. Now the fields
  the player did not change come back, and quitting restores them all.
* **A challenge and the driving tutorial fought over the map** (medium): a
  pick now ends the tutorial (ESC > Tutorial continues it) and a tutorial
  started during a challenge ends the challenge, settings restored.
* **A build refused at the session start left the challenge running with
  no way to end it** (medium): the run is attached, marked refused, listed
  and endable, and never counts.
* **A malformed entry in the progress file's challenges section crashed
  ESC** (medium): ignored with a note.

## Deviations

* **The brief's "reuse the measuring logic of brake_script,
  skidpad_limit_script and accel_script"**: the references drive with those
  scripts' drivers (`BrakeDriver`, `StraightDriver`) and `LapDriver`; the
  meter is new, because a player's attempt must be measured live from the
  Sim's state and those scripts measure inside their own drivers.
  `skidpad_limit_script` finds a steady-state limit speed with an open-loop
  ramp, which is not "mean a_y over a lap": the skidpad metric is a flying
  lap (the tutorial's rule), and its reference is `LapDriver`.
* **Thresholds are the reference x the medal multipliers**, not three
  separately measured runs: one measured run per challenge, recorded in the
  file (`ref`), every threshold derived from it.
* **The goal's metric parameters** (`v0_kmh`, `distance_m`) live in `goal`
  next to `metric` and `threshold`.
* **A challenge changes the session's class** (and puts it back); changing
  the map / car / engine / surface on the settings page ends it.

## Shape of it

| file | what |
|---|---|
| `drive/challenges.py` (new) | the file format and `validate` / `load_all`, `build_stats`, `refusals`, `stars_for`, `Meter`, `ChallengeRun`, the page rows, the reference runs (`measure`, `--measure`, `--write`); self-check |
| `drive/data/challenges/*.json` (new, 8) | the challenges, with their measured references and derived thresholds |
| `drive/drive.py` | `Sim.challenge` / `challenge_pick` / `challenge_end` / `challenge_build`; the step, event and reset hooks; the box via `hud_data`; the pause row and the two pages (`_challenge_event`); `_challenge_class`, `_challenge_switch`; `run_interactive_cli` / `_interactive_session` plumbing; V35 |
| `drive/validate.py` | `challenges` in `MODULES` |
| `drive/CONTRACT.md`, `README.md` | module map, the flow, the Challenges section |
