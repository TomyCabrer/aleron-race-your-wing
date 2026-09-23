# Task 23 — the driving tutorial

The brief (`PLAN-steam-engagement.md`, T23): a new `drive/tutorial.py` with
data-driven steps `{id, map, text, predicate(HudData / sim state), hint}`; an
overlay box; skippable; progress in `runs/progress.json`; offered on the
first launch and in the menu. The steps: throttle and brake -> steering ->
arena turn 1 -> reset (R) -> what the assists do -> **the wing on the
skidpad**: a lap with the wing off, a lap with it on, the measured lateral-g
difference shown -> sectors, the PB ghost, medals -> one valid lap.
Acceptance: the self-check drives every predicate with synthetic HudData;
**V34** runs the whole tutorial headless with `ScriptedInput`.

```
python3 -m drive.tutorial             ->  37/37   (new)
python3 -m drive.progress             ->  11/11   (new)
python3 -m drive.render               ->  32/32   (+1: the tutorial's box is drawn, stays in R_TUTOR)
python3 -m drive.drive --self-check   ->  ALL PASS  (V34 new; V20 sha256 ed41b7f781e959ea unchanged; V21 RTF 11.84)
python3 -m drive.validate             ->  82/82  pass  0 HARD  0 soft   [117.1 s]
python3 -m drive.validate --modules   -> 110/110 pass  0 HARD  0 soft   [322.4 s]  (108 + progress, tutorial)
python3 -m drive.ml                   ->  ALL PASS  32/32
```

## What it does

Eleven steps, in the plan's order (`drive/tutorial.py`, `STEPS`):

| # | id | map | kind | passes when |
|---|---|---|---|---|
| 1 | `pedals` | arena | drive | 50 km/h reached, then under 3 km/h |
| 2 | `steer` | arena | drive | a yaw rate of 6 deg/s each way, above 20 km/h |
| 3 | `turn1` | arena | drive | from the line (the step resets to it) past turn 1's exit + 15 m without leaving the road; off the road, R is another go |
| 4 | `reset` | arena | drive | R (or CREATE, or the menu's reset) |
| 5 | `assists` | - | page | Continue: ABS / TC / steering aid / gearbox, each with its state and what it does |
| 6 | `wing_off` | skidpad | drive | a FLYING lap, line to line, on the road, wing off throughout |
| 7 | `wing_on` | skidpad | drive | the same with the wing on |
| 8 | `wing_result` | - | page | the two laps' mean lateral g and time, the difference |
| 9 | `timing` | arena | page | sectors, the flash colours, the PB ghost, ghost 2, the delta, this class's medal targets |
| 10 | `lap` | arena | drive | a valid lap that went ROUND (95 % of the length counted from the crossing, the records' rule) |
| 11 | `done` | - | page | your lap, the wing's difference, what was skipped, what next |

* **A drive step** is a box on the road, left, between the state panel and
  the pedals (`render.R_TUTOR`, `HudData.tutorial`, drawn whether the HUD is
  on or not): the step, what to do, a live status line ("  34 of 50 km/h",
  "turn 1: 120 m to go", "lap 64 %   mean 0.71 g"), why the last attempt did
  not count ("off the circle: this lap does not count"), the hint after 20 s
  in the step, and a "done:" line for 3 s after a step passes.
* **A page step** is a paused menu page (keyboard, pad, mouse like every
  page): Continue, or End. ESC continues too.
* **Skippable**: ESC > *Tutorial* (a row on the pause page) skips the step,
  starts over, or ends it; ended, the same row continues at the saved step.
* **Maps**: a step names its map; the tutorial asks for a restart and
  `run_interactive_cli` moves `settings.track` there before the next session
  (so TAB cannot leave a map the tutorial is on). arena -> skidpad -> arena.
* **The wing laps need a flank wing.** A car with none (the default car has
  none) drives those two laps with the library's published plate fitted
  (`tutorial.wing_car`, "<name> + tutorial plate", in memory; no per-map
  build is recorded for it) and the player's own car comes back at the next
  session -- or at once when the tutorial is ended on it.
* **Offered once**: a player session opens `runs/progress.json` at launch;
  when its `tutorial` section was never offered, the first session opens on
  **WELCOME** (start / not now) instead of the pre-race page. Not now marks it
  offered; the pause menu always has it.
* **Progress** (`drive/progress.py`, new): `runs/progress.json`, kind
  `carsim-progress-1`, one section per feature (`tutorial` now, task 25's
  `challenges` next); a save re-reads the file and replaces only its own
  section. A corrupt or foreign file is ignored with a note and moved aside
  as `progress.json.bad-<stamp>` on the next save; a torn one falls back to
  its `.bak`; an unreadable one is never written over. Only a player session
  opens it.

## A finding on the way: the two wing laps were not the same kind of lap

The first version measured each wing step as "one full circle from where the
step starts". The wing-OFF step starts from a standstill on the line (the
step resets there), the wing-ON step starts at speed -- so the OFF circle
included the launch and the page showed the wing adding **+0.17 g** (V34:
0.561 g off, 0.727 g on) that was really the standing start. Both are now a
FLYING lap, line to line: V34 reads **0.734 g off, 0.733 g on**, which is the
truth for a scripted driver at a 0.85 margin (it does not drive at the limit,
so the wing has nothing to add). The page says so in plain words, and quotes
the rig's number for what the published plate is worth at the limit (about
+2 % of corner speed, README "The device"): two laps by hand mostly show how
hard each was driven. Inside +-0.02 g the page says "about the same", not a
gain.

## V34

`_v34_tutorial`: an arena Sim with a temporary progress file; ESC > Tutorial
> Start through the real menu; then one frame at a time exactly as
`run_interactive` runs it (the events, 1/60 s of physics, `_tutorial_tick`):
the pedals as a script, a weave on a PathFollower, `LapDriver` through turn
1, round the skidpad (off, then on, with F pressed as an event) and round
the arena; R pressed as an event; every page continued with ENTER; and when
a step asks for another map a new Sim is built there, as the session loop
does. It asserts: every drive step PASSED its own predicate (none skipped),
the four pages were shown, the maps went arena -> skidpad -> arena, the
overlay showed every drive step's head, the progress file says done with
both wing laps and the lap measured.
Measured: **229 s of sim in 19.0 s**; wing off **0.734 g**, on **0.733 g**
(LapDriver at 0.85 is not at the limit), the lap **62.141 s**.

## Integration (scratch smoke test, not a gate)

`run_interactive_cli` driven with `Sim.run_interactive` replaced by a script,
in a temporary working directory (so no player file of the repo was touched):
launch -> WELCOME -> Start -> steps skipped to the assists page -> Continue ->
restart on the **skidpad with the tutorial plate** (+10.1 kg of wing, the
wing armed) -> the wing page -> Continue -> restart on the **arena with the
player's own car** -> ESC on the timing page continues -> ESC > Tutorial > End
-> `runs/progress.json` says step `lap`, offered, not done; a second launch
shows no WELCOME and the pause row "Tutorial: continue at step 10".

## What the review found

Two lenses (semantics in play; acceptance, rules and rendering), 2 finders
and 12 verifiers. Confirmed and fixed in this commit:

* **F could not switch off a wing that started armed** (high, older than this
  task: 048716a). The keyboard folds F into its own copy of the toggle and
  `Sim.handle_event` flips `Sim.wing_on` too; `step_physics` ORs the two. A
  car that starts armed (every garage build, `--wing`, and the tutorial's
  plate car) went HUD "OFF", physics ON on the first F -- so the tutorial's
  "wing OFF" lap was driven with the panel armed, and its difference was
  driving noise. Reproduced with a real `BlendedInput(KeyboardInput)` (V34's
  ScriptedInput has no keyboard state, so it could not see it). Now the
  harness's toggle is the one truth: `handle_event('wing')` clears the
  keyboard's copy. Checked: start armed, F -> off / off, F -> on / on.
* **Found by the first gate run** (high): `Sim.progress` was already the
  race's unwrapped metres; the tutorial's file shadowed it and V30 crashed.
  Renamed `Sim.progress_file`.
* **The "done:" flash and the hint clock ran on the old session's `sim.t`**
  after a restart (every session starts at t = 0): a flash stayed up 30-70 s,
  a hint was held back. The tutorial now re-bases its clocks on a new session.
* **Time trial > RACE during a wing step** would have filed the tutorial's
  plate car as the skidpad's default build -- no longer (`start_timed` skips it
  on the tutorial car).
* **Start over kept the last run's numbers** (a skipped wing lap showed an old
  one): a run from step 1 starts empty, and a skipped step drops its number.
* **The lap step passed laps the records refuse** (the T wet toggle on, a live
  setting change) and named a medal: it now reads the recorder's verdict and
  says "not recorded (...)".
* **Long status / key lines ran out of the box** onto the road: wrapped when
  too wide; the render check now looks to the right of the box too.
* Low: a hand-edited `results` entry crashed the wing page -- malformed
  results are dropped with a note.

Refuted: that V34 does not exercise the session plumbing (the scratch smoke
test below does; V34 is what the brief asks for).

## Deviations

* **Predicates read a frame, not `HudData`.** The plan says
  "predicate(HudData/sim state)"; a predicate here gets `frame_of(sim)`: the
  HUD's numbers (speed, lateral g, yaw rate, on the road, the wing toggle,
  the lap time and validity) plus the sim's `s`, the `LapTimer` events and
  the discrete commands since the last frame -- what a lap or a reset needs
  and `HudData` does not carry. The self-check builds these frames
  synthetically, as the plan asks of HudData.
* **The wing laps are flying laps** (above), which costs one out-lap circle.
* **The wing laps use the published plate** on a car without a flank wing
  (the brief assumes the car has one).
* **TAB cannot leave the tutorial's map** while it runs (End it first).
* **No new key**: the tutorial is driven by the pause menu, so the pad and
  the mouse come for free; nothing was added to the key table.

## Shape of it

| file | what |
|---|---|
| `drive/tutorial.py` (new) | `Step`, the 11 `STEPS` and their predicates / status lines, `frame_of`, `Tutorial` (`tick`, `advance`, `skip`, `end`, `overlay`, `page`), the menu rows, the WELCOME text, `wing_car`; self-check |
| `drive/progress.py` (new) | `Progress`: `runs/progress.json`, sections, merge-on-save, corrupt / foreign / torn / unreadable handling; self-check |
| `drive/drive.py` | `Sim.progress_file` / `tutorial` / `tutorial_car`; `reset` passes the command; `handle_event('wing')` clears the keyboard's copy of F; `start_timed` never files the tutorial car; `run_interactive` calls `_tutorial_tick`; `hud_data` sets `HudData.tutorial`; the pause page's Tutorial row, the TUTORIAL / WELCOME / page-step pages (`_tutorial_event`); `run_interactive_cli` opens the progress file, moves the map, fits and removes the tutorial plate; `_interactive_session` attaches it, offers it, skips the pre-race page under it; V34 |
| `drive/render.py` | `HudData.tutorial`, `R_TUTOR`, `_draw_tutorial` (+ a cached word-wrap); the drawn-and-fits check |
| `drive/validate.py` | `progress`, `tutorial` in `MODULES` |
| `drive/CONTRACT.md`, `README.md` | module map, the tutorial's flow, the pause menu's rows |
