# Task 35 — the wing mode, and the air brake: all three wings at once

Owner (2026-09-24): *"Also user should have the ability to deploy 3 wings at
the same time."*

```
python3 -m drive.airbrake             ->  PASS    (new: the cycle, a matching pair, the pedal hysteresis, the law on the free path, ALL 3, the HUD light)
python3 -m drive.tutorial             ->  PASS    (+1: the wing ON lap refuses LEFT / RIGHT / ALL 3)
python3 -m drive.drive --self-check   ->  ALL PASS  (V38 new; V20 sha256 ed41b7f781e959ea unchanged; V21 RTF 11.92)
python3 -m drive.validate             ->  82/82  pass  0 HARD  0 soft   [130.4 s]
python3 -m drive.validate --modules   -> 120/120 pass  0 HARD  0 soft   [454.2 s]  (119 + airbrake)
python3 -m drive.ml                   ->  ALL PASS  32/32
```

## What was there

The physics could always do it. `Controls.wing_cmd = (left, right, top)`
(task 18's free wings) commands each wing directly, gated by the arm toggle.
Both flanks out is an air brake: their side forces cancel and their drags
add. The player could not use it properly, though:
* **Hidden mode.** The G key had a mode 2 that put both flanks out, with the
  top wing left on its own law. The keyboard's own copy of G only cycled
  auto / left / right.
* **Invisible.** The HUD never showed which mode was on, and nothing
  deployed the top wing on command.

## What it does

* **G / TRIANGLE cycles a named wing mode** (`drive/airbrake.py`, new):

  | mode | what |
  |---|---|
  | **AUTO** | the published law: the OUTER flank in a corner; the top wing by its own mode |
  | **AIR BRAKE** | AUTO's outer flank, plus **all three out while you brake**: pedal 0.30 or more, back in under 0.15, above 5 m/s |
  | **ALL 3** | all three out, all the time |
  | LEFT / RIGHT | one panel, as before |

  All of it stays gated by F / CIRCLE (armed), as every wing always was.
  The mode shows on the aero panel's header (`AERO  AIR BRAKE`), lit while
  the air brake is out, and a note names it when G changes it. It is kept
  across a restart, as J is.
* **AIR BRAKE stays on the free path** (`wing_cmd`) for as long as its mode
  is on. Off the brake it commands the outer flank exactly as the published
  law picks it (`AirBrake._law`: the same deadband, the same 0.3 s hold, no
  side change while a panel is out). A brake on or off therefore never
  switches the physics between its two deploy states, the law's and the free
  path's, which are separate: a switch would make a panel jump. It only
  READS the car's deadband and deploy fractions; it writes nothing into it.
* **A car whose flanks are not a matching pair** (one flank, or two
  different panels or stations without the mirror) would pull sideways with
  both out. Its flanks stay on the published law under AIR BRAKE and ALL 3;
  the top wing still deploys.
* **The physics is untouched.** The mode is merged into the Controls only
  when nothing else commanded the wings (`wing_cmd is None`), and only for a
  mode other than AUTO. A script, a replay, a bot and every acceptance run
  are the same bits (V20 unchanged). The recorded controls log carries
  `wing_cmd`, so a lap driven on the air brake replays exactly (task 19).

## What it is worth (measured)

A headless single-process rig, built the way a session builds the car; the
Corsa, ABS, full brake. The stopping distance with all three out (the plate on
both flanks plus the rear-s1223 top wing), against no wing:

| from | braking at the speed | wings already out |
|---|---|---|
| 100 km/h | -0.4 % | -1.1 % |
| 150 km/h | -2.3 % | -2.2 % |
| 200 km/h | (the Corsa cannot reach it on the strip) | -3.1 % |

**Why so small:**
* **The flanks brake with drag only.** The flank panels make side force and
  drag, no downforce. With both out, only their drag brakes: 285 N at 150
  km/h, against the car body's own 702 N.
* **The downforce is small.** The top wing makes 605 N at 150 km/h, 6 % of
  the car's weight.
* **The deploy lag costs speed.** The 0.45 s deploy lag eats the fastest
  part of a stop.

The tyres do nearly all of the stopping. It is a real, correct effect, just a
small one with these wings (task 36's *Air brake from 150* makes it the
difference between two stars and three).

**The owner's decision (2026-09-24): keep it as it is.** The air brake stays a
small, correct effect; no bigger wings, no physics change.

## V38

V38 stops the tuned Corsa from 150 km/h on the dragstrip, with the plate on
both flanks and the rear-s1223 top wing. The car is built as a session builds
it (`challenges.measure`, which now takes a wing mode, a build JSON and a
probe). It checks:
* in **AUTO**, no wing command reaches the physics;
* in **AIR BRAKE**, none before the brake, all three out (at least 0.95) during
  the stop, and the stop is shorter: 89.69 m against AUTO's 90.28 m (-0.7 %;
  with this build AUTO's active top wing already deploys under braking, so
  the difference is the flanks' drag);
* in **ALL 3**, the wings are out on the way up too, so the car is slower to 150;
* a **one-flank build** under the air brake keeps its heading (under 0.5 deg);
* **trail-braking on the skidpad** in AIR BRAKE (the plate pair, two brake
  pulses mid-corner): both flanks come out, and no flank moves more than
  0.005 in a 1 ms step (before the review's fix, a full 1.0 in one step).

## What the review found

A review workflow ran one finder and verifiers. Five findings were
confirmed, all low once verified. The finder's two unverified ones held up
when checked by hand. All seven are fixed.

* **Every brake on / off made the flank panels jump** (the review's lead
  finding). The physics keeps the published law's deploy state apart from
  the free path's. AIR BRAKE flipped between the two, so the outer panel's
  force vanished in one step at brake-on and came back from zero at release.
  The HUD row, the servo sound and the seed-lap rows all showed it.
  * The fix keeps AIR BRAKE on the free path and runs the published law's
    choice there (above). V38 now asserts the smoothness.
  * The obvious alternative, copying the deploy state across at each
    switch, was rejected: it writes into the car outside `Vehicle.step`, so
    a recorded lap would no longer replay from its controls.
* **An unmirrored build with two different flanks pulled under the air
  brake** (1.9 deg of heading from 150 km/h). `pair()` now requires a
  matching pair: every number of the two panels, station and incidence
  included.
* **The HUD's air-brake light lit with the wings disarmed, or none
  fitted.** It now needs the wings armed and something to deploy.
* **The kept wing mode reached the tutorial's wing laps.** LEFT / RIGHT /
  ALL 3 would spoil the "wing ON" lap with the tutorial blaming the driver.
  That lap now refuses them and names the key back to AUTO.
* **A reference build given as JSON would crash the challenges CLI**, and a
  typo in `ref.wing_mode` was a KeyError. The CLI now prints a name, and
  `validate` rejects a bad `wing_mode` or `build`.
* **The keyboard's own old three-state G cycle was dead state**, with a test
  and comments saying the opposite. It is gone; G is handed to the Sim.
* **The docs mixed "against no wing" with "wings already out".** The numbers
  now say which is which, and what the air brake adds over AUTO (0.7 % from
  150).

Refuted: that AIR BRAKE should keep the outer panel's grip while
trail-braking. The mode means all three out while braking, and their side
forces cancel.

## Found after landing: every launch crashed

`python3 -m drive.drive` crashed at launch from this commit (95d7baa) on:
`AttributeError: 'Namespace' object has no attribute 'wing_mode'`.
* **The cause.** `_interactive_session` restores the G mode with
  `if getattr(opts, "wing_mode", 0) in CYCLE: sim.wing_side_mode = opts.wing_mode`.
  The guard is safe, but the assignment read the bare attribute, which exists
  only after the restart loop has set it. A first session has none.
* **Why no gate caught it.** No check runs a real first session through
  `main()`.
* **Who found it.** The other session working on this checkout (task 38), which
  fixed it in the working tree with a `getattr`; its uncommitted V41 builds a
  first session the way `main()` does.
* **The fix.** Committed on its own, the same one line. Reproduced and verified
  in a clean worktree of the commit: before, `--render offscreen` exits 1
  with the error; after, the session starts.

## Shape of it

| file | what |
|---|---|
| `drive/airbrake.py` (new) | `AUTO / AIR / ALL / LEFT / RIGHT`, `CYCLE`, `LABELS`, `WHAT`, `next_mode`, `pair` (a matching pair), `top_fitted`, `AirBrake.command` / `_law`, `showing`; self-check |
| `drive/drive.py` | `Sim._airbrake`; `step_physics` merges `AirBrake.command` (only when `wing_cmd` is None); G cycles `next_mode` with a note; `HudData.wing_mode` / `air_brake`; the mode kept across a restart (`opts.wing_mode`); `KEYS_HELP`; V38 |
| `drive/challenges.py` | `ref_build('tall')` or a build JSON; `REF_WING_MODES`, `REF_BUILDS`, `ref.wing_mode` (validated); `measure(probe=)`; the CLI names a JSON build |
| `drive/render.py` | `HudData.wing_mode`, `air_brake`; the aero panel's header `AERO  <mode>` |
| `drive/input.py` | the help texts: F wings armed, G / TRIANGLE the wing mode; the keyboard's old G cycle removed |
| `drive/tutorial.py` | the wing ON lap refuses LEFT / RIGHT / ALL 3 (the frame's `wing_mode`) |
| `drive/validate.py` | `airbrake` in `MODULES` |
