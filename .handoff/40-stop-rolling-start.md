# Task 40 — the stops start rolling, and the box keeps the last stop

Owner (2026-09-24):
* *"For challenges that we have to stop from certain speed make it so that
  the car has already initial speed, don't make the user start from 0."*
* *"It is not clear what was the latest stopping time."*

```
python3 -m drive.challenges           ->  PASS    (every stop rolls above its v0; the box's line; the rolling pose)
python3 -m drive.drive --self-check   ->  ALL PASS  (V35: all eight to three stars, values = the files', stops start rolling, R rolls again, the box keeps LAST STOP; V20 sha256 ed41b7f781e959ea unchanged; V21 RTF 11.75)
python3 -m drive.validate             ->  82/82  pass  0 HARD  0 soft   [124.2 s]
python3 -m drive.validate --modules   -> 121/121 pass  0 HARD  0 soft   [446.7 s]
python3 -m drive.ml                   ->  ALL PASS  32/32
```

## What was wrong

* **Every stop started from standing.** Stop from 100, Wet stop from 80 and
  Air brake from 150 put the car still on the dragstrip's line: accelerate
  past the speed, then brake. With the tuned engine, that is half a minute
  of full throttle before every 5 s attempt.
* **The result was hard to find.** It was only the box's small green line
  at the top. The big status line under it went straight back to
  "accelerate past 100 km/h (0), then brake to a stop" once the car stood
  still.

## What it does

* **A stop challenge starts rolling** at its goal's new `start_kmh`, 20 %
  over the speed the distance counts from:

  | challenge | counts from | starts at | gear (automatic) |
  |---|---|---|---|
  | Stop from 100 | 100 km/h | 120 km/h | 4th |
  | Wet stop from 80 | 80 km/h | 96 km/h | 3rd |
  | Air brake from 150 | 150 km/h | 180 km/h | 5th |

  The car starts 2.5 m down the strip (the scripted runs' staging, so all
  four wheels are on it), coasting, in the gear the automatic holds there
  on full throttle. It coasts for 4 to 6 s before it slows through the
  speed, which is plenty of time to brake.
  * **R and SHIFT+R both put it back there.** The box says "R: again from
    120 km/h".
  * **Accelerating past the speed and braking still counts**, as before.
* **The box keeps the LAST STOP**, in its big line, until the next stop
  starts counting: `LAST STOP 42.47 m [***]   R: again from 120 km/h`, then
  `LAST STOP ... brake now: 118 km/h` on the next run-in. A result that did
  not count says `(not counted)`, with the reason on the red line as before.
  The green line with NEW BEST stays.
* **The reference drives it the same way.** `challenges.measure` attaches
  the challenge and calls the Sim's own reset, exactly as the session
  start does. So the thresholds are the rolling start's (plan D4: ref x
  1.12 / 1.06 / 1.02).
* **`validate` refuses a stop with no `start_kmh`**, or one not above
  `v0_kmh`, or one over twice it. Only the stop metric rolls: laps and the
  skidpad still stand.

## The numbers

| challenge | ref before (standing) | ref now (rolling) | 1 star | 2 stars | 3 stars (was) |
|---|---|---|---|---|---|
| Stop from 100 | 42.70 m | 42.47 m | 47.57 m | 45.02 m | 43.32 m (43.55) |
| Wet stop from 80 | 40.39 m | 40.07 m | 44.88 m | 42.48 m | 40.88 m (41.20) |
| Air brake from 150 | 87.89 m | 87.91 m | 98.46 m | 93.19 m | 89.67 m (89.65) |

The five other challenges re-derived byte for byte.

**Why 20 % and not 10 %.** Two things decided it, both measured.
* **At 10 % (110 km/h), the dry stop's third star needed the brake inside
  about a second.** The reference, braking on the first step, stopped in
  42.05 m. A player braking at 107 km/h or later got 42.9 to 43.0 m, over
  the 42.89 m line: the pedal and the ABS were still building up when the
  car passed 100.

  At 120, every brake point from 120 down to 103 km/h stops in 42.5 to
  43.0 m, all three stars. The wet stop, at 96, gets 40.1 to 40.4 m against
  a 40.88 m line. That is the difficulty the stops had before: full pedal
  before the speed.
* **At 10 % (165 km/h), the air brake's third star no longer needed
  wings.** A rolling car starts at static loads, not squatting under full
  throttle, and every stop from it is 0.3 to 0.6 m longer. That is a real
  effect, and the physics is untouched. The reference went to 88.53 m (3
  stars at 90.30), and a wingless car at 90.12 m made three stars.

  At 180, the sweep of brake points from 180 down to 155 km/h gives:

  | build | stop | stars |
  |---|---|---|
  | every wing, AIR BRAKE | 87.9 to 88.6 m | 3 |
  | the plate pair, AIR BRAKE | 88.8 to 89.7 m | 3 |
  | no wings | 90.2 to 90.5 m | 2 |

  Task 36's balance is back, with its thin margin. A wingless car that
  accelerates to 175 and brakes from full throttle stops in 89.74 m, 0.07 m
  over the line.

## Checks

* `drive.challenges` self-check:
  * every stop file rolls, above its v0;
  * the box's line says "brake now", why a voided stop cannot count, and
    promises no roll in a race;
  * a stop with no `start_kmh`, one at v0, or one at three times v0 is
    refused;
  * the rolling pose is `START_S`, `start_kmh` and a gear over 1, and a lap
    stands.
* **V35:**
  * every reference starts where it should: a stop at `START_S` and
    `start_kmh` in a gear over 1, everything else at 0;
  * R pressed mid-stop rolls again at 120 and the meter re-arms;
  * after the stop, the box's line is
    `LAST STOP <the value> [...]   R: again from 120 km/h`.
* **The real session start** (`_interactive_session`, offscreen, a temporary
  settings / progress / library, never the player's) for all three: 2.5 m,
  120 / 96 / 180 km/h, gears 4 / 3 / 5, "brake now" from the first frame.
  After 2 s of coasting, 116 / 93 / 171 km/h.

## What the review found

One reviewer agent found two low defects, both fixed, and nothing
structural.
* **A voided stop said "brake now".** With T (the wet toggle) on, the box
  said "brake now" for a stop that could not count, and gave no reason. It
  now says `not counting: the wet toggle (T) is on`. The same goes for
  another class, a refused build, or slow motion; the self-check covers it.
* **The box promised a roll in a race.** A race survives the restart into a
  challenge, and in a race R puts the car on the line, standing. The box
  still said "R: again from 120 km/h". In a race it now shows the old
  "accelerate past" line.

Checked and ruled out:
* **A stale speed in the meter after a teleport.** A crossing needs
  `armed`, which the reset clears.
* **Records, ghosts and the pre-race page.** The dragstrip has none of them.
* **A class change.** It ends the challenge first.
* **The tutorial and the gearboxes.** No conflict with either.
* **V38.** Its stop has no `start_kmh`, so it still starts standing.

## Shape of it

| file | what |
|---|---|
| `drive/challenges.py` | `START_S`; `ChallengeRun.rolling(tr, pt_p)` (the pose, or None); `ChallengeRun._status` (the box's line keeps LAST STOP); `_collect` remembers whether the last one counted; `validate` needs `start_kmh` for a stop; `measure` resets through the Sim and returns `start`; self-check |
| `drive/drive.py` | `Sim.reset`: a stop challenge's pose for R and SHIFT+R; `_interactive_session`: one reset after the challenge is attached; V35 extended |
| `drive/data/challenges/01, 02, 05` | `goal.start_kmh` 120 / 96 / 180, the blurbs, Air brake's driver `brake:1.0:150` (it brakes at once from 180), thresholds by `--write` |
| `README.md`, `drive/CONTRACT.md`, `.handoff/README.md` | the stops' paragraph, the module row, the index |
