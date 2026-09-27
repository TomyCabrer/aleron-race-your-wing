# Task 42 — the stops start at their speed, held there until the brake is in

Owner (2026-09-25):
* *"I want that for the stopping challenges that the initial speed is the
  same or lower than the required."*

```
python3 -m drive.challenges           ->  PASS    (every stop starts AT its v0, never above; the hold; the box's held line)
python3 -m drive.drive --self-check   ->  ALL PASS  (V35: all eight to three stars, values = the files', DOWN pressed late with the keyboard's ramp stops exactly as the reference; V20 sha256 ed41b7f781e959ea unchanged; V21 RTF 11.79)
python3 -m drive.validate             ->  82/82  pass  0 HARD  0 soft   [124.3 s]
python3 -m drive.validate --modules   ->  122/122 pass  0 HARD  0 soft   [445.9 s]
python3 -m drive.ml                   ->  ALL PASS  36/36
```

## What was wrong

* **The stops started above their speed.** Task 40 put Stop from 100 on
  the strip at 120 km/h, Wet stop from 80 at 96 and Air brake from 150 at
  180, so the pedal was full before the distance counted. A challenge
  called "from 100" that starts at 120 reads wrong.
* **The references declutched, and players do not.** `BrakeDriver` (the
  brake scripts' driver, and the stops' references) pushes the clutch in as
  it brakes. A player on the automatic never does, and the engine braking
  is worth metres. Measured on task 40's own starts, DOWN pressed at once,
  the clutch left to the automatic:

  | challenge | the reference | a keyboard player | the 3-star line |
  |---|---|---|---|
  | Stop from 100 | 42.47 m | 40.30 m | 43.32 m |
  | Wet stop from 80 | 40.07 m | 39.74 m | 40.88 m |
  | Air brake from 150, no wings | 90.20 m | **85.30 m** | 89.67 m |

  So a wingless car made the air brake's third star, which task 36 built
  the challenge to refuse.

## What it does

* **A stop challenge starts AT its speed**: `goal.start_kmh` = `v0_kmh`
  (100 / 80 / 150 km/h), 2.5 m down the strip, in the automatic's gear
  (3rd, 3rd, 5th).
* **The car holds that speed until the brake is in** (`challenges.StopHold`).
  `Sim.step_physics` carries it on down the strip at exactly v0, in the
  pose `Sim.reset` gives (the centre line, static loads, wheels rolling), no
  physics. Throttle and steering do nothing while it is held. It lets go
  when the brake pedal is at full (the keyboard's, 0.2 s after DOWN), or at
  30 % or more and held still (within 0.05) for 0.1 s: a squeeze on a pad's
  trigger. A pedal on its way up or down never lets go of it, so a tap, a
  pedal still falling from the last stop, or a finger resting on L2 leave
  the car held. The distance counts from the step it lets go. So the
  pedal's travel counts for nobody: the keyboard and a pad slam stop
  exactly the same.
  * Why: at v0, with the distance counting from the first touch of the
    pedal, the keyboard's 0.2 s ramp costs 3.5 m from 100 (43.03 m against
    39.50 m) and about 3 m from 150. A pad's trigger, read raw, does not. The
    input device would decide the stars.
  * **The box says** `held at 100 km/h: brake when ready`, then
    `stopping: ...`, then `LAST STOP 39.50 m [***]   R: again from 100 km/h`.
  * **It never runs out of strip.** Past 300 m before the end (43 s at 100,
    29 s at 150) it goes back to the start, as R does.
  * R and SHIFT+R put it back, held again. The keyboard's brake ramp starts
    from 0 there, so a DOWN still held at R takes its 0.2 s again, and
    letting go of it inside that time keeps the car held.
* **The references drive the automatic's clutch**, as a player does:
  `brake:1.0:100:0` (the new optional fourth field is the clutch pedal
  while braking; it defaults to 1, so the brake scripts and V38 are
  unchanged). They brake at once, on the first step.
* **`validate` takes a stop's `start_kmh` above 0 and at most `v0_kmh`.**
  At v0 it is held. Below v0 (no file does this) the car rolls free, and
  the player accelerates through v0 and brakes: the crossing rule, as
  before. The box then says `accelerate past 100 km/h`.
* **The meter is told the pose's speed** (`ChallengeRun.placed`), so a stop
  held at v0 counts from its first braked step, the reference's included.
  It is not told while the run is void (T, slow motion): the held steps arm
  it once the run counts again.

## The numbers

| challenge | ref before (task 40) | ref now | 1 star | 2 stars | 3 stars (was) |
|---|---|---|---|---|---|
| Stop from 100 | 42.47 m | 39.50 m | 44.25 m | 41.87 m | 40.29 m (43.32) |
| Wet stop from 80 | 40.07 m | 39.91 m | 44.70 m | 42.30 m | 40.71 m (40.88) |
| Air brake from 150 | 87.91 m | 84.53 m | 94.67 m | 89.60 m | 86.22 m (89.67) |

The five other challenges re-derived byte for byte.

**A keyboard player gets the reference's number exactly**, whenever they
press DOWN: 39.50 / 39.91 m (three stars) through the real session start.

**The air brake's balance is back** (full pedal from the hold; the lane
keeper steering, as a player would):

| build | stop | stars |
|---|---|---|
| every wing, AIR BRAKE (the reference) | 84.53 m | 3 |
| every wing, AUTO | 85.35 m | 3 |
| the plate pair, AIR BRAKE | 85.96 m | 3 |
| no wings | 86.76 m | 2 |

The wingless car misses the third star by 0.54 m, the plate pair makes it by
0.26 m. The blurb says "about 2.5 % shorter than no wings".

**Seen and not touched:** the plate pair on AIR BRAKE with the steering
left alone (hands off the keys) yaws 2.7 degrees and drifts 1.3 m, and stops
in 87.60 m, two stars. The full build and no wings stay straight. A small
correction on the steering gives 85.96 m. It is the flank air brake's own
behaviour, the same under task 40's start, and the owner kept the air brake
as it is (no physics change).

## Checks

* `drive.challenges` self-check:
  * every stop starts at its v0, and its reference drives the automatic's
    clutch;
  * a stop with no `start_kmh`, one at 0 or one above v0 is refused, one
    below v0 is taken;
  * the hold: only at v0; it lets go at full pedal (the keyboard's ramp, 200
    steps) or a squeeze still for 0.1 s (100 steps); never with no brake, a
    170 ms tap, a pedal falling from 0.9 (R just after DOWN) or a finger
    resting at 3 %; past its room it goes back to the start;
  * the box's line: held, brake now, the wet toggle's reason, R again after
    it let go, no roll promised in a race, and under v0 "accelerate past";
  * a reset's pose arms the meter, but not while the run is void;
  * a meter placed at v0 counts from the first braked step (v^2 / 2a).
* **V35:** every reference as before (all eight to three stars, the files'
  values), R mid-stop rolls again and re-arms, the box keeps LAST STOP, and
  now: a 170 ms tap of DOWN at 1.0 s leaves the car held; DOWN at 1.5 s,
  with the 0.2 s ramp and the clutch left alone, is held at exactly 100 km/h
  until then (the box says so, the meter armed and not counting), lets go at
  1.700 s at full pedal, and stops in the reference's 39.50 m, three stars.
* **The real session start** (`_interactive_session`, offscreen, a
  temporary progress file and library, never the player's), for all three:
  after 2 s of the real loop, the car at exactly 100 / 80 / 150 km/h, 55.6 /
  44.4 / 83.3 m further on, held, "brake when ready". DOWN: 39.50 m [***],
  39.91 m [***]; the air brake 84.53 m [***] on AIR BRAKE with every wing
  (R, then G), 86.76 m [**-] with none.

## What the review found

Two reviewer agents (the live game; the files, docs and tests) and one
judge. Two defects, both fixed:
* **The hold let go on a pedal that was not braking.** It let go of any
  pedal that stopped rising for 0.1 s, a falling one included. Three ways
  to throw a stop away, measured on Stop from 100: R pressed just after
  letting go of DOWN (93.06 m), a 170 ms tap (62.54 m), a finger resting on
  L2 at 3 % (64.45 m), all 0 stars, with the coasting in the distance. Now a
  moving pedal never lets go, a still one needs 30 %, and R zeroes the
  keyboard's brake ramp. All three give 39.50 m, three stars.
* **R under T left a stale "did not count".** The reset armed the meter
  while the run was void, so the next step called it an attempt, and the
  warning stayed after T went off. The reset now places the meter only on
  a run that counts.

Ruled out: ALL 3 gains nothing from the hold (every stop lets go from the
reset's state, wings in, as the reference does: 84.53 m either way, 0.36 m
at most in it); a test gap on a start under v0 (added anyway); a vacuous
lap case in the hold check; CONTRACT's hook paragraph (the module row has
it).

## Shape of it

| file | what |
|---|---|
| `drive/challenges.py` | `HOLD_SETTLE`, `HOLD_MIN`, `HOLD_STILL`, `HOLD_ROOM`; `StopHold` (`holds(brake, dt)`, `advance(dt)`); `ChallengeRun.hold(tr, pose)`, `ChallengeRun.placed(sim, V)`; `Meter.placed(V)`; `_status`'s held line; `validate` takes 0 < start_kmh <= v0; `ref_driver`'s clutch field; `measure(driver=)`; self-check |
| `drive/drive.py` | `Sim.stop_hold`; `step_physics` holds instead of stepping; `Sim._hold_step`; `Sim.reset` sets the hold, places the meter and zeroes the keyboard's brake ramp; `BrakeDriver(clutch=)`; V35 extended |
| `drive/data/challenges/01, 02, 05` | `goal.start_kmh` 100 / 80 / 150, the drivers `brake:1.0:<v0>:0`, the blurbs, thresholds by `--write` |
| `README.md`, `drive/CONTRACT.md`, `.handoff/README.md` | the stops' paragraph, the module row, the index |
