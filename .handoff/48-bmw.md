# Task 48 — the BMW 540i: it spun "just by giving it thrust"

Owner (2026-09-28), verbatim:

> *"After doing that fix the bmw. just by giving it thrust it spins arround and
> it is very difficult to control in general."*

then *"I am going to sleep don't ask me anything and finish all the tasks"*.

(Numbered 48 -- and the leaderboards 47 -- because another session's `/goal`
the same night is task 46: `.handoff/46-goal-eight-asks.md`.)

## Diagnosis (a workflow: 3 lenses + a judge; scripts in the session scratchpad `bmw/`)

1. **The owner's saved seat**: `runs/settings.json` had car `540i`, engine
   **Sport**, **TC off, ABS off**. Sport is x2 torque on EVERY car (it was the
   global default, chosen for the 75 hp Corsa): the 540i became 880 N.m /
   ~570 PS. Rear drive force vs rear grip at Sport: 29.5 / 17.6 / 11.6 kN in
   1st / 2nd / 3rd against ~12-15 kN. Arena rolling start, keyboard UP only, no
   steer: **spins at 2.67 s** (beta 86 deg, rear slip pinned at 1.5, 4.4 s of 6
   on the limiter). Stock with TC off, same keys: beta <= 1.1 deg, no spin.
   Sport with TC on: no spin.
2. **TC was the Corsa front axle's** (slip 0.12 / 0.20, the traction peak): on
   a rear axle 0.12 slip has already cost 40-50 % of its cornering force, so
   with TC ON the 540i still spun when power went on in a wet corner (R 40 m:
   Stock beta 31 deg, Sport 87 deg).
3. **The steer aid was the Corsa's** (k_us 7.82 deg/g, L 2.491 m; the 540i is
   12.6 deg/g, 2.83 m): its soft lock allowed 77-84 % of the angle the 540i's
   limit needs (the Corsa gets 115 %), so it pushed wide and was turned with
   the throttle -- "very difficult to control in general".
4. Not causes: task 45's gearbox (A/B against 82eba99: identical spin times),
   braking with ABS off (the fronts lock first, straight), the open diff (it
   behaves as one).

## What was changed

* **Each car keeps its own Engine** (`Settings.engines`, like Paint): a car
  never given one opens on **Stock** (`ENGINE_DEFAULT` is now `stock`, task 47).
  A settings file from before loads its one engine as the **Corsa's** -- so the
  owner's 540i opens on **Stock** next launch and the Corsa keeps Sport (the
  first screen's note is below, with TC's). Sport stays one Settings press
  away, for that car only. (No physics; the owner's file is not edited: it is read
  that way and saved with the new table.)
* **Each car keeps its own TC** too (`Settings.tcs`; review finding): a car
  never given one opens with TC **on**, and the old file's TC off is the
  Corsa's. With the 540i on Stock and TC off, full throttle in 1st with any
  steer, or a kickdown mid-corner, still spun it (a real E39 has ASC as
  standard). So the owner's next launch opens the BMW on **Stock with TC on**;
  the first screen says *"Engine and TC are now kept per car: the BMW 540i is
  on Stock (was Sport), TC ON - Settings"*, and both are one press away for
  that car alone.
* **The 540i's own TC slip targets**: `CarSpec.tc_slip = (0.06, 0.12)` (None =
  the module's 0.12 / 0.20 on every other car, the same floats: bit-for-bit),
  read in `Vehicle._tc`. TC ON only: every TC-off number and rig is untouched;
  TC-on scripted runs of the 540i (the medals' aids-on laps, the challenge
  references) move and need the rebuild below. Measured with the real field
  (TC on):

  | case | before | after |
  |---|---|---|
  | wet R 40 power-on, Stock | spins, beta 31 deg | no spin, beta 3.4 deg |
  | wet R 40 power-on, Sport | spins, beta 87 deg | no spin, beta 6.8 deg |
  | dry R 40 power-on, Stock / Sport | no spin | no spin, beta 3.9 / 4.2 deg |
  | 0-100 km/h Stock / Sport | 6.99 / 6.00 s | 6.96 / 6.44 s (a real 540i auto: ~6.9 s) |

* **The 540i's own steer aid**: `CarSpec.aid_own` -- the keyboard's / pad's
  soft lock on its own wheelbase, grip and measured understeer (16.25 deg at
  20 m/s where its peak needs 14.13; was 11.23), without the rest of
  `own_aids` (the scripted drivers keep the Corsa calibration).
* **TC off is said**: a session in a car with its own TC calibration, driven
  with TC off, says once a launch *"TC is OFF: the BMW 540i spins its rear
  wheels on full throttle - Settings > TC"* (kept beside the start's other
  notes, not overwritten by them).

Checks: `input` (the 540i's own aid clears its peak angle, the Corsa's did
not), `V48` in `drive --self-check` (the migration, per-car memory, `--car` /
`--engine`, TC targets reach `_tc` and a car without them is bit-for-bit, TC
off untouched, the note once).

## Not done -- the owner's call

* **A tyre-model artefact** (found by the diagnosis): MF6.2's kappa-induced
  side force (RVY1 0.05654, RVY2) is not in tyre.py's symmetrisation, and
  `TYRE_MIRROR` flips it with sign(alpha), so a tyre putting down power pushes
  AWAY from straight ahead (a 2 x SVyk step at alpha = 0; the "C1 at alpha = 0"
  docstring holds only at kappa = 0). A no-steer full-throttle launch drifts the
  540i -8.6 deg in 8 s (the Corsa +3.2 deg); zeroed, 0.001 deg. It changes
  EVERY car's physics (and tyre.py's golden points pin the force), so the
  judge's recommendation is a vehicle-layer DEVIATION 3 (the Vehicle's tyre
  copies get RVY1 = RVY2 = 0) plus a rebuild of medals and refs -- only with
  the owner's OK.
* The MX-5 still spins on the wet circle with TC on (it would want tc_slip
  ~0.04 / 0.08); the AUTO box kicks down to 1st at full throttle up to ~55 km/h
  on a rear-drive car; the `540i_arena_plate` bot is still broken since task 45.

## Medals and challenge references

The 540i's new `CarSpec` fields make `drive/data/medals.json` STALE (the hash
covers every car spec) and move its TC-on challenge references. A full
rebuild was started and then STOPPED on purpose: the task-46 session is
changing the car roster (MX-5 and bus out, a rally car in) and runs the full
`python3 -m drive.medals --build` and `python3 -m drive.challenges --measure
--write` after merging its lanes into main, which covers this change too.
Until that runs, `python3 -m drive.medals` reports STALE (a soft fail) and
the medals still load.

**Done (2026-09-28, by task 46's session after merging its lanes):**
`challenges --measure --write` -- only the 540i's 20 combos moved (this
task's TC change); `medals --build --workers 8` -- 252 classes with medals
(7 lap maps x 4 cars x 3 engines x 3 surfaces), 4748 s, the 540i's
`tc_slip` included. Before the medal build: `drive --self-check` ALL PASS.
