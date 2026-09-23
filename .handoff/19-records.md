# Task 19 — lap records: your top 5 per class

The brief (`.handoff/PLAN-steam-engagement.md`, T19): record every lap, keep
the valid ones as the class's top 5 in `runs/records/<class>.json` with the
time, the sectors, the date, the build (name + the whole `CarBuild` JSON), the
assists, a 50 Hz pose trace in `_Replay`'s schema and a controls log, and prove
that the log is the lap: re-simulating it gives the time back bit for bit.

```
python3 -m drive.records               ->  27/27   (new)
python3 -m drive.drive --self-check    ->  ALL PASS  (V31 new; V20 sha256 ed41b7f781e959ea, unchanged; V21 RTF 12.17)
python3 -m drive.validate              ->  82/82  pass  0 HARD  0 soft   [115.9 s]
python3 -m drive.validate --modules    -> 105/105 pass  0 HARD  0 soft   [286.7 s]  (104 + records)
python3 -m drive.ml                    ->  ALL PASS  32/32
```

Baseline before any edit, same machine: validate 82/82, `--modules` 104/104,
`drive --self-check` ALL PASS, `drive.ml` 32/32.

## What it does

A class is D1's `track|car|engine|surface` (`records.class_key`). Every lap
of a windowed session on a lap map is recorded by a `records.LapRecorder`
hung on `Sim.recorder`; a lap that the timer calls valid, and that nothing
disturbed, goes into `RecordBook.insert` and the class file. The HUD's timing
panel has a new row, `PB 1:00.729` (the class's, not the session's), and the
bottom line says where a lap landed for a few seconds: `NEW PB 1:00.729
(-0.412) P1 of top 5`, `LAP 1:00.837 P2 of top 5 (PB 1:00.729)`, `outside
the top 5`, or `not recorded (reset)`.

## The one decision worth arguing with: the controls are quantised

The brief asks for the controls actually handed to `Vehicle.step`, and for a
re-simulation from them that is bit for bit, in <= 300 KB a lap. Those three
together do not fit with full floats: a keyboard lap's steering changes on
most of its 60 000 steps and a pad or a bot changes on all of them, and a
float64 series that is not piecewise linear does not compress below ~5
bytes a sample -- ~1 MB a lap for steering and pedals alone.

So while a lap is being recorded `Sim.step_physics` hands the recorder the
`Controls` right before the step and the recorder ROUNDS the five continuous
fields in place: the road-wheel angle to 2^-24 rad (6e-8 rad, 3.4e-6 deg) and
the four pedals to 2^-20 of their travel. The rounded value is the one the
car is driven with, so the log is integers, exact by construction, and the
re-simulation sees the same inputs to the bit. It is on the input side of the
step; `Vehicle.step` is untouched; and only a Sim with a recorder does it --
no scripted or headless run has one, so V20 and every acceptance number are
the same bits as before (the gates below).

What it costs, measured: the scripted arena lap (LapDriver, margin 0.90, the
V31 set-up) is 60.727861 / 60.834534 s without a recorder and 60.728602 /
60.836791 s with one -- +0.7 ms and +2.3 ms. That is not the rounding's size
(6e-8 rad), it is how far a closed-loop lap amplifies ANY perturbation in
120 s; a human's own lap-to-lap scatter is a thousand times larger. The
recorder's per-step cost does not show: RTF 11.62 with and without.

Encoding, all exact: each field run-length encoded by step index (the offsets
where it changes and the values there, both delta-coded, packed in the
narrowest int width, zlib level 6, base64). The trace is fixed point (ms, mm,
1e-4 rad, mm/s) delta-coded the same way. The smooth LapDriver lap is the
worst case the suite has and it is **174 KB**; a 19.6 s skidpad lap is
22 KB. Encoding and writing a 60 s lap takes 28-31 ms on the filing thread
(measured at load average 95), off the physics step.

## The start state, and why the replay is exact

A lap opens at the END of the step in which the car crossed the line, so the
snapshot is exactly the state the next step starts from, and closes at the
end of the step with the next crossing (after that step's sector event, which
`LapTimer.update` emits after the lap event). The snapshot is generic, not a
list of fields someone has to keep up to date: every attribute of the
`Vehicle` except the six that are the car rather than its state (`car cfg der
pt_p tyre pos`, plus the diagnostics `tel` / `guard_events`) -- the
`VehicleState`, the `PowertrainState`, the `PtInput` scratch, the ABS gains
and learned levels, the TC gain, the published arrays -- serialised by
`records.ser`, which tags tuples, numpy arrays and dataclasses and RAISES on a
type it does not know rather than dropping state. JSON writes a float as the
shortest repr that reads back to the same bits. Plus the harness's surface
samples (sampled at 200 Hz on `n % 5`, so the replay restores `n` too), and
the `LapTimer`. The car and the `VehicleConfig` go in whole (`CarSpec` with
ballast and wing masses, `DevAero` / `TopAero`), so `resimulate` needs no
library and no settings file.

Checked two ways: V31 (the stock Corsa, arena, three laps, sectors) and
`records.self_check` (a BALLASTED MX-5 at 1.5x power with ABS and TC on and a
designed panel on both flanks, on the skidpad) -- both bit for bit, lap and
sectors.

## What is not a record

* A lap the timer calls invalid (all four wheels off, `LapTimer.lap_valid`).
* A lap with a reset in it -- `R` (the timer keeps running across a sector
  reset, so without this the next lap event would carry the teleport),
  `SHIFT+R`, a race start or restart, the swarm page's seed lap.
* A lap with a live setting change in it (anything on the settings page but
  sound and camera): the engine swaps `pt_p`, the gearbox can put the car in
  neutral, ABS / TC / the steer aid are the lap's assists.
* A lap with the `T` wet toggle on or toggled: the class surface is the
  setting, and `T` is global wet on top of it.
* A lap that did not go ROUND (a sector line out of order, under 95 % of the
  length), any of it in slow motion or single-stepped, or longer than 4 min.
* The DRAGSTRIP: `LapTimer` never fires a lap on an open track. It is
  **excluded**, not given an invented finish line -- a standing-start run to
  a gate is a different game mode (the plan's "or exclude it -- say which":
  excluded). Also a skidpad of another radius or direction (CLI only): a
  30 m pad is not the 50 m pad's track, and D1's key has no radius.
* `--ml-drive`, `--headless`, `--script`: no recorder at all.

## What an adversarial review found, and what changed

The diff went through a four-lens review (determinism, lifecycle, contract,
robustness), 4 finders and 13 verifiers each told to refute; the confirmed
defects, all fixed before this commit:

* **A lap that never went round was filed as the PB** (high). `LapTimer`
  swallows a backwards crossing (its > 10 m jump guard) and fires a lap after
  the 3 s lockout at the next forward one: cross the line, reverse over it,
  drive over it again and the arena "lap" was 12.166 s, P1 for ever, with
  sector 3 = 12.166 in the class's best sectors; a 30 m circle on the open
  map's pad that touched the line filed 15.7 s. Now a lap is filed only if
  every sector event arrived in order and the centreline progress reached
  0.95 of the length (`LAP_MIN_FRACTION`); the HUD says `not a full lap`.
  The timer itself is untouched (V25).
* **A live engine change kept the old class** (high, found twice). The engine
  is in D1's key but is a live setting, so tuned laps went into the stock
  table and the stored assists were the session's first ones.
  `LapRecorder.retarget` now runs on every live `apply_setting` -- and takes
  map, car and surface from the RUNNING key, because the settings page holds
  a PREVIEWED map / car / surface in `settings` until ENTER (the review's
  third high: a previewed map plus a live engine change filed arena laps in
  the open map's table).
* **Closing a lap stalled the step ~0.45 s** (zlib level 9 over int64
  arrays, inside `step_physics`). Now each series is packed in the narrowest
  int width, zlib level 6 (20x faster, 4 % bigger), and the encoding and the
  write run on one worker thread: the crossing step only inserts a LIGHT
  record (time, sectors, build, assists) so the rank, the PB and the HUD are
  right at once, and the thread adds the trace, log and start state and
  writes. A half-built lap is never written. The session flushes the thread
  when it ends, before the next session's book reads the file.
* **A failed write killed the drive** (medium): `PermissionError` out of
  `step_physics` on a read-only install. Now the book stays in memory, the
  HUD says `NOT SAVED (...)`, nothing raises; the self-check drives a lap
  into a chmod-500 directory.
* **A lap in slow motion or single-stepped was a record** (medium; judged
  "not a defect under the letter of the spec" by one verifier, fixed anyway:
  a quarter-speed lap is not the lap a table is for).
* **Robustness**: a file that cannot be READ (a lock, a permission) is no
  longer treated as corrupt -- it was moved aside and the next save wrote a
  one-lap file over the owner's top 5; now the class is marked unreadable and
  never written over that session. `save` merges what is on disk before it
  writes (a second writer loses nothing), writes through a per-process temp
  file with `fsync`, and keeps the previous file as `.bak`, which `load`
  falls back to when the main file is torn. An open lap is dropped after
  4 minutes (`LAP_MAX_S`): a parked car held ~360 MB an hour of log. Any
  exception in `load` (an out-of-range number) is a bad file, not a crash.

Rejected: nothing material. The note itself was flagged missing (it was
being written).

## Deviation from the plan

* **File names.** `runs/records/arena__corsa__sport__patch.json`, not
  `arena|corsa|sport|patch.json`: `|` is illegal in a Windows file name and
  the Steam build ships there. The key inside the file is D1's, unchanged.
* **The controls are rounded** before the step while a lap is recorded (above).
  The plan's "Controls actually handed to `Vehicle.step`" holds literally --
  the rounded ones are the ones handed over.

## Shape of it

| file | what |
|---|---|
| `drive/records.py` (new) | class key; `RecordBook` (top 5, best sectors, per-build bests and the best medal for tasks 20 / 21, `last_builds.json` for task 20; a corrupt / old file moved aside to `*.bad-<stamp>` with a note, an unreadable one never written over, `.bak` fallback, merge-on-save, one lock); `LapRecorder` (the three `Sim` hooks, `retarget`, the filing thread); `ser` / `deser`, `vehicle_snapshot` / `restore_vehicle`; the encodings; `resimulate`; `session_recorder`, `lap_note`; self-check |
| `drive/drive.py` | `Sim.recorder` + three calls in `step_physics`; `discard` in `reset`, `apply_setting` and the `T` toggle; the HUD note and `pb_lap` / `lap_rank`; `_interactive_session` attaches the recorder; `_apply_design` carries the build name + JSON; V31 |
| `drive/render.py` | `HudData.pb_lap`, `lap_rank`; the timing panel is 18 px taller for the PB row |
| `drive/validate.py` | `records` in `MODULES` |
| `drive/CONTRACT.md`, `README.md` | module map, the recorder hooks, the records section, layout |

## Not done

* Nothing reads the controls log yet except `resimulate` and V31: score
  verification is task 28's.
* The trace is recorded but not drawn: the ghost is task 22.
* A lap recorded before a garage rebuild of the same build name stays filed
  under that name with the build JSON it was actually driven in; the pick
  page (task 20) matches by name.
