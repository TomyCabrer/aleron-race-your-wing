# Task 22 — two ghosts and the live delta

The brief (`PLAN-steam-engagement.md`, T22 and decision D2): ghost 1 = the
class PB's trace, ghost 2 = a slot (none / the reference bot / any of your
top-5 laps; default the reference bot), both drawn with `_Replay` and the
existing ghost drawing, top-down and in the 3-D chase view, one toggle key;
ghosts start at the line and restart every lap; a live delta to the PB read
by track progress (the race gap's approach), large at the top centre, green /
red; a sector flash (purple = best sector ever for the class, green = better
than the PB lap's, red = worse). Acceptance: render self-check additions, V22
green with two ghosts, V33 (the delta against a lap's own trace stays about
zero, tolerance stated).

```
python3 -m drive.ghosts                ->  17/17   (new)
python3 -m drive.render                ->  31/31   (+2: the delta and the flash are drawn; a ghost under the car in chase)
python3 -m drive.drive --self-check    ->  ALL PASS  (V33 new; V20 sha256 ed41b7f781e959ea unchanged; V21 RTF 12.19)
python3 -m drive.validate              ->  82/82  pass  0 HARD  0 soft   [113.3 s]
python3 -m drive.validate --modules    -> 108/108 pass  0 HARD  0 soft   [295.2 s]  (107 + ghosts)
python3 -m drive.ml                    ->  ALL PASS  32/32
```

## What it does

`drive/ghosts.py`: a `GhostSet` per session (with records), on the
recorder's book and class.

* **Ghost 1** is the class PB: the 50 Hz trace task 19 files with every
  record, as a `drive.drive._Replay`, drawn as a ground silhouette labelled
  `PB` through `HudData.ghosts` (the swarm's and the race's path: plan view
  and chase view alike, culled to the view radius).
* **Ghost 2** is a slot: `ref` (default; the reference bot -- the lap the
  class's author medal time was set with, `medals.reference_trace`, 20 Hz),
  `none`, or `top2`..`top5` (your P1 IS the PB; offering it stacked two
  identical ghosts). The pre-race page has a **Ghost 2** row;
  LEFT / RIGHT (or ENTER, or a click) steps it. The choice and the J toggle
  survive a session restart.
* **Clocked from the line**: a ghost's time is `sim.t - LapTimer.t_lap_start`,
  so both appear when you cross the line and restart at every crossing; on
  the out-lap there is none, and a ghost that finishes first waits on the
  line.
* **`J`** hides and shows both ghosts (the delta stays). A new key: README key
  table, the menu help, `--help`. No pad button is free for it; the pad
  drives the pre-race page's Ghost 2 row instead.
* **The live delta**: at your centreline distance `p` from the line the PB
  had been driving for `t_pb(p)` (the PB trace carries `s`, task 19), you
  for `tau`, delta = `tau - t_pb(p)`: by track position, as the race gap is
  read -- `p` is COUNTED from the crossing (reset at 'start' / 'lap', ds
  added only on the ribbon), not read off the wrapped `s`. Large (the speed
  font) at the top centre under the timing panel, green ahead, red behind;
  none before the first crossing, without a PB, off the ribbon (the open
  map's pad), behind the line, or on a lap the recorder has dropped.
* **The sector flash**, under the delta for 2.5 s at each sector line:
  purple when the sector is the best ever driven in the class (the class's
  best sectors, which every valid lap updates, not only the top 5), green
  when it beats the PB lap's own sector, red otherwise, with the difference
  to the PB's sector -- and NO colour for a lap that cannot count (the
  out-lap, all four wheels off, a lap the recorder dropped).
* A new PB races the very next lap. It lands as a LIGHT record at the line
  (task 19's filing thread encodes and writes the rest), and the light
  record now also carries the lap's raw trace in memory (`_trace_arr`,
  never written): the ghosts reload when the book's version moves and use
  it at once -- measured, the PB ghost and the delta are live 0.05 s after
  the line. A live engine change moves the class, and the ghosts follow it.

## A finding on the way: the filing thread and the GIL

Before the in-memory trace, the new PB's ghost appeared seconds into the
next lap, and in a tight headless loop never within it. Measured: filing a
60 s lap is 19.6 ms of encoding and 1.4 ms of writing alone, but **4.93 s
wall** against a busy Python physics loop -- a GIL convoy: the encoder's many
short numpy / zlib calls each give the GIL up and wait up to the 5 ms switch
interval to get it back. The interactive loop sleeps in `tick_busy_loop` and
leaves the thread room, but the fix does not depend on it: the ghost no
longer waits for the file.

## What the review found

Two lenses (semantics in play; render and acceptance), 2 finders and 3
verifiers; all confirmed defects fixed in this commit:

* **The delta vanished on exactly the laps you were beating the PB** (high):
  the first version unwrapped `s` by a heuristic (`p > L/2` and `tau <
  t_end/2` meant "just over the line, backwards"), which a car AHEAD of its
  PB at half distance also meets -- blank for up to the lead's seconds, on
  the symmetric open map and skidpad for any lead at all. Now the progress
  is counted from the crossing, the race gap's way; a self-check case sits
  just past half distance ahead of the PB.
* **The sector flash coloured laps that can never be filed** (the out-lap,
  after all four wheels were off, a dropped lap): "purple, best ever" twice,
  meaning nothing. Now no colour for those.
* **In the chase view a PB within ~8 m was hidden under the 3-D car**, label
  and all. Labels (and, in chase, outlines) are drawn after the car now; the
  render self-check puts a PB 4 m ahead and counts it (22 847 px), and the
  chase budget runs with five ghosts, the delta and a flash.
* Low: ghost 2 = "your P1" stacked on the PB (dropped); the delta read
  numbers off the ribbon on the open map's pad (none now).

## V33, and the tolerance

`_v33_ghost_delta`: one arena lap (LapDriver) is recorded, then
re-simulated from its controls log (`records.resimulate`, now with
`on_step` / `on_event` hooks) with a `GhostSet` whose PB IS that lap, reading
the delta and the ghost's pose every 20 ms (the replay starts just past the
line, so the GhostSet is handed that crossing at its first step, as a live
session's is). **Tolerance 5 ms** -- the trace's
time is stored to 1 ms and its distance to 1 mm, sampled at 50 Hz, and the
delta is read between samples. Measured: **max |delta| 0.17 ms** over 3037
reads; the PB ghost within **3 mm** of the car it recorded; the three sector
flashes purple (a lap's own sectors are the class's best).

V32 (task 20's page flow) gained the Ghost 2 row: RIGHT steps the slot
`ref -> none` and the row says so, ENTER steps it again (`top1`), and `J` on
the road toggles the ghosts.

## Frame budget

V22 now draws the busiest time-trial frame: the PB and ghost 2 plus a full
race grid of three bots, all in view and labelled, the delta and a sector
flash: mean 4.06 ms, p99 4.59 ms against the 12 / 16 ms budget (the T19
baseline frame, without them, measured mean 4.07 / p99 5.02 on this machine
-- the ghosts and the delta cost nothing measurable). The render self-check also
asserts the delta and the flash are DRAWN (green ahead, red behind, purple
flash, nothing without) by counting their pixels.

## Shape of it

| file | what |
|---|---|
| `drive/ghosts.py` (new) | `GhostSet` (the two ghosts, `Curve` for the delta, the sector flash, `sync`, the slot), `slot_label`; self-check |
| `drive/drive.py` | `Sim.ghosts`; its event hook in `step_physics`; `J`; `hud_data` (ghosts, `delta_s`, the flash); the pre-race Ghost 2 row; `_interactive_session` builds it; the slot survives a restart; V33 |
| `drive/render.py` | `HudData.delta_s` / `sector_flash` / `flash_col`, `_draw_delta`, `C_FLASH`; ghost labels / chase outlines after the car (`_draw_ghost_tops`); V22's and the chase budget's busy frames; the drawn-delta and the ghost-under-the-car checks |
| `drive/records.py` | `resimulate(on_step=, on_event=)` |
| `drive/input.py` | `J` -> `ghosts`, the help rows |
| `drive/prerace.py` | the Ghost 2 help row |
| `drive/validate.py` | `ghosts` in `MODULES` |
| `drive/CONTRACT.md`, `README.md` | module map, the flow, the key |
