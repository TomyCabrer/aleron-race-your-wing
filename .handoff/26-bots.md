# Task 26 — bot training and race flexibility

The brief (`PLAN-steam-engagement.md`, T26): the swarm stays open-ended by
default (`gens=0`: no prompts, no "train N more"), with a progress panel --
the best lap per generation, the medal lines, the owner's PB for the class
and the gap to each -- and ESC stops it as before. Free values instead of
fixed lists: population any integer in [4, 128], sim time in [20, 240] s.
Race grid: `RACE_GRID_MAX` to the largest N that keeps V22 green, measured
and written down; each slot follows D3 (name + colour + checkpoint + saved
build). A saved bot carries the build it was bred in (checkpoint metadata),
so racing or ranking a bot uses its own car. Acceptance: V30 still passes;
**V36** an N-bot race; the swarm self-check passes.

```
python3 -m drive.race_grid            ->   9/9    (new)
python3 -m drive.swarm_panel          ->  10/10   (new)
python3 -m drive.ml.swarm             ->  ALL ok  (the swarm self-check; exit 0)
python3 -m drive.render               ->  32/32   (V22 and the chase budget now draw the full grid of five)
python3 -m drive.drive --self-check   ->  ALL PASS  (V30 ok, V36 new; V20 sha256 ed41b7f781e959ea unchanged; V21 RTF 11.64)
python3 -m drive.validate             ->  82/82  pass  0 HARD  0 soft   [116.2 s]
python3 -m drive.validate --modules   -> 114/114 pass  0 HARD  0 soft   [379.2 s]  (112 + race_grid, swarm_panel)
python3 -m drive.ml                   ->  ALL PASS  32/32
```

## How big the grid can be: measured

Each bot is a second `Vehicle` stepped at 1 kHz in the same loop as yours.
The physics of the whole grid, per second of sim, on this machine (the
arena, the built-in driver in every slot; `scratchpad` script, load average
about 6 at the time):

| bots | 0 | 1 | 3 | 5 | 7 | 9 |
|---|---|---|---|---|---|---|
| real-time factor of the physics | 12.93 | 5.68 | 2.74 | **1.79** | 1.35 | 1.07 |
| physics per 60 fps frame (16.7 ms / RTF) | 1.3 ms | 2.9 ms | 6.1 ms | **9.3 ms** | 12.3 ms | 15.6 ms |

The render's side (V22, now with the full grid of five bots plus the PB and
ghost 2 in view, the delta and a flash): mean **4.41 ms**, p99 **5.21 ms**.
Five bots: 9.3 + 5.2 = 14.5 ms of a 16.7 ms frame -- real time with a margin.
Seven: 12.3 + 5.2 = 17.5 ms -- the loop falls behind (the accumulator drops
time, the game runs slow). So **`RACE_GRID_MAX = 5`**. V22 by itself would
allow far more: drawing a ghost costs next to nothing; the bound is the
physics, and "keeps V22 green" is read as "keeps the frame inside its
budget" (deviation below). V36 re-measures it on every run: RTF 12.7 alone,
**1.78 with the grid = 9.4 ms of physics in a 16.7 ms frame**.

## What it does

* **The progress panel** (`drive/swarm_panel.py`, new), in the swarm
  window's text panel: the best lap of each of the last eight generations with
  a bar (the fastest, the longest) and how many cars lapped; the class's
  author / gold / silver / bronze (drive/medals.py) and your PB in the class
  (drive/records.py, read only in a player session), each with the swarm's
  best lap's gap to it (`-` = the swarm is faster). The class is the map, the
  car the swarm breeds in, your engine and surface. The swarm's laps are at
  its 2 ms training step and the panel says so; the saved bot is re-measured
  at 1 ms as before. The swarm is still open-ended: no prompts; ESC stops it.
* **Free values**: the Deploy-swarm page's *Cars* is any integer from 4 to 128
  (LEFT / RIGHT one, ENTER eight at a time and round to 4 past the top) and
  *Sim time* any whole second from 20 to 240 (5 s, ENTER 20 s); `--swarm N`
  and `--swarm-T` are clamped to the same ranges, with a note when they are.
* **The grid** (`drive/race_grid.py`, new): five slots, rows of two 7 m
  apart behind the line; each slot is a NAME (the checkpoint's), a COLOUR
  (orange, blue, violet, rose, lime -- the page row says which), a CHECKPOINT
  and a CAR, and the car defaults to **its own**: the car the bot was bred in.
  With three or more bots the HUD's bottom line gives the slot number and the
  gap, so it fits the bar.
* **A saved bot carries its car**: `run_swarm_cli` sets `Swarm.bred`
  (`race_grid.bred_meta`: the car, stock or yours with your ballast and the
  wings' mass, the garage build's JSON, the engine's power scale); it goes into
  the saved checkpoint's meta and the swarm's state file (a resume records
  the car it breeds in then). *Its own* car (`race_grid.own_car`) rebuilds
  exactly that; a bundled checkpoint, which only names a car and a published
  wing, drives that stock car with that wing; the built-in driver, and a swarm
  bot saved before this version, drive yours. The RACE page's *Test
  bot 1 in every car* (the ranking) lists its own car first.

## V36

`_v36_grid`: the full grid on the arena, five slots (the built-in driver and
four bundled checkpoints, each in its own car): your car bit-identical with
and without the grid (V30's rule, now with five cars); every bot leaves the
line from its own slot and keeps going; each drawn in its slot's colour; the
540i checkpoint drives its own 540i (with its plate) while the built-in driver
drives yours; the HUD carries five ghosts and five gaps; the render's frame
budget is held to the same grid (`render.V22_GRID_BOTS == RACE_GRID_MAX`);
and end to end for D3: a Swarm with a bred car -> its state file -> resumed
-> its best as a checkpoint -> `own_car` rebuilds that car (the MX-5, stock,
the build's flank wing, the tuned engine). It reports the grid's cost.
Measured on the final run: RTF 12.8 alone, **1.75 with the grid = 9.5 ms of
physics in a 16.7 ms frame**.

V30 still passes; one expectation in it was written against the old car
list (`RIGHT` from *same* was "the second entry"); it now reads "the entry
after *same*" -- the same assertion on the new list, not a weaker one.

## An incident on the way (not in the code)

A scratch smoke test of the swarm window called `run_swarm_cli` at module
top level with no `if __name__ == "__main__":` guard. macOS starts the
swarm's pool workers with *spawn*, which re-imports the script in each
worker, so every worker started a swarm of its own: a recursive run that
wrote ~5,300 temp directories and ~5,200 killed-mid-run XFOIL scratch
directories (13.8 MB each) into `$TMPDIR` -- about 88 GB in under two hours --
and filled the disk (every tool failed with ENOSPC; the owner's other jobs
that write logs would have too). The runaway was stopped, only the
directories it made were removed (13,032, created in the last 2.5 h, none open
by a live process), and the disk is back where it was (89 GB free). Nothing in
the repo or `runs/` was touched. The smoke test was rewritten with the guard,
4 cars, a hard time limit and XFOIL off, and it passes. The game code was not
at fault: `drive.drive` has its guard.

## What the review found

Two lenses (the grid and the bot's own car; the swarm page and the panel), 2
finders and 10 verifiers: nine confirmed, all fixed in this commit.

* **Your designed wings rode along in a bot's own car** (high): a bot bred
  with no wings, or with the published panel, got the session's designed
  flank and top wings on top of its own (a pure-legacy build's kwargs carry
  no devices, so the session's survived). The build's devices now default to
  none; checked with a designed session and a wingless bot.
* **Your existing swarm bots lost their wings under *its own*** (high): a
  swarm bot saved before this task records only `car` and `wing='off'` (a
  three-slot build reports the legacy wing as off) -- it was bred in YOUR
  car. Such a bot (meta has `swarm`, no `bred`) now drives yours, as it did
  before; only the bundled checkpoints use the stock car + published wing.
* **A resumed swarm kept the first run's car** (medium, two findings): the
  saved bot now records the car it breeds and is measured in NOW, with a note
  when a resume changed it.
* **`--race` and the bot put on the grid after a swarm still defaulted to
  *same***, and `--help` still said three bots (medium): *its own* and five.
* **The HUD's slot numbers shifted when a slot failed to load** (medium): a
  bot carries its slot.
* **The panel compared a rolling-start FIRST lap with flying-lap medals**
  (medium): with the page's default 70 s on the arena no car does two laps,
  and the lap it does is from the rollout's 12 m/s at the line. The swarm now
  marks each lap flying or not; a first lap is shown as that, with no gaps.
* **An off-standard skidpad showed the standard skidpad's medals** (medium):
  no medals or PB for a class the records do not keep, and the panel says why.

## Deviations

* **"The largest N that keeps V22 green"** is read as the largest grid whose
  PHYSICS plus the V22 frame fit the 16.7 ms frame. V22 (a render-only
  budget) stays green with many more ghosts; the loop's real limit is five
  bots' physics. V22's busy frame now draws the full grid of five.
* **Free values by stepping**, not typing: LEFT / RIGHT reach every value
  (and the pad's d-pad does too); ENTER takes a big step.
* **The race page's car row defaults to *its own***, where it was *same as
  mine*; for the built-in driver (the default bot) the two are the same car.

## Shape of it

| file | what |
|---|---|
| `drive/race_grid.py` (new) | `GRID_MAX`, `grid_slot`, the colours, `bred_meta`, `own_car`; self-check |
| `drive/swarm_panel.py` (new) | `panel_lines`, `lap_rows`, `best_lap`, `clamp_pop`, `clamp_T`, `step_value`; self-check |
| `drive/drive.py` | `RACE_GRID_MAX` / `RACE_GRID` / `C_RIVALS` / `RACE_BOT_CARS` from `race_grid`; `_race_car(name, meta)`, `_bot_meta`, `Sim.garage_lib`; the compact multi-bot HUD; the swarm page's free values; `run_swarm_cli`: the clamp, `Swarm.bred`, the panel; V36; V30's adapted expectation |
| `drive/ml/swarm.py` | `Swarm.bred` in the saved bot's meta and in the state file |
| `drive/render.py` | `V22_GRID_BOTS`; V22 and the chase budget draw the full grid |
| `drive/validate.py` | `race_grid`, `swarm_panel` in `MODULES` |
| `drive/CONTRACT.md`, `README.md` | module map, the grid, the panel, the free values |
