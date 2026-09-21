# Task 17 — "deploy a swarm": a genetic algorithm over driving genomes, bred from the user's own lap

The brief: N learning cars on the map and car the user is driving; every
generation the best reproduce; the swarm can be seeded from **a lap the user
drove**; the best is a deliverable the game drives. `train.py` (task 9) is an
evolution *strategy* — one mean genome and a cloud around it. This is the
other classic: a **population**, elites kept, the rest bred by crossover and
mutation. Same package, same anchor, same `Policy`, no new dependency.

```
python3 -m drive.ml                  ->  ALL PASS  28/28   (17 through wave 7)
python3 -m drive.ml.swarm            ->  17 checks, one process, pop 6, 6 s rollouts   (also --self-check)
python3 -m drive.ml.clone            ->  the clone recovers a known trim, both heads
python3 -m drive.render              ->  29/29   (27 before: ghosts and overlay)
python3 -m drive.drive --self-check  ->  ALL PASS
```

`drive/` still imports nothing from `drive.ml`: the subprocess check in
`drive.ml.__main__` imports `drive.drive / vehicle / render / input` and
asserts no `drive.ml` module in `sys.modules`; a grep of `drive/` outside
`drive/ml` finds the two lazy imports only (`_ml_input`, behind `--ml-drive`;
`run_swarm_cli`, behind `--swarm`), both inside functions.

## Shape of it

| file | what it is |
|---|---|
| `drive/ml/swarm.py` (648 lines) | `Swarm`: population, tournament + BLX crossover + gaussian mutation, self-adaptive per-genome step, resumable state, `save_best` -> a plain `Policy` checkpoint. Headless CLI. No pygame. |
| `drive/ml/clone.py` (247 lines) | behavioural cloning: a seed-lap file -> (observation, residual target) pairs -> Adam in numpy -> a genome that drives like the user |
| `drive/ml/policy.py` | the **free-wings head**: 5 outputs (steer, pedal, wing_l, wing_r, wing_top), 373 parameters; `Policy.widen` lifts a 4-output checkpoint; `expand_base` spreads the anchor's one wing verdict over three |
| `drive/drive.py` | the seed lap (`K`, `--seed-lap`, the menu's *Seed lap*), the **Deploy swarm** menu page, `run_swarm_cli` (the window: replay of a scored generation as ghosts while the pool breeds the next), `--swarm / --swarm-seed / --swarm-gens / --swarm-T / --swarm-name / --swarm-resume` |
| `drive/render.py` | `HudData.ghosts` (flat ground silhouettes through `_gpoly`) and `HudData.overlay` (a text panel); empty, the frame is bit-identical |
| `drive/menu.py` | `note=` and one-column help sections for the swarm page |
| `drive/input.py` | `K` -> `seed_lap`; `key_sink` so a `TextPrompt` can own the keyboard |
| `drive/ml/checkpoints/swarm_arena.json` | the deliverable (below) |

## The decisions worth arguing with

**A GA, not a second ES.** The reward has a -60 cliff (off track) and a flat
plateau (a stopped car). An ES estimates one gradient from the whole cloud
and one crashed candidate pulls the mean; ranks fixed that in task 9 but the
mean is still one point. A population keeps several basins alive and an
elite is *never* lost to a bad step: `best never regresses` is a self-check,
and the first thing every seeded swarm does is put the seed itself, unmutated,
at index 0 of generation 0 — so a swarm cannot do worse than what it was
given. Fitness is the existing reward, then lap time as the tiebreak.

**Every genome carries its own mutation step.** The first swarms had one
population-wide sigma, adapted by the 1/5 rule, and it collapsed: 24
near-identical cars on the centreline. Now a child's step is its parents'
mean times a log-normal (`SIGMA_TAU` 0.25) and generation 0 is spread
log-uniform over `SIGMA0 * [0.5, 4.0]`, so one lineage can keep making big
moves while another fine-tunes. The 1/5 rule survives as a nudge on every
genome's step, not as the step.

**Free wings are the default head.** The published policy has one wing
output and the *car* picks the flank. The swarm's genome has one output per
wing, so a car decides for itself, and may run both flanks as an air brake
(`Controls.wing_cmd`; unset it is the published path bit-for-bit). `theta = 0`
with the free head still laps (reward 1289 vs 1290 one-wing, the difference is
`expand_base`'s threshold), and a 4-output checkpoint seeds a free-wings
swarm widened — self-checked that its steer, pedal and wing verdicts are
unchanged. `--legacy-wings` breeds the old head for anyone comparing to a
task-9 checkpoint.

**The seed lap is written by the sim, in the sim's words, with no import of
`drive.ml`.** `K` arms it *before* the lap (the user decides in advance that
this lap is the base, not afterwards that it was a good one). 100 Hz rows of
published state plus the controls, plain JSON, `kind carsim-seed-lap-1`. The
column list lives in `drive.drive.SEED_LAP_COLS` and again in
`drive.ml.clone.SEED_COLS`, and the package self-check asserts the two agree
by name — the only coupling between the two sides is a string tuple.

**The clone learns the residual, not the user.** The target is
`clip((user - baseline) / RESID_GAIN, -1, 1)`: only the part of the driving
the anchor does not already do. On the one real seed lap in `runs/swarm`
(59.70 s, 5 971 rows, wing off) the fit is rmse steer 0.059 of lock, pedal
0.155, mse 0.0205 after 3 000 Adam steps in 2.4 s. The synthetic self-check
(anchor + a fixed trim of 0.15 / -0.20) recovers it under 0.06 / 0.10 with
both heads.

**Every quoted lap is re-measured at 1 ms.** The swarm trains at
`DT_TRAIN` (2 ms); `save_best` runs the best genome at `DT_EVAL` for 3 laps
in a worker of its own so the window keeps drawing (`measure_best_async`),
and that is the number in `meta['measured_1ms']`.

## Numbers

**The deliverable.** `drive/ml/checkpoints/swarm_arena.json` — the best of
the four checkpoints the session bred on the arena, Corsa, **wing off** (the
car the user was driving had no wing fitted; a free-wings genome bred with no
wing is a driver, not an aero policy). Lineage: anchor -> 24 cars x 2 gens ->
24 x 10 -> **64 x 8** (this one, generation 7, individual #419, parents
#381 x #345) -> 64 x 5 (62.388 s, kept in the scratchpad, not the deliverable).
Re-measured here at 1 ms, 3 laps:

```
                              best lap   v_mean    util_f   util_r
swarm_arena.json, wing off     62.205 s  19.874    0.990    1.003
swarm_arena.json, wing plate   62.193 s  19.869    0.990    1.001   deploys 63 %, 92 % outer
anchor (theta = 0), wing off   65.940 s  18.651    0.987    0.985
```

**-3.74 s a lap (5.7 %) over the anchor on the arena with no wing**, and it
still knows what to do with a plate it was never bred with (the widened head
inherits the anchor's outer-flank rule). The value stored in the file is
62.362 s because the session measured with `T_eval = 245 s`; 62.205 here is
`T = 240`. Both are the same three laps to within the lap the window cuts.

**Throughput, measured on this machine (12 cores).** The README had "~3 s a
generation for 32 cars x 70 s" — that was never measured and is wrong:

```
python3 -m drive.ml.swarm --pop 32 --gens 2 --duration 70 --wing off
  gen 0  best 1340.3  mean 1293.3  lap 66.54  lapped 31/32  [14.9 s]
  gen 1  best 1379.1  mean 1309.0  lap 65.22  lapped 30/32  [14.4 s]
python3 -m drive.ml.swarm --pop 8 --gens 2 --duration 40           2.9 s, 2.5 s a generation
--resume ... --gens 1 --out ...                                    2.9 s + 20 s for the 1 ms lap
--seed runs/swarm/seed_arena_corsa_20260921_184040.json --pop 8    clone 2.4 s, then 4.0 s a generation
```

14.4 s is exactly task 9's "32 rollouts of 70 s take ~14 s" — the GA and the
ES pay the same rollout bill. The `secs` column of the four session
checkpoints reads 70.0 for every generation after the first because the
window only collects a generation when the replay of the previous one has
played out at 1x; that column is wall-clock between `start_evaluation` and
`poll`, not compute. `README.md` now says 14.4 s.

**The window path, offscreen.** `--swarm 8 --swarm-T 20 --render offscreen`
scores one generation (1.7 s), replays it, saves state and the best, and
exits — the same code the user's window runs.

## What was tried and rejected

* **One population-wide sigma.** Collapsed on the anchor's line (above).
* **Nesting the seed's whole `meta` in the lineage record.** A checkpoint
  seeded from a checkpoint carried the parent's history *and* the parent's
  seed report, recursively: 10.1 -> 13.1 -> 16.5 -> 19.2 kB down one lineage
  of four. `seed_from` now keeps the parent's meta minus `history`,
  `seed_report` and `curve`, plus its `seed_source` string, so the chain is
  still readable and the file stops growing.
* **`--ml-drive` printing a swarm checkpoint's `meta`.** Forty lines of
  generation history on the console; it now drops `history` as it already
  dropped the ES's `curve`.
* **A `latest` branch in `_swarm_seed_spec`.** Dead: `run_swarm_cli` resolves
  `latest` (newest seed lap *on this map*, by mtime) and `best` (newest
  `swarm_*.json`) before calling it. Removed.

## Not done

* **The one real seed lap is faster than the swarm.** The user's 59.70 s
  against the best genome's 62.2 s. A swarm bred from that lap ran one
  generation here (T = 40 s is shorter than the lap, so none lapped; reward
  713 vs 692 from the anchor at the same T) and no longer. Nobody has yet run
  `--swarm 32 --swarm-seed latest --swarm-gens 20` and reported whether the
  clone's head start survives the GA or gets bred back to the anchor's line.
  That is the experiment the feature exists for.
* **No swarm has been bred with a wing.** All four session checkpoints and
  the deliverable are wing off; `wing_frac` is 0.0 in every stored
  measurement. The free-wings head has therefore never been *selected* on,
  only shown to be harmless.
* **The `secs` history column** is wall-clock including replay time in the
  window (above). It is honest headless. Not changed, because the window's
  pacing is the feature.
* **A 4-output `--legacy-wings` swarm cannot be seeded from a free-wings
  checkpoint** (it raises, by design: there is no honest way to narrow three
  wing rows to one).
* The CONTRACT's `--build` / `--ml-drive` paragraph order around section 8
  was scrambled before this task (HEAD has it) and is the garage's to fix.

## Housekeeping

The four session checkpoints (`swarm_arena_corsa_20260921_{165259,180645,
181831,184102}.json`) were bred by the window with auto-generated names,
before the name prompt existed. `181831` is the deliverable, renamed to
`swarm_arena.json` with `meta['name'] = 'arena'` (the name the README's
`--swarm-seed` example uses); the other three, and the `swarm_t17win.json`
the offscreen smoke test wrote, are in the session scratchpad
(`scratchpad/swarm_checkpoints/`), not deleted. `runs/swarm/` (state files,
seed laps) is gitignored with the rest of `runs/`.
