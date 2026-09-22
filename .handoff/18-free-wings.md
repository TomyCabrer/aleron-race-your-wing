# Task 18 — the bot's three wings, deployed however it wants

The brief: the bot has all three wings (left flank, right flank, top) and
decides for itself how to deploy them — all three to stop, two for a corner,
none on a straight. Task 17 had already built the five-output head, the
`Controls.wing_cmd` physics path and the per-flank deploy states, and the
README promised exactly this. It did not do it.

```
python3 -m drive.ml                  ->  ALL PASS  32/32   (28 through task 17; 4 new)
python3 -m drive.ml.swarm --self-check  ->  17/17
python3 -m drive.ml.clone            ->  both heads recover the trim
python3 -m drive.drive --self-check  ->  ALL PASS (V30 race included)
```

## The finding

`policy.action` composed every output as `clip(base + RESID_GAIN * net)`
with `RESID_GAIN = 0.55`, and the anchor's wing verdict (`baseline_action`,
and `expand_base` spreading it over three wings) was **±1**. So a wing entry
sat in `[-1, -0.45]` or `[0.45, 1]` whatever the network said, and
`WING_ON_THRESH = 0` was never crossed. Measured before the change, arena,
plate, 60 s: a net forcing **+1** or **−1** on every wing output gave the
same `wing_frac` (0.514 four-output, 0.596 free head) and the same reward
(1088.2) as `theta = 0`. On both heads. Every wing decision in every
checkpoint in `drive/ml/checkpoints` — the ES ones and the three swarm
ones — was the hand-written rule's, and the README's "it learned when not to
carry the drag" was the rule reacting to a faster line. The clone had the
same wall: its wing target `(±0.8 − ±1) / 0.55` clipped to ±1 and still
could not move the composed entry across zero.

## The change

**Priors, not orders** (`drive/ml/policy.py`). With the free head the
anchor's three verdicts are `±WING_PRIOR` (0.5) and the wing outputs are
composed at `WING_GAIN` (1.0), full authority, via a per-output gain vector
`_GAIN_FREE = (0.55, 0.55, 1.0, 1.0, 1.0)`. `theta = 0` still drives the
rule (the prior's sign is the verdict); an output past ∓0.5 overrules it.
The **4-output head is untouched**: its wing output is still the rule's, so
every committed ES checkpoint's numbers are unmoved — checked to the bit
(`zeros` 1088.205262477883, `arena_plate` 1129.3261940906082, before and
after), and the new self-check asserts it on every run.

**`Policy.widen` zeroes the wing rows.** It used to copy the 4-output
genome's wing row into all three. That row never had authority, so it is
never-selected noise; handing it real authority would have made a widened
checkpoint deploy at random. Zero rows plus the priors reproduce what that
genome actually did. The swarm and package self-checks assert this on the
composed `wing_on`, not on the raw row.

**What the car did with each wing is now reported.** `Episode` carries
`wing_l_frac / wing_r_frac / wing_top_frac / wing_both_frac` (the last is
the air brake), `lap_time` returns them, `Swarm._score` stores them per
individual (`wings`), the headless log and the window's *all-time best* line
print `wings L R top both`, and `evaluate.compare` prints a line for a
free-wings checkpoint. The rollout trace and the seed-lap ghost carry
`top_deploy` as an 8th column (`_Replay` pads a 7-column state file), so the
hero car in the swarm replay shows its top wing.

**The seed lap records each wing.** `SEED_LAP_COLS` / `SEED_COLS` gained
`wing_deploy_l, wing_deploy_r, top_deploy` (appended; read by name, so older
laps load and the clone infers the side from the yaw rate as before). The
clone's free-head targets come from the user's own per-wing use — the `G`
key's left / right / both and the top wing — and divide by the wing gain.

**`train.py --free-wings`** breeds the 5-output head with the ES too; a
4-output `--init` is widened. Default off.

## Measured

Free head, arena, plate, 60 s at 2 ms, net forced on every wing output:

```
forced -1   wing_frac 0.000                       reward 1088.2
theta = 0   wing_frac 0.596  (the rule)           reward 1088.3
forced +1   wing_frac 0.999  both flanks 99.9 %   reward 1085.6   (the air brake all lap: slower, as it should be)
```

The three swarm checkpoints were bred **wing off**, so their wing rows are
unselected. With a plate fitted they now do what those rows say instead of
the rule: `swarm_arena.json` 1163.1 → 1147.9 (60 s reward; deploys the left
panel 16 % and nothing else), `swarm_arena_corsa_20260921_204340.json`
1139.6 → 1139.5, `swarm_bot_1.json` spun before and spins after (630.5 →
630.6). With the wing off, as bred, none moves.

The user's garage build (`runs/garage_design.json`, "my corsa": flank-new on
both flanks, rear-new on top, all `active`) is what the Rival and the
window's swarm drive. `theta = 0` with the free head on that car: 1294.8 m
in 70 s, 1 lap, L 20 % R 41 % top 60 %, both 0 % — the rule. A swarm bred on
it (pop 16, T 70 s, seed 5, 8 generations, headless, 8 s a generation on
twelve cores), the best car of each generation and what it did with its
wings:

```
  gen 0 best 1324.7 mean 1236.0 lap 67.158 lapped 14/16 wings L 20% R 39% top 63% both 0%  [9.2 s]
  gen 1 best 1342.6 mean 1290.7 lap 66.53 lapped 14/16 wings L 20% R 38% top 65% both 0%  [8.5 s]
  gen 2 best 1345.2 mean 1266.8 lap 66.438 lapped 15/16 wings L 32% R 40% top 84% both 13%  [8.4 s]
  gen 3 best 1374.1 mean 1316.4 lap 65.392 lapped 15/16 wings L 20% R 28% top 66% both 10%  [8.4 s]
  gen 4 best 1377.6 mean 1316.1 lap 65.18 lapped 16/16 wings L 17% R 92% top 57% both 12%  [8.0 s]
  gen 5 best 1377.9 mean 1285.9 lap 65.18 lapped 15/16 wings L 47% R 53% top 57% both 36%  [8.1 s]
  gen 6 best 1405.1 mean 1351.5 lap 64.086 lapped 16/16 wings L 13% R 97% top 65% both 11%  [8.0 s]
  gen 7 best 1409.4 mean 1351.7 lap 63.908 lapped 16/16 wings L 2% R 100% top 92% both 2%  [8.0 s]
```

Eight generations, −3.25 s a lap (67.16 → 63.91 s at 2 ms, not re-measured
at 1 ms), and the wing pattern is the swarm's own, not the rule's: the
right flank panel out for the whole lap (the arena turns left far more than
right, so the right panel is the outer one most of the time), the top wing
up 92 % of the time, the left panel and the air brake nearly dropped. The
rule at generation 0 ran L 20 / R 39 / top 63 / both 0. Whether that is the
fast answer on this car or a local optimum of a 67 s search is not settled
here; what is settled is that the wings are now being selected on at all.
Task 17's checkpoints show `wing_frac` 0.0 in every stored measurement and
no earlier swarm could have moved these numbers.

## Not done

* The observation is unchanged (17 entries, `wing_dep` = the larger flank
  deploy). The bot commands three wings but sees only one deploy fraction;
  per-wing observations would invalidate every checkpoint (`Policy.load`
  refuses), so that is a decision for a wave of its own.
* `--race anchor` is still the 4-output `Policy()`: one wing toggle, the
  car picks the flank. It is the built-in driver; the free head at
  `theta = 0` drives the same rule through `expand_base`.
* The headless `drive.ml.swarm` CLI has no `--build`; a three-wing car
  reaches the swarm from the window (`ESC` → *Deploy swarm*) or from
  `--swarm` on the command line, both through `_session_car`.
* Another session was editing `drive/drive.py` (multi-rival racing) while
  this ran; this task's five touches there (seed columns, `_seed_row`,
  `_Replay`, the hero HUD, the overlay line) are small and were re-verified
  after its writes.
