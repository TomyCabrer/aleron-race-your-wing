# Task 45 — a player tries the game, and every detail gets an agent

Owner (2026-09-26): *"I want you to act as a player that is trying the game.
For every aspector detail of the game you find something that needs change or
could be improved deploy an agent to do it"*.

Built in a separate worktree (`~/Desktop/carsim-t45`, detached on the snapshot
`7eb392a` = main's working tree with tasks 42 + 44 uncommitted). Nothing is
committed; main and `carsim-aerobo` were never touched.

## How

* **Round 1, audit** (read-only): five player lenses, each playing offscreen
  through the real launch path (`drive.main()`, a fresh `runs/` and a copy of
  the owner's save, every screenshot read back) and reading the code:
  1. first launch and menus (title, WELCOME, the tutorial, the pause pages)
  2. driving and the HUD
  3. the time-trial loop
  4. challenges
  5. the garage and wings

  69 findings. One **judge** merged them into **31 fixes** in 5 file-disjoint
  lanes and wrote 5 questions only the owner can answer.
* **Round 2, fix**: **31 fixers** in the **5 lanes** (drive.py and its menus;
  the HUD drawing; the garage; challenge texts; shared menu layout and copy),
  each with its modules' self-checks and a play run. One **integrator**
  merged the lanes, fixed five seams (LAST void for a lap the recorder
  dropped, the list's star count, ...) and listed what was left.
* **Round 3**: the owner's four answers plus round 2's leftovers, in two lanes
  (a: `powertrain.py` + `vehicle.py`, the gearbox; b: everything else, five
  fixers one after the other), then this integrator: merge, the pinned
  numbers, `challenges --measure --write`, an adversarial review of the round-3
  diff, play runs, docs, the suite.
* **After the review**: the round-3 review found the arena's rolling start
  set the car down 16 m before the R = 35 m T6 hairpin, so holding UP was the
  gravel in 2 s. A fixer rewrote `Sim._rolling_pose` (row 12). Last, one
  fixer took the three small leftovers: the pose's docs, the pause page's
  challenge row (row 13) and the garage menu at other window sizes (row 14).

## The owner's four answers (round 3)

1. **The automatic gearbox: fix both** (a physics change; re-baseline and
   rebuild the references). Done in `powertrain.py`, gated on the automatic:
   the full-throttle upshift line is capped a quarter of the soft band under
   where the fuel starts to fade (`N_UP_SOFT_MARGIN`; Corsa 6150 -> **6050**,
   bus 2480 -> 2439.5 rpm), the 0.8 s lockout still lets it change up from the
   fade band at full pedal, and at the end of a stop it opens the clutch and
   idles **in gear** instead of lugging to a stall (a brake hold, a locked-wheel
   release for ABS-off stops, and the engage ramp never closing the clutch
   further than the anti-stall assist). Manual and clutch modes unchanged.
2. **Challenges keep the player's setup**, and say what rules out stars: a
   YOUR SETUP section on every challenge page (yours / stars set / an amber
   `!` row per difference with its cost), the RULES row a build breaks in
   amber, and a result short of a star names the differences.
3. **A precision stop board** for the 3rd star of the three stops: an amber
   brake marker 3 s of the speed past the start, a checker board the 3-star
   distance past it; the 3rd star is the nose within 1 m of the board (short
   or past) plus the build rule. No physics change; refs.json is not moved by
   it (the references still brake at once; V35 drives each to its board).
4. **Ready-made-wing hint + a bigger tutorial wing**: `Try ready-made wings`
   on the TIME TRIAL page of a car with no wings (the garage opens with its
   `W` pressed on the first empty slot), F's no-wings note points at `W`, and
   the tutorial's wing page now shows the wing's own push on lap 2 (see
   *Decisions* for the wing itself).

## What a player notices now

| # | change | where |
|---|---|---|
| 1 | **Every timed lap starts rolling**, on every closed map (open map and skidpad too), lap and circle challenges, the tutorial's laps; a time trial opens rolling whether or not the TIME TRIAL page opens (round 3); `R` on the out-lap or in the first sector is a rolling start with a fresh clock, past a split it is that sector line; every R / SHIFT+R says what it did | `drive.py` `_rolling_pose`, `reset`, `roll_time_trial` |
| 2 | **The automatic changes up under the limiter** and **never stalls at a stop**: Corsa 0-100 14.80 -> **14.31 s**, 0 s on the limiter in any corner case the finders had (the Express at 12 deg of steer still spins its inside wheel, see *Not done*); the bus's wet stop ends `1 AUTO 550 rpm`, not `STALL` | `powertrain.py` |
| 3 | **Stop board**: amber marker, checker board, cones and boards in chase; the box's amber row `board: 14.2 m ahead` -> `board: 0.3 m short [***]` | `challenges.py`, `render.py`, `world.py`, `scenery.py` |
| 4 | **YOUR SETUP** on a challenge page and the setup named in a result; a 3rd-star rule the build breaks is amber; the WINGS line uses the garage's names (`Rear wing (stock)`, `Side plate (stock)`) | `challenges.py`, `menu.py` (`Warn`, `note_under`) |
| 5 | **Try ready-made wings** on the TIME TRIAL page; F with no wings: `no wings fitted - BACKSPACE, then W: try a ready-made wing`; the tutorial's wing page ends with `wing's push +0.010 g (95 N) toward the inside, lap 2` and the live line shows the push | `prerace.py`, `garage.py`, `drive.py`, `tutorial.py` |
| 6 | **BEST never takes a lap that did not count** (T, a setting change, slow motion, not a full lap); the live delta level with the PB reads `0.00`, neutral; a stall on the automatic says `engine stalled: S to restart` | `drive.py` `LapTimer`, `render.py` |
| 7 | **Menus**: Main menu on the pause page and in the garage (back to the title), Quit to desktop asks twice; a short challenge pause page; TAB on the pause and TIME TRIAL pages; the challenge list has Car / Wings at the top; every challenge page opens on Start; pause footer `R sector line   SHIFT+R restart lap` | `drive.py`, `input.py` |
| 8 | **HUD**: H goes race HUD -> full -> off; force arrows off by default (`V`, saved) and a HUD row in Settings; `SHIFT+G` steps the wing mode back; every key's note is a toast above a bottom bar that only carries flags; `LAP 1` / `LAST void`; km/h beside the number; a real PAUSED card; `GOLD in 0.759 s` on the results card; map names on the minimap | `render.py`, `drive.py`, `results.py` |
| 9 | **Garage**: R twice (U undoes) and DEL twice; named ready-made wings (`Side fin (1 of 4) - ...`); a BUILD line; a car panel with corner speed / drag / weight; hints no longer over the key bar; the ESC menu fits 1280x800 with a pad (17 + 8 help rows) | `garage.py`, `garage_ui.py`, `wing_tutorial.py` |
| 10 | **Challenges from before task 44 count again** (as the Corsa / FULL WING pick); the box's green line is the result and its warn row what it lacks | `challenges.py` |
| 11 | **Tutorial**: the circle's roll-in is gentle (0.3 g, 44 km/h) and not judged before the line; continued in a new session a step starts from its own start; step 7's text says to skip F if the wing is already on | `tutorial.py`, `drive.py` |
| 12 | **The rolling start never sets the car down before a corner** (found by the round-3 review: the arena's pose was 16 m before the R = 35 m T6 hairpin, holding UP was the gravel in 2 s; a fixer rewrote the pose). Rule, on a straight inside the last sector: (1) the straight into the line where it is 45 m or more (`ROLL_RUNIN_MIN_M`), from its start, at most 250 m out (`ROLL_RUNIN_MAX_M`), 22 m/s; (2) else the nearest straight back from the line with 3 s at V0 before its corner (`ROLL_CLEAR_S`); (3) else the nearest straight of 15 m or more, V0 cut to its length / 3 s; (4) no straight: the circle start. Poses (V45 pins them): arena 1106.2 m (143 m to the line, 36 km/h out of T6, case 3), linden 1060.4, kestrel 1843.3, ashdown 1330.5 (case 1), open 1492.7 (case 2), skidpad 267.0 (case 4) | `drive.py` `_rolling_pose`, `ROLL_*` |
| 13 | **The pause page's challenge row counts the pick**: `Challenges: 3 of 24 stars (this car + wings)`, the car + wings the list opens on, not `1 of 456` over every car x config; the list's subtitle and the row share `challenges.pick_stars` | `challenges.py`, `drive.py` |
| 14 | **The garage's ESC menu fits at 1280x720, 1440x900 and 1600x900 too** with the wing tutorial running and a 10-character build name: the default row is `Set as Express default  (now: my express)` (was `Make it the Express default ...`, the widest row: the help beside it wrapped past the footer) | `garage.py` |
| 15 | **The wing tutorial continues where the row says, and a first launch's build is named for its car** (two garage leftovers a fixer saw while playing). `continue at step N` opens step N's page, and the box and the hint both say step N; it used to show step 1 (`the design page started again`) under a `Step 8 of 10` hint. A step whose design-page work went with an earlier garage (a new launch: steps 3-8; step 9 with its wing gone) goes on from step 2 on the mission page, and the box and the hint say `continuing from step 2: the design page needed a fresh start`. In the garage that did the work, it goes back to the design page as it was left. Step 9 now passes in a new garage (the progress file keeps step 8's wing). The driving tutorial's pointer continues the same way. Name: a fresh save switched to the Express in Settings opened the garage on `my corsa` (a real player can reach this; it is not only the hand edit). The garage now says `my express` in the BUILD line, the save row and S's prompt (`garage.other_car_name`). The drive calls it the same by the same rule (`drive.py` `_drive_name`, in `_drive_design`): the TIME TRIAL page, the pause and Settings subtitles, the Settings Build row, the title's BUILD, the lap label, a PICK's autosave (`my express (autosave)`) and the per-map memory's name all say `my express` on the Express and `my corsa` back on the Corsa. Only the session's copy is renamed: the working build, `runs/garage_design.json` and the library keep `my corsa` (the memory entry keeps the whole build under its own name; it never points into the library, so `my express` being in no library file is safe). A name the player chose, and the library's own `my corsa`, are kept, as in the garage. New rows: `wing_tutorial` (continue at each step in the working garage and in a new one, box step = hint step = page, the note fits the box; step 9 in a new garage; the pointer), `garage` (the name) and `drive` V45g (the drive's name, the memory, the title, a PICK's autosave, a challenge copy; saved files untouched) | `wing_tutorial.py`, `garage.py`, `drive.py` |

## Self-checks changed

* **Moved only by the owner-approved gearbox change** (each commented "task
  45" where it lives): `drive.py` `accel run` band 14.5 -> **14.0** s lower
  bound (14.802 -> 14.312 s); `powertrain` `auto_upshift_needs_the_road` (text
  computed: 6051 rpm, line 12.84 m/s), `t41_why_n_soft_scales` rewritten (a
  120 rpm band no longer strands the bus in 2nd), ACCEL_0_100 rig 14.69 ->
  14.70 s, `auto_creeps_from_rest` 14.5 -> 12.6 m, bus 0-50 10.5 -> 10.4 s.
  Values that moved but stay in band: V27 auto 20.0 -> 20.2 m/s, V29 stock
  0-100 14.93 -> 14.44 s, V31 laps 60.73 -> 61.00 s, validate's scripted arena
  lap 60.84 -> 61.04 s.
* **refs.json re-measured** (`python3 -m drive.challenges --measure --write`,
  152 ok, 8 unavailable, 131 s): only gearbox-reachable values moved -- stops
  -0.04 .. +0.34 m (the clutch now opens at walking pace, so the driven wheels
  lose the engine's push and lock a little earlier; Corsa `airbrake_150|full`
  84.53 -> 84.87 m), laps -0.43 .. +0.19 s (the box changes up on corner
  exits; Express wet laps -0.36 .. -0.43 s, MX-5 open lap -0.21 s), skidpad
  <= +0.008 g (the Express no longer held in 2nd). No other field changed.
* **V20** (determinism): sha **unchanged**, `ed41b7f781e959ea`: its 6 s on the
  skidpad in 3rd never reach an upshift line, the limiter or a stop.
* New rows: `powertrain` t45 x2 (99/99), `vehicle` T45a (full throttle in a
  corner, every car) and T45b (a stop from 80, dry / wet / ABS off, every car:
  never stalls, pulls away) (38/38); `drive` V25 / V31 (BEST never a voided
  lap), V32 (Try ready-made wings), V34 (the push), V35 (each stop reference
  braked for its board; 2 m late is 2 stars), V45 (a time trial opens
  rolling; the tutorial circle at 0.3 g); `challenges` (board rules, marks on
  all 52 stop combos, YOUR SETUP, the result's setup words), `render` (stop
  board painted plan and chase, `0.00` delta, the stall wording), `menu`
  (Warn, note_under), `garage` (menu fits at 1280x800 in 5 cases,
  try_ready_made), `prerace`, `tutorial`, `input`.
* After the review: `drive` V45 pins each map's rolling-start pose (row 12);
  `garage` lays the menu out at 1280x720, 1440x900 and 1600x900 too (the wing
  tutorial on, 'my express' on the Express, keyboard and keyboard + pad) and
  every size's row now asserts no help line wrapped; `challenges`' pause-row
  check reads the pick (`Challenges: 0 / 3 / 2 of 24 stars (this car +
  wings)`), and `total_stars` keeps its own row; V45e checks the pause row
  on the Corsa / FULL WING and again after the list's pick moved to the Citaro.

## Decisions made without asking

* **End of a stop on the automatic**: the car stays **in gear with the clutch
  open** at its own idle (Corsa 758 rpm, bus 554), not in N, so pulling away
  needs no 0.7 s gear change. Released, it creeps.
* **The tutorial still lends the Side plate.** The owner's option named
  `flank-e423` as "the bigger wing", but measured the plate is the bigger one
  (car page: +2.2 % corner speed vs +0.9 %; skidpad limit 71.72 vs 71.41
  km/h; CL x S 0.44 vs 0.18 m^2). No ready-made side wing makes two laps by
  hand differ visibly at ~72 km/h, so the lesson's payoff is now the wing's
  own measured push on lap 2 (~95 N, 0.010 g), positive whatever the laps
  did. One line (`tutorial.TUTORIAL_WING`) if the owner still wants e423.
* **Board place**: marker at 3 s of v0, board = marker + the 3-star distance
  (1.02 x ref), so a reference-quality stop braked AT the marker ends 0.02 x
  ref short: inside 1 m on Stop from 100 and every wet stop but the bus's
  (1.04 m), outside on Air brake from 150 (1.7-1.9 m) -- there the player
  brakes 1-2 m after the marker. The window is 2 m: about 4 frames at 100 km/h, 3 at 150.
* **Setup marks**: TC is not flagged on a stop (no throttle there); ballast
  counts as setup; ABS / gearbox are read when the result comes in.
* **Old stars**: three stars saved on a stop before the board stay three;
  pre-task-44 stars count as the Corsa / FULL WING combo's.
* **A time trial opens rolling** on every session start (one rule), and the
  RACE on the page rolls again; the Swarm page's seed lap still stands.
* **Garage menu**: names cut to 10 characters in the menu rows only; help
  rows merged (C with the camera, the stick pairs), none over 32 characters.
  After the review: the default row lost 5 characters (`Set as Express
  default`) rather than the help a font step (`menu.py` is shared with every
  drive page); 1280x720 keeps 3 characters to spare, the tightest size.
* **The pause row's pick** is the one the list opens on (`Sim._ch_combo`):
  the last pick browsed there, else the car being driven with FULL WING --
  so after browsing the Citaro, the row counts the Citaro's, as the list does.
* **A voided lap gives nothing**: no BEST and no sector bests, including ones
  it set before it was voided; out-lap sectors in a recorded session no
  longer count.

## Not done

* **The dragstrip timing slip** (the fifth owner question: 0-100, 201 / 402 /
  1000 m with trap speeds and saved bests): proposed, not answered, not built.
* **A 3-2-1 countdown** before a rolling start: not built.
* **Front-drive wheelspin in a tight corner**: at 8-12 deg of steer the
  Express's unloaded inside wheel spins at twice road speed and TC lets it,
  so the engine sits in the fade band in 2nd (4.7 s at 12 deg). A gearbox
  change cannot fix it (the box rightly refuses an upshift road speed
  disagrees with); it needs a brake-based diff lock or a TC change.
* Round 3's three leftovers (the garage menu wrapping at 1280x720 / 1440x900
  / 1600x900, the pause row's `1 of 456 stars`, the arena's rolling start 16 m
  before its hairpin) were done after the review: rows 12-14.

## Medals

**Rebuilt** after the gearbox change (the lead, 2026-09-26, in a separate
worktree at the round-3 snapshot, so nothing edited the tree under it):

```
python3 -m drive.medals --build --workers 8  ->  270 classes with medals (315 in the table, 45 'no lap
                                                 on this map', as before), 2586 runs, 5136 s
```

Nothing would have flagged it: `medals.inputs_hash` covers tracks, cars,
engines and the reference drivers but, by design, not the vehicle /
powertrain physics, so `is_stale()` stayed False on the old table.

Author times against the old table: 175 of 270 within 0.05 s, median 0.000 s,
14 faster by more than 0.3 s, 2 slower. LapDriver's own laps barely moved; the
two big moves are the trained bots:

* `arena|mx5|stock|none` -1.97 s: `mx5_arena_plate` now finishes a lap (55.008)
  where it used to crash at 111 s.
* `arena|540i|tuned|patch` +4.00 s: `540i_arena_plate` now crashes at 14 s
  (was 56.148), so LapDriver's 1:00.149 is the author time.

**The trained bots are brittle to the physics change.** Most bot laps got
faster (the box changes up out of corners), but four runs that used to finish
now don't: `arena_plate` (the Corsa's) spins in `arena|corsa|stock|none`
aids off and `arena|corsa|tuned|*` aids on, and `540i_arena_plate` in
`arena|540i|tuned|patch` aids on. The same bots are race-vs-bot rivals, so on
those classes a rival may spin. Retraining them (a swarm run per car) is left
to the owner; nothing else depends on it.

## Landing on main

Ported onto main's working tree on 2026-09-26 at about 23:10, uncommitted,
on top of tasks 42 and 44. Nothing is committed.

Another session (carsim-44) had edited main at 22:47 for two owner asks:
"There should be an option to go back to the title page" and "In the title
screen they should be deployed with all the wings". Their files were
`drive/drive.py`, `drive/garage.py` and `drive/title.py`. Those three were
3-way merged, base `7eb392a`; every other file is task 45's own.

Agreed with that session:

* **One way back to the title: task 45's.** The pause page and the garage
  menu each have a `Main menu` row, and `run_interactive_cli`'s
  `title(again=True)` shows the title again. Their `Title screen` rows and
  their `show_title` loop were dropped.
* **Kept from theirs:**
  * the title cars' full wing set, all out (`title.WINGS_CONFIG`,
    `title.wing_hud`, their self-check row 8);
  * `_title_screen(lib=...)`;
  * `Sim.has_title`, which gates the pause page's `Main menu` row, so there is
    none on a `--garage` / `--race` / offscreen launch;
  * their V44 back-to-title runs, on task 45's harness (`garage_acts=`).

V45c and V45d set `has_title` on their stub Sims.

V30's `car cycle` once failed wherever the bot checkpoints shared one
mtime (a fresh copy, the one-version merge, and any install).
`race_bot_choices()` and `race_newest_bot()` broke the tie differently,
so bot 1 disagreed with the list. Both now sort by `_bot_age_key` =
(mtime, name) (`drive.py`, 2026-09-27), and the self-check passes on the
merged main.

