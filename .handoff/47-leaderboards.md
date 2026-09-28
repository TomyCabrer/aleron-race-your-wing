# Task 47 — leaderboards: one per map, car and wing mode, you against your bots

Owner (2026-09-28), verbatim:

> *"Make a clear way for the leaderboard to be seen should be done for the play
> and for the bot. 1 leaderboard for every track and car type. Best time and
> name of car should appear. Also compared to the bot. Shoud be for default
> car. Also for race there should be the same modes as challange for the wing
> types so separate leaderboards. Non stock engines are only for messing
> around and don't go to leaderboards (this should be mentioned)"*

Asked once, answered:

| question | owner's answer |
|---|---|
| which bot is on the bot side | **"My trained bots only"** -- a checkpoint, never the built-in driver; a board stays empty until one of your bots laps there |
| surface | **default surface only** (*Dry, wet patches*) |
| a timed lap on the build as designed (no wing mode) | **pick a mode to race**: time trial / race use one of the challenges' four wing modes; FREE is still driven but never on a board |

Then: *"I am going to sleep don't ask me anything and finish all the tasks"* --
so every other choice below is the lead's, and says so.

## What was built

* **`drive/leaderboard.py`** (new; `validate.MODULES`, CONTRACT §1). A board is
  `track|car|wings`: `records.LAP_TRACKS` (4 circuits, open, the 50 m skidpad)
  x `cars.CAR_ORDER` (5) x the challenges' `CONFIGS` (4) = **120 boards**,
  `runs/leaderboard_local/<track>__<car>__<wings>.json` (kind
  `carsim-leaderboard-1`, atomic, merged with the disk on save, a corrupt
  file moved aside). Each board: **you** = the best lap of each build, by
  name ("name of car"), with assists and date; **bots** = the best of each
  trained bot, with the build it drove and how (race / test). Top 10 each.
  `why_not` is the one rule: Stock engine, surface `patch`, a wing mode (not
  FREE), wings within limits, a lap the records filed as valid.
* **The Wings setting** (`Settings.race_wings`: FULL / ONLY TOP / ONLY TOP
  FIXED / TOP FIXED + SIDE / FREE; default **FREE**). A session in a mode
  drives `challenges.config_build` of the build (exactly what a challenge
  drives: own wings where it has them, stock ones lent, sides off on a
  top-only mode, top fixed on a FIXED one), and G is limited as in a
  challenge (`g_modes`). It is on the TIME TRIAL page (row *Wings*, LEFT /
  RIGHT restarts on its page) and on the Settings page (a RESTART_KEY).
  Tutorial sessions and challenges ignore it.
* **Where you see it**
  * title screen: **Leaderboards** (3rd row) -> the LEADERBOARDS page;
  * pause menu: *Leaderboards: every map and car, you vs your bots*;
  * **LEADERBOARDS page**: *Map* and *Wings* rows (LEFT / RIGHT), then one row
    per car: `Corsa   you 1:01.250 low drag      bot   59.900 swarm_v46    bot
    ahead by 1.350 s`; the highlighted car's board in full in the help column
    (you top 5, your bots top 5, the gap); the rules otherwise; the note under
    it is the engine sentence. ENTER on a car races that board (its map, car,
    wing mode, Stock, default surface -> a new session on its TIME TRIAL page);
  * **TIME TRIAL page**: a LEADERBOARD section under CLASS (this board: your
    best, your best bot's, the gap; or why this session is not on one).
* **"This should be mentioned"**: `leaderboard.ENGINE_NOTE` on the page; the
  Settings page's Engine row says `(just for fun: not on leaderboards)` on
  Tuned / Sport, its help has a *Leaderboards* line; the TIME TRIAL page's
  section says *Sport engine: just for messing around, not on the
  leaderboards (Stock engine only)*; a session that starts on Tuned / Sport,
  and a live change to one, put it on the HUD's bottom line.
* **Bots**: a race puts every bot in the session's wing mode:
  `race_grid.own_car(config=)` for a bot's own bred build, `Sim._in_wing_mode`
  for a saved build picked by name and for a stock car (the mode's stock
  wings and their mass, a challenge's reference car); 'same' is the session's
  own copy. A bot car past its car's span limits files nothing. `Rival.on_lap` hands on a
  lap that is timer-valid, went round (95 %) and had no respawn; the RACE
  page's *Test* files each car's best flying lap on that car's board. Never
  the built-in driver, never a bot on a Tuned / Sport engine, never a bot
  whose own build is past its limits or cannot take the mode.

## Lead's choices the owner may overrule

* **Default engine is now Stock** (`ENGINE_DEFAULT`, was Sport). The owner's
  words ("for default car", non-stock "only for messing around") and the
  boards only counting Stock. A saved `runs/settings.json` keeps its engine:
  the owner's own file is on Sport, so the owner's next drive is on Sport and
  the game will say it is not on the boards. One constant to revert.
* **Wings default FREE**, not FULL: task 45's owner call was "no default
  wings", and FULL lends stock wings to a build with none. A player picks a
  mode on the TIME TRIAL page to race for a board.
* A board keeps the best lap **per build name** (so it reads as a table of
  cars), and per bot name.
* Existing records (all driven FREE) are not copied onto any board.

## Review

A review workflow (3 lenses: the feature end to end, the BMW / per-car
settings, regressions; each finding checked by a skeptic) confirmed 13
findings; all fixed: a race kept across a restart was rebuilt before the
session's wing mode was set (bots raced FREE, no bot lap filed); stock-car and
saved-build bots now race in the mode (`Sim._in_wing_mode`: the mode's stock
wings / the build in the mode, their mass) and file only within that car's
span limits; a race started after T files nothing; a board picked during a
tutorial is refused with a note; the Engine row during a Car-row preview wrote
the browsed car; the per-car TC (task 48); the challenge restore takes the
car's own engine / TC; the start notes are kept beside the screen notes; the
HUD's WINGS chip shows a top-only mode; docs.

## Gates (main's tree, 2026-09-28 ~06:00, with task 46's lanes C and D merged)

```
python3 -m drive.leaderboard / prerace / race_grid / input / title / records / cars  ->  PASS each
python3 -m drive.validate             ->  82/82 pass  0 HARD  0 soft  [128.7 s]
python3 -m drive.drive --self-check   ->  all pass but V35 skid_dry|540i|top_fixed_side
                                          (task 48 moved the 540i's TC-on reference: refs.json
                                          is re-measured by task 46's post-merge run)
                                          V47 leaderboards ok, V48 the BMW ok
python3 -m drive.ml                   ->  35/36: 'the anchor laps the arena in every car' fails on
                                          the 540i (1054 m, offtrack) -- PRE-EXISTING: the same at
                                          pristine HEAD 4c0e453 (task 45's gearbox leftover)
real launch (SDL dummy, main([]))     ->  title > Leaderboards > ENTER on the Corsa > its TIME
                                          TRIAL page (Wings FULL, LEADERBOARD section) > RIGHT on
                                          Wings > ONLY TOP session (flanks off, G locked, note);
                                          and a copy of the owner's settings.json: the 540i opens
                                          on Stock with TC on, power 1.0, the note on screen
python3 -m drive.medals               ->  STALE (soft) until the full rebuild task 46 runs
```

Re-run after task 46 merged lane B (the Fairfield oval: 7 maps, 140 boards)
and lane A (CAR_ORDER corsa / rally / 540i / express): V47 and V48 ok;
leaderboard, prerace, input PASS; both real-launch tests pass. Task 46 then
runs the full `challenges --measure --write`, `validate --modules`, `drive
--self-check` and `medals --build`; its own race_grid / title rows were still
red at that point (not these tasks').

**Final, on main with tasks 46 + 47 + 48 together (reported by task 46's
session, V47 / V48 re-confirmed here):** validate 82/82, validate --modules
134/134, drive --self-check ALL PASS, drive.ml 36/37 (only the pre-existing
540i arena anchor row), medals current (medals.json, reference_laps.json and
refs.json rebuilt). Nothing committed.
