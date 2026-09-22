# Plan — Steam engagement features (tasks 19–29)

Scope agreed with the owner on 2026-09-23. Roblox is parked; this is the
Steam build only. This file is the brief for tasks 19–29. Each task still gets
its own `.handoff/<NN>-<slug>.md` note when it is done.

## Why

The sim already has the "tweak" (garage, wings, ballast, engine) and the
"drive". What makes a game of it is the loop *tweak -> drive -> get a number ->
beat it -> tweak*, and two things are missing from that loop: **goals**
(medals, challenges) and **someone to beat** (your own ghosts, bots, other
players). Onboarding matters more than any single feature: a new player who
spins in the first corner and cannot find what the wing does refunds inside
Steam's 2-hour window.

## Decisions (settled — do not re-open; owner may edit this list)

| id | decision |
|---|---|
| D1 | **Class key** = `track|car|engine|surface`, e.g. `arena|corsa|sport|patch` (`trk.TRACK_ORDER`, `cars.CAR_ORDER`, `ENGINE_MODES`, `SURFACE_MODES`). Records, medals and boards are per class. The build (wings, ballast) is NOT in the key: designing the car is the game. Assists (ABS, TC, steer aid, gearbox) are stored with each lap and shown as icons, never split into classes. |
| D2 | **Two ghosts**: ghost 1 = your PB for the class. Ghost 2 = a selectable slot: none / reference bot / any of your top-5 laps. Default: reference bot. |
| D3 | **Race grid "characters"**: each bot slot = name + colour + checkpoint + saved build. |
| D4 | **Medal times are derived, never hand-typed**: author = reference-driver lap (headless); gold / silver / bronze = author x 1.02 / 1.06 / 1.12 (constants in `medals.py`). A class with no reference lap has no medals ("—"), not an invented time. |
| D5 | **Online = Steam leaderboards, scores only, no online ghosts.** Bot scores are computed by a headless deterministic run at submission, never typed in. Online boards exist only for surface = `patch`. |
| D6 | **Weekly track** = seeded generator, seed = ISO year + week. |
| D7 | **Out**: photo mode, bot coach, career/economy, tyre wear / fuel / pit strategy, online ghosts, live multiplayer, Roblox. **Later**: track editor. |

## Rules for every task

* `drive/CONTRACT.md` is authoritative. Every new module goes into its §1
  module map in the same commit.
* **The physics path is not touched.** Nothing new inside `Vehicle.step`;
  §0 determinism conventions hold; V20 stays bit-for-bit.
* `drive/` imports nothing from `drive.ml` at module level — lazy imports
  inside functions only, exactly as today (`_ml_input`, `run_swarm_cli`).
* New logic lives in **new modules**. `drive.py` and `garage.py` are ~5.8k
  lines each; they get hooks only.
* Every new module has `self_check()` runnable as `python3 -m drive.<module>`
  and is added to `validate.MODULES`. New harness checks in `drive.py` are
  numbered **V31 upward** (V30 is the race check).
* **Frame budget**: `render.py` V22 (p99 <= 16 ms, load-normalised) stays
  green with two ghosts, particles and a full race grid.
* **Scripted and headless runs never read player files** (`runs/records/`,
  `runs/progress.json`, `runs/leaderboard_local/`) — the same rule
  `runs/settings.json` already follows.
* Player data is JSON under `runs/`, each file with a versioned `kind`
  field in the style of `SEED_LAP_KIND`. A corrupt or old file is ignored with
  a note, never a crash.
* **Input**: every new screen takes keyboard, pad and mouse, like the garage;
  every new key goes into the README key table and the menu help.
* **Gates** (all must pass before a task is committed; paste the numbers
  into the note):
  `python3 -m drive.validate`, `python3 -m drive.validate --modules`,
  `python3 -m drive.drive --self-check`, `python3 -m drive.ml`.
* **Concurrency**: a second session may edit this repo at the same time.
  Re-read a file right before editing it, use exact-match edits, never revert
  a change you did not make, never `git add -A` — stage your own paths.
* One commit per task, message style `area: what it does (task NN)`.

---

## Phase A — the single-player loop

### T19 Lap records: your top 5 per class

New `drive/records.py`; hooks in `Sim` where `LapTimer.update` emits
`("lap", ...)` events.

* Record **every** lap, not only K-armed seed laps:
  * a pose trace at 50 Hz in the `_Replay` schema
    (`t, x, y, psi, u, wing_deploy, wing_side, top_deploy`), reusing the seed-lap
    recorder's pattern;
  * a **controls log**: the `Controls` actually handed to `Vehicle.step`,
    run-length encoded by step index. Kept locally for future score
    verification; not uploaded, not shown.
* On a valid lap (`LapTimer.lap_valid`) insert into the class's top 5 and
  persist `runs/records/<class>.json`: time, sector times, date, build name +
  full `CarBuild` JSON snapshot, assists, trace, controls log. Target
  <= 300 KB per lap (rounding or an `.npz` sidecar are both fine).
* Invalid laps never enter. Dragstrip: records only if it has a finish time;
  if it does not, either add a finish line at its end or exclude it — say which
  in the note.
* HUD: PB, and the lap's position in the top 5 when it lands.

Acceptance
* `records.self_check`: insert / order / evict / persist / corrupt-file tolerance.
* **V31**: a scripted 3-lap headless run leaves the right top-5 file, and
  re-simulating one lap from its controls log reproduces its lap time
  **bit-for-bit**.

### T20 Pre-race screen and per-track builds

* Shown when a timed session starts on a lap map (launch, map change) and from
  a new pause-menu item **Time trial**. Shows: class, selected build, top 5 for
  the class, medal targets.
* Default build = the last build used on this track
  (`runs/records/last_builds.json`).
* **RACE**: ENTER / ✕. An unchanged build starts in one press.
* **EDIT**: opens the garage and comes back here.
* **PICK**: any saved build from the garage library, built on any track.
  Each row shows that build's best time on *this* track, if it has one.
* Scripted and headless runs skip the screen.

Acceptance: **V32** menu-flow check in the style of V26 (driven by events,
no window).

### T21 Medals

* New `drive/medals.py`. Table `drive/data/medals.json`, generated by
  `python3 -m drive.medals --build`: per class, author time = the best of the
  reference drivers run headless on the stock build (anchor policy, bundled
  checkpoints that match car + track, `LapDriver`). Thresholds per D4.
* Results show the medal earned; the best medal per class is persisted; the
  pre-race screen shows the targets.
* Self-check: every class covered or explicitly "no reference lap"; ordering
  bronze > silver > gold > author; staleness against a hash of the track
  definitions, car specs and engine modes (stale -> soft fail that prints the
  regenerate command).

### T22 Ghosts and live delta

* Ghost 1 = the class PB trace. Ghost 2 = the D2 slot. Both drawn with
  `_Replay` and the existing ghost drawing, top-down and chase 3D. One toggle
  key (pick an unused one).
* Ghosts start when the player crosses the start line and restart every lap.
* Live delta vs the PB, by track progress `s` (the `RACE_GAP` trail approach):
  large `+/-0.00` at top centre, green / red.
* Sector flash at each sector line: purple = best sector ever for the class,
  green = better than the PB lap's sector, red = worse.

Acceptance: render self-check additions; V22 green with two ghosts; **V33**:
delta against a lap's own recorded trace stays about 0 (state the tolerance).

**STOP after Phase A.** Report and ask the owner to playtest.

---

## Phase B — depth

### T23 Driving tutorial

* New `drive/tutorial.py`: data-driven steps
  `{id, map, text, predicate(HudData/sim state), hint}`; overlay box;
  skippable; progress in `runs/progress.json`; offered on first launch and
  in the menu.
* Steps: throttle and brake -> steering -> arena turn 1 -> reset (R) -> what the
  assists do -> **the wing on the skidpad**: a lap with the wing off, a lap with
  it on, and the measured lateral-g difference shown -> sectors, PB ghost,
  medals -> one valid lap.

Acceptance: self-check drives every predicate with synthetic `HudData`;
**V34** runs the whole tutorial headless with `ScriptedInput`.

### T24 Wing-design tutorial

* A guided pass through the garage's existing gated DESIGN navigator:
  mission -> aerofoil (screening weights) -> planform -> end plates / blend ->
  results -> fit to the car -> save the build -> drive it (the pre-race screen
  picks it).
* Hints anchored to the existing garage widgets. `garage.py` gets minimal
  hooks: one tutorial object that the page draw and event code query.
* Plain words: downforce vs drag, why a flank wing, what each number means.

Acceptance: the garage self-check still passes; step predicates are tested
against garage state.

### T25 Challenges

* New `drive/challenges.py` + `drive/data/challenges/*.json`. Schema:
  `id, title, blurb, class, constraints, goal {metric, threshold}, stars`.
  * `constraints`: max wing area per slot, max fitted wing mass, max ballast,
    allowed slots, max CdA.
  * `stars`: 1 = pass, 2 = a tighter metric, 3 = the metric plus an
    efficiency constraint.
* Metrics: `lap_time`, `stop_distance` from V0, skidpad mean `a_y` over a lap,
  dragstrip time and trap speed. Reuse the measuring logic of `brake_script`,
  `skidpad_limit_script` and `accel_script`.
* Constraints are checked before the start; a refusal quotes the reason, in
  the garage's gating style.
* Challenge page in the pause menu with star counts; progress in
  `runs/progress.json`.
* Ship **8 challenges** covering braking, skidpad, drag, lap and wet. Every
  threshold comes from a measured headless run (recorded in the note). Every
  3-star result is shown to be achievable by a scripted or bot run.

Acceptance: the self-check validates every JSON file and the constraint
checker cases; **V35** runs the achievability checks.

### T26 Bot training and race flexibility

* The swarm stays open-ended by default (`gens=0`): no prompts, no "train N
  more". Add a progress panel: best lap per generation, medal lines, the
  owner's PB for the class, and the gap to each. The user stops with ESC, as
  today.
* Free values instead of fixed lists: population any integer in [4, 128];
  sim time in [20, 240] s.
* Race grid: raise `RACE_GRID_MAX` to the largest N that keeps V22 green.
  Measure it and write it down. Each slot follows D3.
* A saved bot carries the build it was bred in (checkpoint metadata), so
  racing or ranking a bot uses its own car.

Acceptance: V30 still passes; **V36** N-bot race; the swarm self-check passes.

### T27 Polish (no photo mode)

* Results screen: time, delta to PB, sectors, medal, top-5 position, and an
  animation for a new record.
* PB and medal chime, procedural in `audio.py` (no asset files).
* Tyre smoke tied to slip (pooled, budgeted). Subtle camera shake on kerbs and
  off-track, with a settings toggle.

Acceptance: the render and audio self-checks pass; V22 green.

**STOP after Phase B.** Report and ask the owner to playtest.

---

## Phase C — online and the weekly track

### T28 Leaderboards (scores only)

* New `drive/leaderboard.py` with a backend interface:
  `submit(board, score, details)`, `top(board, n)`, `around_me(board, k)`.
  * `LocalBackend`: JSON under `runs/leaderboard_local/`. Used by the tests
    and in development.
  * `SteamBackend`: optional import of a Steamworks Python binding. If the
    binding is missing, the online UI shows "offline" and nothing crashes.
* Boards: human `<class>`, bot `bot|<class>`, weekly
  `week|<YYYY-Www>|<car>|<engine>`. Surface `patch` only (D5).
* Human submission: the best valid lap, on a new PB. Bot submission: the game
  runs the checkpoint with its build headless and submits that result.
* UI: top 10, your rank, and the entries 2 above and below you. Shown on the
  pre-race screen and on a Leaderboards menu page.
* **Owner actions the agent cannot do**: Steamworks partner account, SDK
  download, app ID. Write the exact steps into the note; until they are done
  the game runs on `LocalBackend`.
* Cross-machine determinism: a script that prints the hash of a fixed headless
  lap. The owner runs it on the Mac and on a Windows PC and the result goes in
  the note. Auditing bot boards depends on it.

Acceptance: the leaderboard self-check passes on `LocalBackend`; a test that
the game works without the Steam binding installed.

### T29 Track generator and weekly track

* New `drive/trackgen.py`: a seeded generator that outputs a closed `Seg` list
  through `track.build`. Weekly seed from the ISO year + week. Appears in the
  map list as `Weekly <YYYY-Www>`.
* Constraints:
  * closes (end pose = start pose within a tolerance);
  * no self-intersection, including track width and run-off;
  * minimum radius >= R_min;
  * length 1.5–4 km;
  * at least 3 sector lines;
  * an optional wet patch.
* Weekly medals: a headless reference run at first load, cached under `runs/`.
* Dev option: `python3 -m drive.trackgen --save <seed>` exports a generated
  track to a curated pool.

Acceptance:
* 200 seeds -> 100 % valid.
* `LapDriver` completes a lap in all 3 cars on 50 seeds.
* Generation < 1 s per track.

---

## Not in this plan

* Track editor (later).
* Steam packaging and the Windows build (separate plan).
* **Fictional car names in the UI before release**: Opel / Mazda / BMW
  names and shapes are trademarks (separate small task).
* Roblox.
