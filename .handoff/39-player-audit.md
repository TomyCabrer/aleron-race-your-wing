# Task 39 — a player's audit, and the fixes

Owner (2026-09-24, 02:13): *"become an exigent player / boss and deploy agents
that fix issues or things that could be done better that you find. Focus on
general aspects and details and how a person will interact with the game."*
Plus: *"don't go crazy with token usage"*, so the audit was deliberately small.

Started at 09:30, after the other carsim sessions went quiet (tasks 33-37
committed, 38 written but not committed). Committed on the owner's go-ahead:
task 38 on its own (`0656b15`), then this batch -- see *Committing* at the end.

## How

* **Audit** (read-only): five player lenses, each playing offscreen (real
  frames, screenshots read back) and reading the code:
  1. first launch and menus
  2. driving and the HUD
  3. the time-trial loop
  4. the garage, challenges, bots and the swarm
  5. robustness, settings and copy

  38 findings. One *boss* agent spot-checked every candidate against the code,
  merged duplicates, kept **15**, turned **6** down, and wrote down the questions
  only the owner can answer.
* **Fix**: six agents on file-disjoint groups (`drive.py` in two passes one after
  the other), each running its modules' self-checks. One adversarial reviewer
  went over the batch's own diff (separated from task 38 with
  `git archive HEAD` + task 38's diff) and fixed three seams itself.
* Two corrections by the lead before the fixes:
  * The proposed car help said "every car drives its front wheels". That is
    **wrong**: MX-5 and 540i are really rear-driven since task 11. The old
    help ("FWD-only ... traction is fiction") was stale, and now says RWD.
  * The generic-pad fix was **narrowed**. Nothing guesses the button order,
    because it can't be tried on hardware here, and the DualSense `ps` tables
    are untouched.

## What a player notices now

| # | change | where |
|---|---|---|
| 1 | **Time-trial retries are a rolling start 150 m before the line** (at most 15 % of the lap), at a corner-safe speed in a sensible gear. The clock starts **6-9 s** after RACE / SHIFT+R; before, it took a 60-86 s out-lap. The HUD counts down `OUT LAP  <m> m to the line`. Only on the four circuits, time-trial sessions, no rivals / challenge / tutorial. The seed lap and every other map still use a standing start (`Sim.reset(standing=True)`). | `drive.py` `Sim._rolling_pose`, `Sim.reset`, `start_timed`, `hud_data` |
| 2 | **The running lap shows INVALID the moment it's voided**, with the reason (`off track`, `reset`, `not recorded (...)`). Before, the HUD showed the *previous* lap's validity. SHIFT+R / RACE **keep BEST, LAST and the sector bests** (`LapTimer.restart`). A lap with an `R` in it can no longer become BEST. | `drive.py` `LapTimer`, `hud_data`; `render.py` timing panel |
| 3 | **A race HUD by default.** `minimal` is the new default: speed / gear / rpm, timing, minimap, a one-line wing chip (only with wings), warnings. `full` is the engineering view (loads, state, aero, pedals, g-g). **H and C are remembered** (`Settings.hud`, `Settings.camera`) across TAB, restarts and launches. `LIMITED BY` only shows near the limit. Text no longer runs out of its panels at 1280x800 (`_blit_fit`). | `render.py`, `drive.py` `Settings`, `_view_event` |
| 4 | **Tutorial flow.** ESC during the tutorial opens the TUTORIAL menu; on a page, ESC no longer skips the page (ENTER continues). R / SHIFT+R / BACKSPACE can't skip WELCOME or a tutorial page. TAB doesn't break the tutorial. | `drive.py` `_menu_open`, `_tutorial_event`, `_menu_event`; `tutorial.py` |
| 5 | **Plain words.** No file paths, class names or "traction is fiction" on the Settings / pause / race pages. The engine help follows the selected car. The pause subtitle is map · car · lap · wings · gearbox. The race help says bots 1..5. | `drive.py` help tables |
| 6 | **Nothing crashes on a bad file.** A settings file of the wrong shape clamps to defaults. An unparsable one is moved to `.bad`. Saves are atomic. Quitting from the garage keeps the design. Load / save problems show on screen once. Same in the garage: a wrong-shaped design or build loads as defaults, read-only saves say `could not save: ...`. **Build names that differ only in case / punctuation no longer overwrite each other** (`-2`). | `drive.py` `Settings`, `run_interactive_cli`; `garage.py`, `aero/library.py` |
| 7 | **G / F / T say what they did**: no wings fitted -> `no wings fitted - BACKSPACE: the garage`; wings off -> `(wings are OFF - F arms them)`; `WET everywhere: 63% grip (T again for dry)`. | `drive.py` `handle_event` |
| 8 | **Garage first screen**: `W  try a ready-made wing` / `D  design your own` on empty slots, a *Start here* line, every side-panel line wrapped to the panel. The wing tutorial is offered the first time D is pressed. | `garage.py` |
| 9 | **Pad**: the generic pad's help rows tell the truth (SDL order), the d-pad drives menus on a generic pad, **focus loss or unplugging the pad pauses into the menu**, a hot-plugged pad (and one found at launch) gets the car's steering lock. The menu legend's words follow each page's footer; no more `LEFT / RIGHTchange value`. | `input.py`, `menu.py`, `drive.py` (lock at launch) |
| 10 | **Challenges explain themselves**: a refused build's page starts with *Fix it in the garage*. A wing challenge warns `your car has no wings: the third star needs them`. The withheld third star says why. | `challenges.py` |
| 11 | **The next medal**: the TIME TRIAL page and the results card show `next  AUTHOR 57.184  (0.572 s to go)` / `all medals won`. Won medal rows get a `*`. The subtitle uses proper names (`Linden park · Opel Corsa C 1.2 · ...`). | `prerace.py`, `results.py`, `render.py` card |

Self-checks changed: V32's two RACE checks now expect the rolling pose, not
`s == 0`. `render.py`'s V30 boot renderer is `hud='full'` explicitly, so V22's
busiest frame still measures the full HUD. New checks were added in `input`,
`menu`, `garage`, `aero.library`, `challenges` and `prerace`.

## Turned down (with the reason)

* **16:9 HUD re-anchoring and fullscreen**: they depend on the fullscreen
  question below, and span three files.
* **Map / Car rows on the TIME TRIAL page**: they need `track.py`'s
  `TRACK_ORDER`, which is task 38's uncommitted work.
* **Race laps / positions / result card vs bots**: a new system; question below.
* **Bundled bot checkpoints** (which ship, their names, delete protection): a
  content decision.
* **Hiding the developer keys** (T L K M N X V B O) and a TAB confirm: a
  design call.

## Owner decisions

1. **Skidpad challenges are 3-starrable with no wings** (0.738 g against the
   plate reference's 0.733 dry; 0.463 against 0.462 wet). Options: set t3 to the
   reference x 1.00, add a *needs a flank wing* 3-star rule, or accept it. Only
   a warning was added.
2. **Fullscreen for Steam**: (a) `pygame.SCALED` on the 1280x800 surface, with
   black bars on 16:9, or (b) native resolution with the HUD re-anchored?
   An F11 / Alt+Enter toggle and a Settings row?
3. **Developer keys** hidden behind a developer setting in the Steam build?
   TAB mid-lap needs a second press?
4. **Race vs bot**: a Laps row, positions and a result card next?
5. **Bot checkpoints**: which ship, what they're called, protected from
   Delete? Can `swarm_arena_corsa_20260921_204340` go?
6. **TIME TRIAL page Map / Car rows** once task 38 is in?
7. **Default camera** for a new player: `car_up` (today) or chase?
8. **Generic (Xbox) pad button order.** `DEFAULT_PAD_BUTTONS` looks like
   XInput order (pygame on Windows); the names are SDL order (macOS HIDAPI).
   The help now matches SDL order, so on Windows a generic pad's rows may be
   wrong. Moving to `pygame._sdl2.controller`, which normalises the order,
   is the real fix; it can't be tested here without an Xbox pad.
9. **Auto-pause on focus loss** can't be exercised with the dummy SDL driver.
   If macOS sends a stray focus-lost when the window is created, a session
   would open paused. Worth one real launch to check.

## Committing

Owner's choice (19:00): task 38 alone, then this batch. Task 38 was staged from
its 09:25 diff (`git apply --cached --3way`; two union conflicts with task 40:
`START_S` beside the `T_MAX` comment, and the 38 / 40 index rows) plus its later
Paint-row follow-up (15 hunks in `drive.py`, the Paint row paints the car the Car
row shows), `drive/paint.py` and its note -> `0656b15`. This batch was every
remaining hunk except another session's value-row arrow-click fix
(`garage_ui.py` + its `garage.py` self-check), which is left uncommitted for its
owner. Checked first on the combined tree: validate 82/82, drive --self-check
ALL PASS.
