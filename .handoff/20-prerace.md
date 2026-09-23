# Task 20 — the pre-race screen, and a build per map

The brief (`PLAN-steam-engagement.md`, T20): a timed session on a lap map
starts on a screen that shows the class, the build, the class's top 5 and the
medal targets; RACE in one press, EDIT through the garage and back, PICK any
saved build with its best time here; each map opens with the build last used
on it; the same page from a pause-menu item *Time trial*; never for a script
or a headless run; keyboard, pad and mouse; V32.

```
python3 -m drive.prerace               ->  14/14   (new)
python3 -m drive.menu                  ->  ALL PASS  (+2: the mouse, a scrolling list)
python3 -m drive.input                 ->  ALL PASS  (+2: the mouse in and out of a menu)
python3 -m drive.drive --self-check    ->  ALL PASS  (V32 new; V20 sha256 ed41b7f781e959ea unchanged; V21 RTF 12.16)
python3 -m drive.validate              ->  82/82  pass  0 HARD  0 soft   [114.2 s]
python3 -m drive.validate --modules    -> 106/106 pass  0 HARD  0 soft   [286.8 s]  (105 + prerace)
python3 -m drive.ml                    ->  ALL PASS  32/32
```

## What it does

Every windowed, human-driven session on a map with a lap (`prerace.wanted`:
not a script, not headless or offscreen, not `--ml-drive`, not the dragstrip or
a non-standard skidpad) opens PAUSED on the **TIME TRIAL** page:

* the class (map, car, engine, surface, as titles), the build's name;
* the class's top 5 -- time, build, assists (`ABS TC AID AUTO`), date;
* the medal block: author / gold / silver / bronze from `drive.medals`
  (task 21; "no reference lap for this class" when it has none) and the
  best medal you hold, with your PB;
* rows **RACE** (the cursor opens on it: ENTER, CROSS or a click is one
  press -- every car to the line, `reset(to_checkpoint=False)`, the clock
  starts at the next crossing, i.e. the flying lap, exactly as a player's
  first timed lap always has), **Build** (PICK), **Edit this build in the
  garage** (EDIT).

PICK lists every build in the garage library (`runs/library/builds/`),
whichever map it was designed on, each with its best lap **in this class**,
and marks the one being driven; a build being driven that is not saved is
listed first as "(driving, not saved)". Selecting another restarts the session
on it (`Sim.prerace_pick` -> `run_interactive_cli` makes it the design); a car
being driven that is in no library file is first saved there as `<name>
(autosave)`, so a pick never loses a car. EDIT (only when the session has a
garage) ends the session into the garage; the garage's ENTER starts a new
session, which opens on this page again. ESC is the pause page; *Resume*
drives on from where the car is, and those laps count too (the recorder files
every lap of the session -- RACE is the clean start, not a switch for
records). R / SHIFT+R / BACKSPACE keep working as the pause menu's hotkeys. The pause menu has a new row, *Time
trial: your top 5, medals, the build*, whenever the session has the page.

**A build per map.** Every player session (`_player_session`: not
`--ml-drive`, headless or `--render off / offscreen`) records the build it
drives as its map's: `runs/records/last_builds.json[track] = (name, CarBuild
JSON)` (`_track_build_used`; RACE writes it too). `run_interactive_cli` reads
it at launch and on every map change (`_track_build`) and, when it differs
from the current design, makes it the design -- IN MEMORY: the garage's own
file `runs/garage_design.json` is written only by the garage, so the car on
the ramp survives a map change. The arena opens with the car last driven on
the arena and the skidpad with the skidpad's; leaving the dragstrip for the
arena and coming back gives back the dragstrip car (checked with the real
`run_interactive_cli` and faked sessions). Never over a car the garage has
just built (EDIT, BACKSPACE), never at a launch that named a car (`--build`,
`--wing...`), and after a swarm the next session skips the page
(`opts.prerace_skip`: you came back to race the bot you bred, not to be
timed).

**The mouse, on every menu page.** `Menu.draw` now records each row's
rectangle; `hover:X:Y` moves the cursor to the row under the pointer, a
press (`click:X:Y`) ARMS the row under it and the release on that same row
(`release:X:Y`) runs it exactly as ENTER would (`Menu.hit`, `row_centre`). A
press within `CLICK_GUARD_DRAWS` (15) frames of a page change is ignored, so
the second click of a double-click cannot run a row of the page the first
opened. A list longer than the window scrolls (the PICK page's library can
grow). The keyboard half of the input turns pygame's mouse into those
commands only while a menu is up (`input._menu_mouse`: motion, left press /
release, right click = back, the wheel = up / down); with the menu closed the
mouse does nothing.
This reaches the pause, settings, swarm and race pages too, which had no
mouse before. The pad drives both new pages through the existing menu mode.

## V32

`_v32_prerace`, events only, no window (V26's style): the page opens paused
on RACE; one `select` races (menu closed, unpaused, car on the line, timer
reset, `last_builds.json` written with this build); the pause menu's Time
trial reaches it; PICK lists the two library builds with their class bests
(1:00.950 and 1:02.100) in order; ESC comes back from PICK; picking the other
build sets `prerace_pick` and restarts; EDIT stops into the garage; ESC on
the page is the pause page; then the page is DRAWN on an offscreen surface
and a hover, a press and a release at the RACE row's centre (after the
double-click guard's 15 frames) race. And `prerace.wanted` is
false for headless, scripted, dragstrip.

## What the adversarial review found, and what changed

Three lenses (session flow, page / input, contract), 3 finders and 9
verifiers told to refute; every confirmed defect is fixed in this commit:

* **The map default overwrote the garage's car on disk** (high, found twice).
  The first version switched to the map's last-RACED build with
  `design.save()`, so a car built and driven but never raced was in no file
  after a launch or a TAB. Now the switch is in memory, every session records
  its build as its map's, and a PICK away from an unsaved car autosaves it.
* **"ESC: Resume drives on, untimed" was false** -- the laps are filed -- and
  only RACE recorded the map's build. The text now says the laps count, and
  every session records the build.
* **PICK credited a saved build with a lap an edited car drove under its
  name** -- `build_bests` was keyed by name. Now by content
  (`records.build_id`).
* **A double-click carried over** onto the next page (double-clicking *Time
  trial* ran *Edit* and left the session; *Settings* changed the engine).
  Now a row fires on release over the armed row, with the guard above.
* **PICK had no scrolling** (with ~18 builds rows went off the window).
* **A top-5 lap with a non-dict `build`** (another version, a hand edit)
  crashed the page and so every launch in that class; rows are read
  defensively and a failing page falls back to the pause page.
* Low: `--render offscreen` read / wrote the per-map builds (now
  `_player_session`), and EDIT was offered with no garage and quit the game
  (now hidden).

## Deviation / interpretation

* **"Each row shows that build's best time on this track"** -- shown as the
  build's best **in this class** (this map with this car, engine and
  surface). A time set in another car or on another surface is not a time
  this build can be compared by on the screen you are about to race from.
  The class file keeps a per-build best for every valid lap
  (`RecordBook.build_best`, task 19's book), not only the top 5, so a build
  whose best is outside the top 5 still shows it.
* "The last build used on this track" is recorded at the START of every
  session on that map, not only on RACE (the review showed RACE-only lost
  cars and disagreed with "used").
* The screen shows on EVERY drive session start on a lap map, not only
  launch and map change: a car / surface / ballast change is a new class and
  a new session, and EDIT has to come back to it. It is one press.

## Shape of it

| file | what |
|---|---|
| `drive/prerace.py` (new) | `PreRace` (rows, help sections, PICK rows), `wanted`, `default_build`, `assists_text`; self-check |
| `drive/drive.py` | `Sim.prerace` / `prerace_pick`; the Time trial row; `open_prerace`, `_menu_show_prerace(_pick)`, `start_timed`, `_prerace_event`; `_interactive_session` builds and opens the page; `run_interactive_cli`: `_track_build`, `_track_build_used`, `_autosave_build`, `_player_session`, the pick, `from_garage`, `prerace_skip`; `_resolve_design` carries the library; V32 |
| `drive/menu.py` | row rectangles, `hit`, `row_centre`, `hover:` / `click:` / `release:`, the double-click guard, the scrolling window |
| `drive/input.py` | `_menu_mouse`; the menu help's mouse row |
| `drive/records.py` | `build_id`; `build_best` / the build bests filed by content |
| `drive/validate.py` | `prerace` in `MODULES` |
| `drive/CONTRACT.md`, `README.md` | module map, the input / menu mouse, the page's flow |

The medal block reads `drive.medals` lazily: in this commit that module is
not in the tree yet (task 21), so the page shows "no reference lap for this
class" until it is -- by design, not an error path.
