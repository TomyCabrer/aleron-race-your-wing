# Task 24 — the wing-design tutorial

The brief (`PLAN-steam-engagement.md`, T24): a guided pass through the
garage's existing gated DESIGN navigator: mission -> aerofoil (screening
weights) -> planform -> end plates / blend -> results -> fit to the car ->
save the build -> drive it (the pre-race screen picks it). Hints anchored to
the existing garage widgets. `garage.py` gets minimal hooks: one tutorial
object that the page draw and event code query. Plain words: downforce vs
drag, why a flank wing, what each number means. Acceptance: the garage
self-check still passes; step predicates are tested against garage state.

```
python3 -m drive.wing_tutorial        ->  21/21   (new; every step's predicate against a real headless garage)
python3 -m drive.garage               ->  ALL PASS  (unchanged checks)
python3 -m drive.tutorial             ->  38/38   (+1: the Tutorial page's wing-design row)
python3 -m drive.drive --self-check   ->  ALL PASS  (V20 sha256 ed41b7f781e959ea unchanged; V21 RTF 11.99)
python3 -m drive.validate             ->  82/82  pass  0 HARD  0 soft   [115.7 s]
python3 -m drive.validate --modules   -> 111/111 pass  0 HARD  0 soft   [326.5 s]  (110 + wing_tutorial)
python3 -m drive.ml                   ->  ALL PASS  32/32
```

## What it does

`drive/wing_tutorial.py`: ten `WStep(id, title, text, do, check, anchor)`,
in the order the garage's own gates enforce:

| # | id | passes when (`check(garage, mem, action)`) | the hint points at |
|---|---|---|---|
| 1 | `open` | the MISSION (or DESIGN) page is open (`D`, `L3`) | - (the car page) |
| 2 | `mission` | the mission is STATED and the DESIGN page is open | the mission form's *state this mission* row |
| 3 | `screen` | the AIRFOIL group has a ranking (`L` / ENTER on *library screening*) | AIRFOIL > library screening |
| 4 | `section` | the AIRFOIL group is FINISHED (a section taken and fitted, `F`) | AIRFOIL > ranking |
| 5 | `plates` | the end plates are designed, flown FLAT, or locked | ENDPLATE > library screening |
| 6 | `planform` | the design box was looked at, then left for another WING / RESULTS step | WING > design box |
| 7 | `results` | a RESULTS step is open | RESULTS > summary |
| 8 | `fit` | the designed wing is saved (not dirty) and is the one in its slot (`S`) | the key bar |
| 9 | `build` | the car as it stands is a saved build, by content (`prerace._same_build`) | the library's builds list |
| 10 | `drive` | the garage returns `'drive'` | the key bar |

* Each step's box says what the step IS in plain words (downforce and drag,
  why a flank wing, what an aerofoil is, what the screening weights price --
  clmax, L/D, thickness, stall, moment -- what the planform numbers are, what
  F, D, L/D and the lap in the results mean) and what to press, keyboard and
  pad. It sits over the page's lower right; an outline is drawn round the
  anchored widget with a line to it. The anchor is found where the page's
  last draw put it: every garage widget keeps its own `_rect` / `_hits` for
  the mouse, and `anchor_rect` reads them.
* The order is the garage's, not the brief's: the navigator gates ENDPLATE
  on a finished AIRFOIL and the whole WING group (type = mount / blend, box
  = the planform) on a finished ENDPLATE, so the end plates come BEFORE the
  planform (deviation below).
* **Hooks** (`garage.py`): `Garage.tutor` and `Garage.progress`; `frame()`
  calls `tutor.update(self, action)` after the events and `tutor.draw(self)`
  after the page; the pause menu gets the tutorial's rows (start / continue,
  or skip this step / end); `H` hides the box. The garage's pause menu now
  also takes the **mouse** (point, press, release on the row, wheel, right =
  back), through the drive's own `input._menu_mouse`: it was keyboard and pad
  only, and the rule is that every new screen takes all three.
* **Where it starts**: the garage's menu (`ESC` on the car, `OPTIONS`), or
  the drive's ESC > *Tutorial* page (*Wing-design tutorial*: to the garage,
  where it starts or continues), which the driving tutorial's last page
  points at. A tutorial left half way rides on `opts.wing_tutor` across the
  garage <-> drive trips and continues on the next garage visit; ended, the
  menu continues it from the saved step (`runs/progress.json`, section
  `wing_tutorial`). The last step passes on the garage's `drive`, and the
  session that follows opens on the TIME TRIAL page with that build, whose
  *Build* list has it on every map.

## Integration (scratch smoke test, not a gate)

`run_interactive_cli` with `Sim.run_interactive` and `Garage.run` scripted,
in a temporary working directory (no player file of the repo touched):
drive ESC > Tutorial shows `tut_start, wt_garage, tut_back` -> *Wing-design
tutorial* -> the garage opens with the tutor on `open` and the progress file
-> `D` passes it (`mission`) -> `H` hides the box -> ESC, ESC: the garage menu
lists `wt_skip`, `wt_hide`, `wt_end` -> a MOUSE hover, press and release on *skip*
skips `mission` (now `screen`) and closes the menu -> the garage returns
`drive` -> the tutor is still on `opts.wing_tutor` at `screen`, and
`runs/progress.json` holds `wing_tutorial: {step: screen, skipped:
[mission]}` next to the driving tutorial's section.

## What the review found

Two lenses (the tutorial in play; acceptance and rules), 2 finders and 10
verifiers. Confirmed and fixed in this commit:

* **"Save the car as a build" passed on LOADING a build** (medium): one click
  on the outlined builds list loads its top build, and the loaded car IS a
  saved build -- the step passed with the designed wing off the car. The step
  now also needs the designed wing still fitted.
* **The designer redesigns the wing already in the slot, and S saves over it**
  (medium; the garage's edit-in-place model, older than this task): a
  returning player's own wing would be replaced with no warning. Steps 1 and
  7 now say so and name the ways round it (library N: a new wing; N: rename).
* **The box did not take the mouse** (medium): a click on it reached the rank
  row under it and took a section the player could not see. The garage now
  gives a mouse event on the box to the box (`WingTutor.hit`).
* **Keyboard-only instructions, and H with no pad / mouse way** (medium):
  every step names the pad path too, and the menu has a hide / show row.
* **Steps were never re-checked** (medium, two findings): re-opening the
  design page re-locks its section gates, and a new garage (the next launch,
  a drive and back) has no stated mission -- the box then pointed at a shut
  step with no way on. The tutor now walks back to the first step that is
  open, and says so.
* **Text wrapped by character count ran out of the box** (medium): wrapped
  by the pixel width of the font drawing it; checked, the widest line fits.

Refuted: that the hook self-checks bypass the garage (the smoke test and the
real `Garage._handle` click in the self-check cover the path).

## Deviations

* **The end plates come before the planform**: the garage's navigator
  gates it that way (ENDPLATE opens on a finished AIRFOIL; the WING group,
  whose type step is the mount and the blend and whose design box is the
  planform, opens on a finished ENDPLATE). The tutorial follows the garage.
* **The solver is not a step**: RESULTS open as soon as the WING group does
  (the lap is computed on every edit), so the results step asks for the
  RESULTS page and names O (optimise) rather than requiring a run -- the
  garage itself calls the optimiser optional.
* **The garage's pause menu takes the mouse** (a small hook beyond the
  tutorial object): the new rows live there. And a mouse event on the
  tutorial's box is the box's (one more hook in `Garage._handle`).
* **No V-number**: the brief's acceptance is the garage self-check and the
  predicates against garage state, both in `python3 -m drive.wing_tutorial`
  and `python3 -m drive.garage`.

## Shape of it

| file | what |
|---|---|
| `drive/wing_tutorial.py` (new) | `WStep`, the 10 `STEPS` and their predicates, `anchor_rect`, `WingTutor` (`update`, `draw`, `menu_rows` / `menu_action`, `skip`, `end`), `menu_row`, `saved_state`; self-check on a real headless garage |
| `drive/garage.py` | `Garage.tutor` / `progress`; `frame` calls `update` / `draw`; the menu's rows and actions; the menu takes the mouse; `H`; help rows |
| `drive/drive.py` | `run_interactive_cli` gives the garage the progress file and the tutor and keeps it across trips; the Tutorial page's `wt_garage` row (`Sim.wing_tutor_start`) |
| `drive/tutorial.py` | `menu_items(garage=)`: the wing-design row; the done page points at it |
| `drive/validate.py` | `wing_tutorial` in `MODULES` |
| `drive/CONTRACT.md`, `README.md` | module map, the hooks, the garage key table (`H`), the paragraph in *Design the wings* |
