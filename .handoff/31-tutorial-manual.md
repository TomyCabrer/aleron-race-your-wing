# Task 31 — the tutorial: the manual gearbox, and the way to the wing tutorial

Owner feedback on task 23 (2026-09-24): *"Tutorial is nice. Last step or step
10 (skippable) how to drive manual. Also tutorial on wing design."* Numbered
31: 28-29 are the plan's Phase C, 30 is the look-and-sound work.

```
python3 -m drive.tutorial             ->  PASS    (+8: the manual predicate x4, the optional page, the switch in the progress file, the group skip, the done page / WELCOME rows)
python3 -m drive.drive --self-check   ->  ALL PASS  (V34: 8/8 drive steps, 5/5 pages, the manual box for its step and back; V20 sha256 ed41b7f781e959ea unchanged; V21 RTF 11.66)
python3 -m drive.validate             ->  82/82  pass  0 HARD  0 soft   [127.3 s]
python3 -m drive.validate --modules   -> 119/119 pass  0 HARD  0 soft   [487.5 s]
python3 -m drive.ml                   ->  ALL PASS  32/32
```

The modules gate failed once on the final tree: `drive.world` (handoff 30's
code, untouched here), whose check asserts the slowest of 39 backdrop frames
is under 5 ms of wall time. The machine's load was about 12 at the time (review agents
measuring in parallel). Standalone it passed 3 of 3, and the whole gate
re-run passed. A wall-clock maximum is load-sensitive: worth a margin or a
median in that check (not changed here, it is not this task's code).

## What it does

* **Two optional steps before the last page** (the tutorial is 13 steps now,
  about ten minutes):
  * **11, a page: *The manual gearbox (optional)*.** Why you would drive it
    (the gear for the exit, engine braking in, no change up mid-corner); the
    keys (`E` / `R1` up, `Q` / `L1` down); the shift lights over the rev bar;
    changing down under braking, one gear at a time, with the box blipping;
    pulling away (the box works the clutch, it cannot stall); and the clutch
    mode (`Z` / `SQUARE`, launch on it, it can stall, `S` restarts). Its rows
    are **Try it**, **Skip it (the gearbox stays as it is)** and *End the
    tutorial*. *Skip it* skips both steps (`Step.group`, `Tutorial.skip(group=True)`).
  * **12, a drive step on the arena: *Manual gearbox (optional)*.** From the
    line, up to 3rd by hand, then brake and change down one gear at 30 km/h or
    more. A shift passes through neutral, so a gear is read only while one is
    in (3 -> N -> 2 is one downshift); a reset (stopped, back in 1st) is not a
    downshift; a downshift before 3rd does not count. The box's status line
    says the gear and what is next.
* **The box is switched for step 12 and back** (`Step.gearbox = 'manual'`,
  `Sim._tutorial_gearbox`). An **automatic** goes to *Manual (auto clutch)*
  when the step starts, as the settings page would do it (saved, the lap in
  progress discarded, the class retargeted), and back to *Automatic* when the
  step is over: passed, skipped, the tutorial ended or started over, a
  challenge picked (which ends the tutorial), or the game quit mid-step
  (`run_interactive_cli`'s `finally`). A session restart mid-step keeps it
  (the saved setting is manual, the tutorial still remembers the automatic).
  **Any gearbox change you make during the step makes the box yours**, and
  nothing is put back. A player already on a manual box (either kind) is not
  touched. The switch is also written to the progress file, so a run that
  dies mid-step (a kill, a crash) is undone at the next launch.
* **The wing-design tutorial is one press away**:
  * the last page's rows are *Drive on* and **Next: the wing-design
    tutorial (in the garage)**, which finishes the driving tutorial and opens
    the garage on task 24's tutorial;
  * the first launch's WELCOME page has a **Wing-design tutorial** row too,
    for a player who can already drive.

  Both appear only when the session has a garage (the pause menu's Tutorial
  page already had it, `wt_garage`).
* The last page lists skipped steps by title, not by id.

## V34

It drives the manual step as well: a scripted driver on the manual box
(full throttle, up at 5200 rpm to 3rd, then brakes and asks for one gear
down under 60 km/h). V34 asserts the step passes on its own predicate, that
the Sim was on the manual box during it, and that it was back on automatic
afterwards. The run is 8/8 drive steps, 5/5 pages, arena -> skidpad -> arena,
238 s of sim.

**A bug in V34 itself, fixed:** the loop's time bound counted only the
sessions already finished, so a step that never passed in the last session
ran forever (it did, once, while the new step's scripted driver was wrong).
It now counts the running session too.

## Probed by hand (headless, a fake renderer)

| flow | result |
|---|---|
| WELCOME with a garage | rows start / wing-design / not now |
| WELCOME's wing-design row | marks the offer seen, quits to the garage with `wing_tutor_start` |
| page 11's *Skip it* | the done page; the box never changed |
| the done page's wing row | tutorial done and saved, to the garage |
| step 12 on an automatic, then *Start over* or *End* | automatic again |
| the player sets the clutch box during step 12, then skips | it stays the clutch box |
| the quit path (`_tutorial_gearbox_restore`) | automatic again |
| no garage | no wing rows |

## What the review found

One reviewer, who checked each claim by driving scratch Sims headless. It
found no high-severity bugs; all five of its findings are fixed in this
commit and re-probed.

* **Ending the tutorial left the game paused** (medium, OLDER than this task,
  reachable from the new page too). A row's select hides the menu before its
  action runs, so `_tutorial_stop`'s "close the menu if it is open" never
  closed the PAGE: still paused, the keyboard still in menu mode, and only P
  got out. It now closes whenever a page is up.
* **The manual step could pass without a downshift** (low). R keeps the speed
  and may land in any gear (in 1st at 87 km/h after an upshift's neutral),
  and a spell on the automatic counted its own shifts. Now only a change UP
  by hand climbs toward 3rd; R starts the gear reading over, and the
  automatic clears it.
* **A player already on a manual box could not pick Automatic during the
  step** (low): the next frame switched it back. The switch is now looked at
  once per step.
* **Choosing Manual during the step was undone at its end** (low), although
  the page says Settings > Gearbox keeps it. Any gearbox change the player
  makes during the step now makes the box theirs, and nothing is put back.
* **A run that died mid-step left the box on Manual for good** (low: a kill,
  a closed terminal, a crash; the `finally` does not run). The switch is also
  written to the progress file (`tutorial.gearbox_prev`), and the next launch
  puts the box back (`tutorial.gearbox_left`) if it is still the one the
  tutorial set.

Checked by the reviewer and found fine:
* restarts mid-step (TAB, a map or car change);
* BACKSPACE to the garage mid-step;
* *Start over*, *End*, a challenge pick, quit;
* the recorder: the switch and the restore each discard the open lap, and a
  lap closes before the tick, so no lap is filed with the wrong assists;
* old saved step ids resume by id;
* the group skip, and both wing-design rows.

## Deviations

* **Two steps, not one.** A page (with a real *Skip it*) plus the drive
  step. The owner asked for a step that can be skipped; a drive step alone
  can only be skipped through ESC > Tutorial, and the explanation does not
  fit the box on the road.
* **Where it sits.** Steps 11-12, after the valid lap and before the last
  page ("last step or step 10"): the done page stays last.
* **"Also tutorial on wing design":** it already existed (task 24, in the
  garage, and in ESC > Tutorial). This task makes it reachable from the two
  places a new player actually is: the end of the driving tutorial and the
  first-launch WELCOME page.

## Shape of it

| file | what |
|---|---|
| `drive/tutorial.py` | `Step.gearbox`, `Step.group`; `_chk_manual`, `_st_manual`, `_pg_manual`; the two steps; `frame_of` adds `gear`, `gearbox`; `Tutorial.gearbox_prev` / `set_gearbox_prev` (saved in the progress file), `gearbox_seen`, `gearbox_wanted`, `skip(group=)`; `gearbox_left`; the page rows (Try / Skip, the done page's wing row); `offer_items(garage)`; skipped steps by title; self-check +8 |
| `drive/drive.py` | `Sim._tutorial_gearbox`, `_gearbox_live`, `_wing_tutor_go`; `_tutorial_ctx.garage`; the `tut_skip_group` / `tut_wing` / offer `wt_garage` actions; `apply_setting('gearbox')` hands the box to the player; `_tutorial_stop` closes the page (the old paused-after-End bug); `_tutorial_gearbox_restore` (in `_challenge_switch` and the CLI's `finally`) and the launch's `gearbox_left` undo; V34 drives the manual step, and its time bound is fixed |
| `README.md`, `drive/CONTRACT.md`, `.handoff/README.md` | the tutorial paragraph, the module map, the flow |
