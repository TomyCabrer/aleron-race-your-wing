# Task 33 — Phase A's open questions, answered

The owner (2026-09-24), on the five questions left open after Phase A (tasks
19-22): *"Choose whether makes most sense / is better, or ask me
individually."* Each is decided here, with the reason. Two changed, three
stay.

```
python3 -m drive.prerace              ->  16/16   (+2: the Ghosts row; session_start through ten sessions)
python3 -m drive.drive --self-check   ->  ALL PASS  (V32: the Ghosts row and J on the page; V20 sha256 ed41b7f781e959ea unchanged; V21 RTF 11.82)
python3 -m drive.validate             ->  82/82  pass  0 HARD  0 soft   [123.3 s]
python3 -m drive.validate --modules   -> 119/119 pass  0 HARD  0 soft   [426.6 s]
python3 -m drive.ml                   ->  ALL PASS  32/32
```

## Changed

* **J had no pad button** (task 22): a pad could not hide the ghosts.
  Every pad button is taken, and a chord would be a surprise mid-lap. The
  **TIME TRIAL page** (ESC > Time trial, where the ghosts already are) now
  has a **Ghosts** row, *shown / hidden (J)*, that does exactly what J does.
  It sits above *Ghost 2*; ENTER or LEFT / RIGHT toggles it, J does too while
  the page is up, and it is kept across a restart as J is. V32 drives both by
  events.
* **The TIME TRIAL page opened on every restart** (task 20). It now opens
  only when there is something new on it (`prerace.session_start`): when the
  session's class (map, car, engine, surface) or build (its name and its
  content, `records.build_id`) is not the PREVIOUS session's, and back from
  the garage (EDIT's round trip still ends there).
  * **Opens:** the first drive of a launch, a map change (through the
    dragstrip, which has no page, and back too), an engine change, a PICK.
  * **Drives straight on:** a restart that keeps both, such as a ballast
    change, a tutorial or challenge ending on the same class, or a race. The
    page would have shown exactly what was just on it, for one more press.

  It is always one press away (ESC > Time trial). The rule is one pure
  function, `prerace.session_start`, which the self-check drives through ten
  sessions in a row.

## Kept, and why

* **The controls are rounded while a lap is recorded** (task 19): the
  steering to 2^-24 rad (3.4e-6 deg) and the pedals to 2^-20 of their
  travel. Nothing a player can feel changes. This is what makes a lap's log
  exact and small (22-174 KB), and it is what task 28's score verification
  needs: a lap re-simulates bit for bit. The alternative, full floats, is
  about 1 MB a lap and does not replay exactly.
* **PICK shows each build's best in THIS class** (task 20), not on the
  track in general. A time set in another car or on another surface cannot
  be compared on the screen you race from, and the class is what the top 5,
  the medals and the ghosts are all keyed on.
* **Medals are set with the aids on as well as off** (task 21). The author
  time is the best of the reference driver's laps both ways, so the medals
  are as hard as the car allows, whichever aids you drive with. Plan D1:
  aids never split a class, they are stored with each lap and shown next to
  it. Setting them aids-off only would make the medals easier for anyone
  driving with aids off, the opposite of the point.

## What the review found

A review workflow ran one finder and verifiers. Three findings were
confirmed, all low, and all are fixed. Two more were judged "by design";
they are fixed as well, because the design itself changed:

* **The page opened again with nothing new** after WELCOME's *Not now*,
  which shows it without recording it: the first rule remembered only the
  page opened at a session's start. The rule now compares each session with
  the PREVIOUS session, whatever was shown.
* **J did nothing on the page** although its row says "(J)": the menu's keys
  had no J. J now toggles the ghosts there.
* **Nothing checked the session-start decision.** It is one pure function
  now, and its self-check runs ten sessions: first, ballast, dragstrip and
  back, engine, PICK, garage edit, garage return, swarm return, headless.
* Judged by design, changed anyway:
  * **a map change through the dragstrip and back** landed without the page;
  * **PICK of the same car under another name** did too.

  Both open it now, which matches the README's "a new class or build".

## Shape of it

| file | what |
|---|---|
| `drive/prerace.py` | `PreRace.ghosts_on` and the `set:pr_ghosts` row; `seen_key`, `due`, `session_start`; PR_HELP; self-check +2 |
| `drive/drive.py` | `_prerace_sync` sets `ghosts_on`; `_prerace_event` toggles it, and J on the page; `_interactive_session` asks `session_start` (`opts.prerace_seen`, `opts.prerace_force`, set by the garage's return); V32 drives the Ghosts row and J |
| `drive/input.py` | J in `MENU_KEYS` |
| `README.md`, `drive/CONTRACT.md`, `.handoff/README.md` | the J key's pad row, when the page opens, the flow |
