# Task 32 — the results card, in the HUD's look, and kept

Owner feedback on task 27 (2026-09-24): *"It should be a card following the
aesthetics but with the possibility of seeing it in settings page."*

```
python3 -m drive.results              ->  PASS    (+3: summary, page_rows, the kept numbers / other class / long reasons)
python3 -m drive.render               ->  54/54   (+1: the panel look, the size, the settled gold edge, every kind of card inside itself)
python3 -m drive.menu                 ->  ALL PASS (+1: a page's own drawing)
python3 -m drive.drive --self-check   ->  ALL PASS  (V37 new; V20 sha256 ed41b7f781e959ea unchanged; V21 RTF 11.73)
python3 -m drive.validate             ->  82/82  pass  0 HARD  0 soft   [124.0 s]
python3 -m drive.validate --modules   -> 119/119 pass  0 HARD  0 soft   [419.8 s]
python3 -m drive.ml                   ->  ALL PASS  32/32
```

(On the first run, before the review fixes, `drive.audio` gave one SOFT miss:
its synthesis-time budget, p99 1.52 ms against 1.5, while the machine was
under load. It passed 50/50 standalone and on the final run. That module is
handoff 30's; this task does not touch it.)

## What it does

* **The card in the HUD's own look** (`render.Renderer.draw_card`,
  `card_size`). Task 27 drew the card as a flat dark square box. Task 30 then
  gave every HUD panel rounded, top-lit, hairlined corners
  (`_hud_panel_surface`), and the card was the one panel left out. The card
  now uses that panel. It sits under the delta and is exactly as wide as
  the timing panel above it (`R_RESULTS = (460, 196, 360, 140)`). Its height
  follows its rows: 100 px for a lap, 54 px for one that does not count.
  * **Header.** *LAP*, *NEW PB* or *LAP NOT COUNTED* (in red), and on the
    right the medal as a tag in its own colour.
  * **Time.** The lap time large, with the delta (green / red) and the place
    on its baseline. *first lap in this class* goes in the header, as words.
  * **Sectors.** Each sector over a 4 px bar in its colour (purple = the
    class's best ever, green = better than the PB lap's, red = slower, grey =
    nothing to compare with).
  * **NEW PB.** The header and the panel's edge pulse gold (five cached
    shades), and the card drops in as before.
* **Settings > Last lap** (only in a session with records): the row shows the
  last lap in one line (`1:00.900  -0.200  P1  GOLD  NEW PB`). `ENTER` opens
  the **LAP RESULTS** page:
  * the last card itself, drawn by the renderer into the page, settled (no
    drop; a NEW PB's edge gold);
  * *THIS SESSION*: the session's last ten laps, newest first, with those
    that did not count and why;
  * `ESC` / *Back* return to the Settings row.
* **The page's drawing** is a small generic hook: `Menu.show(art=f,
  art_h=px)`. `f(screen, rect)` gets `art_h` px at the top of the first help
  column and the sections go below it. Every `show()` resets it, and a
  drawing that raises is dropped with a note, never taking the menu down.
* **The tutorial / challenge box** (the same `_draw_tutorial`) gets the same
  panel, keeping its yellow strip, now inside the left edge.

Screenshots (headless, 1280x800) were checked: the card in chase and plan,
NEW PB, a slower lap, a lap that does not count, the tutorial box, and the
LAP RESULTS page after five laps.

## What the review found

A review workflow ran two finders, one on the page flow and state, one on
the drawing, and an adversarial verifier on each of their top findings.
Seven were confirmed, all low once verified; all are fixed here, along with
the four the verifiers had no time for.

* **Lap numbers.** THIS SESSION renumbered the laps once more than ten were
  kept (the 15th lap showed as "lap 10"). Each kept card now carries its
  number in the session.
* **A failed save carded one lap twice.** A lap outside the top 5 whose
  file write failed was carded again by the filing thread's repeat. Older
  than this task, but the list made it stick. The repeat now says so
  (`records`: `refiled`) and gets the note only.
* **A live engine change mixed classes in the list** under one class's
  name. A row from another class now says whose engine.
* **The Settings row was the widest item** and the menu's `v N more` hint
  ran over it. The row is short now (time, delta, medal). A menu that scrolls
  now makes room for its hint beside the widest label; that is general, the
  pause and PICK pages had it too.
* **"first lap in this class" ran under a six-letter medal tag** at some
  window sizes. The header words are cut to what clears the tag.
* **A long reason ran off the page and out of the card** (a recorder
  "snapshot failed (...)" path). Page rows are cut at 60 characters. The
  card cuts a word too long for its line with `...`, and keeps at most
  three lines.
* **The page's drawing ignored the surface it was handed.**
  `draw_card(surf=)` now draws where it is told.
* **No drive-level check covered the page. V37 does:**
  * no row without records;
  * thirteen laps, the refiled repeat not carded, kept as 4..13;
  * ENTER opens the page, newest first;
  * the card goes through the art hook onto the given surface;
  * ESC returns to the row;
  * R on the page drives on.
* **The render check was weak.** It now also draws every kind of card
  (lap, not counted, first in class with a bronze tag, a 200-character
  reason) onto a blank surface and asserts that nothing lands outside the
  card.

Refuted: that a restart emptying the list is a defect. A restart is a new
session, as the lap counter is (see Deviations).

## Deviations

* **"Seeing it in settings page"** is read as *seeing the card again from
  the Settings page*: a row there opens it. An on / off switch for the card
  is one more row if wanted; it was not asked for.
* **"This session"** is the running session: a restart (a map or car
  change) starts a new list, as the lap counter does.

## Shape of it

| file | what |
|---|---|
| `drive/render.py` | `R_RESULTS` (the timing panel's x and width), `CARD_BAR`; `_draw_results` -> `card_size`, `_card_why` (cut, cached), `draw_card(surf=)`; `_draw_tutorial` on `_hud_panel_surface`; self-check +1 (the panel look, the size, the settled gold edge, every kind inside its card) |
| `drive/menu.py` | `Menu.art` / `art_h` via `show(art=, art_h=)`, drawn at the top of column 0; a scrolling list makes room for its `v N more`; self-check +1 |
| `drive/results.py` | the card's `key` and `n`; `summary(short=)`, `page_rows(log, key)`, `LOG_N`, `ROW_CHARS`; self-check +3 |
| `drive/drive.py` | `Sim._results_log`, `_results_count`; `_rec_lap` keeps each card, numbered, and skips a `refiled` repeat; the Settings row `lap_results`; `_menu_show_results` (page `'results'`); its events; V37 |
| `drive/records.py` | the filing thread's repeat of a failed save is marked `refiled` |
| `README.md`, `drive/CONTRACT.md`, `.handoff/README.md` | the Settings table, the results paragraph, the module map, the menu hook |
