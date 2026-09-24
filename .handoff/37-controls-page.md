# Task 37 — the controls: a DualSense drawn with what every button does

Owner (2026-09-24): *"difficult to see what the ps controls do in
setting(s)"*.

```
python3 -m drive.controls_page        ->  PASS    (new: every bound PS button drawn, and labelled with its command; the rows; inside its rect at three sizes)
python3 -m drive.menu                 ->  ALL PASS
python3 -m drive.drive --self-check   ->  ALL PASS  (V39 new; V20 sha256 ed41b7f781e959ea unchanged; V21 RTF 11.80)
python3 -m drive.validate             ->  82/82  pass  0 HARD  0 soft   [125.4 s]
python3 -m drive.validate --modules   -> 121/121 pass  0 HARD  0 soft   [442.2 s]  (120 + controls_page)
python3 -m drive.ml                   ->  ALL PASS  32/32
```

## What was wrong (seen in headless screenshots)

* **The pause page** listed the keyboard and the DualSense in two help
  columns. With its long item labels, the second column, the pad's, ran off
  the panel's right edge: "throt…", "shift…", the actions cut off.
* **The settings page** showed no pad at all, apart from "ENTER / CROSS" in
  its footer. Its help, every section for every setting at once, ran off the
  panel at the right and the bottom.

## What it does

* **A CONTROLS page** (`drive/controls_page.py`, new), from the pause menu
  and from Settings:
  * **the DualSense drawn:** the shoulder buttons, the d-pad, the four face
    buttons with their symbols in their colours, the sticks, the touchpad,
    CREATE / OPTIONS, the PS button;
  * **a label per control, with a leader line to it:** the button's name
    over what it does while driving (L2 brake, R2 throttle, TRIANGLE the wing
    mode, CIRCLE wings armed, and so on), with *in a menu* below;
  * **a Keyboard keys row** for every key.

  It is drawn through the menu's art hook (task 32) and fits any window: the
  labels' font shrinks to its column. The bindings are `drive/input.py`'s;
  the self-check asserts that every bound PS button is on the drawing.
* **The pause page shows one help column**: the pad's when one is
  connected, else the keyboard's. A line points to Controls for the rest.
* **The settings help's texts were rewritten to fit their column** (about
  45 characters): the CAR and GEARBOX rows used to run off the panel.
* **The settings page's help follows the highlighted row**
  (`Menu.show(help_for=)`, `SETTINGS_ROW_HELP`). It shows what that setting
  does, with its keys and pad buttons (Gearbox: E / Q or R1 / L1 to shift, Z
  or SQUARE the clutch; Camera: C or R3, the zoom keys or L3; Map: TAB), under
  one *On this page* section with the page's own keys, pad buttons included.

## What the review found

A review workflow ran one finder and verifiers. Five findings were
confirmed, one was refuted, and the finder's last one was checked by hand.
All are fixed.

* **My general help-row wrapping broke the garage's menu and the RACE
  page** (high, then medium once verified). It wrapped rows to the narrowest
  column the page could have, about 9 characters in the garage, and pushed
  both pages off the bottom of the screen. It was the wrong fix: the
  settings texts now fit their column on their own. The wrap is gone, and
  those pages draw as before.
* **L3 was labelled "zoom"**; it resets to AUTO zoom. It now says so.
* **The Gearbox help named only S to restart after a stall.** A pad has no
  starter button; the clutch fully in restarts it. It now says both.
* **LEFT / RIGHT on the CONTROLS pages jumped the cursor to the top row**,
  and the next CROSS opened the wrong page. The pages re-show at the cursor
  now; V39 presses LEFT on *Back*.
* **The d-pad was drawn lopsided.** Its arms are symmetric now, with a hub.
* **The labels were copied text** that no check tied to the bindings. The
  self-check now holds each bound PS button's label to its command in
  `PS_PAD_BUTTONS` (a rebinding fails it).
* **A generic (non-PS) pad got the DualSense page silently.** The finding
  was refuted as a defect, but the page now says the connected pad's
  buttons differ, and the pause page lists them.

Screenshots (headless, 1280x800, a PS layout) were checked: the pause page,
CONTROLS, the keyboard's page, and the settings page on its Car and Gearbox
rows. Everything is inside the panel.

## Shape of it

| file | what |
|---|---|
| `drive/controls_page.py` (new) | `CONTROLS` (each control, its anchor on the drawing, its label, what it does), `MENU_PAD`, `pad_rows`, `kb_rows`, `draw_pad(screen, rect)`, `BUTTON_CONTROL`; self-check (every bound PS button drawn, the rows, the drawing inside its rect at three sizes) |
| `drive/menu.py` | `Menu.help_for` via `show(help_for=)`: the help sections for the highlighted row |
| `drive/drive.py` | the pause page's Controls row and its one help column; the settings page's Controls row, `SETTINGS_NAV`, `SETTINGS_ROW_HELP`, `_settings_help`; `_menu_show_controls` / `_controls_kb` and their events (Back returns to the row it came from); V39 |
| `drive/validate.py` | `controls_page` in `MODULES` |
