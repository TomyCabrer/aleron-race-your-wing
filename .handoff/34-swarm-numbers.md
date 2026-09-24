# Task 34 — the swarm's numbers: fixed stops, or type them

Owner feedback on task 26 (2026-09-24): *"when determining the amounts of
bots easier to fixed numbers or let the user write it (if not going 1 by 1
to 128 takes a lot of time)"*.

```
python3 -m drive.swarm_panel          ->  PASS    (the fixed numbers, the cycle, typing and BACKSPACE, the row)
python3 -m drive.menu                 ->  ALL PASS (+1: a digit reaches only a declared row)
python3 -m drive.input                ->  ALL CHECKS PASS
python3 -m drive.drive --self-check   ->  ALL PASS  (V40 new; V20 sha256 ed41b7f781e959ea unchanged; V21 RTF 11.83)
python3 -m drive.validate             ->  82/82  pass  0 HARD  0 soft   [137.3 s]
python3 -m drive.validate --modules   -> 119/119 pass  0 HARD  0 soft   [544.1 s]
python3 -m drive.ml                   ->  ALL PASS  32/32
```

## What it does

On the Deploy-swarm page, **Cars** (the population) and **Sim time** (the
seconds each car gets):

* **LEFT / RIGHT jump between fixed numbers** (the pad's d-pad too):
  * Cars: 4, 8, 16, 24, 32, 48, 64, 96, 128;
  * Sim time: 20, 30, 45, 60, 70, 90, 120, 150, 180, 240 s.

  From a value typed between two, they go to its neighbour on that side, and
  they never wrap (RIGHT on 128 stays 128).
* **ENTER / CROSS cycles** the same numbers and wraps round to the first.
* **The keyboard types any value.** The digits 0-9 (the number row or the
  keypad) on either row set the number directly:
  * up to three digits (a fourth starts a new number);
  * **BACKSPACE takes a digit back**, of the number being typed or of the
    row's value (on these two rows it is not the garage's hotkey);
  * the value is clamped into the range as you type.

  The row shows what you typed, and the clamped value when it is out of
  range: `Cars  1_ = 4`, then `12_`, then `128_`. The number lasts until you
  act: ENTER sets it (no cycle), and LEFT / RIGHT, moving to another row or
  leaving the page end it (the row drops its `_`, and the next digit starts
  a new number). There is no clock.
* The digits are routed only to rows a page declares
  (`Menu.show(typed=...)`); on every other page they do nothing.

## What the review found

A review workflow ran one finder and verifiers. Three findings were
confirmed, all low once verified, and all are fixed.
* **After a pause of 3 s, ENTER cycled to the next fixed number** while the
  row still showed the typed one. The first design ended a typed number
  after 3 s idle, but nothing redrew the row. There is no clock now: what the
  row shows is what ENTER sets.
* **BACKSPACE, the natural key to fix a digit, went to the garage.** It is
  the menu's garage hotkey, so it ended the session and the page's values
  were lost. On the two number rows it now takes a digit back.
* **Moving off the row and back continued the old number.** A move now ends
  it, and the row is redrawn.

The review also pointed out that V40 depended on the wall clock. That was
refuted, and moot now: there is no clock.

## Shape of it

| file | what |
|---|---|
| `drive/swarm_panel.py` | `POP_PRESETS`, `T_PRESETS`; `step_value` (the next fixed number), `cycle_value`, `Typed` (`digit`, `backspace`, `shown`, `clear`), `row_value`; self-check rewritten for them (+3) |
| `drive/menu.py` | `Menu.typed` via `show(typed=)`; `handle('digit:N')` -> `'type:N:<action>'` on such a row; self-check +1 |
| `drive/input.py` | the digits (and the keypad's) in `MENU_KEYS` as `digit:N` |
| `drive/drive.py` | `Sim._swarm_typed`; `_swarm_row`; `_swarm_step` on the fixed numbers (ENTER after typing sets); the `type:` action; BACKSPACE on a number row; a move ends the number; the page's help and footer; V40 (the steps, typing, ENTER after typing, the clamp, BACKSPACE, a move, Sim time) |
