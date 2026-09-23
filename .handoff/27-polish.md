# Task 27 — polish

The brief (`PLAN-steam-engagement.md`, T27): a results screen (time, delta to
PB, sectors, medal, top-5 position, an animation for a new record); a PB and
medal chime, procedural in `audio.py` (no asset files); tyre smoke tied to
slip (pooled, budgeted); subtle camera shake on kerbs and off-track, with a
settings toggle. No photo mode. Acceptance: the render and audio self-checks
pass; V22 green.

```
python3 -m drive.results              ->   6/6    (new)
python3 -m drive.render               ->  36/36   (+4: the smoke pool in V22's frame, the smoke, the results card, the shake)
python3 -m drive.audio                ->  11/11   (+1: the chime)
python3 -m drive.drive --self-check   ->  ALL PASS  (V20 sha256 ed41b7f781e959ea unchanged; V21 RTF 11.80)
python3 -m drive.validate             ->  82/82  pass  0 HARD  0 soft   [120.8 s]
python3 -m drive.validate --modules   -> 115/115 pass  0 HARD  0 soft   [384.1 s]  (114 + results)
python3 -m drive.ml                   ->  ALL PASS  32/32
```

## What it does

* **The results card** (`drive/results.py`, new; `render._draw_results`):
  when a timed lap closes, a card drops in at the top centre (under the
  delta) for 6 s -- over the drive, never stopping the car, since a time trial
  is lap after lap: the lap time, the delta to the PB it was driven against
  (green / red; "first lap in this class"), each sector in its colour (purple
  = the class's best ever, green = better than that PB lap's sector, red =
  slower), its place in the top 5 and its medal in the medal's colour. A
  **new PB** drops in with *NEW PB* pulsing gold for 2.5 s. A lap that does
  not count gets a short card with the reason. Built in `Sim._rec_lap` from
  the recorder's result and the book the lap was just filed in; its age runs
  in sim time, so a pause holds it.
* **The chime** (`audio.chime_wave`, `CarSound.chime`): synthesised like the
  engine -- a rising arpeggio of decaying sines with a touch of the octave:
  four notes to the octave (E5 G#5 B5 E6) for a new PB, three (C5 E5 G5) for a
  new best medal (which is always a new PB too: the rarer one sounds) -- once
  per lap, on mixer channel 1, over the engine's stream on channel 0,
  at the session's volume. Peak 0.40 (under the engine's 0.62), 0.77 s.
* **Tyre smoke** (`render.SmokePool`): a FIXED pool of 160 particles; each
  frame at most 2 new ones, taken round the tyres that slide (|slip ratio| >
  0.15 or |slip angle| > 0.14 rad, above 4 m/s), at their contact patches,
  each living 1.3 s (2 x 60 x 1.3 = 156: a full pool never recycles a puff
  still fading), growing and fading into whatever is under it (its colour is
  read from the frame), drawn under the car -- plan views at the view's px/m,
  the chase view by each puff's camera depth, culled at the ground-near plane
  and capped in size. A long slide costs exactly what a short one does. A
  pause freezes it; a full reset clears it with the skid marks.
* **Camera shake** (`render.shake_offset`, `HudData.shake`): a deterministic
  mix of three frequencies, at most 2.5 px: 0.35 of it with a wheel on a
  kerb (the corners' inner 0.8 m of the ribbon, `render._draw_kerbs`' rule)
  or off the ribbon, all of it with four off, both scaled by speed (full at
  20 m/s); none under a pause. The plan views move the view's anchor,
  the chase view the eye by a few centimetres; the HUD, the menus and the
  minimap stay put. **Settings > Shake** (On / Off, saved, applied at once;
  not part of the class, so it never costs a lap its record).

## Numbers

* V22 (the busiest time-trial frame, now also with four tyres smoking: 140 of
  160 puffs live): mean **5.05 ms**, p99 **7.22 ms** against the 12 / 16 ms
  budget (task 26's frame without smoke: 4.41 / 5.21; reading each puff's
  background colour is most of the difference).
* The chime: first note measured at 656 Hz (E5 = 659 Hz, the FFT bin), peak
  0.40, the tail at 0.023 after 0.77 s.

## What the review found

Two lenses (in play; rendering cost and checks), 2 finders and 10
verifiers: seven confirmed, all fixed in this commit.

* **Riding a kerb never shook** (medium): the kerbs are painted ON the ribbon,
  so the "wheels off the ribbon" test missed them; a wheel on a kerb strip
  (the renderer's own kerb rule) now counts. Probed: a car on turn 1's inside
  kerb, all four wheels on the ribbon, shakes at 0.35.
* **The chase view's smoke was ~12x too small** (medium): it was scaled by
  the distance to the cull point 70 m ahead, not the eye; now by camera
  depth, culled at the ground-near plane, capped.
* **Under a pause the view kept shaking and the tyres kept smoking**
  (medium, two findings): both frozen now.
* **A lap whose filing failed ran the card and the chime twice** (medium):
  the filing thread's repeat is the note only.
* **The medal chime could never sound** (low): a new best medal is always a
  new PB; the medal chime now wins.
* **Old puffs off the road were tarmac-coloured discs** (medium): each fades
  into the colour under it.
* **A saturated pool recycled puffs half-way through their fade** (low): the
  emit cap and life now fit the pool.

Refuted: that the chase shake exceeds its limit (the eye moves by a few
centimetres; SHAKE_PX bounds the plan views, the chase view is by design an
eye offset).

## Deviations

* **The results screen is a card, not a page**: a page would pause the
  drive after every lap of a time trial, and the next lap is already
  running; the TIME TRIAL page (ESC) keeps the full table.
* **"On kerbs"**: the sim has no kerb surface (the kerbs are paint on the
  corners' inner 0.8 m of the ribbon), so a wheel is "on a kerb" by the
  renderer's own kerb geometry; a wheel over the ribbon's edge anywhere
  shakes the same.

## Shape of it

| file | what |
|---|---|
| `drive/results.py` (new) | `card`, `view`, `drop`, `pulse`; self-check |
| `drive/audio.py` | `CHIMES`, `chime_wave`, `CarSound.chime`; the chime check |
| `drive/render.py` | `HudData.results` / `wheels_xy` / `shake`, `R_RESULTS`, `SmokePool`, `shake_offset`, `update_camera(shake=)`, `_draw_smoke`, `_draw_results`; V22 with the smoke; the smoke / card / shake checks |
| `drive/drive.py` | `Settings.shake` and its row; `_rec_lap`: the card and the chime; `hud_data`: the card, the patches, the shake; `run_interactive` hands the shake to the camera; a full reset clears the smoke |
| `drive/validate.py` | `results` in `MODULES` |
| `drive/CONTRACT.md`, `README.md` | module map, the flow, the Shake setting, the paragraph |
