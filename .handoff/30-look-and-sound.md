# 30 — the look and the sound

Owner request of 2026-09-23, outside `PLAN-steam-engagement.md`: *"Improve
the graphics (more detail, improve the car chase view, improve how the track
and outside the track looks, improve how the setting looks and polish
details) and improve the sounds."* Numbered 30 so it cannot collide with the
plan's 19–29.

Built by five agents in isolated worktrees (world, props, car + camera,
effects, audio). Each part got an adversarial review, then two fix rounds
with a fresh verifier each, and was then merged. Task 27 (smoke, shake,
chime) landed on `main` while this work was in flight and overlapped it; the
merge keeps **one** system for each (below).

## What changed on screen

* **The world** (`drive/world.py`, `drive/scenery.py`, new):
  * **Sky and landscape.** A sky gradient with clouds, a per-map panorama
    (far range, ridge, tree line) scrolled by the chase camera's *view*
    heading, and grass with mowing stripes.
  * **Beside the road.** Verges; gravel traps outside the slow corners (R <
    60 m, with borders and rake lines); painted run-off outside the fast
    ones; raised kerbs at the apex *and* the exit.
  * **On the tarmac.** A rubbered racing line, repair patches and cracks,
    standing water that reads as water, and inset edge lines.
  * **Paint and haze.** Grid boxes exactly where `race_grid` lines the cars
    up. The skidpad's guide circles and the dragstrip's lane, launch box and
    numerals. A horizon haze.
  * **Plan views.** Grass instead of the dark background, with a sparse
    world-anchored speed cue.
  * **Generic.** It is all laid out from the track's own geometry, so a
    generated track (task 29) gets it too; tested on two.
* **Props** (`drive/props.py`, new):
  * **Circuit.** Tree belts (five species), tyre walls and armco with made-up
    sponsor boards (colour blocks only), catch fences, two grandstands, the
    pit building and race-control tower, the start gantry with lights,
    300/200/100 m boards before the slow corners, marshal posts,
    floodlights and flags.
  * **Other maps.** The open map is a test centre (hangars, tower, windsock,
    fence, cone stacks, car park). The skidpad has a timing hut and masts.
    The dragstrip has its walls, start-light tree, timing boards,
    bleachers and poles.
  * **Clearance.** No collision exists, so every solid prop is ≥ 30 m from
    the edge. The exceptions are thin and justified in the docstring:
    gantry pillars ≥ 4 m, boards ≥ 6 m, dragstrip walls ≥ 8 m, the
    start-light tree ≥ 3 m.
* **The car and the chase camera** (`drive/render.py`):
  * **Body styles.** One per fitted car: the Corsa a 5-door hatch, the MX-5
    an open roadster, the 540i a saloon. They are generic shapes with no
    badges, placed on each car's own a / b / track, in its own colour.
  * **Motion.** The front wheels steer, the rims spin (from `omega`), and the
    body rolls with `phi`.
  * **Lights, detail, shading.** Brake lamps glow, with a reverse light; mirrors,
    sills, grille, plates, exhaust and a spoiler with a third brake light.
    Sky ambient + sun lambert + specular shading, and a soft translucent
    cast shadow.
  * **The camera.** A spring: it trails under acceleration, closes in under
    braking, swings out and looks ~2–3° into a corner, and never rolls the
    horizon. The focal length stays constant (speed widening was tried and
    removed: it forced panorama rebuilds).
  * **Ghosts.** In chase they are translucent low-poly cars that cross-fade,
    never pop. One level with you is drawn before your car with its
    outline, and a near one keeps its outline and label.
  * **Plan view and HUD.** A nicer plan-view car. Rounded HUD panels and a
    segmented rev bar with shift lights; no panel, text or counted colour
    moved.
* **Effects** (`drive/fx.py`, new):
  * **Smoke.** Tyre smoke starts *past* the grip peak, per car (onset |α|
    11.0 / 11.7 / 13.8°, measured with the ramp-steer rig on every car at
    8–40 m/s). A well-driven corner and an ABS stop do not smoke.
  * **Dust and spray.** Dust off the road (grass bits, gravel stones) and
    spray on the wet.
  * **Pool and cost.** One pool of 160 (80 at low detail), resolution-scaled.
  * **Jolt.** The chase camera's jolt is surface-aware: a fine vertical buzz
    on a kerb, bumps on grass or gravel.

## What changed in the sound (`drive/audio.py`, rewritten)

* **Engine.** Per-cylinder firing pulses through a pipe echo and exhaust
  resonances, with flow noise, intake roar and a valve tick.
  * **Per-car profiles**, keyed by `HudData.car_key`: the Corsa's small I4,
    the MX-5's rorty I4, and the 540i's cross-plane V8, whose banks are
    split per stereo side and asymmetric so a mono fold keeps the burble.
  * **Events.** Pops only on a genuine lift from high rpm (a per-lift budget,
    never on a shift's declutch). An ignition-cut limiter and the starter.
    **One** clunk per shift (the physics' g → 0 → g′ is one shift) with an
    upshift cut. A faint final-drive whine, louder in reverse.
* **Tyres and road.**
  * **Tyres.** Scrub from 80 % of the grip (you hear the limit coming),
    squeal past it panned to the sliding side, a harsher lock-up, and ABS
    chatter.
  * **Road, per wheel** from `HudData.surf4`: a tarmac hum, wet spray, a
    kerb's rumble strip at V / 0.35 m, grass swish, gravel crunch.
* **Air and the wing.** Wind with gusts, and the active wing's servo whine
  only while a panel is moving.
* **Cues and mix.** Sector chimes on the flash's rising edge. A bus
  compressor, a DC blocker and a soft clip; stereo when the mixer grants it.
  Every HUD read is guarded, and one non-finite chunk resets the synth (one
  silent chunk, then sound).
* **Cost and start-up.** Synthesis is ≤ 1.5 ms a chunk in the worst case (V8,
  stereo, everything on) and ~0.65 ms typical. `warm_up()` imports
  scipy.signal off-frame at session start.

**Nobody has listened to it.** Every layer was judged by spectra, envelopes
and levels, by the builder and two verifiers. Listen to
`runs/selfcheck/audio_demo_{corsa,mx5,540i}.wav`, which `python3 -m
drive.audio` writes (~38 s each: idle, launch, shifts, a stop with ABS, a
slide, a kerb, the wing moving, the limiter, pops, grass, gravel, wet,
reverse, a stall, the starter). Points to check by ear:

* The V8's stereo width (L/R correlation 0.3–0.5 at WOT).
* The Corsa's overrun at high rpm, which is still close to a tone.

## Where it overlapped task 27, and what was kept

| | task 27 had | now |
|---|---|---|
| tyre smoke | `render.SmokePool`, grey discs | `drive/fx.py` (smoke, dust, spray, depth-sorted, past the grip peak). `Renderer.smoke` is a view of fx's pool, so the full reset's `smoke.clear()` and V22's pool check still hold (V22 now drives a real lock-up, past fx's hold) |
| shake | `hud.shake` → `update_camera(shake=)`; anchor in plan, eye in chase | the same strength and **Settings > Shake**. In chase the motion is fx's surface-aware jolt, applied only while that strength is above 0. Plan views keep `shake_offset`. The kerb test in `hud_data` reads the *drawn* kerbs (`surf4`) |
| lap chime | `CarSound.chime('pb' / 'medal')` from `_rec_lap` | kept as is: it knows a *new best* medal, which a HUD edge cannot. The rewritten audio's own P1 / medal edge chimes were removed; its sector chimes stay |

## Harness hooks (`drive/drive.py`)

* `hud_data`: `surf4 = scenery.wheel_surfaces(track, wheel_world(),
  on_track4, mu, scenery=renderer.cfg.scenery)` and `car_key =
  settings.car`.
* `Settings.graphics`, `GRAPHICS_MODES = ("full", "low", "classic")`:
  * a row on the settings page, live through `Renderer.set_look`;
  * saved;
  * not part of the class (it never discards a lap);
  * the session's ViewConfig is built with `render.look_config`.
  * *Classic* is the original plain look.
* The interactive session calls `audio.warm_up()` at its start.

## Numbers

Frame build, `gfx_shots` (12 scenes × 2 cameras, 1280×800, SDL dummy
driver, machine under the other session's load):

| | original | now |
|---|---|---|
| chase, mean of scene means / worst scene | 9.5 / 17.6 ms | **5.7 / 7.5 ms** |
| plan (car_up) | 3.0 / 4.1 ms | 3.1 / 3.9 ms |

**Why the chase view got faster with everything added.** `_draw_dashes` and
`_draw_kerbs` called the near-plane clipper once per dash and per kerb block,
about 100 numpy calls a frame and ~10 ms in the worst scenes. Every chase
track layer is now projected in one call, and only what crosses the near
plane meets the clipper. The per-layer costs are in `world.py`'s docstring.

**Garbage collection.** A full collection used to stall 3–4 ms. The fix is
`gc.unfreeze(); gc.collect(); gc.freeze()` at the end of
`Renderer.__init__`. At most one stale session is held across restarts:
leak scripts show 1 of 9 old renderers alive, against 9 of 9 with a plain
freeze.

Self-checks:

| module | result |
|---|---|
| render | 53/53 |
| world | 10/10 |
| scenery | 14/14 |
| props | 55/55 |
| fx | 42/42 |
| audio | 50/50 |
| results | pass |
| race_grid | pass |

The gate suite is recorded at the end of this note.

## Known limitations

* **No collision with scenery.** The physics was not touched. The clearance
  rule keeps solid things 30 m out; a car that goes further drives through
  them.
* **Off the road, beside a wall.** 30 m or more out, close to a wall, the
  painter's order can still be wrong for a face that straddles the car's
  depth.
  * A pixel audit of 10,504 off-road poses gives ~57 k px of a wall top
    drawn over the roof and ~37 k px of car showing through.
  * The pre-fix rule gave ~6 k and ~270 k. Two alternatives were tried and
    were worse.
  * Inside a hangar the wall hides the car, which is correct 3-D, but the
    player cannot see the car.
* **Wing deployed at the limit.** Smoke can start a little before the peak
  a_y (22 of 198 rig runs, 1–15 puffs). The fronts really are past their own
  peak there.
* **The sun is never in frame.** `SUN_DIR` is 44° up, as the car was already
  lit; only its glow shows in the sky.
* **Grid bars.** They are placed for the Corsa's nose; a 540i covers them.
* **Timings.** Measured on the SDL dummy driver, without a real window or
  vsync.

## Gates

Run on the merged build (this work on top of `04b99dc`, task 27), load ~4–6:

```
python3 -m drive.drive --self-check   ->  ALL PASS
python3 -m drive.validate             ->  82/82   pass  0 HARD  0 soft   [125.7 s]
python3 -m drive.validate --modules   -> 119/119  pass  0 HARD  0 soft   [423.9 s]  (115 + scenery, world, props, fx)
python3 -m drive.ml                   ->  ALL PASS  32/32
```
