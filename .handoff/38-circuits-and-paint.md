# Task 38 — three new circuits, and a paint for every car

Owner (2026-09-24): *"Add more circuits similar astetics and abiilty to give
cars different colours"*.

```
python3 -m drive.track                ->  ALL TRACK CHECKS PASS (+ each circuit, closure re-solved)
python3 -m drive.paint                ->  PASS  (new)
python3 -m drive.render               ->  55/55 (+1: paint)
python3 -m drive.props                ->  82/82 (was 55: the three circuits)
python3 -m drive.medals               ->  8/8   (189 classes, 162 with an author time; current)
python3 -m drive.drive --self-check   ->  ALL PASS  (V41 new, V26 relative to TRACK_ORDER + paint)
python3 -m drive.validate             ->  82/82  pass  0 HARD  0 soft   [122.7 s]
python3 -m drive.validate --modules   -> 122/122 pass  0 HARD  0 soft   [443.5 s]
python3 -m drive.ml                   ->  ALL PASS  36/36 (the anchor laps every circuit in every car)
```

## The circuits

Three closed circuits in the arena's own look. Nothing in the scenery was
drawn for them: world, scenery and props lay a circuit out from its geometry
(kerbs, gravel under R 60, run-off, grid boxes, pits, stands, tyre walls,
armco, boards, trees), seeded by the map's name, so each gets its own
dressing, clouds and ridges in the same style.

| map | title | length | turn | corners | pit straight | wet patches | sectors |
|---|---|---|---|---|---|---|---|
| `linden` | Linden park | 1110.4021 m | anticlockwise | 7, R 30..60 m: tight, technical | 150 m | WET_T5 850-905 full width; WET_T2_ENTRY 400-435, n -6..0 | 0 / 370 / 690 |
| `kestrel` | Kestrel ring | 1913.3440 m | anticlockwise | 7, R 55..100 m: fast | 390 m (320 m back straight) | WET_T2 466-544; WET_T4_ENTRY 830-870, n -6..0 | 0 / 610 / 1140 |
| `ashdown` | Ashdown circuit | 1390.0362 m | **clockwise** | 7, R 30..120 m, a hairpin | 211 m | WET_T5 830-886; WET_T4_ENTRY 495-530, n 0..6 | 0 / 505 / 930 |

* **Registry** (`drive/track.py`). `TRACK_ORDER` is now arena, linden,
  kestrel, ashdown, open, skidpad, dragstrip: the circuits sit together, so
  TAB from the arena goes to the next circuit. The new `CIRCUITS` tuple names
  the race circuits (`Track.closed` cannot: the open map and the skidpad are
  closed too). `make_track` dispatches every name and honours `surfaces`.
  An unknown name still builds the arena, because `evaluate` hands it
  `'arena,open'`.
* **Geometry.** Straights and arcs like the arena, closed by two solved
  straights stored to 7 dp (closure 2e-8..6e-8 m). `solve_closure` re-derives
  them in the self-check (`CLOSURE_FREE`). Every circuit is 12 m wide, its
  line mid-straight with ≥ 37 m of R > 80 behind it (six painted grid rows,
  the race grid's three), sections ≥ 60 m apart (`CLEARANCE_MIN`, so two
  24 m run-offs never meet), and every wet patch covers n = 0 (the bots and
  the anchor sample grip on the centreline only).
* **Kestrel was reshaped once.** The first draft (357 / 376 m straights into
  R 110 / R 90, and later an R 130 T3 into an R 45 T4) put the anchor, then an
  MX-5 with LapDriver, off the road. T3 is now R 100 and T4 R 55 after a
  140 m straight; its neighbourhood was swept (T3 R 90-110, the straight
  120-160 m, T4 R 45-60) and keeps every class the chosen shape laps.
* **LapDriver 0.90 laps all three in every car**, stock engine, patches on and
  off, aids on and off: 54 / 54 clean flying laps, max |n| 4.18 m. Flying
  laps, patches, aids off, Corsa / MX-5 / 540i: Linden 59.1 / 56.9 / 57.6 s,
  Kestrel 79.0 / 76.7 / 77.7 s, Ashdown 65.0 / 63.5 / 63.4 s.
* **Frame build** (chase / car_up, ms): arena 5.72 / 2.96, linden 5.57 / 2.87,
  kestrel 4.97 / 2.59, ashdown 5.04 / 2.69.

### Everything that knew the map list by name

* `records.LAP_TRACKS`: records, the pre-race page, ghosts and the results
  card work on the circuits.
* `medals.T_MAX`: linden 280, kestrel 460, ashdown 340 s (the slowest
  runs, LapDriver 0.60 all-wet: 252 / 333 / 278 s). The build schedules the
  longest laps first.
* `challenges`: the lap-time whitelist is `LAP_TRACKS`. No challenge file
  was added (the counts are fixed at 8).
* `aero/mission.TRACKS`: the garage optimiser can target the circuits.
* The self-checks that listed maps: scenery (the themes by name, the grid
  check on every circuit), props (`REQUIRED` and the names: 55 -> 82), world
  (the clouds on every map).
* `drive.ml`: the anchor laps each circuit twice in every car. The RACE
  page's bot Test runs `evaluate.bot_test_T(tr)` seconds: 150 on the arena and
  Linden, 167 on Ashdown, 230 on Kestrel (and 197 on the open map, whose
  perimeter is longer than the arena). It scales up by lap length, never
  down, and the page prints that figure. `bot_lap` is still handed the arena
  budget, so nothing is scaled twice.
* `drive.py`: the headless default driver is `LapDriver` on every
  `CIRCUITS` map (it was PathFollower off the arena, and it left the road at
  Linden's and Ashdown's slow corners), built with the session's surfaces
  and car. `--script lap` drives the requested circuit. The Deploy-swarm
  page's default Sim time scales with the circuit (`swarm_T`: Kestrel 107 s),
  and a time the player set is kept when the map changes
  (`_swarm_menu_kept`).

### The medal table was rebuilt

`python3 -m drive.medals --build --workers 7`: 1626 runs, 3273 s. All
189 classes: 162 with an author time (every lap class, the new 81 included),
27 dragstrip 'no lap on this map'. **The 108 old entries and their 81
reference traces are bit-for-bit unchanged** (the build is deterministic).
New author times come from LapDriver (70), its 0.80 margin (2) and the
anchor (9). No bundled bot names a new circuit, so these medals can be a
little softer than the arena's, where bred bots set some author times.
Inputs hash `8168f04119ad`. `reference_laps.json` is 1.8 MB (was 0.7).

## The paint

**Settings > Paint**, under Car: the car on the road's paint, one per car,
saved. LEFT / RIGHT / ENTER change it live: the car repaints on the next
frame, nothing restarts, and the lap being timed is kept. A swatch beside
the row shows the colour.

* **Palette** (`drive/paint.py`, new, pure data): *factory* (each car's own:
  the Corsa yellow, the MX-5 red, the 540i blue), signal yellow, rosso red,
  estoril blue, arctic white, silver, racing green, midnight purple, cobalt
  blue, teal, burgundy.
* **Why fixed, not an RGB picker.** The self-checks count exact colours on
  screen. A naive white (232,234,238) or silver (174,180,191) puts 53 / 12
  stray px of HUD text colour into the tutorial box's strip, and orange,
  sky blue, violet, rose, lime, mint and light grey are the bots', the
  ghosts' and the wing's. Every paint is checked at every plan tone against
  them, and it stays ≥ 40 units from the bots, the ghosts and the wing (the
  closest pair: arctic white vs the REF ghost, 30).
* **Looks only.** Paint is never in the class key, a ranking, a medal, a
  ghost, the CarSpec or the build JSON. A lap record's settings snapshot
  lists it, as it lists Graphics, and nothing reads it back.
* **Where it shows.** The chase car, the plan-view car, the minimap dot (it
  was always Corsa yellow; now the car's colour, with a light ring so a dark
  paint reads) and the garage's car. Both the full and the classic looks.
  The bots and ghosts keep their slot colours. The swarm viewer is factory.
  Headless and scripted runs never read the paint, so every acceptance
  number and screenshot is as before.
* **How** (`drive/render.py`). `set_paint(rgb)` (None = factory),
  `Renderer.set_paint` for the live change, `factory_colour(car)` (a spec or
  a key). The colour is in `CarGeom.key` and `car_geom()`'s key, so every
  mesh and plan cache keyed on them refreshes; the ghost mesh and the shadow
  hull key on the new `shape_key`. The plan tones go through the
  reserved-colour guard, which now includes the HUD text colours.
* `drive/garage.py`: `build_car_mesh(paint)`, `GarageView.set_paint`,
  `Garage.set_paint`. `drive/menu.py`: `Menu.show(swatches={action: rgb})`
  draws a swatch after that row's label.

## Found on the way (task 35)

**`python3 -m drive.drive` crashed at launch** on main since 95d7baa
(task 35): `_interactive_session` read `opts.wing_mode` when only the
restart loop sets it (`AttributeError: 'Namespace' object has no attribute
'wing_mode'`). Fixed here with a `getattr`, one line; V41 builds a first
session the way `main()` does and would have caught it.

## What the reviews found

Each part was built by an agent and checked by an independent adversarial
verifier (then fixed and re-verified), and a last three-lens review of the
whole diff verified every finding with a skeptic.
* **Kestrel's T4** lost an MX-5 with aids off on the dry surface: the
  reshape above.
* **`factory_colour('mx5')` returned Corsa yellow** (it read `.name` off a
  key string): `car_style` now takes a key; the swatch and the garage would
  have shown a factory MX-5 yellow.
* **The Deploy-swarm Sim time** carried from one map to the next after the
  first Deploy: `_swarm_menu_kept`.
* **Doc numbers**: Kestrel's pit straight is 390 m, not 320; paint is in a
  record's settings snapshot; the reference lap is ~12 KB.

## Shape of it

| file | what |
|---|---|
| `drive/track.py` | the three circuits (segments, origins, patches, factories with `title`), `CIRCUITS`, `CLOSURE_FREE`, `CLEARANCE_MIN`, `solve_closure`, `make_track` dispatch; self-check per circuit (closure re-solved, length, round trip, clearance, radii, sectors, grid zone, patches on the centreline) |
| `drive/paint.py` | new: the palette, `rgb()`, self-check (in `validate.MODULES`) |
| `drive/render.py` | `set_paint`, `Renderer.set_paint`, `factory_colour`, `car_style` by key, paint in the geometry keys, `shape_key`, the plan-tone guard, the minimap dot; self-check +1 (55) |
| `drive/garage.py`, `drive/menu.py` | the garage car's paint; the swatch side channel (+1 menu check) |
| `drive/drive.py` | `Settings.paint` (per car), `paint_of`, `paint_rgb`, the Paint row, its row help, live apply, the session / garage / swarm-viewer paint; the circuits in the help, the headless and lap-script drivers, the bot Test's duration, the swarm Sim time; V26 relative to `TRACK_ORDER` plus paint; **V41** (paint on the road, in the garage, the swarm viewer, the swarm time) |
| `drive/records.py`, `medals.py`, `challenges.py`, `scenery.py`, `props.py`, `world.py`, `aero/mission.py`, `ml/__main__.py`, `ml/evaluate.py` | the map lists above |
| `drive/data/medals.json`, `reference_laps.json` | rebuilt |
| `README.md`, `drive/CONTRACT.md` | the maps, the Paint row, the APIs |

## Left open

* The garage preview is still the old hatch shape for every car (it was
  before); it is painted now.
* No bot has been bred on a new circuit. The bundled bots race there as
  transfer ("never seen this track").
* Author times on the circuits come from LapDriver and the anchor only.
