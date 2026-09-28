# Task 46 — eight asks: tip devices, floating pylons, an oval, a rally car, saved wings and cars everywhere

Owner (2026-09-28), a `/goal`, verbatim:

> *"Wing tip not connected properly to wing in pylon.
> 1 more track 2 straights and 2 very big curves.
> I don't want the buss to big not very useful or the roofless car too short. Instead add a rally car.
> There should be a way to save the wings (same as saving cars) to test them on different cars. (Can only be fitted in cars they fit). Make it easy to see the list of wings (a small diagram)
> .I did a wing for the Renault and when changing car they appear on the other car (Renault wings are very big so didn't fit properly. This happened in challenges. So to fix this, you can only race with cars that have been saved (default cars are also an option)
> I want there in the garage or similar to see the different saved cars.
> And for bot and racing bot should be the one selected by name same as in race
> Also when pushing the wing forward it sometimes is floating in the air. (Make it so that the wing placed both pylons (or in endplate pylons case the endplate) are both connected to a physical part of the car."*

Then: *"I am going to sleep don't ask me anything and finish all the tasks"* --
so nothing below was asked; every choice is listed under *Decisions*.

## 1. The tip device follows its wing (garage.wing_polys, render.wing_mesh3, aero/blend.py)

**Cause.** `blend.plate_rings` swept every plate with its chord along x. A
tip device on a pylon-borne wing caps the TIP SECTION, which is turned by the
wing's incidence (and, drawn in the garage, its twist): on a side wing at
10-12 deg the leading edge poked out past the plate and the plate sat half
off the section (measured: 75-95 % of the tip section on its plate).

**Fix.** `plate_rings(..., chord_dir=)`: a fence / canted / blended tip
device on a pylon (or no-mount) wing is swept in the tip section's own frame
-- its chord along the tip chord line, its line turned the same way. The
plates that CARRY a wing (endplate mount) stay along the flow, as a real
carrying plate is. Drawing only.

Checks: garage `wing drawing (task 46): a pylon wing's tip device caps its
whole tip section` (vertical and canted: every tip-section point on the
plate's outline seen along the span; blended: its first ring along the tip
chord, 12 deg, twist 0 / 3); the older rows now measure a turned plate in its
own frame (`plate_ends(turned=True)`, render `_chord4`), and the "exactly the
old boxes" rows compare against the old boxes turned by the wing's angle.

## 2. Nothing floats (garage.side_land / top_x_floor, render._side_land3)

* **Side-wing struts land on the body.** They used to run to the body's
  widest line at every height, so over a bonnet (the default side slot's
  upper strut is 0.27 m above the Corsa's bonnet at the front wheels) and
  beside the glass they ended in mid-air. `side_land(geo, x, z)` reads the
  drawn section at the strut's station: level with a side (the upright flank,
  the sill's tuck, the glass) the strut is the level box it always was, to
  that surface; above the body's top there it braces down to the nearer top
  corner (a bonnet's wing edge, else the roof edge); below the floor, up to
  the floor corner. The chase view does the same on its own section.
* **Carrying side plates** are set down on the side at their own station
  (the nose and the tail are narrower than the widest line) and height, and
  the stay to the belt stands there too.
* **A top wing wider than the body** on carrying plates set its feet down
  beside the car at its waist, in the air: a bracket now runs in to the side
  (kind `sidebracket`).
* **The top wing goes no further back than where its structure stands on
  the car** (`CarBuild.clamp` via `top_x_floor`): a pylon wing's swan-neck
  feet are a gap and a pylon's width behind its trailing edge, a carrying
  plate's chord overhangs the tip -- the band's old floor (0.15 m inside the
  tail) put a 0.30 m wing's pylons 6 cm behind the Corsa. Pushed against it,
  the garage says `top wing: its pylons must stand on the car - it goes no
  further back`. This is the one change here that can move a saved build (a
  top wing at the very back moves forward to the floor), and so its physics.

Checks: garage `every side-wing strut lands ON the drawn body` (140 struts
over the Corsa's and the Express's whole bands, each body end within 2 cm of
the mesh sliced at its station, 37 of them braced) and `the top wing goes no
further back than where its pylons' feet (or its endplates) stand on the car`;
render `car: each car builds its own style...` now measures each strut's body
end against the drawn section, not the widest line.

## 3. A rally car; the MX-5 and the Citaro retired (lane A)

* `cars.CAR_ORDER = corsa, rally, 540i, express`. **`ESCORT_RS1800`** (`rally`,
  "Ford Escort rally"): Ford Escort RS1800 Mk2, Group 4 tarmac trim (~1979),
  RWD (the driveline refuses AWD), 980 kg, L 2.407 m, 50 % front, BDA 1975 cc
  217 N.m @ 6750 (published) / 180 kW @ 8500 (est), cut 9000, ZF close-ratio
  5-speed x 4.90, 195/50R15, `mu_scale` 1.10 (CALIBRATION); `own_aids`,
  `rev_scaled`, `brk_valve='scaled'` (a tarmac bias, front locks first).
  0-100 6.4 s, top 188.7 km/h (limiter in 5th), 0.93-0.98 g, scripted arena
  lap 60.49 s (Corsa 61.04), LapDriver flying arena lap 58.25 s.
* Its own body (`bodies` style `rally`: 0.19 m ground, flank slot (0.97, 0.90),
  top wing on the boot at (-1.72, 1.40)), factory red with a white livery,
  spot lamps and mud flaps in the chase view, a BDA engine voice, smoke onset
  12.1 deg.
* `cars.RETIRED = {mx5, bus}`: out of `CARS`, read only by the physics
  self-checks that exercise the task-41 optional fields. An old save naming
  them opens the Corsa; `cars.build_car_key` reads a build made for a retired
  (or unknown) car as a Corsa build everywhere. `mx5_arena_plate.json` is
  kept (it drives the session's car). `STOCK_CARS = corsa, 540i`.
* `refs.json`: 128 combos (8 challenges x 4 cars x 4 configs).

## 4. The Fairfield oval (lane B)

`fairfield`, "Fairfield oval": 1902.4778 m anticlockwise, a 480 m pit straight
(the line mid-way), T1 R150 x 180 deg, a 480 m back straight, T2 R150 x 180
deg; **16 m wide** (on 12 m the ML anchor put the 540i off T1: its correction
scales with the road's width, and in a 14 s bend it drifted to full lock).
Sectors 0 / 730 / 1170; patches DAMP_T2 (mu x0.80 into T2's apex) and
WET_T1_ENTRY (standing water, the braking strip). Rolling start 1662.5 m at
22 m/s. props counts a 150-deg-plus bend as a real corner (brake boards,
tyre walls, a stand) -- every other map's props are bit-identical. LapDriver
0.90 and the anchor lap it 24 / 24 in the Corsa / 540i / Express (Corsa
LapDriver 55.6-56.6 s).

## 5. Saved wings and saved cars (lane D, garage.py)

* `K` / `SHIFT+K` (and a pause-menu row) save the selected slot's wing under a
  name, like `S` for a car; `WingSpec.made_for` records the car (old files
  decode as ""; WingLab's commit records it too).
* **SAVED WINGS** (`L`; `SHIFT+L` the old text list): a card per wing with a
  planform + front-view diagram, span, area, the car it was made on, and
  whether it FITS the selected slot of this car (`wing_fit`: the role, and the
  span within this car's limit at the slot's height -- W's rule, x3 under
  Unlimited). A wing that does not fit is dimmed with the reason in red, and
  ENTER says what would make it fit.
* **SAVED CARS** (`G`): a card per saved build with a picture of its car and
  wings (the car page's own drawing, cached), ENTER loads it -- another car's
  build takes the drive's garage to that car with that build
  (`_garage_car(..., build=)`).

## 6. Race only saved cars; bots' cars by name (lane C)

* **Challenges:** the Car row is every car's STOCK car (the config's stock
  wings) and every SAVED build by name, each only on its own car
  (`race_grid.build_home` / `car_choices`); `ch_pick` stays (car, config) and
  `Sim.ch_build` / `opts.ch_build` carry the build. The drive's working build
  never rides into a challenge (the owner's Renault wing on the Corsa); an
  unsaved one is named with "save this car in the garage (S) to race it in
  challenges".
* **Race vs bot / Deploy swarm:** car rows offer its own, same as mine, each
  stock car (now with NO wings -- they used to carry yours) and each saved
  build (`build:<name>`, that build on its own car with its wing mass); a
  bred checkpoint records the saved build so "its own" puts it back.

## Decisions (owner asleep: none asked)

* Tip devices turn with the tip section; carrying plates stay along the flow.
* Floating: side struts brace onto the body where it has no side at their
  height (drawing only); the top wing's rearmost station is the one change
  that can move a saved build (and so its physics).
* The rally car is the RS1800 Mk2 in Group 4 trim (RWD: AWD is not modelled);
  its factory colour is palette red (white would clash with the Express).
* MX-5 / bus specs kept in `cars.RETIRED` for the physics self-checks; their
  records, ghosts and builds stay on disk (a build for them reads as a Corsa
  build and is offered on the Corsa -- lane A's rule kept over lane C's
  "offered nowhere").
* The oval is 16 m wide and its curve patch damp, not standing water (measured
  reasons in `track.py`).
* The TIME TRIAL PICK page still lets a player drive another car's build on
  purpose (tagged, task 41's design); only challenges and bots were changed.

## Integration

Built in four background worktrees from 4c0e453 (lanes A-D) plus the lead's
wing work on main; merged per file with `git merge-file` into main's tree,
which also carried session carsim-cd's UNCOMMITTED tasks 47 (leaderboards) and
48 (the 540i's TC) -- both kept. Seams fixed by hand: `_drive_design`
(a challenge drives its saved pick; else task 47's Wings mode; else fitted),
the RACE label + `own_cfg`, the self-check lists, `race_grid`'s retired-car
rows, garage's fake Settings (`take_car_engine`), README/CONTRACT/title.

## Gates (main's tree after every merge; carsim-cd's tasks 47 / 48 included)

```
python3 -m drive.challenges --measure --write  ->  128 combos: 128 ok, 113 s (only the 540i's 20 moved: task 48's TC)
python3 -m drive.medals --build --workers 8    ->  252 classes with medals (every lap class: 7 lap maps x 4 cars
                                                   x 3 engines x 3 surfaces), 2448 runs, 4748 s; Corsa and Express
                                                   author times bit-identical, 540i moved (task 48), rally + Fairfield new
python3 -m drive.validate                      ->  82/82   pass  0 HARD  0 soft
python3 -m drive.validate --modules            ->  134/134 pass  0 HARD  0 soft  [662 s]
python3 -m drive.drive --self-check            ->  ALL PASS (V20 sha256 unchanged)
python3 -m drive.garage                        ->  195/195 before lane A's merge; in --modules after
python3 -m drive.ml                            ->  36/37: 'the anchor laps the arena in every car' fails on the
                                                   540i (1054 m, offtrack) -- PRE-EXISTING at 4c0e453 (task 45's
                                                   gearbox), not this task's
real launches (scratch copy + the owner's runs/): --render offscreen --car rally --track fairfield,
                                                   --garage (the owner's 540i settings): no traceback
```

Lane diffs (as built, before merging): `.git/task46-lanes/lane{A,B,C,D}.patch`.
Nothing is committed.

## Not done / known

* No bred ML checkpoint for the rally car or the oval (medals use LapDriver
  and the anchor there).
* Wings are saved as copies under new names; a library wing still cannot be
  renamed. A pad can only accept an offered name.
* The MX-5's and the bus's old records and ghosts stay on disk, unlisted.
