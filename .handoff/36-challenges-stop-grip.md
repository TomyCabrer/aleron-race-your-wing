# Task 36 — the challenges: stopping and grip, not straight-line speed

Owner (2026-09-24): *"Challenge mode to reach certain meters in the minimum
time doesn't make sense: aero is built to stop and grip, not to be slick."*

```
python3 -m drive.challenges           ->  PASS    (the new set; a removed id's saved entry never read)
python3 -m drive.drive --self-check   ->  ALL PASS  (V35: all eight to three stars, values = the files'; V20 sha256 ed41b7f781e959ea unchanged; V21 RTF 11.12)
python3 -m drive.validate             ->  82/82  pass  0 HARD  0 soft   [141.4 s]
python3 -m drive.validate --modules   -> 120/120 pass  0 HARD  0 soft   [470.5 s]
python3 -m drive.ml                   ->  ALL PASS  32/32
```

## What changed

Two of task 25's eight challenges measured straight-line speed. Both are
replaced; the other six are untouched.

| out | in |
|---|---|
| `drag_400` **Quarter mile** (402 m from standing, sport engine) | `airbrake_150` **Air brake from 150**: stop from 150 km/h, the tuned Corsa on the dragstrip. Reference: every wing (the plate on both flanks and the rear-s1223 top wing), on the wing mode's AIR BRAKE (task 35), braking from 165 km/h with ABS and full pedal. 3 stars also need no ballast |
| `trap_1000` **Speed at 1000 m** (tuned engine) | `lap_open` **Proving-ground lap**: one valid lap of the open map's road with the 150 hp engine, corners to hold and stops into them. Up to 15 kg of wing; 3 stars with 10.5 kg or less (lap_arena's rule) |

New ids, never the old ones. A player's saved best for `drag_400` (a time in
seconds) must not be read as a stop distance: old entries stay in
`runs/progress.json`, unread (task 25 showed nothing reads an id that is not
loaded).

## The numbers (plan D4: every threshold is the measured reference x 1.12 / 1.06 / 1.02)

| challenge | reference | 1 star | 2 stars | 3 stars |
|---|---|---|---|---|
| Air brake from 150 | 87.89 m (34.8 s sim) | 98.44 m | 93.17 m | 89.65 m |
| Proving-ground lap | 1:06.513 (135.9 s sim) | 1:14.494 | 1:10.504 | 1:07.843 |

(As the challenges page formats them.) The six others were re-derived by
the same `--write` run and came out byte for byte: their files did not
change.

**Why the stop's reference brakes from 165, not at 150.** The meter counts
from the moment the car slows through 150. A reference that presses the
pedal exactly at 150 puts the pedal's and ABS's build-up and the wings'
0.45 s deploy inside the measured distance. A player who brakes a little
above 150 skips all of that. With the first reference, braking at 150
(89.69 m), a wingless car braking early stopped in 89.82 m, which is three
stars: the wings decided nothing. Braking from 165 (the car is fully
stopping, wings out, when it passes 150) measures the settled stop. Still
D4, with no special band.

**What the wings are worth on it, measured.** From 150 km/h, the tuned
Corsa, full pedal, ABS:

| car | wing mode | brakes at | stop | stars |
|---|---|---|---|---|
| no wings | AUTO | 150 | 91.11 m | 2 |
| no wings | AUTO | 165 | 89.82 m | 2 |
| no wings | AUTO | 175 (its best) | 89.74 m | 2 |
| plate pair | AUTO | 165 | 89.90 m | 2 (on a straight the plates are only weight) |
| plate pair | AIR BRAKE | 165 | 88.60 m | 3 |
| plate pair + top wing | AUTO | 165 | 88.65 m | 3 (the active top wing comes out under braking) |
| plate pair + top wing | AIR BRAKE | 165 | 87.89 m (the reference) | 3 |

The third star now takes wings that brake: 2.1 % between the full build
and no wings, settled. The margin for a perfect wingless stop is thin:
89.74 m against 89.65 m.

## Known issue (not this task's, not fixed)

**The open map's tarmac pad is 45 m off the road loop** (`track.make_open`:
the pad's centre is `0.5 * sy`, the loop's is `R + 0.5 * sy`). So on the
Proving-ground lap the top straight and its corners have grass right off
the 12 m road, and the bottom straight has 51 m of tarmac outside it. Running
wide at the top voids the lap or bogs the car; at the bottom it is free. It
predates this task: the open map's time trial and medals have always had
it. The reference follows the centreline and is not affected.

Fixing it changes the open map's surfaces, and with them every open-map
record and the medal table (`drive.medals --build`, a long run). That is
the owner's call.

## What the review found

A review workflow ran one finder and verifiers. Four findings were
confirmed, all low once verified, and one was refuted; plus one extra
finding checked here by hand.
* **The stop's reference measured the braking build-up**, so the wings
  decided nothing and the handoff's numbers undersold them. It now brakes
  from 165 (above).
* **The open map's pad** (Known issue, above): flagged, not fixed.
* **Documentation slips.** The README's list swallowed the shared paragraph
  under its last bullet, two figures disagreed, and two numbers here were
  rounded differently from the page. All fixed.
* **No check covered a saved entry of a removed challenge.** The self-check
  now asserts that `drag_400` / `trap_1000` entries count for nothing and are
  never read.
* **The lap's 3-star rule excluded no wing** (0.4 m^2 a slot fits every
  library wing). The verifier refuted this as a defect, but it was right
  that the rule did nothing: it is now 10.5 kg of wing, lap_arena's rule,
  which excludes the full build.

## Shape of it

| file | what |
|---|---|
| `drive/data/challenges/` | `05_drag_400.json`, `06_trap_1000.json` removed; `05_airbrake_150.json`, `06_lap_open.json` added (thresholds by `--write`) |
| `drive/challenges.py` | the self-check's set: stops (one on the air brake), circles, laps, wet, the open map; the malformed-entry test on a loaded id; a removed id's saved entry is never read |
| `README.md`, `drive/CONTRACT.md`, `.handoff/README.md` | the list, the metrics note, the index |
