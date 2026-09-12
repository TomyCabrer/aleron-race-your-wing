# Task 3 — "Rear car wing should be able to have pylon or endplate"

`WingSpec.mount` is now a real choice with a real aerodynamic consequence:
`'pylon' | 'endplate' | 'none'` (`drive/aero/wing.MOUNTS`). It applies to any
wing, flank or top, but it is the **top/rear** wing the owner asked about and
that is where the numbers below are measured.

## What was there before

The top wing had exactly one mount treatment and it was implicit: `strut_cd()`
always charged **two pylons** (wetted-area friction on `S_ref`, chord 0.10 m,
× 1.3 form factor), and tip plates were a separate continuous knob
(`plate_h`, 0…0.30 m) with no structural meaning. So "pylon" *was* the
default, unconditionally, and there was no way to say otherwise.

## The model, and where each number comes from

| mount | lattice | drag charged | mass |
|---|---|---|---|
| `pylon` **(default)** | `plate_h` as dialled | `strut_cd`: 2 pylons × `standoff`, wetted-area friction × **1.3** | skins + plates + 2 pylons |
| `endplate` | `plate_h` forced up to `MOUNT_PLATE_H` | no strut term; the plates' own wetted area through the existing `cd_pl` | skins + the (bigger) plates, no pylons |
| `none` | `plate_h` as dialled | nothing | skins + plates |

* The **1.3 form factor** on the pylon is the pylon/wing junction interference
  allowance — Hoerner, *Fluid-Dynamic Drag*, ch. 8: a faired strut junction
  runs 20–40 % above flat-plate friction. It was already in `strut_cd`; the
  commit records what it is for rather than adding a second term, which is
  what keeps the default bit-for-bit.
* The **endplate lift benefit is not a correlation.** The plates are real
  panels in the vortex lattice (`vlm.Lattice(plate_h=…)`), so the reduced tip
  loss falls straight out of the VLM solve. That is much stronger than an
  effective-AR fudge and it is why the model is honest.
* `MOUNT_PLATE_H` = 0.06 m flank / 0.12 m top is an **estimate** (marked as
  such): the plate has to carry the whole wing load in bending into two body
  hardpoints, so it needs a flange deep enough for two fasteners plus edge
  distance. The top wing's is larger because its load is 4–5× the flank
  panel's (594 N at 40 m/s against 130 N).
* `wing_mass()` is a **bottom-up floor, not a weight sheet**: two skins over
  the planform plus two plates at 1.5 mm 2024-T3 (2780 kg/m³ × 0.0015 =
  4.17 kg/m²), plus two 6 × 100 mm bar pylons at 1.6 kg/m when pylon-mounted.
  No ribs, no fasteners, no body reinforcement.

## Measured, on the 1.40 × 0.30 m s1223 top wing at h = 0.45 m

```
  mount      CLa     e       cd0      mass      (wing.self_check)
  bare       4.131   1.199   327 ct   3.24 kg
  endplate   4.595   1.505   340 ct   3.52 kg
  pylon      4.131   1.199   363 ct   4.96 kg   <- the unchanged default
```

The result is worth stating plainly: **on this wing the endplate mount is
lower-drag *and* higher-lift *and* lighter than two pylons.** The 23 counts
the pylons cost buy nothing aerodynamically; the 13 counts the plates cost buy
11 % of lift slope (CLa 4.131 → 4.595) and span efficiency 1.199 → 1.505. It
is not a free lunch in reality — the plates have to carry bending the pylons
carried in compression, and `MOUNT_PLATE_H` is the honest admission that they
must be substantial — but within what this model can see, endplate wins.

## "…and changes where the load is carried into the body"

The owner asked for this and I am **not** modelling it, deliberately. For a
rigid body the transfer path is irrelevant: only the resultant force and its
line of action enter the equations of motion, so whether the load enters
through two pylons on the bootlid or through two plates at the tips makes no
difference at all to load transfer or to the tyre normal loads. Modelling it
would be modelling nothing. What *does* change physically, and is modelled:
the mount's own drag, and the fact that the two mounts package the wing
differently (a pylon mount needs `standoff` clearance above the deck; an
endplate mount does not) — which the user expresses through the slot's `h`.
If the owner wants mount **compliance** (a flexible mount lets the wing move
in pitch under load and sheds downforce at speed) that is a different and
genuinely interesting model, and it is not in this batch.

## Wiring

* `drive/aero/wing.py` — `MOUNTS`, `MOUNT_PLATE_H`, `SKIN_KG_M2`, `PYLON_KG_M`,
  `WingSpec.mount`, `WingSpec.plate_h_flown`, `strut_cd` gated on the mount,
  `wing_mass()`; `analyse()` returns `mount` and `mass`. JSON round-trips.
* `drive/garage.py` — a `mount` row on the DESIGNER page, directly under **end
  plates** (the parameter it interacts with), cycling like the section does.
  It re-analyses on change, because it changes the lattice.
  `CarBuild.hud_kwargs` now carries `dev_mount` / `top_mount` and reports
  `plate_h_flown` rather than `plate_h`, so the renderer draws the plates the
  aero actually flew.
* `drive/render.py` — `HudData.dev_mount` / `top_mount`; the 3-D chase view
  draws the difference (pylon: two struts on the deck; endplate: plates that
  reach the deck and no struts); the HUD's TOP row names it `PYL` / `EPL` / `--`.
* **`drive/vehicle.py` is untouched, and does not need to be.** The mount's
  entire effect is already inside `CZ` and `CD`, which `TopAero.from_aero`
  reads off `analyse()`. That is the contract working as designed: the garage
  hands the physics frozen numbers and the physics never learns what a mount is.

## Evidence

* `python3 -m drive.aero.wing` → ALL PASS, 27 checks (was 21), six of them the
  mount group, including `mount: 'pylon' is the default and reproduces the
  pre-mount analysis` which asserts `cd0`, `CLa` and `cd_strut` are **equal**
  to the pre-mount values.
* `python3 -m drive.render` → 26/26 (was 24/24). `chase: the top wing shows
  which mount it is on` counts strut polygons off the mesh: pylon 12, endplate
  0, and the endplate's plates reach 0.47 m down against 0.17 m.
* `python3 -m drive.garage --selfcheck` → ALL PASS, and the optimiser group is
  still bit-identical (BO 3.270 / random 2.052 / start 0.944).
* `runs/render_chase3d_endplate.png` — the top wing carried on two plates down
  to the deck, no pylons, HUD reading `TOP v 100% FIX EPL`.

## Assumptions / not verified

* The mount belongs to the **wing** (`WingSpec`), not to the slot: one library
  wing bolted on two different cars keeps its mount. Arguable — a case could be
  made for it being a slot property like incidence — but the mount changes the
  lattice the wing is *solved* on, so it has to travel with the wing or the
  cached `spec.aero` would be wrong.
* Every wing already in `runs/library` keeps `mount='pylon'` on load, so no
  saved design changes behaviour.
* Not driven interactively; the mount row's feel and the 3-D plates were
  checked from headless frames only.
* `wing_mass()` is reported but **nothing charges it to the car yet** — task 7
  (adjustable mass) is the natural consumer and should add it.
