# Task 14 — "the wing design should follow exactly the same procedure as the UROP"

The UROP tool is `~/dev/urop-bo-aero` (AeroBO). `drive/aero/` was already a
port of its car-wing *physics* — the README claims the lattice to 1e-12 and it
holds. What did **not** match was the *procedure*: the order the design is
arrived at, and the vector it is arrived at over.

## 1. What the two procedures were

AeroBO designs a car wing in three steps, and `cartrack.py` / `carsection.py`
each open by arguing for their own step:

| | AeroBO | carsim, before |
|---|---|---|
| **mission** | `cartrack.py`: a LAP. Its docstring measures the alternative — a `CD_budget = 0.11` allowance admitted **1857 of 1857** feasible draws of the car-wing box, i.e. decided nothing, while a two-point requirement pair admitted 5 | nothing. The design speed was `V_REF[role]`, a module constant; the requirement pair sat *below* the planform rows, inside the OPTIMISER group |
| **section** | `carsection.py`: designed in 2-D, "where a candidate costs milliseconds", and the wing asked afterwards whether the winner helped | the SECTION rows were first on the page already |
| **wing** | `carwing.py:evaluate_car_wing` | the DESIGNER page's PLANFORM rows |

And the design vectors were different shapes:

```
AeroBO   x = [taper, twist_root, twist_tip, alpha, endplate_h, ride_height, (S,) b]
carsim   x = [span, chord, taper, twist_tip, plate_h, inc]
```

Five differences, not one: **two** twist rows against one, a **ride-height**
row against none, **area + span** sizing against **span + root chord**, the
span row meaning overall width **including plates**, and the reverse ordering.

## 2. What changed

**`drive/aero/wing.py`.** `DESIGN_VARS` is now AeroBO's order, and
`design_table(role, area)` is the one place the vector is assembled:

```
taper, root twist, tip twist, incidence, end plates, ride, (reference area,) span
```

* **`twist_root_deg`** is new. The lattice's twist law was `tw * eta` — zero at
  the root by construction. It is now `tw_root + (tw_tip - tw_root) * eta`,
  AeroBO's two-row law, of which the old law is exactly `twist_root_deg = 0`.
* **`ride_h`** is new: the distance to the wall the wing is imaged in. This is
  AeroBO's `ride_height_m` in the same role — the gap ground effect is a
  function of — and it is deliberately **not** the slot's mount height. For a
  top wing it is the track (and the slot's `h` *is* that number, so the row
  writes both and the slot keeps no second opinion). For a flank panel it is
  the deployed standoff to the car's own flank. The mount height stays a
  packaging row: it sets the roll arm and the sill/roof span fit, and the image
  plane never sees it.
* **root chord stops being a row.** AeroBO sizes by area and span and lets the
  chord fall out (`c = 2S / (b (1 + taper))`); carrying a free chord *alongside*
  a free span and a reference area would let a candidate be scored against an
  area it does not have. `apply_design` derives it, and the page shows it and
  the reference area as read-only.
* **the area row** appears only when a band is declared (`area=True`), one row
  ahead of the span, exactly as `CarWingProblem` does it — and the note that
  the score then has to be read in forces travels with it.
* **endplate cant and blend** are NOT rows, which matches AeroBO: they are
  problem settings there too. At its published cant of 90 deg with no blend,
  `developed_semispan` reduces to the span itself, so carsim's span row already
  means what AeroBO's does. Nothing to change, stated so it is not re-checked.

**`drive/garage.py`.** The DESIGNER page lists the vector in that order, under
one `PLANFORM (the design vector, in order)` label — the slot's **incidence**
sits in the middle of the planform rows because that is where AeroBO's
`alpha_deg` sits. Above the section rows there is now a **MISSION** group:
design speed, force floor, drag cap. The optimiser reads `self.V_design`, not
`V_REF[role]`, and its group no longer re-states the requirement.

## 3. The bug the exercise exposed — the flank standoff had three values

The gap the flank panel's body image is built across was written down three
times, and they disagreed:

```
drive/aero/wing.py    FLANK_STANDOFF      = 0.45      the module default
drive/garage.py       DEV_OUT0 + DEV_OUT1 = 0.60      passed on every designed wing
drive/aero/library.py standoff            = 0.45      a third default, passed explicitly
```

So a panel's lattice numbers depended on which door it came through, and
`library.analyse_wing` overruled whatever the wing had been designed at. A
design *row* cannot carry three defaults. They are one number now and it is the
one the car deploys to, **0.60** — `DEV_OUT0 + DEV_OUT1`, the same constants
the renderer draws the deployed panel with.

`TOP_PYLON_L = 0.45` is split out and held at the old value. A top wing's
`strut_cd` and `wing_mass` were charging a pylon of the *flank* standoff, which
the top wing's lattice never reads; unifying the flank's number would have
moved every top wing for no reason. It is **unmodelled** and says so: a real
pylon is `ride_h - deck_z(x)` long and shortens as the wing comes down, which
is what `carwing.MountSpec` carries. Separate change, separate numbers.

## 4. The numbers

Every library wing, `analyse_wing` at its own design point:

```
wing          role    what moved
------------  ------  ----------------------------------------------------------
rear-new      top     UNCHANGED
rear-s1223    top     UNCHANGED
flank-e423    flank   CL0   0.5143 -> 0.5016  (-2.47%)   e     1.2861 -> 1.2365  (-3.86%)
                      CLa   2.7871 -> 2.7181  (-2.48%)   cd0   0.0970 -> 0.0975  (+0.52%)
                      CL_max 2.0868 -> 2.0890 (+0.11%)   cd2   0.1785 -> 0.1836  (+2.86%)
                      cd_strut 0.0015 -> 0.0021 (+33.4%) mass  4.4260 -> 4.9060 (+10.85%)
flank-new     flank   CL0   0.7309 -> 0.6867  (-6.05%)   e     1.6064 -> 1.4852  (-7.54%)
                      CLa   2.8699 -> 2.6959  (-6.06%)   cd0   0.0172 -> 0.0175  (+1.62%)
                      CL_max 2.0331 -> 2.0493 (+0.79%)   cd2   0.1364 -> 0.1472  (+7.90%)
                      cd_strut 0.0007 -> 0.0009 (+33.3%) mass  8.5040 -> 8.9840  (+5.64%)
```

**The whole flank delta is section 3 and nothing else.** Attributed, not
asserted: the *old* code called with `standoff=0.60` reproduces the *new*
code's numbers **identically** on both flank wings, every key, to 5e-6. Less
downforce because the wall is further away (less image augmentation), a longer
strut and a heavier pylon because both are charged per metre of standoff.
Direction and cause both check out.

And these numbers are not new to the garage: it was already analysing at 0.60.
What moved is `wing.py`'s and `library.py`'s answer, into agreement with it.

**Geometry is bit-identical** — `S`, `AR` and `chord` unchanged on all four
wings. Deriving the chord from the area is a no-op on a wing that already had
one, which is the point: `area = 0.0` resolves to the planform's own
`b c (1 + taper) / 2`, so a saved wing decodes to the area it was analysed at.

**The two-row twist is a no-op** on every existing wing (`twist_root_deg = 0`
reproduces the old law exactly), and **the ride row is a no-op on top wings**:
a wing saved before the row existed recorded its height under
`aero['ride_h']`, and `from_json` seeds the row from it. Without that seeding
`clamp` filled the role default instead and the library's own `rear-new` — which
sits at 1.85 m — was silently re-flown at 1.30 m and handed **+1.05 % of CLa it
had not earned**. Caught by this table, which is what the table is for.

**The published study is untouched.** `drive.validate` 82/82, 0 HARD, 0 soft.
The closed-form device (CONTRACT section 4) never enters the lattice path, so
the README's 2.1996 parity and the 222.1 N at the R = 100 limit are the same
numbers they were.

## 5. What is still not AeroBO

Stated so it is not mistaken for done:

* **no lap.** The mission rows are a speed and a two-point requirement pair —
  which is what AeroBO's own docstring says a budget should have been — but not
  `cartrack.py`'s integral over a circuit. The exchange rate between downforce
  and straight-line drag is still a designer's number here, not an integral.
* **the top pylon length** is the constant of section 3.
* **no multi-section / slotted element.** `carwing_multi.py` and
  `carsection.py` have no counterpart; the flank panel is a single surface.
* **the flank panel has no AeroBO counterpart at all.** `carwing.py` is a rear
  wing and maps onto the top slot. `fin.py` is an aircraft's directional-
  stability fin, not a lateral device on a car. So "the same procedure" is
  literal for the top wing and an adaptation for the flank — the ride row is
  the one place that shows, and it is the standoff there because that is the
  gap the image plane actually sits across.
