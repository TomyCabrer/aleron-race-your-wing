# Task 16 — "for garage to design wing I want it to be by sections just as the UROP app"

**Status (2026-09-21, polish pass):** done, uncommitted. Measured after the
pass: `drive.aero.{mission, section, screen, blend, wing, vlm, optimize,
library, polar}` PASS, `drive.garage` ALL PASS, `drive.drive --self-check`
ALL PASS, `drive.render` 29/29, `drive.vehicle` 33/33, `drive.input` and
`drive.menu` ALL PASS. The four new modules are on `validate.MODULES`. The
legacy single-wing DESIGN page, still reachable through the airfoil page's
back route, is gone (sections 12-14 said it had no separate page; now the
code agrees). What is explicitly not done is section 10.

Clarified before building, because "by sections" is three different things in
`~/dev/urop-bo-aero` and only one of them was asked for:

| reading | what it is there | asked for |
|---|---|---|
| a slotted TWO-ELEMENT section | `carwing_multi.py`: four arrangement rows, two-body panel solve, one endplate | **not yet** — "single element first" |
| the aerofoil DESIGNED rather than picked | `carsection.py`: CST weights + a thickness row, scored in 2-D | **yes** |
| a spanwise-varying section | `DESIGN_MULTISECTION.md`, deliberately NOT implemented there | no |

What was asked for, in the user's words: *"first to design the mission then be
able to design airfoil then wing just as in the bo aero"*, on **both** slot
roles, as **three gated pages**, with the mission being **a lap on carsim's own
tracks**. That is what is here.

```
STEP 1  MISSION   drive/aero/mission.py   a LAP of the arena / open / skidpad
STEP 2  DESIGN    a navigator over everything downstream, in the LEFT column:

          AIRFOIL   (the wing's own section)     ENDPLATE  (the tip panels')
            library screening                      library screening
            ranking                                ranking
            section                                section
            shape optimisation                     shape optimisation
          WING                                   RESULTS
            wing type                              summary
            design box                             geometry
            solver                                 loading
            convergence                            evaluations
```

The order is **enforced**, not suggested: `Garage.open_designer` refuses until
the mission is stated, and `ESC` steps back up the chain rather than dropping
to the car. `D`, `L3` and the pause menu all enter at step 1. Inside the
navigator, `TAB` moves the focus between the steps and the selected step's own
rows; the mark in the left margin is `+` done, `>` ready, `-` blocked.

There is no separate wing PAGE any more. The wing is a group of the navigator,
which is what "make sure that wing behaves the same way" asked for: it gets the
same four-stage treatment the two section groups get.

## 1. The mission is the lap task 14 said was missing

`14-urop-parity.md` section 5 listed, under "what is still not AeroBO":

> **no lap.** The mission rows are a speed and a two-point requirement
> pair ... but not `cartrack.py`'s integral over a circuit. The exchange rate
> between downforce and straight-line drag is still a designer's number here,
> not an integral.

It is an integral now. `drive/aero/mission.py` flies a quasi-steady lap:
corners at their steady speed, straights the acceleration profile out of one
corner met by the braking profile into the next.

**Not `cartrack.py`'s synthetic corner list.** `drive/track.py` already
defines every circuit as an ordered list of constant-radius arcs and
straights, which is `TrackSpec`'s shape, so `TrackProfile.from_track` reads
carsim's real geometry and nothing here invents a track. The arena is its own
seven published corners.

**The lateral model is `qss.py` plus exactly one term** — aerodynamic
downforce, which `qss` has no row for because the published device is a
lateral one. At zero downforce the two are the same function:

```
residuals     == qss.residuals     worst |d| 0.0e+00   (bit for bit)
max_ay        == qss.max_ay        worst |d| 1.6e-11
corner_speed  == qss.corner_speed  worst |d| 4.7e-11, and the same limit word
```

and it lands on the repo's own published numbers: R = 130 m gives **31.880
m/s, power-limited**, which is what `track.py`'s T7 comment says; R = 100 m
gives **29.0875 m/s**, which is `wing.V_REF["flank"]` to 6 figures.

A lap costs **27 ms** on the arena, which is what makes it an objective a
search can call. (`qss.corner_speed`'s 200-step damped fixed point is a
bisection on the same monotone residual here, and the power cap is skipped
outright when the grip speed is already affordable — six of the arena's seven
corners. 273 ms -> 27 ms, same answer to 5e-11.)

## 2. Two things the lap says about this car, and both are correct

Written down because both look like bugs and neither is.

* **A downforce wing behind the rear axle makes the car SLOWER.** `b = 1.519 m`
  is CG-to-rear, so a wing behind `x_t = -1.519` puts a NEGATIVE share on the
  front axle — it levers the front UP — and the car is front-limited. At
  `CZ*S = 2.0` with the drag off: **-0.86 s at `x_t = 0`, +0.28 s at
  `x_t = -2.00`.** The DESIGNER page's own station row already said "a top wing
  behind the rear axle unloads the front"; this is the first time it is priced.
* **Downforce is nearly worthless at Corsa speeds and drag is not.** The arena
  is cornered at 16-32 m/s, where `CZ*S = 0.4` is 96-216 N against 9908 N of
  car. A `CD*S` of 0.06 costs **+0.086 s**; the downforce it buys returns
  **0.002 s**. That asymmetry is why the published study hangs a lateral panel
  on the flanks instead of a wing on the roof, and the mission reproduces it
  without being told. A top-wing lap search correctly drives camber to ~0.

## 3. The finding: the study's own panel changes sign

`drive.validate` already reports, as one of its two findings to pass upward,
that `alpha_peak_deg = 7.0` is too low — the Magic Formula front-axle peak at
the R = 100 limit load split is **10.34 deg**. Scrub drag is
`m a_y sin(alpha)`, so the term is 1.47x larger than assumed.

Priced in lap time on the arena, with the published flank panel (S 0.35 m²,
CL 0.70, L/D 3.2):

```
alpha_peak    bare lap     with the panel    what the panel is worth
  7.00 deg    51.8431 s      51.7635 s            -0.0796 s
 10.34 deg    53.1159 s      53.1433 s            +0.0275 s
```

**The device changes sign.** At the larger scrub angle more of the circuit is
power-limited, and in a power-limited corner the panel's drag costs more than
its side force buys — which is `qss.corner_speed`'s own argument running the
other way. So a lap verdict from this module is a verdict AT A STATED SCRUB
ANGLE. 7.0 is `qss`'s, kept so the parity above means something, stated rather
than defended, and a row of `lap(...)` rather than a constant. Pinned by
`mission.self_check` so it cannot drift silently. Neither `qss.py` nor
`crossover.py` has been modified.

## 4. The section is designed, not picked

`drive/aero/section.py`. Ten rows:

```
x = [w_upper(4), w_lower(4), t/c, alpha_deg]
```

* **`N_CST = 4` is a dimension trade, not "more weights buy nothing".** Measured
  over all 34 bundled sections, the fit error away from the leading edge is
  median 2.1e-3 chord at 4 and 7.9e-4 at 8. More weights do fit better; what
  they cost is rows (`d = 2 n_cst + 2`), and AeroBO's own budget law
  (`10.5 + 3.0 d` for 95 %) puts 4 at 40 evaluations and 8 at 65.
* **The box is the library's own per-coefficient hull**, padded 15 %, so all 34
  shipped sections are inside it by construction and the search can always
  reproduce a shape known to fly. It is a BOX and not the hull, so corners no
  section occupies are **refused** (`check_section`), never clamped.
* **Thickness is its own row** for `carsection.py`'s reason, with the same exact
  linear rescale. It lands to 2.8e-17 and moves the camber by under 1e-4.
* **Two degenerate directions, both stated.** The half-thickness scale is undone
  by the rescale (AeroBO's own, accepted for its reason). And **swapping
  `w_upper` and `w_lower` is a no-op** — `airfoil.normalise_loop` orients
  whatever loop it is handed, so the swapped weights rebuild the same section
  (NACA 2412: t/c 0.11987, camber +0.02009 either way). Found by a self-check
  that expected a refusal and did not get one.
* **The incidence row is here AND on the wing page, over the SAME band**, which
  is `carsection.py`'s rule verbatim: the same question asked over two bands
  makes the two stages incomparable. Step 2's winner SEEDS step 3's row.

**This is not AeroBO's bridge, and the difference is in carsim's favour.**
`carsection.SectionBridge` is a calibrated affine reduction with a measured fit
error because its 3-D solver is too expensive per candidate. Here
`wing.analyse` is **1.4 ms**, so the candidate's polar is flown on the role's
reference planform exactly and there is no fit error to report. What survives
is the experimental split — the section may not move the planform — which is
the part that was ever load-bearing. A candidate costs **31 ms** end to end.

The `weighted` 2-D control ships alongside (equal weights on normalised
`cl_max` and `(l/d)_max`, never consulting the lap), so "the lap is the better
objective" can be measured here rather than asserted.

## 4b. The end plate is a design problem, and the hook was already there

`vlm.Lattice` has always taken `plate_a` and `plate_L0` -- *"the plate's own
section (a flat plate by default)"* -- and `wing.build_lattice` has always
hardcoded them to `TWO_PI, 0.0`. So giving the tip panels a designed section
is not new physics, it is a default that was never lifted.

`WingSpec.plate_airfoil` is the row (`""` = the flat plate, so every wing saved
before it decodes to exactly the numbers it was analysed at, and
`plate_polar=None` is **bit-identical** to the shipped path -- checked). The
plate's section is read at the plate's own Reynolds number, which is the wing's
scaled by the taper, because a plate a third of the chord long sits a bank
lower than the wing does.

**It is a CAMBER problem, and the measurement says so.** On `flank-e423` at
`plate_h` 0.12 m, through `analyse`:

```
plate section              CL0      CLa    CL_max       e
flat plate (what shipped) 0.5688  3.0017   1.9580  1.4125
cambered, alpha_L0 -4 deg 0.5942  3.0036   2.0469  1.4097
cambered, alpha_L0 -8 deg 0.6199  3.0055   2.0579  1.3919
a_lin 6.10, alpha_L0 -7.5 0.6164  3.0042   2.0562  1.3952
```

The plate's zero-lift angle is the whole effect; its **slope barely enters**
(6.28 -> 6.10 moves CL0 by 0.004). At `plate_h` 0.14 m an e423 plate is worth
**+10.5 % of CL0** and takes CL_max 1.455 -> 2.434.

**A plate has no incidence row**, and the design vector says so: 9 rows, not
10. The lattice's plates are vertical panels at the tips carrying only a slope
and a zero-lift angle -- there is no angle to bolt them on at -- so the angle
the plate is JUDGED at is the wing's own, and it belongs to the wing.
`section.SectionProblem.target` carries the difference and
`element_from_x` reads a 9-row vector as a plate's.

Gated by a self-check that the candidate really enters in a different place:
the same camber change is worth **+0.0229 CL0 as a plate and +0.3245 as the
wing's own section**.

## 4c. The library screen: chosen on the map it is judged on

`test_the_library_shortlist_is_chosen_on_the_map_it_is_judged_on.py` in
urop-bo-aero is named after the bug it documents -- a screen that ranked on a
live min-max and then scored on a frozen band, so "the sections allowed to
compete were chosen by one map and then judged by another".

carsim had the same shape of problem waiting. The AIRFOIL LIBRARY page ranks
the 34 shipped sections on a weighted composite of L/D, cl_max, |cm| and
thinness. That is a **different map from the lap**, so its winner is not this
one. The screen step therefore scores every library section on the SAME
objective the optimiser will use, and nothing else. ~45 sections in 1.3-1.6 s.

A section whose CST weights or thickness fall outside the design box is judged
at the **clipped** shape, which is not the section it came from. Those are
counted, named in the ranking table (`[CLIPPED]`) and reported on the
EVALUATIONS panel rather than quietly ranked. Typically 2-3 of 45 -- `e61` and
`e63` sit at t/c 0.057 and 0.043 against a band that starts at 0.070.

## 4d. The weights are rows

The ten (or nine) numbers of the design vector are editable `LEFT`/`RIGHT`
rows, not a read-out. Moving one rebuilds the shape, re-runs the panel method
and re-flies the lap -- 31 ms, inside the frame budget. The t/c row lands
exactly where it says (self-check: 0.1214 -> 0.1264 on a 0.005 step), because
the weights are rescaled to hit it and the rescale is linear.

## 5. A bug this exposed — the optimiser's start vector was outside its own box

Pre-existing, in `Designer.optimise`, and it only became visible because the
lap objective is smooth in span where the capped objectives were not.

`design_x0` reports the wing AS IT STANDS. The span band is narrowed at run
time by the sill/roof fit at the slot's height (`_span_band`), so a wing
carrying a span the slot cannot take produced a start vector OUTSIDE the
bounds — measured on the seeded library at `h = 0.90 m`: **span 1.05 m against
a band that stops at 0.88 m**. `optimize.maximise` clips `x0` before evaluating
it, so the BO was fine; `f0` was computed on the UNCLIPPED vector. The
comparison `f_best >= f0` then failed for a packaging reason and the page
reported *"no feasible design found under the constraints; try a wider cap"* —
which is not what had happened, and there was no cap involved.

`x0` is clipped into the bounds now, and the two failure messages say which
failure it was: every candidate refused, or nothing beat the wing as it stands.

**Not fixed, and flagged instead:** a `Designer` opened on a wing whose span
does not fit its slot still ANALYSES it at that span, so the page can show a
wing that cannot be bolted on. Clamping on open would move existing numbers.

## 5b. A second bug: the wing search flew a flat plate

Found by walking the finished navigator. `Designer.update` goes through
`library.analyse_wing`, which reads `spec.plate_airfoil` -- but the optimiser's
inner loop calls `wing.analyse` **directly**, and that call had no
`plate_polar`. So however the end plates had been designed, the wing was
optimised against a lattice with flat ones, and the page beside it was drawing
a different wing. Threaded now, and gated: *"the wing search flew the END PLATE
it was given, not a flat one"*.

## 5c. The two stages are measured on different instruments

Stated because it is the one place they are, and the RESULTS > SUMMARY panel
says it on screen rather than leaving it to be discovered.

Every candidate in the SCREEN and the SHAPE OPTIMISATION is scored on
`polar.estimate_polar` -- 3 ms, against XFOIL's ~2.4 s -- which is the only
reason a 48-evaluation search is affordable at all. The WING then flies the
library section on an XFOIL polar wherever one exists. So **a section that wins
the 2-D screen need not win on XFOIL**, and `X` on the SOLVER step is what
measures the winner properly.

## 6. What is still not AeroBO

* **No slotted element.** Single surface, as asked ("single element first"). The
  section vector is laid out so a flap block drops in ahead of the size rows,
  which is where `carwing_multi` puts it. `panel2d.py` here is single-body
  (215 lines against AeroBO's 1144) and that port is the work.
* **The end plate's CANT and BLEND are still problem settings, not rows** --
  which is what AeroBO does too (`14-urop-parity.md` section 2). Only its
  SECTION is designed.
* **The end plate's section is judged through the wing, never on its own.**
  There is no 2-D control objective for a plate: a plate's `cl_max` means
  nothing on a panel that is there to turn the tip flow.
* **The lap is quasi-steady and its braking is TYRE-limited.** `corsa_c.py`
  carries `brakes = "MISSING - no disc/drum dia, pad mu or F/R torque split"`
  and `dampers = "MISSING"` under the line "Do not ship a lap sim without
  them." So: no transient load transfer, and the deceleration is the one the
  tyres can take. A wing is compared against a wing under one assumption; an
  absolute lap time from this module is not a claim about the real car. For
  scale, the driven laps in `runs/` average 19-21 m/s against this lap's 24.1.
* **The longitudinal friction law is the lateral one** (`qss.fy_max` used for
  `mu_x`). `tyre.py` has the real combined-slip surface and the 1 kHz sim flies
  it; charging it per candidate would cost the milliseconds this layer exists
  to save.
* **The flank panel still has no AeroBO counterpart**, so "the same procedure"
  stays literal for the top slot and an adaptation for the flank — task 14's
  own caveat, unchanged.

## 7. The numbers

Acceptance suite and every self-check, after:

```
python3 -m drive.validate     ->  82/82 pass  0 HARD  0 soft  [109.4 s]
python3 -m drive.garage       ->  ALL PASS  (the three-step chain is now in it)
drive.aero.{mission, section, wing, library, polar, airfoil, panel2d, vlm, optimize}  ->  PASS
```

The published study is untouched: `garage.self_check` still measures the
closed-form plate at **+2.1996 %** against the README's 2.1996 and **222.1 N**
at the R = 100 limit.

One full walk of the chain, left flank, arena, dry, from the seeded library:

```
step 1  mission stated            arena, 1249 m, 7 corners, bare lap 51.8431 s
step 2  section, 16 evals         seeded from e423 (-0.1184 s)  ->  51.5679 s
step 3  wing, 16 evals            -> 51.5269 s,  -0.2618 s vs the slot empty
```

New pages draw in 3-16 ms against 60-90 ms budgets.

---

## 8. The weights were being asked in the wrong step

Reported: *"in airfoil section the weights are decided in library screening,
screens the library and then gives option to optimise. In wing type and design
box there are other parameters that are not considered."*

Both halves were right, and the first one was the deeper mistake.

### 8a. What "the weights" are in AeroBO

`gui/v3/stages/airfoil.py`, module docstring, first two sentences:

> 1. SCREENING picks the best KNOWN section for the mission's design point.
>    The UIUC database is scored on six criteria under **the user's weights**
>    at the mission's design Cl.
> 2. SHAPE OPTIMISATION designs a NEW section: 8 CST weights ... seeded from
>    the screening pick.

So there are two different things called weights, and carsim had put the wrong
one on the form. The CST weights are the SEARCH's design vector -- nobody has
an opinion about `w_upper[2]` before a search has run. The weights a designer
does have an opinion about are the **criterion weights**: how much of the
score is the L/D this surface flies at, how much is the lift it reaches, how
much is the depth a spar needs. Those are the screen's question, and the
screen is what they decide.

`drive/aero/screen.py` is that module -- `airfoil_select.score_candidates` in
carsim's units. Seven criteria (`ldcr`, `clmax`, `cm`, `ldmax`, `cdcr`,
`thick`, `astall`), AeroBO's own two presets (its bulk-sweep set for a surface
that carries the design load; its `FIN_WEIGHTS` for the plate, which its own
car endplate shares with its vertical stabiliser), its `DEAD_CRITERIA` for a
panel at zero lift, its gates and floors, its `GOAL_PENALTY = 3.0` and its
`RHO_ASF = 0.05`.

The navigator's four steps did not move. What is asked in each of them did:

```
library screening   the seven weights, the gates, the floors   <- NEW
ranking             every criterion, and the POINTS it put into the score
section             the CST weights (relabelled THE SHAPE)
shape optimisation  composite | composite-no-worse-than-seed | Tchebycheff
                    ...and lap time, which is carsim's own
```

### 8b. The frozen band, and why a composite run is BLOCKED before a screen

A sub-score is 0-100 over a `(lo, hi)` band. Take the band from the live
population and the map moves with the population, so the composite is not a
fixed function of the shape and a search cannot maximise it. So `screen.bands`
is measured ONCE, over the screened library, and the problem keeps it. The
navigator marks `shape optimisation` **blocked** until it exists and the
optimiser refuses with the reason rather than reporting a failed run. That is
AeroBO's *the shortlist is chosen on the map it is judged on* made binding
instead of said.

Measured, in `drive.aero.section.self_check`: a composite candidate costs
**3.6 ms** against a lap candidate's **31.0 ms**, because J scores a 2-D
section and a composite run therefore never flies the lattice -- AeroBO sends
`wing: None` on one for exactly this reason.

### 8c. AeroBO's GATES delete carsim's library

`GATE_DEFAULTS = {"tc_min": 0.15, "cm_max": 0.08}` is AeroBO's published pair,
over 2120 UIUC sections, for an aircraft wing. Measured over the 34 sections
carsim ships, at Re 5e5 and cl 0.9:

```
t/c    min 0.0427   p25 0.0956   median 0.1070   max 0.1610
|cm|   min 0.0000   median 0.0684   p75 0.1856   max 0.3477
```

**That pair admits one of the 34.** Worse, the |cm| ceiling refuses precisely
the sections a downforce wing exists for -- S1223 0.348, S1210 0.306, CH10
0.275, FX74-CL5-140 0.275, E423 0.247 -- because a high-lift section's moment
is what it trades for its lift, and a wing bolted to a car has no tail to trim
it: the mount carries it. A gate inherited unchanged would not have been a
filter, it would have been a deletion of the library.

So the gates are **off by default** here, and the values one comes back ON at
are this library's own quartiles (0.0956, 0.1856), where switching one on
refuses about a quarter of the library -- which is what a gate is for. This is
the one place carsim deliberately does not copy AeroBO's number, and the
reason is measured rather than asserted.

### 8d. What the screen now proves about itself

From `drive.garage.self_check`, all new:

```
the criterion weights are on the SCREENING step      7 weights, a preset, two gates, three floors
the AIRFOIL group opens on AeroBO's wing preset      airfoil 'wing (bulk sweep)', endplate 'plate (symmetric)'
a plate is not ranked on the three it cannot carry   cm, ldcr, ldmax all at zero
a composite search is BLOCKED before a screen        refused, with the step named
the screen scores every section on the WEIGHTS       39 ranked, 0 gated, 0.3 s
a weight edit re-chooses the shortlist               preset -> 'ag35',  thickness alone -> 'e837'
a gate DROPS candidates, not ranks them low          24 of 39 refused at t/c >= 0.12
the ranking table carries every criterion + points   L/D@cl 72.2(+34.4)  clmax 1.68(+8.1)  ...
```

and in `drive.aero.screen.self_check`, the scalarisers:

```
a candidate past the band is NOT clipped                 125.0 points
a trade the plain composite calls even is refused        plain 50.000 = 50.000, goal 35.000
the ASF is zero at the seed and negative below it        worst criterion named
a zero-weight criterion is DROPPED from the ASF          not admitted at lambda 0
```

The last one matters: a zero-weight term is identically 0, so admitting it
would pin the Tchebycheff min at 0 for every candidate and silently turn the
objective off.

## 9. The wing: a design box is a table of BANDS

The second half of the report. `w.box` was showing the design vector's
VALUES under the heading "design box", while the bands the optimiser actually
searched were module constants in `wing.BOUNDS` that nothing on any form could
reach. In AeroBO stage 3 the design box IS the band table.

Now: every design variable carries a `min` and a `max` row, `Designer.box`
holds them, and `Designer.search_bounds()` is the ONE function both the rows
and `optimize.maximise` read -- the same trap `_span_band` was written for,
one level up. A band may only narrow the packaging band, may not be inverted,
and the span's ends still show the sill/roof fit rather than what was typed.

**The reference area became a design variable.** `wing.design_table(role,
area=True)` has carried AeroBO's `(S_m2,)` row -- one row ahead of the span --
since the port, and no form ever set `area=True`. WING TYPE now asks it:
fixed (the span row IS the aspect ratio, every candidate on one reference) or
searched (two candidates no longer share a reference, so the score has to be
read in forces). The bounds, the start vector, the labels and the decode all
move with it because all four walk the one table.

**The plate's section became a wing-type row.** It was a read-only read-out on
the solver step, settable only by designing one. It is a real lattice choice
and is now a choice.

### 9a. A name collision in `design_bounds`, found by the new check

`design_bounds(role, area=False, **override)` could not be given an override
for the AREA row: the design variable is called `area` and so is the switch
that decides whether it is in the vector at all. Nothing had hit it because
nothing had ever overridden more than the span. An editable design box
overrides every row, so it hit it immediately:

```
TypeError: design_bounds() got multiple values for keyword argument 'area'
```

Fixed by adding a `bands=` dict merged under `**override`, so the garage's
`span=` idiom still reads the same and the collision is documented at the
signature rather than discovered again.

### 9b. Two UI clips that were hiding real numbers

* `ListBox` clipped every sub-line at a flat **70 characters**, in a panel with
  200 px of unused width beside it. It was cutting the ranking table's columns
  mid-number. Now clipped to the panel.
* The ranking row is genuinely wider than the panel with all seven criteria on
  it, so the columns are ordered **by weight, heaviest first**: what survives
  the cut is what the score is mostly made of, and a zero-weight criterion
  goes last, which is where it belongs on a table that exists to say where a
  score came from.

## 10. Still not AeroBO

Named so nobody goes looking for them:

* **The plate's CANT** (`endplate_cant_deg`) and **chord law**
  (`endplate_chord_follows`). carsim's plates stand normal to the wing and
  carry its tip chord. The BLEND and its SHAPE were ported after this note
  was written (section 14): they are WING TYPE settings, as AeroBO's
  `blend_frac` / `blend_shape` are, and not rows of the design vector.
* **`flown_reynolds`.** `library.polar` snaps to `RE_BANK` and caches per
  bank entry; reading a polar at the flown Reynolds number would need an
  interpolation path that does not exist. AeroBO measures the difference at
  its own design point as CZ +0.660 %, CD +1.152 %.
* **`pareto`.** AeroBO's fifth section objective returns a FRONT, not a
  winner. The three scalar ones are ported; there is nowhere on the navigator
  to draw a front.
* **The free chord law** (`chord_order`, `chord_law`, `chord_limits`). carsim
  derives the chord from area, span and taper on purpose -- carrying a free
  chord alongside a reference area would let a candidate be scored against an
  area it does not have.
* **AeroBO's remaining car objectives** (`efficiency`, `downforce_plus_drag`,
  `cd`, `drag`). carsim's four per role cover lap time, the capped force and
  the force/drag ratio; the rest were not asked for.

## 11. The numbers, after

```
python3 -m drive.validate     ->  82/82 pass  0 HARD  0 soft  [112.2 s]
python3 -m drive.garage       ->  ALL PASS
drive.aero.{screen, mission, section, wing, library, polar, airfoil,
            panel2d, vlm, optimize}  ->  PASS
```

The published study is still untouched: +2.1996 % against the README's
2.1996, and 222.1 N at the R = 100 limit.

Worst navigator panel draw: **5.5 ms at `af.screen`**, against a 90 ms budget.

---

## 12. The gate was a mark, not a gate — and the mouse did nothing

Reported: *"still not the same. shouldn't let it go to next section before
finalising the one before. Also doesn't let the user interact with it (the
mouse should also work not only the keyboard)."*

Both were true, and the first one was written into `Nav`'s own docstring as a
deliberate choice:

> A blocked step can still be looked at -- the gate belongs to the action, not
> to the eye.

That is not what the UROP app does. `gui/v3/app.py`:

```python
def select(stage, view=None):
    state = session.stage_states(S)[stage][0]
    if state == "locked":
        reason = session.stage_states(S)[stage][1]
        ui.notify(reason or f"{stage} is not available yet", type="info")
        return
```

It **returns without moving**, and it notifies the stored REASON --
`stage_states`' own docstring says why: *"A locked stage keeps the reason it
is locked, because a greyed-out node with no explanation is the thing this
shell exists to avoid."* So the mark was carrying the right information and
nothing was enforcing it.

### 12a. What is gated, and what is not

Two different shapes, and they are gated differently on purpose.

**The section groups are a PROCEDURE**, so their four steps are sequential:

```
library screening   always open
ranking             needs a screen
section             needs a section taken from the ranking
shape optimisation  needs a section that flies (+ the band, on a composite)
```

**The wing's four steps and the results' four are VIEWS OF ONE PROBLEM** -- a
design box and a solver are two ways of looking at one wing, and AeroBO's tab
strip enables all of a stage's views together. So they are gated as a GROUP
and are open together.

Group to group:

```
AIRFOIL   -> ENDPLATE   fit the section to the wing (F)
ENDPLATE  -> WING       fit a plate section, OR say the plates fly FLAT
WING      -> RESULTS    the lattice has solved this wing
```

### 12b. BLOCKED and LOCKED are different sentences

`Nav.MARK` carries four states now, and the two unavailable ones say different
things:

* **blocked** (`-`) is *not yet*. The step before it has not been finished,
  and finishing it opens this one.
* **locked** (`x`) is *not here*. There is nothing on this vehicle for the
  step to decide and no amount of work upstream opens it. The end plates at
  zero height are exactly that, and AeroBO's own endplate stage locks on the
  same condition with the same reason: *"its height is pinned at zero, so
  there is no surface here to give an aerofoil to"*.

### 12c. Fitting is what finishes a group — not the search

The gate could have been read as "you must optimise everything", which would
be wrong twice over. AeroBO is explicit that **step 2 is optional** ("that is
the point of ordering them this way"), and a screen whose winner you accepted
is a complete decision. So a group is finished by **F, fit it to the wing** --
screen, take a winner, fit: three keys, no search.

And because a gate that makes a decision compulsory is not a gate but a
demand, the ENDPLATE group carries the way past it:
**`or fly FLAT plates and move on`**. An explicit answer, because "I looked
and chose flat" and "I never came here" are different states and only one of
them finishes a step.

Every gate opens with one action. Nothing is ever trapped.

### 12d. Finalising is also what moves you on

`fit_and_advance` is the single path both F and the row's own action take.
The mouse walk found this: fitting through the ROW left the navigator sitting
on `af.section` while fitting through the KEY moved to the endplate, because
only the key had the second half. One function now, and the same for the
screen action, which lands on the ranking whichever way it was run.

The hot keys also stopped reaching across the page. `L` used to fall back to
the AIRFOIL group from anywhere, so pressed on a wing step it screened a
library the user was not looking at and jumped the navigator two groups back.
A key that acts on something off screen is the same defect as a gate that
lets you skip a step.

### 12e. The mouse

Every page took the keyboard only; the mouse existed solely to orbit the 3-D
car. A navigator with sixteen steps and a form with twenty-odd rows is a lot
of arrow keys.

`Nav`, `ParamList` and `ListBox` now record the geometry they drew and expose
`hit(pos)` / `click(pos)` / `wheel(dy)`. **The mouse is not a second way in,
it is the same way in**: a click lands on exactly the row the drawing put
under the cursor and does there what the keyboard would do on that row.

* a **navigator step** selects -- and is refused by the same gate, with the
  same reason, as an arrow key
* a **value row**: the value's two halves ARE the two arrows. Click the left
  of the `< value >` and it is a LEFT, the right and it is a RIGHT. That is
  why the value has been drawn with those arrows on the selected row all
  along.
* an **action row** selects on the first click and fires on the second --
  firing on the click that also moved the cursor would let a mis-aimed click
  start an optimiser
* a **ranking row** selects, and a second click is the ENTER on it
* the **wheel** moves the selection in whatever is under the cursor
* **hover** lights the row, so what is clickable is visible before it is
  clicked

A full walk of the chain, mouse only, nothing typed:

```
start             af.screen
click w.box       af.screen   REFUSED: "the wing flies a section, and this one has not..."
click 'screen'    af.rank     54 sections ranked
click a ranking   seed = flank-sec-10
click 'fit'       ep.screen   AIRFOIL finished
click 'flat'      w.type      ENDPLATE answered
click w.box       w.box
click summary     r.summary
```

### 12f. One more clip that was hiding the point

The refusal is the whole reason for refusing, so `Nav` drew it under the tree
-- clipped to 56 characters, which threw away the half that said what to do
about it. Text is wrapped by MEASURED pixel width now (`_wrap_px`), against
the font actually drawing it, and the refusal gets up to six lines under a
`NOT YET` heading.

### 12g. The numbers, after

```
python3 -m drive.validate     ->  82/82 pass  0 HARD  0 soft  [113.7 s]
python3 -m drive.garage       ->  ALL PASS
drive.aero.{screen, mission, section, wing, library, polar, airfoil,
            panel2d, vlm, optimize}  ->  PASS
```

New in `drive.garage.self_check`, all of it driven through the real handlers:

```
a fresh page opens on the FIRST step and everything else is shut   1 of 16 open
jumping to the wing is REFUSED, with the reason                    reason quoted
...and so is a click on it
screening opens the ranking and the section                        ep.* still shut
fitting the section to the wing finishes the AIRFOIL group         ENDPLATE opens
...but the WING is still shut on the plates' question
declining the plates is an ANSWER and opens the wing
a plate at zero height is LOCKED, not blocked                      a different sentence
a click on the navigator selects that step
a click on the right of a value row steps it up                    clmax 0.20 -> 0.25
...and on the left, down                                           back to 0.20
a click on an action row runs it                                   screened by the mouse
```

---

## 13. The defaults, measured instead of guessed

Reported: *"Make the defaults for the optimiser the same as the one in AeroBo
and make sure that the default weights are reasonable."* Then, mid-work:
*"Also wing can be pylon or endplate. Same as AeroBo. end plates carry a
section shouldn't be there. It should be the one decided in airfoil."*

### 13a. The optimiser: AeroBO ships a MEASUREMENT where carsim had a guess

`data/search_budget.json` in urop-bo-aero is the frozen payload its
`experiments/budget_analysis.py` emits. Everything below is read off it and
nothing is chosen here.

**The budget law**, `evals = intercept + per_dim x d`, fitted per convergence
target:

```
target  intercept  per_dim   rms   cases   max measured
 0.90      10.03     2.22     8.0    13         47      "quick"
 0.95       9.61     3.08    10.7    13         64      "balanced"
 0.99       2.62     5.97    11.9    13         91      "thorough"
```

The RMS travels with it, because a law with an 8-11 evaluation residual is a
SIZING RULE and not a prediction. (Earlier notes in this file quoted it as
"10.5 + 3.0 d" from memory. The payload says 9.61 + 3.08.)

**The Sobol split**, and this is the one that mattered:
`n_init = clamp(round(0.5 x d), 4, 16)`. It is MEASURED, against exactly the
rule carsim was using -- over 15 of the study's cases the seed sizes rank

```
0.5 x d  ->  1.00      2.0 x d  ->  2.88
1.0 x d  ->  2.25      4.0 x d  ->  4.25        (mean rank, lower better)
```

and the airfoil class re-measures it on its own cases and agrees. carsim was
spending **16 of 48** on the initial design of a 10-row section -- 1.6 x d,
between the two worst arms -- and 8 of 32 on the wing.

What the defaults become:

```
                     was                  now
section  (d = 10)    48 evals, 16 Sobol   40 evals, 5 Sobol
plate    (d =  9)    48 evals, 16 Sobol   37 evals, 4 Sobol
wing     (d =  7)    32 evals,  8 Sobol   31 evals, 4 Sobol
wing     (d =  8)    32 evals,  8 Sobol   34 evals, 4 Sobol   [area freed]
```

MEASURED on carsim's own section problem, three seeds each:

```
flank   old 16+32 = 48   mean lap 51.6259 s   (spread 0.0018)
flank   new  5+35 = 40   mean lap 51.6249 s   (spread 0.0006)
flank   new  5+27 = 32   mean lap 51.6253 s   (spread 0.0017)
top     old 16+32 = 48   mean lap 51.8479 s   (spread 0.0008)
top     new  5+35 = 40   mean lap 51.8478 s   (spread 0.0002)
```

Better on 17 % fewer evaluations, and tighter across seeds. Even at 32.

An `effort` row carries AeroBO's three settings, and the budget follows the
law through it. The budget stays editable, because a rule with a 10.7-evaluation
RMS should be.

**Not ported**: the stop rule. The payload gives the wing class
`patience 40, tol 0.002`, and records that the AIRFOIL class ships NO stop
rule at all -- "no rule stayed inside the 3 % miss budget". carsim's
`maximise` has no early stop and did not grow one here.

**The constrained seed rule is NOT used, and that is measured too.** The
study's constrained arm seeds at `1.0 x d` clamped [8, 48], because on its
reported 20-D box only 4.7 % of draws fly and a thin seed rides a box corner.
carsim's box is not that box: a uniform draw over the section box is refused
**42 % of the time on the top role and 36 % on the flank**, so ~60 % fly and
a 5-point seed is all-refused with probability ~1 %.

### 13b. The weights: |cm| ranks a car wing BACKWARDS

The shipped default was AeroBO's `gdp-sweep` verbatim, which weights |cm| at
0.20. AeroBO itself defines the transformation for a surface whose moment
something else carries -- its `wing-trimmed` job moves that weight to cruise
L/D, because |cm| is LOWER-better and therefore rewards REFLEX, "which is
what an aerofoil does instead of having a tail".

**A wing bolted to a car has no tail either.** Its moment is carried by the
MOUNT, in bending, and the beam is sized for it. And reflex is the opposite
of what a downforce wing wants.

Measured over the 39 shipped sections, on all three circuits and both roles
-- Spearman's rho between the composite ranking and the LAP ranking, and the
lap the winner actually does against the best section available:

```
flank, |cm| ALONE                 rho -0.97    it ranks the library backwards
flank, AeroBO bulk sweep          rho +0.96    mean +0.0059 s  (worst +0.0108)
flank, that set with cm = 0       rho +0.97    mean +0.0059 s
flank, cm split over the three
       lift criteria  (ADOPTED)   rho +0.98    mean +0.0031 s  (worst +0.0054)
```

So the default is AeroBO's bulk-sweep preset with the |cm| weight removed and
its 0.20 redistributed across `clmax`, `ldmax` and `ldcr`. It halves the lap
cost of the section the screen picks and improves the rank correlation on
every one of the six cases. `ldcr` stays the largest single weight, which is
AeroBO's own ordering: a heavier `ldmax` scores better on the flank still
(+0.0019 s) but it is read at a lift the surface does not fly, and it is
worse on the top slot. The original set stays on the menu as
**"AeroBO bulk sweep"**.

### 13c. A carsim END PLATE is not AeroBO's end plate

AeroBO's plate is a FENCE at zero toe on a symmetric car:
`CarWingEndplateProblem` refuses a cambered section outright, because the two
plates' side loads cancel in CY and the camber is a trap. carsim's plates are
lifting tip panels in the same lattice as the wing, and camber POINTS their
load with the wing's -- `WingSpec.plate_airfoil` has recorded +9.0 % of CL0
for a plate at `alpha_L0 = -8 deg` since it was written.

Measured, as the correlation between a plate section's |camber| and the lap
it buys:

```
FLANK plate   arena +0.96   open +0.52   skidpad +0.97      camber HELPS
TOP plate     arena -0.96   open +0.97   skidpad -0.97      sign FLIPS
```

The shipped AeroBO plate preset picked the **worst plate in the library** on
the flank (+0.0053 s out of a 0.0053 s spread, rho -0.57). So the plate's
preset is keyed on the ROLE now, which is what AeroBO's own `JOB_WEIGHTS`
does for its four surface jobs:

* **flank** -> a camber-seeking set. Right on all three circuits.
* **top** -> AeroBO's drag-led `FIN_WEIGHTS` kept. Its sign flips with the
  circuit and the whole library is worth 0.0003-0.0119 s there, so there is
  no default that is right rather than merely neutral, and the neutral one is
  the honest answer for a panel at zero design lift.

**|cm| comes OFF the plate's dead list.** AeroBO retires it -- "a symmetric
section's |cm| about the quarter chord is zero identically, so this weight
ranks noise" -- and that is true of ITS plate, whose library is restricted to
symmetric sections. carsim's is not. Measured, |cm| is the single most
informative criterion a carsim plate has: rho +-0.97 against the lap on
either role.

### 13d. The screening lift was a bug, and AeroBO had already answered it

`cl_design` was the reference wing's own CL at the incidence it currently
sits at. On the shipped TOP slot that is **1.718** (mean local strip cl
1.616), which is above the 2-D cl_max of **30 of the 39** sections -- so the
screen threw three quarters of the library away as right-censored, and did it
because of a mount angle the WING page owns and is about to move. A screen
whose shortlist changes when another page's row moves is not a screen.

AeroBO answers this exact case in `session.REFERENCE_CL`:

> A family that declares no design lift -- the car rear wing MAXIMISES
> downforce under a drag budget -- still needs the mission form to open on a
> number. It opens on the load CL = 1.0 implies at that family's own track
> point: a plain reference chosen by this shell, labelled as such, and not a
> published target.

carsim's wings are that family. So the row opens on **1.0**, is labelled a
reference on the page, and is EDITABLE -- with `or read it off the wing as it
stands` beside it as an offered action, never an automatic one. 39 of 39
sections rank now, on both roles.

### 13e. Three defects the walk found

**A composite winner was bolting the wing on at a random angle.** A composite
run never builds the lattice -- that is what makes it 4 ms against a lap's
31 ms -- so its incidence coordinate is INERT: every value scores the same
and the optimiser's choice is noise. Fitting wrote that noise onto the slot's
mount angle. Measured: a flank composite search left the row at **-5.933
deg**, the bottom of its own band. `SectionProblem.prices_inc` now says
whether an objective read the row, the write-back is gated on it, and the row
itself says "(this objective does not price it)".

**`apply_design` is documented as `design_x0`'s exact inverse and was not.**
`WingSpec.area` carries 0.0 for "take the planform's own"; `clamp` resolves
it, `design_x0` resolves it, `apply_design` read it literally. The chord then
decoded to `2 x 0 / ...` = 0 and the lattice refused **every** candidate with
"chord collapses to zero somewhere on the span" -- a true sentence about a
wing nobody proposed. It survived because every path that touched a design
row called `clamp` on the way past; it bites on a library wing opened and
searched without a single row being edited first, which is exactly what the
new gated walk does.

**A drag the mission could not integrate.** The estimate polar's build-up
returned `cd0 = 6.6e7` for one pathological shape the box admits, which the
lap handed to `_resistance` as a drag area of 2.6e7 m^2 and overflowed
float64 in the straight integrator. The lap still REFUSED it, so no answer
was ever wrong -- but it got there by overflowing, spraying numpy warnings
across the terminal of anyone running a search. Guarded at `CD0_SANE = 1.0`,
sized off the library rather than picked: the 39 shipped sections span cd0
0.0097 to 0.0321, and 300 uniform draws over the design box reached 0.0617 at
worst.

### 13f. Two rows that should not have been there

* **The mount offers two layouts**, `pylon` and `endplate`, as AeroBO's own
  car wing does. `none` stays in `wing.MOUNTS` because a stored legacy panel
  is decoded with it and `clamp` must accept it, but its own docstring has
  always said it is "not a buildable car" -- so it is not a thing the design
  form offers. `MOUNTS_BUILDABLE` is that distinction.
* **"end plates carry a section" is gone from WING TYPE.** It was asking, in
  the wing's words, a question the ENDPLATE group had already answered, and
  two homes for one answer is how they drift apart. The solver step shows
  what that group decided, as a read-out.

### 13g. The numbers, after

```
python3 -m drive.validate     ->  82/82 pass  0 HARD  0 soft  [113.3 s]
python3 -m drive.garage       ->  ALL PASS
drive.aero.{screen, mission, section, wing, library, polar, airfoil,
            panel2d, vlm, optimize}  ->  PASS
```

One full gated walk, left flank, arena, dry, from a clean library:

```
bare lap                                            51.8431 s
AIRFOIL   'wing (downforce)', cl 1.0, 40 evals (5 Sobol)
          screen   -> e423, 83.72 points
          optimise -> BO 91.26 vs random 81.08 vs start 83.72
          fit      -> mount angle UNCHANGED (a composite does not price it)
ENDPLATE  'plate (cambered)', 37 evals
          screen   -> 84.22 points
WING      pylon, 31 evals (4 Sobol)
          optimise -> BO 51.7882 s vs random 51.8019 vs start 51.8191
FINAL                                               51.7883 s   -0.0548 s
```

---

## 14. The wing and the plate now MEET

> *"for endplate there should be transition from wing and plate like in AeroBo"*

### 14a. What was wrong

`vlm.Lattice` bolted the tip plates on at a right angle and changed
**everything about the surface in one step** at the junction panel:

| at the junction edge | inboard (wing) | outboard (plate) |
|---|---|---|
| direction | in the wing plane | 90 deg out of it |
| zero-lift angle | the section's (E423: -2.3 deg) | the plate's (0 deg flat) |
| twist | the wing's tip twist | 0 |

Two surfaces sharing an edge, not a wing with end plates. AeroBO's own note
on the same defect is blunt: a "blended" corner joined a 0.21 m chord to a
0.62 m one *"across a corner the junction model was simultaneously giving a
fillet credit for"*.

### 14b. `drive/aero/blend.py` -- AeroBO's geometry, to the bit

A port of `geometry.winglet_turn_angle` / `_turn_law` / `winglet_path` /
`winglet_projection` / `winglet_tip_height`, `vlm.transition_ramp` and all of
`junction.py`. Three turn laws (`arc` the circular fillet, `smooth` the
smoothstep, `spiral` the clothoid at `SPIRAL_RAMP_FRAC = 0.25`), the path
integrated unit-speed on a fixed 32-node Gauss-Legendre rule, one rule per
smooth piece of the law.

The gate is parity, the way `vlm.self_check` pins the lattice:

```
12 tip positions (3 shapes x 4 blends) vs geometry.winglet_path  ->  |delta| = 0
winglet_projection(0.2, 90, 0.7, spiral)   0.08506688624821367   exact
winglet_tip_height(0.2, 90, 0.7, spiral)   0.14506688624821368   exact
junction.blend_radius_min(0.2,90,0.8,sp)   0.07639437268410978   exact
```

`plate_h` becomes the plate's **developed arc**, which is its height while it
is straight -- so `plate_blend = 0` is the published right-angle corner
**bit-for-bit**, checked across every wing in the library on eleven stored
coefficients.

### 14c. What ramps, and on what

`ramp(t) = psi(t)/phi` -- the turn angle over the cant. The plate's **section**
`(a, alpha_L0)` and its **toe** are interpolated on it, so the surface
finishes becoming the plate exactly where it finishes turning into it.
Measured, flank panel, blend 0.6 spiral:

```
alpha_L0  -2.30 deg (the wing's)  ->  -1.00 deg (the plate's), monotone
toe       -2.99 deg (the tip's)   ->   0.00 deg,              monotone
```

The chord does not ramp because carsim's plate **carries the wing's tip
chord**; AeroBO's has its own, up to 3x, and ramps it on this same `w`.

### 14d. THE WING PAYS FOR THE REACH

Arc length is the invariant, so a blended plate is the same size of plate --
what it trades is **tip height for outboard reach**. That reach comes out of
the span row, because the span row is what the car is allowed to be wide: a
flank panel's span is its vertical extent between sill and roof, a top wing's
is the car's width. `wing.plate_flown` is where the three dependent numbers
(the wall, the arc after its clearance clip, the span left) are worked out
once.

This is load-bearing, and the measurement says so. With the reach unpaid:

```
blend  0.0   0.2     0.4     0.6     0.8     1.0
paid  +0.0000 +0.0046 +0.0079 +0.0120 +0.0168 +0.0221 s      e 1.41 -> 1.82
free  +0.0000 -0.0062 -0.0138 -0.0205 -0.0263 -0.0317 s      e 1.41 -> 1.70
```

The free column is AeroBO's own fixed bug reproduced exactly -- *"nothing in
the model opposed curling the plate into a quarter-round winglet"*.

The clearance clip is inverted through `height_fraction` rather than applied
to the arc: a blended plate does not climb as far, so it may be **longer**
before it touches the wall. The path is homogeneous of degree one in the arc,
so that inverse is one division.

### 14e. The corner's own drag -- an ADD-ON, off by default

A lifting-surface method values a corner only through the wake line it draws,
so the interference drag of two surfaces meeting at an angle is invisible to
it -- and that drag is the entire reason to blend. `blend.junction_report`
charges Hoerner's unfilleted correlation at the two corners with a fillet
credit for the blend radius. The credit is a **calibrated shape, not a
measurement** (linear decay to `FILLET_CREDIT_FLOOR = 0.15` at `R/c = 0.10`),
which is why it is its own row, why it reports the **uncredited** number
beside it, and why raising the blend off zero is what switches it on --
AeroBO's *"the junction charge defaults ON with the blend"*.

The junction member is the **PLATE**, so:

* a plate with no section is bare sheet, below the correlation's own root
  (it goes negative under t/c ~ 0.054), and charges **nothing** rather than
  being extrapolated;
* a NACA 0012 plate on the flank panel charges **32.3 ct sharp**, and blend
  0.6 credits that to **5.5 ct**.

### 14f. Does it pay? Measured: no, on this car

2 roles x 3 circuits x 3 shapes x 6 blends, plus a scan over 4 plate depths
and 3 plate sections:

```
flank e423, 0.12 m NACA0012 plate, arena
  blend 0.2  +0.0046 s    e 1.483   S 0.3208   junction 16.9 ct
  blend 0.6  +0.0120 s    e 1.645   S 0.2965   junction  4.1 ct
  blend 1.0  +0.0221 s    e 1.819   S 0.2723   junction  4.5 ct
best case anywhere in the scan:  +0.0013 s  (flank, 0.06 m goe795, blend 0.10)
```

The curve **does turn over inside the range** -- the credit saturates near
blend 0.1, which is where the cost is least -- which is AeroBO's own
`test_blending_is_no_longer_a_free_lunch` shape. It just never gets below the
square corner here: the span the reach costs is worth more than the corner it
smooths. That is an answer about **this car at these plate depths**, not a
defect. AeroBO's endplate is a 3x-chord surface whose junction charge is far
larger, so the same rows turn over below zero there.

### 14g. Where it is asked

`WING TYPE`, beside the mount -- a **setting**, not a design coordinate, which
is how AeroBO carries its own (`replace(prob, blend_frac=...)`, never a row of
X): it decides what geometry is being built, not where inside it to look.

```
mount                  pylon / endplate
plate blend            0.00 .. 1.00          (locked with no plate)
blend shape            arc / smooth / spiral (locked at blend 0)
junction interference  not charged / charged
```

`Param.enabled` takes a **callable** now, read every frame: a row that is only
answerable while something else is true (there has to be a plate to blend) had
a bool captured at build time, and the plate height lives one panel away.

Two views changed so what is drawn is what was flown: the **planform** is
drawn at the flown span with the plates' ribbon reaching outboard of it, and
the **loading** stage gained *"wing and plate, seen from behind"* -- the one
view the planform cannot show, because the transition happens out of the
planform's plane.

### 14h. Not ported, and stated

* **The wing-side arc.** AeroBO can start the turn inboard of the tip and bend
  the wing's own panels into it. `blend`'s law takes the argument;
  `vlm.Lattice` passes 0. AeroBO wants one because a device-side blend has
  `R <= h/cant`, *"a fraction of a tip chord"*, so its credit saturates at
  once -- but that is about ITS scale. A carsim flank plate at h 0.12 m and
  blend 0.6 is already at `R/c = 0.102`, past `FILLET_FULL_R_OVER_C`, so
  there is nothing for a wing-side arc to buy.
* **A variable or signed cant.** carsim's plates stand normal to the wing.
  The functions take a cant because the turn law is normalised by it.
* **The chord ramp.** No chord step exists here to ramp over.

### 14i. The numbers, after

```
python3 -m drive.validate     ->  82/82 pass  0 HARD  0 soft  [114.1 s]
python3 -m drive.garage       ->  ALL PASS
drive.aero.{airfoil, panel2d, polar, blend, vlm, wing, screen, section,
            mission, optimize, library}  ->  11/11 ALL PASS
```
