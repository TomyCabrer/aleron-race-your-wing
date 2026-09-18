# Wave 8 — the fourth cell: the panel on the INNER flank

> *"The previous study swept the section orientation with the flank held at the
> outer flank. That only covers two cells of a two-by-two. What about the inner
> flank with the section turned over?"*

He is right that it is the one cell nobody tested, and it is not obviously
wrong. **The answer is: it is worth +0.02 to +0.48 percentage points of corner
speed, every bit of it is the device's own DRAG yaw moment changing sign, it is
paid for out of the rear axle's margin — the one quantity this whole study is
bounded by — and on the crude published plate it puts the car on its roof in
the arena's wet patch. Default unchanged, and it is now a selectable option so
he can measure it himself.**

Confidence: **high** on the matrix and the mechanism (they are algebra plus a
steady rig, and the mechanism is exact to machine precision). **Low** on
whether the inner flank's panel would make the lift the lattice says it makes,
for a stated reason that no amount of work inside this model can fix. Both are
below.

---

## 1. Ground truth first: which flank, and was it ever chosen?

**It was never chosen. It was assumed, by a chain of models none of which
contains the term that distinguishes the two flanks.** I went looking for a
recorded reason and the absence is itself the finding.

| where | what it says |
|---|---|
| `crossover.gain(k, R, x_w)` | a closed form in `(k, R, x_w)`. **No `y`, no device drag moment at all.** |
| `qss.residuals` | yaw balance `Y_f a - Y_r b + F x_w = 0`; transfer `dFz_tot = (m a_y h_cg - F h_w)/t`. **No `y` in either.** |
| `ledger.ledger` | charges the device's drag as a **power** loss (`s_drag`), never as a moment. |
| `specs/chassis.txt:138` | "`y_dev` … = 0.72 (on the outer flank, so `y_dev = -sgn_dev*0.72`) (derived) :: half the mean track" |
| `specs/chassis.txt:50` | "the drag term is ~50 Nm against ~215 Nm from the side force, **include it for completeness**" |
| `specs/numerics.txt:44, 50` | the outer flank, asserted the same way |
| `vehicle.py:133` | "half the mean track: the panel sits on the **OUTER** flank" |
| CONTRACT §4 / `vehicle.py` DEVIATION 2 | "the drag term yaws the car OUT of the turn, **which a force on the outer flank must do**" — a consistency check on the implemented geometry |
| reconciliation 7 / `numerics.txt:282` | the side-selection argument (`tanh(ay_filt)` vs `sign(steer)`) is entirely about **left vs right**. The outer flank is its premise. |

So the only remark anywhere about the term that distinguishes the flanks is
`chassis.txt`'s **"include it for completeness"** — a reason for not asking the
question, not an answer to it.

There is **no packaging reason** either: the garage fits a panel to *both*
flanks and `CarBuild.mass_points` charges *both* masses
(`.handoff/07-weight.md`), the standoff is the same `FLANK_STANDOFF = 0.45` on
each, and `cars.py:492` records that "each flank is symmetric, so the only
y-offset that could exist cancels". The hardware is on both sides whichever one
deploys. The choice is purely which one is commanded out.

**And `.handoff/04-wings-audit.md` §3 did NOT already measure this.** Its
"inner flank forced" row (−0.9644 % fin / −1.9374 % plate) was a
*counterfactual for the flank-latch bug*: it forced `sgn_dev` to the wrong
sign, which flips the **side force** as well as the station. That measures
cell 3 of the matrix below, not cell 4. The audit says so itself — "two
independent ways of putting the panel on the **wrong** flank".

### One real merit of the outer flank, which the study never claimed

Found on the way and worth recording, because it is the strongest argument for
the shipped configuration and nobody had made it. The device's lift grows with
body slip (`alpha_dev = inc + sgn*(-beta_dev)`), so its yaw moment grows with
slip too — the device is **yaw-destabilising on either flank**. On the outer
flank the drag term partially cancels that; on the inner flank it adds to it.
Measured at a frozen R = 100 state, `d Mz_dev / d beta_dev`:

| panel | outer flank | inner flank | |
|---|---|---|---|
| fin | **−370.4** N·m/rad | **−594.2** N·m/rad | +60.4 % |
| plate | **−385.4** N·m/rad | **−618.2** N·m/rad | +60.4 % |

The ratio is exactly `(x_w + Y_DEV/LD)/(x_w − Y_DEV/LD) = (0.97 + 0.225) /
(0.97 − 0.225) = 1.604`, which is the algebra rather than a coincidence. **The
outer flank buys 60 % less yaw-stiffness loss for the same side force.**

---

## 2. Can the model answer this honestly? Partly, and I am telling you which part

### What sideslip the car actually reaches — measured, not guessed

The inner flank is the **leeward** flank: the apparent wind in a left turn is
`(-u, +|v|)`, blowing right-to-left, so the right (outer) flank is windward.
A rigid-wall image models blockage and a ground-effect-like lift rise. It does
**not** model separation, a wake, or a velocity deficit. So the question is at
what angles the leeward flank has to stay attached.

| rig | \|beta\| | the panel's own \|beta_dev\| |
|---|---|---|
| scripted arena lap, plate, deployed samples | mean **1.21°**, p50 1.06, p90 2.55, p99 3.37, max **4.21°** | ~0.6° less |
| … deployed time above 5° | **0.0 %** | |
| R = 100 open-loop limit, parity, wing off | 4.12° | 3.56° |
| … fin outer / inner | 4.57 / 4.97° | 4.01 / 4.42° |
| … plate outer / inner | 5.24 / 7.12° | 4.69 / 6.49° |
| R = 100 limit, drive model, plate | 9.35° (outer), 12.0° = the rig's abort (inner) | 8.66 / 11.08° |

So: **1–4° in anything driven, 4–7° at the steady grip limit, and past that the
car is departing anyway.**

### What the model does and does not contain

Searched for it rather than assumed: there is **no** representation of
separation, a wake, a leeward/windward asymmetry or a boundary layer anywhere
in `drive/` or `specs/`. The car's entire side aerodynamics is the linear
`Fy_body = -q*A*Cs_psi*beta` with `Cs_psi = 2.2 /rad`, which `ledger.py` swept
over 1.0–3.0 precisely because it is **unmeasured and there is no CFD**
(`specs/chassis.txt:131`). The panel's section polar is a 2-D attached-flow
estimate (or XFOIL), and the panel is handed the **free-stream** dynamic
pressure `q = ½ρV²` on either flank, with no velocity deficit.

### So: one error I can bound, and one I cannot

**Bounded — the wall interference.** The wall-distance sweep (committed with
`wall_side`) says exactly what the image is worth, and the worst case is that
it is worth nothing. `flank-e423` at 5°:

| | CL | CDi | e | vs free air |
|---|---|---|---|---|
| free air (no body at all) | 0.7520 | 0.08040 | 1.1647 | — |
| wall on the **suction** side (cells 1, 3) | 0.7988 | 0.08214 | 1.2862 | **+6.22 %** CL |
| wall on the **pressure** side (cells 2, 4) | 0.7827 | 0.07984 | 1.2706 | **+4.08 %** CL |

A thickened or separated leeward boundary layer makes the wall *softer and
further away*, and the sweep is monotone in wall distance, so the whole
possible error is the inner flank losing its 4.08 % back. **Note which way that
cuts: the shipped outer-flank configuration leans on the image MORE (+6.22 %)
than the inner one does (+4.08 %).** The image is not the inner flank's problem.

**Not bounded — the panel working in a wake.** If the leeward flank at the
panel's station is separated, the panel sees a reduced `q` and a distorted
inflow, and it loses lift roughly in proportion to the deficit. That is a
first-order loss (tens of percent), the lattice will not notice it, and
**nothing in this repo can tell you whether it happens.** What I can say about
it, stated as judgement and labelled as such:

* the panel's station is `x_w = +0.97 m` and the front axle is at
  `a = 0.9715 m` — it sits **on the front axle line**, roughly a quarter of the
  way back from the nose, well forward of where a yawed car's leeward-side
  separation grows;
* at the 1–4° the lap actually reaches, the leeward flank that far forward is
  very likely still attached — the flow there is set by the nose and A-pillar
  acceleration, not by rear-body separation;
* at the 6–7° the steady limit reaches with the plate it is **marginal**, and I
  would not defend a number there;
* the front-wheel wake is unmodelled on **both** flanks, so it is not a
  differential risk; the body's own sidewash is unmodelled on both but IS
  differential in sign, and unbounded.

**What would settle it**: a yawed RANS or a wind-tunnel yaw sweep of the body
with the panel in place at 0 / 3 / 6 / 9° — the same measurement `Cs_psi` has
been waiting for since `ledger.py` was written. Two runs (windward and leeward)
at 6° would do it. Until then the numbers below are **rigid-wall-model
numbers**, and that is the honest label.

---

## 3. The four-cell matrix, on equal terms

`flank-e423`, inc 4°, `x_w = 0.97`, `h_w = 0.90`, R = 100 m open-loop ramp
steer, **`qss_parity`** (CL frozen at `CL0 + CLa·inc`, which is the mode
`wing_ab` calls the answer because it is the only one comparable with
`crossover.gain`). Wing off: 28.9426 m/s, 0.8539 g, `util_f` 0.9926,
`util_r` 0.9376, understeer-margin cap `sqrt(util_f/util_r) − 1 = +2.8915 %`.

Cells 2 and 3 are the same panel with its section mirrored, so `CL0`, `CLa` and
`cd1` all negate — the drag polar mirrors with the lift, as it must.

| cell | flank | suction | CL | F_dev | D_dev | Mz_dev | = lift | + drag | gain | util_f | util_r | limiting |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | outer | inboard | +0.749 | **+128.6** | 14.66 | **+114.2** | +124.7 | **−10.6** | **+1.2653 %** | 0.9936 | 0.9615 | front |
| 2 | outer | outboard | −0.734 | −119.8 | 13.63 | −126.0 | −116.2 | −9.8 | **−1.2682 %** | 0.9916 | 0.9111 | front |
| 3 | inner | inboard | −0.749 | −122.3 | 13.95 | −108.6 | −118.7 | +10.0 | **−1.2151 %** | 0.9917 | 0.9145 | front |
| **4** | inner | outboard | +0.734 | **+126.1** | 14.35 | **+132.7** | +122.4 | **+10.3** | **+1.3329 %** | 0.9937 | 0.9658 | front |

**Cell 4 is the best of the four, by +0.068 pp over the shipped cell 1.** Cells
2 and 3 are the two losers and they lose by essentially the same amount, which
is the point: *what makes them lose is the side force pointing away from the
turn centre, not which flank they are on.* The flank is worth ±0.05 pp on top
of a ±1.27 pp force-direction effect.

The same measurement on the published closed form, where the panel's drag is
the study's crude `CL/3.2` rather than a fitted polar and is therefore much
larger:

| panel | cell 1 (shipped) | cell 4 | change | cell 1 util_r | cell 4 util_r |
|---|---|---|---|---|---|
| fin | +1.1695 % | **+1.4377 %** | **+0.268 pp** | 0.9576 | 0.9708 (front) |
| plate | +2.2029 % | **+2.6755 %** | **+0.473 pp** | 0.9778 | **0.9993 — REAR-limited** |

`+2.2029 %` for the shipped plate in parity is the audit's own published number
(`.handoff/04-wings-audit.md`), reproduced to the last digit, which is the
cross-check that this measurement is on the same rig as the old ones.

### Radius sweep — the gain is real, monotone, and bought with rear margin

Parity, cell 4 minus cell 1, in percentage points of corner speed:

| R (m) | fin | plate | designed | cell 4 steer change | cell 4 limiting |
|---|---|---|---|---|---|
| 30 | +0.070 | +0.128 | +0.018 | −0.06 … −0.12° | front / front / front |
| 50 | +0.118 | +0.222 | +0.030 | −0.12 … −0.41° | front / front / front |
| 75 | +0.176 | +0.467 | +0.044 | −0.29 … −0.62° | front / front / front |
| 100 | +0.268 | +0.473 | +0.068 | −0.40 … −0.84° | front / **rear** / front |
| 130 | +0.482 | +0.107 | +0.141 | −0.17 … −1.26° | **rear/spin** / **rear/spin** / front |
| 175 | +0.435 | −0.020 | +0.103 | −0.27 … −1.07° | **rear/spin** / **rear/spin** / **rear** |

Three things fall out of that table and all three matter:

1. **It grows with radius**, because the drag moment goes as `q` and the
   benefit is the drag moment.
2. **It grows with the panel's DRAG, not its lift.** The fitted `flank-e423`
   polar makes 14.7 N of drag where the published fin's assumed `L/D = 3.2`
   makes 53.7 N, and the designed panel accordingly gains only +0.02 … +0.14 pp.
   **On a well-designed panel the whole effect is nearly nothing** — which is
   the practically relevant case.
3. **Past R ≈ 100 the extra gain is the rear axle going limiting.** `util_r`
   rises in every single pair. The plate's cell 4 is rear-limited at R = 100
   and the ramp steer aborts on `|beta| > 12°` at R = 130. `numerics.txt`'s own
   T17 says this in advance: "a gain above the understeer-margin cap
   `sqrt(mu_r/mu_f) − 1` is only ever reachable by making the REAR axle the
   limiting one, and on a FWD hatch that is a spin, not a lap time."

---

## 4. The mechanism, exact rather than inferred

Evaluated at a **bit-identical state** (`u` 29.757, `v` −2.893, `r` 0.29788,
`delta` 7°, `dep` 1, fin), so nothing can drift between the two runs:

| term | outer | inner | difference |
|---|---|---|---|
| `q`, `V`, `beta`, `dep`, `sgn_dev` | | | **0.000000** |
| `CL_dev`, `alpha_dev` | 0.915602 | 0.915602 | **0.000000** |
| `F_dev` | 171.866031 | 171.866031 | **0.000000** |
| `D_dev` | 53.708135 | 53.708135 | **0.000000** |
| `y_dev` | −0.72 | +0.72 | +1.44 |
| **`Mz_dev`** | **128.040** | **205.380** | **+77.340 = 2·Y_DEV·D_dev** |
| `D_aero`, `Fy_body`, `Mz_body` | | | 0.000000 |

`Mz_dev`'s lift term is `F_dev·x_w = +166.710` N·m either way; the drag term is
**23.2 %** of it and it is the only thing that moves. That is the brief's
"about 23 % of the device's yaw authority", confirmed.

### And the roll consequence is EXACTLY ZERO, which is worth saying plainly

The brief asked whether the inner flank's side force adds to or subtracts from
roll moment and load transfer, since it acts at `h_w = 0.90`. **It does
neither, by algebra: a pure LATERAL force has no lateral moment arm.** The roll
moment is `-F_dev*(h_w - h_ra)` and the transfer share is
`F_dev*(h_cg - h_w)/t_bar`; neither expression contains `y`. Measured at the
identical state above: roll moment **−127.181 N·m** and transfer share
**−42.228 N** on *both* flanks, to every digit.

Roll gradient at the R = 100 parity limit, which is the number a driver feels:

| | deg/g | |
|---|---|---|
| wing off | 5.3325 | |
| fin outer / **inner** | 5.1657 / **5.1659** | a difference of 0.0002 deg/g = **0.004 %** |
| plate outer / **inner** | 5.0349 / **5.0355** | 0.012 % |
| designed outer / **inner** | 5.1670 / **5.1690** | 0.04 % |

The device's whole roll benefit survives the move intact. The absolute roll
angle is a few hundredths of a degree higher on the inner flank only because
the car is going faster (4.515 → 4.539° at 0.8740 → 0.8786 g for the fin).
**Nothing about roll offsets the yaw gain, in either direction.**

---

## 5. Directional stability — where this stops being a bookkeeping change

MATCHED-STEER trim on the rigs' own `_rig` (speed held, wheels free-rolling),
25 m/s, road wheel ramped to 6° over 1 s and held 8 s. Same input, so the
difference is the car:

| panel | flank | a_y (g) | r (rad/s) | beta (deg) | path R (m) | util_f | util_r |
|---|---|---|---|---|---|---|---|
| off | — | 0.8422 | 0.3310 | −3.455 | 75.53 | 0.9643 | 0.9153 |
| fin | outer | 0.8674 | 0.3410 | −3.866 | 73.32 | 0.9728 | 0.9405 |
| fin | **inner** | 0.8786 | 0.3476 | −4.285 | **71.93** | 0.9797 | 0.9592 |
| plate | outer | 0.8869 | 0.3514 | −4.312 | 71.14 | 0.9801 | 0.9606 |
| plate | **inner** | 1.0732 | **2.8192** | **−81.3** | **8.87** | 0.9136 | 0.9387 |

**The fin on the inner flank does exactly what the owner predicted** — 1.4 m
tighter radius for the same steering input, i.e. less steer for a given path.
**The plate on the inner flank departs.** Not at the limit of a ramp steer: at
25 m/s and six degrees of lock, which is an ordinary mid-corner input that the
same car on the outer flank holds at `util_f` 0.98.

The mechanism is the yaw-stiffness table in §1 closing the loop: pro-turn
moment → more yaw rate → more body slip → more device CL → more pro-turn
moment, with 60 % less of the drag term's damping to oppose it. On the outer
flank the drag term is what damps that loop. Nobody designed it to; it is a
by-product of the flank the study assumed.

---

## 6. On a lap — and WHERE, before anything is called aerodynamics

Three findings in this batch turned out to be the arena's `WET_T3` (130 m,
s = 455–585, mu 0.632) wearing an aero costume, so the lap delta is
decomposed by track feature first. Scripted arena lap, `margin 0.90`, wet
patches on, second lap (deterministic to 0.2 ms):

| panel | wing off | cell 1 (outer) | cell 4 (inner) | inner − outer |
|---|---|---|---|---|
| fin | 60.8355 | 60.8076 (**+0.0279 s**) | 60.7477 (**+0.0878 s**) | **−0.0599 s** |
| plate | 60.8355 | 60.6882 (**+0.1473 s**) | 60.7711 (**+0.0644 s**) | **+0.0829 s** |
| designed | 60.8355 | 60.8890 (**−0.0535 s**) | 60.7241 (**+0.1114 s**) | **−0.1649 s** |

`+0.0279` and `+0.1473` are wave 7's own corrected fin and plate lap numbers,
reproduced — so this is the same lap, not a new one. And then the answer to
"where":

**The plate loses on the inner flank inside the wet patch, and it is the
surface, not the aerodynamics.** Of the +0.0844 s it gives away, **+0.0746 s is
`s4` (+0.0493) and `T4` (+0.0253)** — the 50 m straight out of T3 and the
corner after it. The cause is one step earlier: the plate-inner run's `util_r`
peaks at **0.9986 at s = 487.3, mu 0.6322 — inside `WET_T3`** — against 0.9906
for the plate-outer run at the same place. The rear axle goes to its limit in
the wet corner, the driver backs out, and the car leaves `s4` **0.65 m/s
slower** (21.23 against 21.88 at s = 630) *with the panel stowed and mu back at
1.0*, i.e. the loss is a carried exit-speed deficit, not drag. In the dry the
inner flank is ahead at every radius (§3). **The wet patch is where the rear
margin is thinnest, and the extra oversteer moment is what tips it over** —
which is `ledger.py`'s "the wet is this device's best case" seen from the other
side.

**The designed panel's −0.1649 s is the LapDriver, not the flank, and I am not
claiming it.** −0.1651 s of accumulated delta, of which **T6 +0.0632, s7
−0.0757, T7 −0.0884** and the start straight −0.0445. At s ≈ 1090 in T6 the two
runs' paths diverge; from s = 1110 to the line the inner run is 0.6–0.8 m/s
faster *with `util_f` and `util_r` between 0.2 and 0.4* — nothing is
grip-limited anywhere in T7, which is the power-limited corner. The outer run
still has the panel out (`dep` 0.99 against 0.00) at s = 1130, i.e. it is still
correcting. That is a closed-loop controller artefact of exactly the kind
`wing_ab`'s docstring warns about, and it is bigger than the effect being
measured.

**Conclusion on the lap: it cannot resolve this.** The differences are
0.06–0.16 s on a 60.8 s lap (0.1–0.27 %) and the largest single contributions
trace to one corner exit. Only the fin's −0.0599 s is distributed across
corners in the direction the rig predicts. The open-loop per-corner rig stays
the figure of merit, as it has been since `.handoff/10`.

---

## 7. Does the side-selection and deploy/stow logic survive the inversion?

Checked, not assumed — `.handoff/04-wings-audit.md` §3(b) pinned the panel on
the wrong flank for 36.6 % of cornering time, so this logic has been wrong
before. `dev_flank` deliberately does **not** touch the latch: it changes only
which *slot* the latched side reads and the sign of `y_dev`.

**A. the right slot is read.** Two *different* panels fitted (left CL 0.50,
right CL 1.00) so the telemetry says which one was used:

| flank | turn | `sgn_dev` | slot read | expected | `y_dev` | |
|---|---|---|---|---|---|---|
| outer | left | +1 | RIGHT | RIGHT | −0.72 | ok |
| outer | right | −1 | LEFT | LEFT | +0.72 | ok |
| inner | left | +1 | LEFT | LEFT | +0.72 | ok |
| inner | right | −1 | RIGHT | RIGHT | −0.72 | ok |

**B. the timing is bit-identical, as it must be.** Chicane (left 3 s, right
3 s, straight 2 s): the flank swaps at **t = 3.2990 s with `dep` = 0.0001** on
*both* settings — the audit's own 3.299 s — stows in **0.2990 s** after the
wheel centres (`t_ret` 0.30), never has two panels out, and spends 4.32 % of
cornering samples on the not-yet-latched side (the 0.30 s hold), identically.

**C. on the lap**: wrong-flank exposure while cornering **0.16 %** (outer) and
**0.40 %** (inner), mean `dep` 0.5722 / 0.5711, mean `D_dev` 18.162 / 18.144 N.
The 0.16 → 0.40 is the different steering trace of a car that turns slightly
more, not a logic difference. No regression.

---

## 8. What was built, and what was NOT

* `VehicleConfig.dev_flank` = `'outer'` (**default, bit-for-bit**) | `'inner'`,
  and `--dev-flank outer|inner` on the CLI, through `_build`, the lap script,
  the skidpad rigs, `wing_ab` and the interactive session. The telemetry
  sidecar records it.
* `wing.build_lattice(..., wall_side=+1|-1)`, through `analyse` / `spanwise` /
  `Library.analyse_wing`, returned in the stored aero. `vehicle.py` never
  learns what a wall is — the orientation arrives inside a `DevAero`'s CL/CD
  laws, exactly as a mount does.
* **The default is NOT changed**, and nothing in the garage UI selects the
  inner flank. Two reasons. It wins by 0.07 pp on the panel anyone would
  actually build; and it departs at ordinary steering inputs on the panel the
  study publishes. An option that is worth a rounding error when it is safe and
  spins the car when it is not does not belong on by default.
* `--dev-flank inner` on its own only **moves** the panel; it does not turn the
  section over. On the published `fin`/`plate` closed form there is nothing to
  turn over (the law has one CL), so `--wing fin --dev-flank inner` IS cell 4
  for that panel. For a designed panel, build it with
  `wing.analyse(..., wall_side=-1)`. The CLI help says so.

## 9. What this changes in the earlier notes

**`.handoff/13-curved-flow-and-orientation.md` is corrected in place**, not
left to be reconciled. Its orientation conclusion is right and stays, but it is
stated in general terms and it is **conditional on the flank**:

* "There is no configuration in which the suction surface faces outboard and
  the useful force is still inboard" is true **only on the outer flank**. On
  the inner flank, outboard *is* toward the turn centre — that is cell 4, and it
  is the best of the four.
* Its "What would change this answer" paragraph named a rear-limited car. That
  was too narrow: what actually changes it is **moving the panel to the other
  flank**, which is a free choice, not a different car.
* Its lattice table quotes `e = 1.0802` for *both* orientations, which cannot
  come from moving the wall (the induced-drag geometry changes with it: this
  note measures 1.2862 against 1.2706 on `flank-e423`). It looks like the
  section was mirrored with the wall left where it was. `wall_side` moves the
  **body**, which is what physically happens when the same panel is hung on the
  other flank, and it is validated against free air in the distant-wall limit.
  The qualitative result is the same either way — wall on the suction side makes
  more lift for more induced drag — so wave 7's conclusion is unaffected.

Nothing in `04`, `10` or `12` changes. The `y_dev` / `Mz_dev` algebra in
`04-wings-audit.md` §1 is still exactly right for the outer flank and is now
the `flank = +1` case of a two-case formula.

## 10. Recommendation

**Keep the outer flank as the default. High confidence.**

Not because the inner flank loses on the numbers — on the rigid-wall model it
wins everywhere in the dry — but because of what it costs and what the win is
made of:

1. **The win is small where it is safe.** On a panel with a fitted polar
   (`flank-e423`, `L/D` ≈ 53) it is +0.02 to +0.14 pp of corner speed. The
   effect *is* the drag moment, so designing the drag out of the panel — which
   is what the whole `drive/aero` half of this repo exists to do — designs the
   benefit out with it.
2. **The win is large only where it is dangerous.** It reaches +0.47 pp on the
   published plate, and the same configuration is rear-limited at R = 100,
   aborts the ramp steer at R = 130, departs at 25 m/s and 6° of lock, and
   gives away 0.08 s a lap inside the arena's wet patch by driving the rear
   axle to `util_r` 0.9986.
3. **It costs 60 % more yaw stiffness** for the same side force
   (`dMz/dbeta` −370 → −594 N·m/rad), on a car whose rear margin is the
   `sqrt(mu_r/mu_f) − 1 = +2.47 %` ceiling that bounds the entire study.
4. **It puts the panel on the leeward flank**, where this model's rigid wall is
   least defensible and where the one error I cannot bound lives.
5. **It gains nothing in roll** — exactly nothing, 0.004 % of roll gradient.

The honest framing for the owner: *the fourth cell is a real +0.27 pp on the
published fin and it does reduce the steer angle required, which is the
mechanism you predicted and it is confirmed. But it buys that by spending rear
margin, and rear margin is the thing this car has least of and the thing the
study's own cap is written about. On the panel you would actually build it is
worth 0.07 pp — inside the noise of the lap and below the 4 % of CL that the
body-image model is uncertain by.* It is a selectable option;
`--dev-flank inner` and the numbers above are there to be re-run.

**What would change this recommendation**: a yawed-body measurement showing the
leeward flank attached at 6°, *and* a rear axle with margin to sell — either a
stiffer front bar taking `roll_dist_f` up, or the RWD cars in `cars.py`, where
the rear is the limit and an oversteer moment is the wrong direction anyway.
The second half is why this is unlikely to become the right answer for this
car.

---

### Numbers a later reader should not re-derive

| claim | value |
|---|---|
| the drag term's share of the device's yaw moment | **23.2 %** of the lift term (R = 100, fin) |
| the flank's whole effect on `Mz_dev` | `2·Y_DEV·D_dev` = **+77.340 N·m** |
| the flank's effect on `F_dev`, roll moment, load transfer | **exactly 0**, all three |
| cell 4 minus cell 1, parity, R = 100 | fin **+0.268 pp**, plate **+0.473 pp**, designed **+0.068 pp** |
| steer angle change, R = 100 | fin **−0.841°**, plate **−0.807°**, designed **−0.396°** |
| roll-gradient change | **+0.0002 deg/g** (0.004 %) |
| `dMz_dev/dbeta_dev`, outer → inner | **−370.4 → −594.2** N·m/rad (fin) |
| plate on the inner flank, 25 m/s, 6° lock | **departs**, beta −81.3°, r 2.82 rad/s |
| \|beta\| while the panel is deployed on a lap | mean **1.21°**, max **4.21°**, 0.0 % above 5° |
| body image, CL vs free air | **+6.22 %** suction side, **+4.08 %** pressure side |
