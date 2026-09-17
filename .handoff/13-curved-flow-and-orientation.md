# Wave 6 — the owner's two aero questions

> *"The side wings would be affected by the curve AoA. It might be better to
> have the foils in the opposite side (suction side outwards)."*

## Item 1 — the curved-flow angle of attack. He is right, and it matters more than it looks

`alpha_dev` tracked the **body** sideslip only. A body-fixed point at
`(x_w, y_dev)` moves at `V + omega x r`, which with `omega = (0,0,r)` is
`(u - r*y_dev, v + r*x_w)`, so the panel's own flow angle is
`atan2(v + r*x_w, u)` and not `beta`. The `y_dev` term only perturbs the axial
component and is second order in the angle.

**The sign is the opposite of the intuition, and I checked it before writing
any code.** In a left turn `r > 0` and `v < 0`, so `v + r*x_w` is *less*
negative for a panel **ahead** of the reference point: `|beta|` falls, and a
forward-mounted panel therefore sees **less** incidence in a corner, not more.

Measured, R = 100 open-loop ramp steer, `x_w = 0.97`:

| | beta → beta_dev | CL | ramp-steer gain |
|---|---|---|---|
| fin | −5.807 → −5.200° | 0.9503 → 0.9215 (−2.76 %) | +3.8587 → +3.7412 % |
| plate | −7.631 → −6.944° | 1.5789 → 1.5448 (−1.88 %) | +6.0473 → +5.9449 % |

**And on a lap, which is where it now decides something.** The device's
surviving lap benefit is small enough (wave 5 retired the claim that the
baseline cannot lap without it) that a few percent of device force is the same
order as the effect being measured. Scripted arena lap, `margin 0.90`:

| | device worth, uncorrected | corrected | change |
|---|---|---|---|
| fin | +0.0582 s (+0.096 %) | **+0.0279 s (+0.046 %)** | **more than halved** |
| plate | +0.1594 s (+0.262 %) | **+0.1472 s (+0.242 %)** | −7.7 % |

So: **the correction does not overturn the device — the sign of the benefit
survives on both panels, so wave 2's lap re-conclusion stands — but for the
weaker panel it halves it.** That is the headline.

`cfg.dev_curved_flow` defaults **False**, a documentation decision and not a
physics one: CONTRACT §4 writes the published closed form and every W-group
acceptance number is quoted against it. `True` is the physically correct model.
Applied to **both** branches (designed `DevAero` and the legacy fin/plate) so
they cannot disagree; `qss_parity` freezes it exactly as it zeroes `dCLda`.

### Do the ML checkpoints need retraining? No.

Stated plainly because it was asked. `dev_curved_flow` defaults False and
`env.rollout` never sets it, so **every checkpoint was trained and is measured
in exactly the same model** — there is no mismatch to correct. And were it
enabled, the effect is ~2–3 % of a device force that is itself worth 0.14 % of
lap time, i.e. ~0.004 % of lap time: two orders below the ES's own
iteration-to-iteration noise. Retraining would measure noise.

### Does the top wing need the equivalent term? No, and for a stated reason.

The flank term exists because the panel is offset **longitudinally** and yaw
rate therefore changes its local *sideslip*. The top wing's force is vertical,
so the equivalent coupling would be **pitch** rate acting on its `x_t` offset,
changing its local angle of attack. `VehicleState` has no pitch DOF — pitch
enters only as the lagged `dFz_x` load-transfer state — so there is no `q` to
form the term with, and the contract already says the top wing is held at its
mounted incidence because "the car's pitch and heave move it by well under a
degree". Consistent, and not an omission.

## Item 2 — "suction side outwards". THE SIM CANNOT ANSWER THIS YET, and that is the finding

First, what the device is *for*, from the study's own material: `F_dev =
sgn * dep * q * S_DEV * CL` with CONTRACT §4's "`+y` = **inward** for a LEFT
turn", and `CL0 = +0.70`. So the useful force is **inboard**, toward the turn
centre — the panel is a lateral-force generator supplementing the front tyres,
which is why `crossover.py` frames it against lateral grip and `R_cap`.

There are two readings of "suction side outwards" and they are not the same
question:

**Reading A — reverse the force.** Flip the camber so `CL0 = −0.70`. Measured,
same rig, against 8.6084 m/s² wing-off:

| panel | CL_dev | F_dev | peak a_y | gain |
|---|---|---|---|---|
| as shipped (suction inboard) | +0.9503 | +168.85 N | 8.9405 | **+3.8587 %** |
| camber reversed | −0.5284 | −93.88 N | 8.4067 | **−2.3431 %** |
| camber and incidence reversed | −0.8610 | −152.98 N | 8.2857 | **−3.7482 %** |

Unambiguous: the force points outboard and the device becomes a **penalty**.
Not a track-location artefact — I checked, having traced three earlier
"findings" to the arena's surfaces; here the mechanism is simply the sign of
`F_dev`, visible in the steady rig.

**Reading B — same inboard force, but the low-pressure surface facing
outboard into clean air instead of into the gap between panel and body.** This
is almost certainly what he means, and it is an *interference* question.
**The model cannot see it.** `wing.build_lattice` sets `image = None` for
`role == 'flank'` — the flank panel is solved in **free air, with no
representation of the car body at all**. `garage.py`'s mesh comment
("suction side towards the car") is a drawing convention, not physics. With no
wall in the solution, mirroring the section is *exactly* equivalent to negating
`CL`, which is Reading A. **Any answer the sim gives about Reading B today
would be an artefact of the force sign, not a statement about interference.**

### The path to answering it properly, and its cost

The machinery exists. `vlm.Lattice` already takes `image_z` / `image_sign` and
uses them for the top wing's ground plane, and the flank panel is solved in a
frame where its span is the lattice `y` and its **lift is the lattice `z`** —
so a rigid wall perpendicular to the lift, at the panel's standoff from the
body (`DEV_OUT0 + DEV_OUT1`, ~0.45 m deployed and less stowed), is the same
construction as ground effect. Add that and the two orientations genuinely
differ, because the wall is then on the suction side in one and the pressure
side in the other.

The cost, stated up front: it changes `CL0`/`CLa` for **every flank wing in
the library**, so every designed panel and every garage read-out moves, and
the flank polars would need re-analysing. The legacy fin/plate closed form —
and therefore every W-group acceptance number — is untouched, because it does
not go through the lattice.

~~**I stopped here deliberately**~~ — **BUILT IN WAVE 7, and the question is
now answered.** See below.

## Wave 7 — the body image, and the answer

`build_lattice` now gives the flank panel the car's flank as a **rigid-wall
image**, the top wing's ground plane one frame over (commit `3702794`).
Validated against free air in the distant-wall limit and monotone in between:
+5.20 % on the lift curve at the real 0.45 m standoff, +1.27 % at 0.90 m,
+0.30 % at 1.80 m, **+0.01 % at 12 m**. Span efficiency 0.9996 → **1.0802** —
above 1, which is the signature of the image, induced drag falling below the
free-air elliptic limit. The loading is **raised, not redistributed**
(CL +5.2 % for CDi +2.4 %). Wave 3's endplate-beats-pylon conclusion
**survives and strengthens**: the endplate's span-efficiency advantage grows
from +16.9 % to +19.0 %.

### With the wall present the two orientations are genuinely different

They are no longer mirror images, which is exactly why the image was the
blocker. Lattice at 5°, same panel:

| configuration | CL | CDi | e |
|---|---|---|---|
| free air (no body) | 0.6030 | 0.06680 | 0.9996 |
| **suction INBOARD** — wall on the lift side | **0.6344** | 0.06842 | 1.0802 |
| **suction OUTBOARD** — wall on the pressure side | 0.6267 | **0.06677** | 1.0802 |

So the interference genuinely differs in kind: with the suction surface facing
the body the panel makes **more** lift for **more** induced drag; facing
outboard it makes slightly less lift for essentially free-air induced drag.

### But the answer is still: keep the suction side inboard

**Lift points from the pressure surface to the suction surface.** That is
geometry, not modelling, and it is what settles it: if the suction side faces
outboard then the force points outboard, *away* from the turn centre, and the
device subtracts from cornering instead of adding to it. There is no
configuration in which the suction surface faces outboard and the useful force
is still inboard. The body image changes the **magnitude** of each orientation;
it cannot change that sign.

Vehicle level, R = 100 open-loop ramp steer, both orientations analysed **with
the image**, against 8.6084 m/s² wing-off:

| orientation | CL_dev | F_dev | peak a_y | gain |
|---|---|---|---|---|
| suction inboard (as shipped) | +0.6272 | **+111.75 N** | 8.8489 | **+2.7948 %** |
| suction outboard | −0.5619 | **−100.13 N** | 8.3835 | **−2.6116 %** |

**A swing of 5.4 percentage points, and the default is the right way round.**

### And I checked where it loses before calling it aerodynamics

Three failures in this batch turned out to be the arena's surfaces wearing an
aero costume, so: this is a **steady rig at a single radius with no surface
patch in it**, the loser is front-limited exactly as the winner is
(`util_f` 0.9918 against 0.9935, both `limiting = 'front'`), and the only thing
that differs is the sign of `F_dev`. It is the aerodynamics, and it is
geometric rather than subtle.

**Default unchanged**, as instructed — and in this case unambiguously correct
rather than merely conservative. A selectable flipped orientation was *not*
added: an option whose only effect is to make the device work backwards is a
footgun, not a feature, and the numbers above are the answer he asked for.

### What would change this answer

Only something that makes an outboard force useful — a car that is
**rear**-limited in the corner where the device acts, so that unloading the
front helps. This one is front-limited in every measured configuration
(`limiting = 'front'` throughout), which is the whole premise of the study and
is why the device points inboard.

## What a reader of the earlier notes would now be misled by

* `.handoff/04-wings-audit.md` and anything quoting the device's per-corner
  gain: those numbers are the **uncorrected** AoA. Subtract ~0.1 pp
  (fin +3.86 → +3.74 %, plate +6.05 → +5.94 %) for the corrected model.
* `.handoff/10-lap-reconclusions.md`: its *sign reversal* conclusion stands,
  but its magnitudes are uncorrected. The fin's lap benefit halves.
* Any statement that the baseline "cannot lap without the device" — **withdrawn
  in wave 5**; both failures were inside `WET_T3` and had nothing to do with
  the wing. Three configurations land within 3 ms of each other.
